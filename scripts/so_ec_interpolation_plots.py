import argparse
import numpy as np
import pandas as pd
import torch
import logging

from collections import defaultdict
from matplotlib import pyplot as plt
from pathlib import Path
from torchvision import transforms as T
from tqdm import tqdm

from src.core.models.species_detector import SpeciesDetector
from src.core.constants import LOG_MEL_SPECTROGRAM_PARAMS
from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main(
    model_info_path: Path,
    scores_path: Path,
    save_dir: Path,
    seed_num: int = 0,
    device_id: int = 0,
) -> None:
    device = f"cuda:{device_id}" if device_id is not None else "cpu"
    # load model info
    model_info = pd.read_parquet(model_info_path)
    # filter to SO EC
    df = model_info[(model_info["dataset"] == "SO") & (model_info["scope"] == "country=='EC'")].copy()
    df = df.rename(columns={"model_name": "model_class"})
    df.loc[df["model_class"] == "NIFTI", "delta_t"] = 0.0
    df.loc[df["model_class"] == "Smooth NIFTI", "delta_t"] = 1.0
    # init audio dataset and audio params
    log.info("loading SO EC dataset")
    spectrogram = LogMelSpectrogram(**LOG_MEL_SPECTROGRAM_PARAMS)
    hops_per_second = spectrogram.sample_rate / spectrogram.hop_length
    frame_length_seconds = 192 / hops_per_second
    frame_length_hops = 192
    transforms = T.Compose([spectrogram, T.CenterCrop(size=[7488, 64])])
    data = SoundingOutChorus("~/data/sounding_out", sample_rate=spectrogram.sample_rate, test=False)
    test_data = SoundingOutChorus("~/data/sounding_out", sample_rate=spectrogram.sample_rate, test=True)
    # define the species we want to render examples and generate interpolations for
    log.info("identify the habitat where selected species occur most frequently")
    species_params = [
        { "species_name": "Coereba flaveola_Bananaquit", "file_name": "FS-11_0_20150806_0600.wav", "t_start_seconds": 32.4, "delta": 55 },
        { "species_name": "Poliocrania exsul_Chestnut-backed Antbird", "file_name": "TE-03_0_20150716_0635.wav", "t_start_seconds": 25.2, "delta": 65 },
        { "species_name": "Ramphocelus flammigerus_Flame-rumped Tanager", "file_name": "PO-11_0_20150815_1815.wav", "t_start_seconds": 0.0, "delta": 65 },
        { "species_name": "Leptotila pallida_Pallid Dove", "file_name": "PO-11_0_20150818_0625.wav", "t_start_seconds": 6.5, "delta": 65 },
        { "species_name": "Capsiempis flaveola_Yellow Tyrannulet", "file_name": "PO-03_0_20150815_0640.wav", "t_start_seconds": 16.0, "delta": 65 },
        { "species_name": "Mionectes oleagineus_Ochre-bellied Flycatcher", "file_name": "FS-16_0_20150802_0645.wav", "t_start_seconds": 0.2, "delta": 65 },
        { "species_name": "Microbates cinereiventris_Tawny-faced Gnatwren", "file_name": "TE-06_0_20150716_0645.wav", "t_start_seconds": 33.5, "delta": 65 },
        { "species_name": "Manacus manacus_White-bearded Manakin", "file_name": "FS-04_0_20150802_0650.wav", "t_start_seconds": 41.2, "delta": 65 },
    ]
    # identify the habitat where each species occurs most
    # for each species, we use that habitat average embedding as our background template for interpolation
    for params in species_params:
        counts = data.metadata.merge(
            data.labels.astype(bool).astype(int),
            left_index=True,
            right_index=True
        ).groupby("habitat")[params["species_name"]].sum().reset_index()
        params["habitat"] = counts.loc[counts[params["species_name"]].idxmax()].habitat
    # load final test scores
    log.info("loading scores")
    scores = pd.read_parquet(scores_path).reset_index()
    scores = scores[
        (scores["model_id"].isin(df["model_id"])) &
        (scores["species_name"].isin([s["species_name"] for s in species_params])) &
        (scores["dataset"] == "SO") &
        (scores["scope"] == "country=='EC'")
    ]
    # sort by model class for figure order, load and cache all models
    log.info("loading and caching VAEs and CLFs")
    name_map = {
        "Base": "Classic VAE",
        "NIFTI": "Shift Invariant\nVAE",
        "Smooth NIFTI": "Shift Invariant \&\nTemporally Smooth\nVAE",
    }
    df["model_class"] = pd.Categorical(df["model_class"], categories=name_map.keys(), ordered=True)
    # pick the first 3 models of each class
    df = df.sort_values("model_class").groupby("model_class").nth(seed_num).reset_index()
    vaes = []
    clfs = []
    for i, row in df.iterrows():
        # download the VAE
        vae = load_artefact(row.artefact_id).eval().to(device)
        # load species logistic regression model weights
        checkpoint = torch.load(row.clf_checkpoint_path)
        clf = {
            param.split(".")[1]: checkpoint["state_dict"][param]
            for param in checkpoint["state_dict"].keys()
            if param.startswith("classifiers") and param.endswith("weight")
        }
        vaes.append(vae)
        clfs.append(clf)
    # encode the habitat mean representation, encode each individually and compute the average
    log.info("encoding habitat-wise silent embeddings")
    z_model_habitat_features = defaultdict(lambda: defaultdict(list))
    habitats = data.metadata[data.metadata.country == "EC"].habitat.unique()
    with torch.no_grad():
        for habitat in habitats:
            file_names = data.metadata[data.metadata.habitat == habitat].file_name
            for file_name in tqdm(file_names, total=len(file_names)):
                x = transforms(data.load_sample(file_name))
                for (i, row), vae in zip(df.iterrows(), vaes):
                    z = vae.encode(x.unsqueeze(0).to(device))[0].chunk(2, dim=-1)[0]
                    z_model_habitat_features[habitat][row.model_class].append(z)
    z_model_habitat_means = {}
    for habitat in habitats:
        z_model_habitat_means[habitat] = {}
        for i, row in df.iterrows():
            z_mean = torch.cat(z_model_habitat_features[habitat][row.model_class]).mean(dim=[0, 1], keepdims=True)
            z_model_habitat_means[habitat].update({row.model_class: z_mean})
    log.info("building plot, rendering spectrograms and interpolated reconstructions")
    # plot spectrograms and interpolated reconstructions
    vmax, vmin = 0.0, -80
    fig, axes = plt.subplots(
        nrows=len(df) + 1,
        ncols=len(species_params) + 1,
        figsize=(2 * len(species_params), 1.5 * (len(df) + 1)),
        height_ratios=[*[1.0 / (len(vaes) + 1) for _ in range(len(vaes) + 1)]],
        width_ratios=[0.01, *[0.95 / (len(species_params)) for _ in range(len(species_params))]],
        constrained_layout=True,
    )
    with torch.no_grad():
        for i, params in enumerate(species_params):
            species_name, file_name, t_start_seconds, delta, habitat = params.values()
            if i == 0:
                title_ax = axes[0, 0]
                title_ax.set_ylabel("Real\nExample")
                make_ax_invisible(title_ax)
            ax = axes[0, i + 1]
            t_start = int(t_start_seconds * hops_per_second)
            t_end = t_start + int(frame_length_seconds * hops_per_second)
            # some of the examples are in the test set
            try:
                wav = data.load_sample(file_name)
            except:
                wav = test_data.load_sample(file_name)
            x = transforms(wav).squeeze()
            x = 20 * np.log10(x[t_start:t_end].exp())
            # plot the original
            plot_mel_spectrogram(
                x.squeeze().t(),
                **LOG_MEL_SPECTROGRAM_PARAMS,
                vmax=vmax,
                vmin=vmin,
                ax=ax,
                cmap="Greys"
            )
            ax.set_xticks([0, 191], [0.0, 1.536], rotation=90)
            ax.set_title("\n".join(species_name.split("_")))
            ax.tick_params(labelbottom=False, bottom=False)
            ax.set_xlabel("")
            if i != 0:
                ax.tick_params(labelleft=False, left=False)
                ax.set_ylabel("")
            for j, model_class in enumerate(["Base", "NIFTI", "Smooth NIFTI"]):
                row = df[df["model_class"] == model_class].iloc[0]
                vae, clf = vaes[j], clfs[j]
                if i == 0:
                    title_ax = axes[j + 1, 0]
                    title_ax.set_ylabel(name_map[row.model_class])
                    make_ax_invisible(title_ax)
                # load scores for this species and mode;
                model_species_scores = scores[
                    (scores["model_class"] == row["model_class"]) &
                    (scores["model_id"] == row["model_id"]) &
                    (scores["species_name"] == species_name)
                ].iloc[0]
                # fetch habitat silent embedding
                z = z_model_habitat_means[habitat][row.model_class]
                # fetch weights of log reg model
                log.info(f"generating {species_name} with {row.model_class}")
                W = clf[species_name]
                # linear interpolation across the hyperplane by delta
                # delta needs to be tuned per species
                norm = torch.linalg.norm(W)
                z_tilde = z + ((z @ W.T / norm) + delta) * (W / norm)
                # decode using VAE with alignment factor dt
                if row.model_class == "Base":
                    x_tilde = vae.decode(z_tilde).cpu()
                else:
                    # dt is tunable by VAE
                    x_tilde = vae.decode(z_tilde, torch.ones(1, 1, 1, device=z.device) * row.delta_t).cpu()
                # map to decibels
                x_tilde_db = 20 * np.log10(x_tilde.exp())
                ax = axes[j + 1, i + 1]
                # plot reconstruction
                plot_mel_spectrogram(
                    x_tilde_db.squeeze().t(),
                    **LOG_MEL_SPECTROGRAM_PARAMS,
                    vmax=vmax,
                    vmin=vmin,
                    cmap="Greys",
                    ax=ax
                )
                ax.set_xticks([0, 191], [0.0, 1.536], rotation=90)
                if j != len(df) - 1:
                    ax.tick_params(labelbottom=False, bottom=False)
                    ax.set_xlabel("")
                if i != 0:
                    ax.tick_params(labelleft=False, left=False)
                    ax.set_ylabel("")
                AP = np.format_float_positional(model_species_scores['mAP'], precision=2)
                auROC = np.format_float_positional(model_species_scores['auROC'], precision=2)
                ax.set_title(f"AP: {AP}, auROC: {auROC}")
    fig.suptitle("SO EC")
    fig.savefig(save_dir / f"so_ec_{seed_num}_interpolation_w_scores.pdf", format="pdf", bbox_inches="tight")
    log.info(f"figure saved to {(save_dir / f'so_ec_{seed_num}_interpolation_w_scores.pdf').expanduser()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-info-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--seed-num",
        type=int,
        default=0,
        help="model number",
    )
    args = parser.parse_args()
    main(**vars(args))
