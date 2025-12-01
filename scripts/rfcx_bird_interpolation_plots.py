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
from src.core.data.rainforest_connection import RainforestConnection
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
    # filter to RFCX EC
    df = model_info[(model_info["dataset"] == "RFCX") & (model_info["scope"] == "taxa=='bird'")].copy()
    df = df.rename(columns={"model_name": "model_class"})
    df.loc[df["model_class"] == "NIFTI", "delta_t"] = 0.0
    df.loc[df["model_class"] == "Smooth NIFTI", "delta_t"] = 1.0
    # init audio dataset and audio params
    log.info("loading RFCX birds dataset")
    spectrogram = LogMelSpectrogram(**LOG_MEL_SPECTROGRAM_PARAMS)
    hops_per_second = spectrogram.sample_rate / spectrogram.hop_length
    frame_length_seconds = 192 / hops_per_second
    frame_length_hops = 192
    transforms = T.Compose([spectrogram, T.CenterCrop(size=[7488, 64])])
    data = RainforestConnection("~/data/rainforest_connection", sample_rate=spectrogram.sample_rate, test=False)
    test_data = RainforestConnection("~/data/rainforest_connection", sample_rate=spectrogram.sample_rate, test=True)
    # define the species we want to render examples and generate interpolations for
    species_params = [
        { "species_name": "Coereba flaveola_Bananaquit", "file_name": "0a9cdd8a5.flac", "t_start_seconds": 49.2, "delta": 50 },
        { "species_name": "Turdus plumbeus_Red-legged Thrush", "file_name": "9cb1f4a34.flac", "t_start_seconds": 28.8, "delta": 50 },
        { "species_name": "Setophaga angelae_Elfin Woods Warbler", "file_name": "88b5c9c1b.flac", "t_start_seconds": 40.5, "delta": 40 },
        { "species_name": "Vireo altiloquus_Black-whiskered Vireo", "file_name": "c4b778e64.flac", "t_start_seconds": 28.25, "delta": 50 },
        { "species_name": "Patagioenas squamosa_Scaly-naped Pigeon", "file_name": "21e2f2977.flac", "t_start_seconds": 23, "delta": 40 },
        { "species_name": "Nesospingus speculiferus_Puerto Rican Tanager", "file_name": "1702d35a0.flac", "t_start_seconds": 29.1, "delta": 40 },
        { "species_name": "Spindalis portoricensis_Puerto Rican Spindalis", "file_name": "16553d5cd.flac", "t_start_seconds": 33.5, "delta": 50 },
        { "species_name": "Melanerpes portoricensis_Puerto Rican Woodpecker", "file_name": "745171bf2.flac", "t_start_seconds": 4.5, "delta": 50 },
    ]
    # load final test scores
    log.info("loading scores")
    scores = pd.read_parquet(scores_path).reset_index()
    scores = scores[
        (scores["model_id"].isin(df["model_id"])) &
        (scores["species_name"].isin([s["species_name"] for s in species_params])) &
        (scores["dataset"] == "RFCX") &
        (scores["scope"] == "taxa=='bird'")
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
    # encode the mean representation
    log.info("encoding silent embeddings")
    z_model_features = defaultdict(list)
    with torch.no_grad():
        file_names = data.metadata.file_name
        for file_name in tqdm(file_names, total=len(file_names)):
            x = transforms(data.load_sample(file_name))
            for (i, row), vae in zip(df.iterrows(), vaes):
                z = vae.encode(x.unsqueeze(0).to(device))[0].chunk(2, dim=-1)[0]
                z_model_features[row.model_class].append(z)
    z_model_means = {}
    for i, row in df.iterrows():
        z_mean = torch.cat(z_model_features[row.model_class]).mean(dim=[0, 1], keepdims=True)
        z_model_means.update({row.model_class: z_mean})
    # log.info("building plot, rendering spectrograms and interpolated reconstructions")
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
            species_name, file_name, t_start_seconds, delta = params.values()
            if i == 0:
                title_ax = axes[0, 0]
                title_ax.set_ylabel("Real\nExample")
                make_ax_invisible(title_ax)
            ax = axes[0, i + 1]
            t_start = int(t_start_seconds * hops_per_second)
            t_end = t_start + int(frame_length_seconds * hops_per_second)
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
                # fetch silent embedding
                z = z_model_means[row.model_class]
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
    fig.suptitle("RFCX Birds")
    fig.savefig(save_dir / f"rfcx_birds_{seed_num}_interpolation_w_scores.pdf", format="pdf", bbox_inches="tight")
    log.info(f"figure saved to {(save_dir / f'rfcx_birds_{seed_num}_interpolation_w_scores.pdf').expanduser()}")

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

