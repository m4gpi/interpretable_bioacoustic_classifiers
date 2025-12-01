import argparse
import hydra
import numpy as np
import pandas as pd
import torch
import rootutils
import logging
import warnings
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
from matplotlib import pyplot as plt
from pathlib import Path
from torchvision import transforms as T
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.data.soundscape_embeddings import SoundscapeEmbeddingsDataModule
from src.core.models.species_detector import SpeciesDetector
from src.core.models.base_vae import BaseVAE
from src.core.models.smooth_nifti_vae import SmoothNiftiVAE
from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils import tree
from src.cli.utils.instantiators import instantiate_transforms

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODELS = dict(
    smooth_nifti_vae=SmoothNiftiVAE,
    nifti_vae=SmoothNiftiVAE, # same model, just with smooth_prop set to 0
    base_vae=BaseVAE,
)

def main(
    data_dir: Path,
    scores_path: Path,
    save_dir: Path,
    seed_num: int = 0,
    device_id: int = 0,
) -> None:
    device = f"cuda:{device_id}" if device_id is not None else "cpu"
    # load model info
    index = pd.read_parquet(data_dir / "index.parquet")
    # filter to SO EC
    df = index[index["scope"] == "SO_EC"].copy()
    # FIXME: check temporal shift parameter...
    df.loc[df["model_name"] == "nifti_vae", "delta_t"] = 0.0
    df.loc[df["model_name"] == "smooth_nifti_vae", "delta_t"] = 1.0
    # init audio dataset and audio params
    log.info("loading SO EC dataset")
    with open(rootutils.find_root() / "config" / "transforms" / "cropped_log_mel_spectrogram.yaml", "r") as f:
        transform_conf = yaml.safe_load(f.read())
        transforms = instantiate_transforms(transform_conf)
        log_mel_spectrogram_params = transform_conf["log_mel_spectrogram"]
        del log_mel_spectrogram_params["_target_"]
    spectrogram = transforms.transforms[0]
    hops_per_second = spectrogram.sample_rate / spectrogram.hop_length
    frame_length_seconds = 192 / hops_per_second
    frame_length_hops = 192
    data = SoundingOutChorus("~/data/sounding_out", sample_rate=spectrogram.sample_rate, test=False)
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
        (scores["species_name"].isin([s["species_name"] for s in species_params])) &
        (scores["scope"] == "SO_EC")
    ]
    # sort by model class for figure order, load and cache all models
    log.info("loading and caching VAEs and CLFs")
    name_map = {
        "base_vae": "VAE",
        "nifti_vae": "SIVAE",
        "smooth_nifti_vae": "TSSIVAE",
    }
    df["model_class"] = df["model_name"].map(name_map)
    df["model_class"] = pd.Categorical(df["model_class"], categories=name_map.values(), ordered=True)
    df = df.sort_values("model_class").groupby("model_class").nth(seed_num).reset_index()

    vaes = []
    clfs = []
    z_model_habitat_means = defaultdict(tree)
    for i, row in df.iterrows():
        # load the pretrained VAE
        with open(rootutils.find_root() / "config" / "model" / f"{row.model_name}.yaml", "r") as f:
            model_conf = yaml.safe_load(f.read())
            vae = hydra.utils.instantiate(model_conf)
        checkpoint = torch.load(data_dir / row.vae_checkpoint_path)
        vae.load_state_dict(checkpoint["model_state_dict"])
        vaes.append(vae)
        log.info(f"Loaded {row.model_name} from {row.vae_checkpoint_path}")
        # load species logistic regression model weights
        # checkpoint = torch.load(row["clf_checkpoint_path"])
        # clf = {
        #     param.split(".")[1]: checkpoint["state_dict"][param]
        #     for param in checkpoint["state_dict"].keys()
        #     if param.startswith("classifiers") and param.endswith("weight")
        # }
        # clfs.append(clf)
        # compute the habitat model average embedding
        dm = SoundscapeEmbeddingsDataModule(
            root=data_dir,
            model=row.model_name,
            version=row.version,
            scope="SO_EC",
            transforms=None, # FIXME this shouldnt be required
        )
        dm.setup()
        # FIXME hack in the habitat labels using the file name
        embeddings = dm.data
        embeddings.labels = embeddings.labels.reset_index()
        embeddings.labels["habitat"] = embeddings.labels.file_name.str.split("-", expand=True)[0]
        embeddings.labels.set_index(["file_i", "file_name", "country", "habitat"])
        # encode the habitat mean representation for this model
        z_mean = (
            embeddings
            .features
            .iloc[:, range(128)]
            .merge(embeddings.labels[["file_i", "habitat"]], on="file_i", how="left")
            .drop("file_i", axis=1)
            .groupby("habitat")
            .mean()
        )
        for habitat in z_mean.index:
            z_model_habitat_mean = torch.tensor(z_mean.loc[habitat])
            z_model_habitat_means[habitat][row.model_name] = z_model_habitat_mean.unsqueeze(0).unsqueeze(0)

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
            x = transforms(data.load_sample(file_name)).squeeze()
            x = 20 * np.log10(x[t_start:t_end].exp())
            # plot the original
            plot_mel_spectrogram(
                x.squeeze().t(),
                **log_mel_spectrogram_params,
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
            for j, row in df.iterrows():
                vae, clf = vaes[j], clfs[j]
                if i == 0:
                    title_ax = axes[j + 1, 0]
                    title_ax.set_ylabel(name_map[row.model_class])
                    make_ax_invisible(title_ax)
                # load scores for this species and mode;
                model_species_scores = scores[
                    (scores["model"] == row["model_name"]) &
                    (scores["version"] == row["version"]) &
                    (scores["species_name"] == species_name)
                ].iloc[0]
                # fetch habitat silent embedding
                z = z_model_habitat_means[habitat][row.model_class]
                # fetch weights of log reg model
                log.info(f"generating {species_name} with {row.model_name}:{row.version}")
                W = clf[species_name]
                # linear interpolation across the hyperplane by delta
                # delta needs to be tuned per species
                norm = torch.linalg.norm(W)
                z_tilde = z + ((z @ W.T / norm) + delta) * (W / norm)
                # decode using VAE with alignment factor dt
                if row.model_name == "base_vae":
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
                    **log_mel_spectrogram_params,
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
        "--data-dir",
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
        required=False,
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
