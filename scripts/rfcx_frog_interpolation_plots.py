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
from src.core.data.rainforest_connection import RainforestConnection
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils import tree
from src.cli.utils.instantiators import instantiate_transforms

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def wrap_sentence(sentence, max_length):
    words = sentence.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + (1 if current_line else 0) <= max_length:
            if current_line:
                current_line += " "
            current_line += word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'legend.fontsize': 13,
})

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
    df = index[index["scope"] == "RFCX_frog"].copy()
    df.loc[df["model_name"] == "nifti_vae", "delta_t"] = 1.0
    df.loc[df["model_name"] == "smooth_nifti_vae", "delta_t"] = 0.0
    # init audio dataset and audio params
    log.info("loading RFCX frog dataset")
    with open(rootutils.find_root() / "config" / "transforms" / "cropped_log_mel_spectrogram.yaml", "r") as f:
        transform_conf = yaml.safe_load(f.read())
        transforms = instantiate_transforms(transform_conf)
        log_mel_spectrogram_params = transform_conf["log_mel_spectrogram"]
        del log_mel_spectrogram_params["_target_"]
    spectrogram = transforms.transforms[0]
    hops_per_second = spectrogram.sample_rate / spectrogram.hop_length
    frame_length_seconds = 192 / hops_per_second
    frame_length_hops = 192
    data = RainforestConnection("~/data/rainforest_connection", sample_rate=spectrogram.sample_rate, test=False)
    # define the species we want to render examples and generate interpolations for
    species_params = [
        {'species_name': 'Eleutherodactylus antillensis_Red-eyed coquí', 'file_name': 'train/0313e82cf.flac', 't_start_seconds': 21.472, "delta": 10},
        {'species_name': 'Eleutherodactylus brittoni_Grass coquí', 'file_name': 'train/13c678c1d.flac', 't_start_seconds': 0.864, "delta": 10},
        {'species_name': 'Eleutherodactylus coqui_Common coquí', 'file_name': 'train/06c44d203.flac', 't_start_seconds': 1.28, "delta": 10},
        {'species_name': 'Eleutherodactylus gryllus_Cricket coquí', 'file_name': 'train/157a50231.flac', 't_start_seconds': 35.7867, "delta": 10},
        {'species_name': "Eleutherodactylus hedricki_Hedrick's coquí", 'file_name': 'train/251569711.flac', 't_start_seconds': 18.2133, "delta": 10},
        {'species_name': 'Eleutherodactylus locustus_Locust coquí', 'file_name': 'train/2bf32cf03.flac', 't_start_seconds': 21.0347, "delta": 10},
        {'species_name': 'Eleutherodactylus portoricensis_Forest Coqui', 'file_name': 'train/068f1b8e2.flac', 't_start_seconds': 22.864, "delta": 10},
        # {'species_name': 'Eleutherodactylus richmondi_Bronze coquí', 'file_name': 'train/055088446.flac', 't_start_seconds': 58.992, "delta": 20},
        # {'species_name': 'Eleutherodactylus unicolor_Dwarf coquí', 'file_name': 'train/10dae79ed.flac', 't_start_seconds': 30.192, "delta": 20},
        # {'species_name': 'Eleutherodactylus wightmanae_Melodius coquí', 'file_name': 'train/0c2124550.flac', 't_start_seconds': 18.2293, "delta": 20},
        # {'species_name': 'Leptodactylus albilabris_Caribbean White-lipped Frog', 'file_name': 'train/05b9c974c.flac', 't_start_seconds': 31.984, "delta": 20},
    ]
    # load final test scores
    log.info("loading scores")
    scores = pd.read_parquet(scores_path).reset_index()
    scores = scores[
        (scores["species_name"].isin([s["species_name"] for s in species_params])) &
        (scores["scope"] == "RFCX_frog")
    ]
    # sort by model class for figure order, load and cache all models
    log.info("loading and caching VAEs and CLFs")
    name_map = {
        "base_vae": "VAE",
        "nifti_vae": "SIVAE",
        # "smooth_nifti_vae": "TSSIVAE",
    }
    df["model_class"] = df["model_name"].map(name_map)
    df["model_class"] = pd.Categorical(df["model_class"], categories=name_map.values(), ordered=True)
    df = df.sort_values("model_class").groupby("model_class").nth(seed_num).reset_index()
    df = df[df["model_name"].isin(["base_vae", "nifti_vae"])]

    vaes = []
    clfs = []
    z_model_means = defaultdict(tree)
    for i, row in df.iterrows():
        # load the pretrained VAE
        with open(rootutils.find_root() / "config" / "model" / f"{row.model_name}.yaml", "r") as f:
            model_conf = yaml.safe_load(f.read())
            vae = hydra.utils.instantiate(model_conf)
        checkpoint = torch.load(row.vae_checkpoint_path, map_location=device)
        vae.load_state_dict(checkpoint["model_state_dict"])
        vaes.append(vae.to(device))
        log.info(f"Loaded {row.model_name} from {row.vae_checkpoint_path}")
        # load species logistic regression model weights
        checkpoint = torch.load(row["clf_checkpoint_path"], map_location=device)
        clf = {
            param.split(".")[1]: checkpoint["state_dict"][param]
            for param in checkpoint["state_dict"].keys()
            if param.startswith("classifiers") and param.endswith("weight")
        }
        clfs.append(clf)
        # compute the model average embedding
        dm = SoundscapeEmbeddingsDataModule(
            root=data_dir,
            model=row.model_name,
            version=row.version,
            scope="RFCX_frog",
            transforms=None, # FIXME this shouldnt be required
        )
        dm.setup()
        # encode the mean representation for this model
        z_mean = dm.train_data.features.iloc[:, range(128)].mean()
        z_model_means[row.model_name] = torch.tensor(z_mean, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    log.info("building plot, rendering spectrograms and interpolated reconstructions")
    # plot spectrograms and interpolated reconstructions
    vmax, vmin = 0.0, -80
    fig, axes = plt.subplots(
        nrows=len(df) + 1,
        ncols=len(species_params) + 1,
        figsize=(2 * len(species_params), 2.5 * (len(df) + 1)),
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
            x = transforms(data.load_sample(data.base_dir / file_name)).squeeze()
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
            ax.set_title(wrap_sentence(species_name.replace("_", ", "), 18))
            ax.tick_params(labelbottom=False, bottom=False)
            ax.set_xlabel("")
            if i != 0:
                ax.tick_params(labelleft=False, left=False)
                ax.set_ylabel("")
            for j, row in df.iterrows():
                vae, clf = vaes[j], clfs[j]
                if i == 0:
                    title_ax = axes[j + 1, 0]
                    title_ax.set_ylabel(name_map[row.model_name])
                    make_ax_invisible(title_ax)
                # load scores for this species and mode;
                model_species_scores = scores[
                    (scores["model"] == row["model_name"]) &
                    (scores["version"] == row["version"]) &
                    (scores["species_name"] == species_name)
                ].iloc[0]
                # fetch silent embedding
                z = z_model_means[row.model_name].to(device)
                # fetch weights of log reg model
                log.info(f"generating {species_name} with {row.model_name}:{row.version}")
                W = clf[species_name].to(device)
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
                AP = np.format_float_positional(model_species_scores['AP'], precision=2)
                auROC = np.format_float_positional(model_species_scores['auROC'], precision=2)
                ax.set_title(f"AP: {AP}\nauROC: {auROC}")
    fig.suptitle("RFCX frog")
    fig.savefig(save_dir / f"rfcx_frog_{seed_num}_interpolation_w_scores.pdf", format="pdf", bbox_inches="tight")
    log.info(f"figure saved to {(save_dir / f'rfcx_frog_{seed_num}_interpolation_w_scores.pdf').expanduser()}")

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
