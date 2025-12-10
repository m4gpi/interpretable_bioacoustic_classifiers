import argparse
import hydra
import numpy as np
import pandas as pd
import pathlib
import torch
import rootutils
import seaborn as sns
import logging
import warnings
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib import colors as mcolors
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

def get_vae(model_dict):
    with open(rootutils.find_root() / "config" / "model" / f"{model_dict['model_name']}.yaml", "r") as f:
        model_conf = yaml.safe_load(f.read())
        vae = hydra.utils.instantiate(model_conf)
    checkpoint = torch.load(model_dict["vae_checkpoint_path"], map_location="cpu")
    vae.load_state_dict(checkpoint["model_state_dict"])
    log.info(f"Loaded {model_dict['model_name']} from {model_dict['vae_checkpoint_path']}")
    return vae

def get_transforms():
    with open(rootutils.find_root() / "config" / "transforms" / "cropped_log_mel_spectrogram.yaml", "r") as f:
        transform_conf = yaml.safe_load(f.read())
        transforms = instantiate_transforms(transform_conf)
        log_mel_spectrogram_params = transform_conf["log_mel_spectrogram"]
        del log_mel_spectrogram_params["_target_"]
    return transforms

plt.rcParams.update({
    'axes.labelsize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
})

@torch.no_grad()
def main(
    data_dir: pathlib.Path,
    save_dir: pathlib.Path,
    format: str = "pdf",
    device_id: int = 0,
) -> None:
    device = f"cuda:{device_id}" if device_id is not None else "cpu"

    df = pd.read_parquet(data_dir / "index.parquet")

    transforms = get_transforms()
    spectrogram = transforms.transforms[0]
    spectrogram_params = dict(
        hop_length=spectrogram.hop_length,
        sample_rate=spectrogram.sample_rate,
        mel_min_hertz=spectrogram.mel_min_hertz,
        mel_max_hertz=spectrogram.mel_max_hertz,
        mel_scaling_factor=spectrogram.mel_scaling_factor,
        mel_break_frequency=spectrogram.mel_break_frequency,
    )

    specs = []

    habitat_map = {"PL": "UK1", "KN": "UK2", "BA": "UK3", "TE": "EC1", "FS": "EC2", "PO": "EC3"}
    for scope, model, version in [("SO_UK", "nifti_vae", "v12"), ("SO_EC", "nifti_vae", "v12")]:
        model_dict = df[(df.model_name == model) & (df.version == version) & (df.scope == scope)].iloc[0].to_dict()
        vae = get_vae(model_dict).to(device).eval()
        dm = SoundscapeEmbeddingsDataModule(root=data_dir, model=model, version=version, scope=scope)
        dm.setup()
        embeddings = dm.train_data
        embeddings.labels = embeddings.labels.reset_index()
        embeddings.labels["habitat"] = embeddings.labels.file_name.str.split("-", expand=True)[0]
        embeddings.labels["habitat"] = embeddings.labels.habitat.map(habitat_map)
        embeddings.labels.set_index(["file_i", "file_name", "country", "habitat"])
        z_mean = (
            embeddings
            .features
            .iloc[:, range(128)]
            .merge(embeddings.labels[["file_i", "habitat"]], on="file_i", how="left")
            .drop("file_i", axis=1)
            .groupby("habitat")
            .mean()
        )
        for habitat, z0 in z_mean.iterrows():
            z0 = torch.tensor(z0.to_numpy(), dtype=torch.float32, device=device).reshape(1, 1, -1)
            x_hat = vae.decode(z0)
            x_hat = 20 * np.log10(x_hat.exp().cpu())
            specs.append((f"{scope.split('_')[0]} {habitat}", x_hat))

    for scope, model, version in [("RFCX_bird", "nifti_vae", "v16"), ("RFCX_frog", "nifti_vae", "v16")]:
        model_dict = df[(df.model_name == model) & (df.version == version) & (df.scope == scope)].iloc[0].to_dict()
        vae = get_vae(model_dict).to(device)
        dm = SoundscapeEmbeddingsDataModule(root=data_dir, model=model, version=version, scope=scope)
        dm.setup()
        embeddings = dm.train_data
        z0 = embeddings.features.iloc[:, :128].mean()
        z0 = torch.tensor(z0.to_numpy(), dtype=torch.float32, device=device).reshape(1, 1, -1)
        x_hat = vae.decode(z0)
        x_hat = 20 * np.log10(x_hat.exp().cpu())
        specs.append((scope.replace("_", " "), x_hat))

    fig, axes = plt.subplots(ncols=len(specs), figsize=(8.1, 1.5), constrained_layout=True)
    vmax, vmin = 0.0, -80
    for i, (ax, (title, x_hat_db)) in enumerate(zip(axes, specs)):
        plot_mel_spectrogram(
            x_hat_db.squeeze().t(),
            **spectrogram_params,
            vmax=vmax,
            vmin=vmin,
            cmap="Greys",
            ax=ax
        )
        if i != 0:
            ax.set_yticks([])
            ax.set_ylabel("")
        if i != 2:
            ax.set_xlabel("")
        ax.set_xticks([0, 191], [0, 1.536])
        ax.set_title(title)

    save_file = save_dir / f"latent_averages.pdf"
    print(save_file)
    fig.savefig(save_file, format="pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
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
        "--format",
        type=str,
        default="pdf",
        required=False,
        help="file format (pdf, png, jpg, etc)",
    )
    args = parser.parse_args()
    main(**vars(args))
