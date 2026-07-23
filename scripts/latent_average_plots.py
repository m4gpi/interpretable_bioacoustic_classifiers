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

from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.data.soundscape_embeddings import SoundscapeEmbeddingsDataModule
from src.core.models.species_detector import SpeciesDetector
from src.core.models.vae import VAE
from src.core.models.sivae import SIVAE
from src.core.data.rainforest_connection import RainforestConnection
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils import tree
from src.cli.utils.instantiators import instantiate_transforms

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

@torch.no_grad()
def main(
    data_dir: pathlib.Path,
    model_dir: pathlib.Path,
    save_dir: pathlib.Path,
    format: str = "pdf",
    device_id: int = 0,
) -> None:
    device = f"cuda:{device_id}" if device_id is not None else "cpu"

    so_model = "dynamic-malta" # "just-drum"
    rfcx_model = "part-armor" # "earthy-vigor"
    model_map = {
        "just-drum": "SIVAE",
        # "dynamic-malta": "SIVAE",
        # "daring-system": "SIVAE",
        "earthy-virgo": "SIVAE",
        # "part-armor": "SIVAE",
        # "secluded-montana": "SIVAE",
        # "lumpy-gibson": "VAE",
        # "slow-partner": "VAE",
        # "unique-tiger": "VAE",
        # "jumpy-engine": "VAE",
        # "quaint-pilot": "VAE",
        # "numb-chef": "VAE",
    }

    specs = []

    habitat_map = {"PL": "UK1", "KN": "UK2", "BA": "UK3", "TE": "EC1", "FS": "EC2", "PO": "EC3"}
    for scope, model, version in [("SO_UK", "sivae", so_model), ("SO_EC", "sivae", so_model)]:
        vae = SIVAE.load_from_checkpoint(model_dir / model / version / "step=180000.ckpt", map_location="cuda")
        data = SoundingOutChorus(root=data_dir / "sounding_out", test=False)
        features = SoundscapeEmbeddingsDataModule(root=data_dir / "soundscape_vae_embeddings" / version / scope).setup().train_data.features
        z_mean = features.loc[:, [f"z_mean_{i}" for i in range(128)]].reset_index().merge(data.metadata.reset_index()[["file_i", "habitat"]], on=["file_i"]).drop(["file_i", "dataloader_idx", "timestep"], axis=1).groupby("habitat").mean()
        for habitat, z0 in z_mean.iterrows():
            z0 = torch.tensor(z0.to_numpy(), dtype=torch.float32, device=device).reshape(1, 1, -1)
            x_hat = vae.decode(z0, torch.zeros(1, device=device).view(1, 1, 1))
            x_hat = 20 * np.log10(x_hat.exp().cpu())
            specs.append((f"{scope.split('_')[0]} {habitat}", x_hat))

    scope, model, version = ("RFCX_bird", "sivae", rfcx_model)
    vae = SIVAE.load_from_checkpoint(model_dir / model / version / "step=180000.ckpt", map_location="cuda")
    data = SoundingOutChorus(root=data_dir / "sounding_out", test=False)
    features = SoundscapeEmbeddingsDataModule(root=data_dir / "soundscape_vae_embeddings" / version / scope).setup().train_data.features
    z0 = features.loc[:, [f"z_mean_{i}" for i in range(128)]].mean()
    z0 = torch.tensor(z0.to_numpy(), dtype=torch.float32, device=device).reshape(1, 1, -1)
    x_hat = vae.decode(z0, torch.zeros(1, device=device).view(1, 1, 1))
    x_hat = 20 * np.log10(x_hat.exp().cpu())
    specs.append(("RFCX", x_hat))

    fig, axes = plt.subplots(ncols=len(specs), figsize=(8.1, 1.5), constrained_layout=True)
    vmax, vmin = 10.0, -60
    for i, (ax, (title, x_hat_db)) in enumerate(zip(axes, specs)):
        vae.front_end.plot(x_hat_db.squeeze().t(), vmax=vmax, vmin=vmin, cmap="Greys", ax=ax)
        if i != 0:
            ax.set_yticks([])
            ax.set_ylabel("")
        if i != 2:
            ax.set_xlabel("")
        ax.set_xticks([0, 191], [0, 1.536])
        ax.set_title(title)

    if save_dir is not None:
        save_file = save_dir / f"latent_averages.pdf"
        log.info(f"Saved plot to {save_file}")
        fig.savefig(save_file, format="pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/data/",
    )
    parser.add_argument(
        "--model-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/checkpoints/",
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
