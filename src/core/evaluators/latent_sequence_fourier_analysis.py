import abc
import torch
import pathlib
import hydra
import itertools
import logging
import lightning as L
import numpy as np
import matplotlib as mpl
import seaborn as sns
import tqdm
import pandas as pd

from pathlib import Path
from torch.functional import F
from matplotlib import pyplot as plt
from torchvision import transforms as T
from typing import Any, List, Dict, Tuple

from src.core.utils.sketch import plot_mel_spectrogram, plot_latent_power_spectral_density_heatmap, multiline
from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["LatentSequenceFourierAnalysis"]

class LatentSequenceFourierAnalysis(Evaluator):
    def __init__(self, results_dir: str | pathlib.Path) -> None:
        super().__init__()
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def __call__(self, trainer, model, data_module, config) -> None:
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path is not None, f"No checkpoint found at {ckpt_path}"

        log.info(f"Encoding <{config.data.get('_target_')}> with <{config.model.get('_target_')}>")
        predictions = trainer.predict(model, data_module.predict_dataloader(), ckpt_path=config.get("ckpt_path"), return_predictions=True)
        df = pd.concat(list(itertools.chain(*predictions)), axis=0)
        df = df.reset_index().merge(
            data_module.data.metadata.file_name,
            left_on="file_i",
            right_on="file_i",
            how="left"
        ).set_index(df.index.names)

        log.info(f"Computing STFT over each latent dimension")

        seq_len = df.index.get_level_values("timestep").max() + 1
        metadata = data_module.data.metadata

        zs = df.loc[:, [f"z_mean_{i}" for i in range(model.latent_dim)]].reshape(-1, int(seq_len), model.latent_dim)
        delta = df.loc[:, "delta"].values.reshape(-1, int(seq_len), 1)

        zs = torch.as_tensor(mu, device=model.device, dtype=torch.float32)
        delta = torch.as_tensor(delta, device=model.device, dtype=torch.float32)

        with tqdm.tqdm(total=mu.size(0)) as pbar:
            for file_name, z in zip(df.file_name.unique(), zs):
                fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 8), width_ratios=[0.97, 0.03], constrained_layout=True)
                row = metadata[metadata.file_name == file_name].iloc[0]
                x = data_module.transforms(data_module.data.load_sample(row.file_path)).squeeze()
                x_hat = model.decode(x.unsqueeze(0), delta[i].unsqueeze(0)).detach().squeeze().T

                mesh = plot_mel_spectrogram(x.T, **model.spectrogram_params, vmin=x.min(), vmax=x.max(), ax=axes[0, 0])
                fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
                axes[1, 0].set_title("Original")

                mesh = plot_mel_spectrogram(x_hat, **model.spectrogram_params, ax=axes[1, 0])
                fig.colorbar(mesh, cax=axes[1, 1], orientation="vertical")
                axes[1, 0].set_title("Reconstruction")

                ts = torch.arange(x.size(0)).repeat(x.size(1)),
                z_norm = ((z - z.mean(axis=0)) / z.std(axis=0)).t(),
                hs = torch.cat([torch.ones(64) * 1e-1, torch.linspace(1e-1, 39/3, 64)])
                lc = multiline(ts, z_norm, hs, ax=axes[2, 0], cmap='jet', lw=2, alpha=0.75)
                cbar = fig.colorbar(lc, cax=axes[2, 1])
                cbar.set_label("Kernel Bandwidth (h)", rotation=90)
                axes[2, 0].set_xlabel("Time ($t$)")
                axes[2, 0].set_ylabel(r"$\mathbf{z}_{\text{norm}}(t)$")
                axes[2, 0].set_title("Latent Time-series by Kernel Smoothness")

                spec_params = dict(audio_sample_rate=model.sample_rate, audio_fft_hop_length=model.fft_hop_length, audio_frame_length_hops=model.frame_window_length)
                mesh = plot_latent_power_spectral_density_heatmap(z, fft_length=seq_len, **spec_params, ax=axes[3, 0], cbar=False)
                fig.colorbar(mesh, cax=axes[3, 1], orientation="vertical")
                axes[0 + 2, 0].set_title("Latent Power Spectral Density")

                fig.suptitle("Spectral Analysis of Latent Time-series")
                fig.savefig(self.results_dir / f"{file_name}.png")
                plt.close()
                pbar.update(1)
        log.info(f"Saved to {str(self.results_dir)}")
