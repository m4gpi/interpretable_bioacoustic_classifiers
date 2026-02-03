import lightning as L
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from matplotlib import pyplot as plt
from typing import Any, Dict, List, Tuple

from src.core.models.ear import EAR
from src.core.models.soundscape_generator import SoundscapeGenerator
from src.core.utils import metrics
from src.core.utils.sketch import plot_mel_spectrogram

plt.switch_backend('agg')

__all__ = ["VQVAESequenceDecoder"]

class VQVAESequenceDecoder(L.Callback):
    def __init__(
        self,
        ckpt_path: str,
        num_per_batch: int = 6,
    ) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self.num_per_batch = num_per_batch
        self.model = EAR.load_from_checkpoint(ckpt_path)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # VQ-VAE reconstructions
        encoding_idx = batch.x[:self.num_per_batch]
        z_q, _ = self.model.quantise.quantise(encoding_idx.flatten().unsqueeze(-1))
        z_q = z_q.view(self.num_per_batch, -1, self.model.latent_dim)
        xs = self.model.decode(z_q).cpu().squeeze(1)
        # AR reconstruction approximations
        encoding_idx = torch.stack([torch.multinomial(outputs["x_probs"][i], num_samples=1) for i in range(self.num_per_batch)], dim=0)
        z_q, encodings = self.model.quantise.quantise(encoding_idx.flatten(end_dim=1))
        z_q = z_q.view(self.num_per_batch, -1, self.model.latent_dim)
        x_hats = self.model.decode(z_q).cpu().squeeze(1)
        # plot alongside each-other
        fig, axes = plt.subplots(nrows=self.num_per_batch, ncols=3, figsize=(15, self.num_per_batch * 3), width_ratios=[0.49, 0.49, 0.02])
        for i, x, x_hat in zip(range(self.num_per_batch), xs, x_hats):
            ax1, ax2, cax = axes[i, :]
            if i == 0:
                ax1.set_title("Original Reconstruction")
                ax2.set_title("AR Approximation")
            vmin, vmax = min(x.min(), x_hat.min()), max(x.max(), x_hat.max())
            plot_mel_spectrogram(x.t(), vmin=vmin, vmax=vmax, ax=ax1, **self.model.spectrogram_params)
            mesh = plot_mel_spectrogram(x_hat.t(), vmin=vmin, vmax=vmax, ax=ax2, **self.model.spectrogram_params)
            fig.colorbar(mesh, cax=cax, orientation="vertical")
        pl_module.logger.experiment.log({f"val/reconstruction": wandb.Image(fig)})
        # unroll the AR model pure generation
        T = 39 # T is the number of frame representations
        K = 16 # K is the number of categorical chunks per frame
        y_idx = pl_module.generate(N=self.num_per_batch * 2, T=T * K)
        y_idx = y_idx.view(self.num_per_batch * 2, T, K).flatten().view(-1, 1)
        z_q_chunk, _ = self.model.quantise.quantise(y_idx)
        z_q = z_q_chunk.view(self.num_per_batch * 2, T, -1)
        x_gens = self.model.decode(z_q).cpu().squeeze(1)
        fig, axes = plt.subplots(nrows=self.num_per_batch, ncols=3, figsize=(15, self.num_per_batch * 3), width_ratios=[0.49, 0.49, 0.02])
        for i, x1, x2 in zip(range(self.num_per_batch), x_gens[:self.num_per_batch], x_gens[self.num_per_batch:]):
            ax1, ax2, cax = axes[i, :]
            vmin, vmax = min(x1.min(), x2.min()), max(x1.max(), x2.max())
            plot_mel_spectrogram(x1.t(), vmin=vmin, vmax=vmax, ax=ax1, **self.model.spectrogram_params)
            mesh = plot_mel_spectrogram(x2.t(), vmin=vmin, vmax=vmax, ax=ax2, **self.model.spectrogram_params)
            fig.colorbar(mesh, cax=cax, orientation="vertical")
        pl_module.logger.experiment.log({f"val/generation": wandb.Image(fig)})
