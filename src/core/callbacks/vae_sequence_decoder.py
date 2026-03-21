import lightning as L
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from matplotlib import pyplot as plt
from typing import Any, Dict, List, Tuple
from torch.distributions.normal import Normal

from src.core.utils import metrics
from src.core.utils.sketch import plot_mel_spectrogram

plt.switch_backend('agg')

__all__ = ["VAESequenceDecoder"]

def load_from_checkpoint(model, ckpt_path, map_location: str = "cpu"):
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model

class VAESequenceDecoder(L.Callback):
    def __init__(
        self,
        model: torch.nn.Module,
        ckpt_path: str,
        num_per_batch: int = 6,
        log_every_n_train_steps: int | None = None,
        latent_hop_length: int = 6,
    ) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self.num_per_batch = num_per_batch
        self.model = load_from_checkpoint(model, ckpt_path)
        self.log_every_n_train_steps = log_every_n_train_steps
        self.latent_hop_length = latent_hop_length

    def setup(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        stage: str,
    ) -> None:
        self.model = self.model.to(pl_module.device)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_every_n_train_steps == 0:
            frame_step = int(self.model.latent_window_length / self.latent_hop_length)
            # decode original reconstruction
            q_z = batch.x[:self.num_per_batch]
            delta = batch.y[:self.num_per_batch]
            mu, log_sigma_sq = q_z.chunk(2, dim=-1)
            z = Normal(mu, (0.5 * log_sigma_sq).exp()).rsample()
            z, delta = z[:, ::frame_step].contiguous(), delta[:, ::frame_step].contiguous()
            xs = self.model.decode(z, delta).cpu().squeeze(1)
            # decode AR generated
            z_hat = pl_module(q_z)["z_hat"]
            z_hat = z_hat[:, ::frame_step].contiguous()
            # z_hat = torch.cat([z[:, :z_hat.size(1), :64], z_hat], dim=-1).contiguous()
            x_hats = self.model.decode(z_hat, delta).cpu().squeeze(1)
            # plot alongside each-other
            fig, axes = plt.subplots(nrows=self.num_per_batch, ncols=3, figsize=(15, self.num_per_batch * 3), width_ratios=[0.49, 0.49, 0.02])
            for i, x, x_hat in zip(range(self.num_per_batch), xs, x_hats):
                ax1, ax2, cax = axes[i, :]
                if i == 0:
                    ax1.set_title("VAE Reconstruction")
                    ax2.set_title("AR Approximation")
                vmin, vmax = min(x.min(), x_hat.min()), max(x.max(), x_hat.max())
                plot_mel_spectrogram(x.t(), vmin=vmin, vmax=vmax, ax=ax1, **self.model.spectrogram_params)
                mesh = plot_mel_spectrogram(x_hat.t(), vmin=vmin, vmax=vmax, ax=ax2, **self.model.spectrogram_params)
                fig.colorbar(mesh, cax=cax, orientation="vertical")
            pl_module.logger.experiment.log({f"train/reconstruction": wandb.Image(fig)})
            plt.close()

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
        frame_window_length = self.model.frame_window_length
        frame_start = pl_module.num_init_frames
        frame_step = int(self.model.latent_window_length / self.latent_hop_length)
        # decode original reconstruction
        q_z = batch.x[:self.num_per_batch]
        delta = batch.y[:self.num_per_batch]
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        z = Normal(mu, (0.5 * log_sigma_sq).exp()).rsample()
        z, delta = z[:, ::frame_step].contiguous(), delta[:, ::frame_step].contiguous()
        xs = self.model.decode(z, delta).cpu().squeeze(1)
        # decode AR generated
        z_hat = pl_module(q_z)["z_hat"]
        z_hat = z_hat[:, ::frame_step].contiguous()
        # z_hat = torch.cat([z[:, :z_hat.size(1), :64], z_hat], dim=-1).contiguous()
        x_hats = self.model.decode(z_hat, delta).cpu().squeeze(1)
        # plot alongside each-other
        fig, axes = plt.subplots(nrows=self.num_per_batch, ncols=3, figsize=(15, self.num_per_batch * 3), width_ratios=[0.49, 0.49, 0.02])
        for i, x, x_hat in zip(range(self.num_per_batch), xs, x_hats):
            ax1, ax2, cax = axes[i, :]
            if i == 0:
                ax1.set_title("VAE Reconstruction")
                ax2.set_title("AR Approximation")
            vmin, vmax = min(x.min(), x_hat.min()), max(x.max(), x_hat.max())
            plot_mel_spectrogram(x.t(), vmin=vmin, vmax=vmax, ax=ax1, **self.model.spectrogram_params)
            mesh = plot_mel_spectrogram(x_hat.t(), vmin=vmin, vmax=vmax, ax=ax2, **self.model.spectrogram_params)
            ax2.axvline(x=frame_start / frame_step * frame_window_length, color="white")
            fig.colorbar(mesh, cax=cax, orientation="vertical")
        pl_module.logger.experiment.log({f"val/reconstruction": wandb.Image(fig)})
        plt.close()

    def on_test_batch_end(
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        z_hat = pl_module.generate(seq=39)
        delta = torch.randn(self.num_per_batch * 2, 39, 1)
        x_gens = self.model.decode(z_hat, delta).cpu().squeeze(1)
        fig, axes = plt.subplots(nrows=self.num_per_batch, ncols=3, figsize=(15, self.num_per_batch * 3), width_ratios=[0.49, 0.49, 0.02])
        for i, x1, x2 in zip(range(self.num_per_batch), x_gens[:self.num_per_batch], x_gens[self.num_per_batch:]):
            ax1, ax2, cax = axes[i, :]
            vmin, vmax = min(x1.min(), x2.min()), max(x1.max(), x2.max())
            plot_mel_spectrogram(x1.t(), vmin=vmin, vmax=vmax, ax=ax1, **self.model.spectrogram_params)
            mesh = plot_mel_spectrogram(x2.t(), vmin=vmin, vmax=vmax, ax=ax2, **self.model.spectrogram_params)
            fig.colorbar(mesh, cax=cax, orientation="vertical")
        pl_module.logger.experiment.log({f"test/generation": wandb.Image(fig)})
        plt.close()
