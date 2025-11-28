import os
import enum
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb

from conduit.data import TernarySample
from dataclasses import dataclass
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.distributions.normal import Normal
from typing import Any, Dict, Tuple

from temporal_vae.transforms import frame_fold, unframe_fold, translation
from temporal_vae.models.components import Activation, init_alignment_encoder
from temporal_vae.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior
from temporal_vae.utils import soft_clip, linear_decay, bounded_sigmoid
from temporal_vae.artefact import save_artefact
from temporal_vae.sketch import plot_mel_spectrogram
from temporal_vae.models.base_vae import BaseVAE
from temporal_vae.utils import detach_values, prefix_keys


__all__ = ["NiftiVAE"]


@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class NiftiVAE(BaseVAE):
    p_dt_step_start: int | None = None
    p_dt_step_end: int | None = None
    p_dt_sigma_min: float | None = None
    p_dt_sigma_max: float = 2.0
    p_dt_step_slope: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.offset_encoder = init_alignment_encoder(
            in_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
            out_channels=self.cnn_block_sizes[-1] * self.cnn_block_width // 4,
            in_features=self.cnn_block_sizes[-1] * self.cnn_block_width // 4 * self.latent_window_length,
            cnn_kernel_size=(1, self.latent_frequency_dim),
            mlp_reduction_factor=2,
            flatten_start_dim=1,
            activation_fn=Activation[self.mlp_activation],
            out_features=1,
        )

    @property
    def p_dt_sigma_params(self):
        return dict(
            x_min=self.p_dt_step_start,
            x_max=self.p_dt_step_end,
            y_min=self.p_dt_sigma_min,
            y_max=self.p_dt_sigma_max,
            k=self.p_dt_step_slope or 1.0,
        )

    def cross_decode_hard(self, q_z: Tensor) -> Tensor:
        # hard method, roll on dimension 1 to switch feature representations w.r.t. translations
        return q_z.unflatten(0, (-1, 2)).roll(1, dims=1).flatten(end_dim=1)

    def cross_decode_soft(self, q_z: Tensor) -> Tensor:
        # soft method, average distributions of frames across translation gap
        q_z_i, q_z_j = q_z.unflatten(0, (-1, 2)).chunk(2, dim=1)
        (mu_x_i, log_sigma_sq_x_i), (mu_x_j, log_sigma_sq_x_j) = q_z_i.chunk(2, dim=-1), q_z_j.chunk(2, dim=-1)
        # average of the means is the mean
        mu_x = torch.stack([mu_x_i.flatten(end_dim=1), mu_x_j.flatten(end_dim=1)], dim=1).mean(dim=1)
        # average of the variance is the sum divided the number of distributions squared
        log_sigma_sq_x = (torch.stack([log_sigma_sq_x_i.flatten(end_dim=1).exp(), log_sigma_sq_x_j.flatten(end_dim=1).exp()], dim=1).sum(dim=1) / 2**2).log()
        return torch.stack([
            torch.cat([mu_x, log_sigma_sq_x], dim=-1),
            torch.cat([mu_x, log_sigma_sq_x], dim=-1)
        ], dim=1).flatten(end_dim=1)

    def forward(self, x: Tensor, cross_decode: bool = False, **kwargs: Any) -> Dict[str, Tensor]:
        x = x.float()
        q_z, _, dt = self.encode(x)
        if cross_decode:
            q_z = self.cross_decode_hard(q_z)
            # q_z = self.cross_decode_soft(q_z)
        mu_x, log_sigma_sq_x = q_z.chunk(2, dim=-1)
        z = Normal(mu_x, (0.5 * log_sigma_sq_x).exp()).rsample()
        x_hat = self.decode(z, dt).view(*x.size())
        return dict(x=x, x_hat=x_hat, q_z=q_z, z=z, dt=dt)

    def encode(self, x: Tensor, **kwargs: Any):
        U = x
        for block in self.feature_encoder:
            U = block(U)
        mu_x, log_sigma_sq_x = self.content_encoder(U).unsqueeze(1).chunk(2, dim=-1)
        log_sigma_sq_x = soft_clip(log_sigma_sq_x, minimum=np.log(self.sigma_x_min ** 2))
        dt = self.offset_encoder(U).unsqueeze(1)
        return torch.cat([mu_x, log_sigma_sq_x], dim=-1), U, dt

    def decode(self, z: Tensor, dt: Tensor | None = None, **kwargs: Any) -> Tensor:
        U = self.content_decoder(z.flatten(end_dim=1))
        for i, conv_block in enumerate(self.feature_decoder):
            if dt is not None and i == len(self.feature_decoder) - 2:
                U = translation(U, dt.unsqueeze(-1), padding_mode="circular")
            U = conv_block(U)
        return U

    def loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        q_z: Tensor,
        z: Tensor,
        dt: Tensor,
        **kwargs: Any
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # maximise the log likelihood of the data under a gaussian prior
        log_sigma_sq_z = self.log_sigma_sq_z_current(self.trainer.global_step)
        nll = negative_log_likelihood(x, x_hat, log_sigma_sq_z).flatten(start_dim=-3).sum(dim=-1)
        mae_frame = (x_hat - x).flatten(start_dim=-3).abs().sum(dim=-1)
        losses.append(nll.mean())
        outputs |= dict(log_likelihood_x=-nll.detach().mean(), sigma_z=(0.5 * log_sigma_sq_z).exp().detach(), mae_frame=mae_frame.detach().mean())
        # frame-wise mean of the KL between posterior and prior
        prior_dkl = gaussian_kl_divergence_standard_prior(q_z).sum(dim=-1)
        losses.append(prior_dkl.mean())
        outputs |= dict(prior_dkl=prior_dkl.detach().mean())
        # maximise the log likelihood of the alignment factor under the prior as a point-wise estimate
        mu_dt_wrt_x = torch.zeros(1).to(dt.device)
        log_sigma_sq_dt_wrt_x= self.p_dt_sigma_current(self.trainer.global_step).pow(2).log()
        intra_frame_nll = negative_log_likelihood(dt, mu_dt_wrt_x, log_sigma_sq_dt_wrt_x)
        losses.append(intra_frame_nll.mean())
        outputs |= dict(log_likelihood_dt=-intra_frame_nll.detach().mean())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        return outputs

    def p_dt_sigma_current(self, t: int) -> Tensor:
        if self.p_dt_sigma_min is None: return self.p_dt_sigma_max
        return torch.tensor(bounded_sigmoid(t, **self.p_dt_sigma_params))

    def metrics(
        self,
        dt: Tensor,
        **kwargs: Any,
     ) -> Dict[str, Any]:
        outputs = super().metrics(**kwargs)
        return dict(
            **outputs,
            p_sigma_dt=self.p_dt_sigma_current(self.trainer.global_step),
            dt=wandb.Histogram(np_histogram=np.histogram(dt.detach().flatten(end_dim=-2).cpu().numpy(), bins=64, range=[-5.0, 5.0]))
        )

    def training_step(self, batch: TernarySample, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        batch.x = frame(batch.x, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        dt = torch.randn(batch.x.size(0), 1, 1, 1).to(batch.x.device) * self.p_dt_sigma_current(self.trainer.global_step)
        batch.x = torch.stack([batch.x, translation(batch.x, dt, padding_mode="circular")], dim=1).flatten(end_dim=1)
        return super().training_step(batch, batch_idx, cross_decode=True, **kwargs)

    def validation_step(self, batch: TernarySample, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        x = frame(batch.x, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        dt = torch.randn(x.size(0), 1, 1, 1).to(x.device) * self.p_dt_sigma_current(self.trainer.global_step)
        x = torch.stack([x, translation(x, dt, padding_mode="circular")], dim=1).flatten(end_dim=1)
        return super().validation_step(TernarySample(x=x, y=batch.y, s=batch.s), batch_idx, cross_decode=True, **kwargs)

    def on_validation_batch_end(self, outputs: Dict[str, Tensor], batch: TernarySample, batch_idx: int, **kwargs: Any) -> None:
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            _, spec_trans = step_outputs["x"].unflatten(0, (-1, 2)).squeeze().chunk(2, dim=1)
            recons, recon_trans = step_outputs["x_hat"].unflatten(0, (-1, 2)).squeeze().chunk(2, dim=1)
            nrows = min(batch.x.size(0), 6)
            specs = batch.x[:nrows].cpu().numpy()
            recons = recons.view(batch.x.size(0), -1, *recons.size()[1:])[:nrows]
            num_timesteps = self.frame_hop_length * recons.size(1)
            recons = unframe(recons, hop_length=self.frame_hop_length, num_timesteps=num_timesteps).cpu().numpy()
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, nrows * 3))
            for i in range(nrows):
                vmin, vmax = min(recons[i].min(), specs[i].min()), max(recons[i].max(), specs[i].max())
                plot_mel_spectrogram(specs[i].T, **self.spectrogram_params, vmin=vmin, vmax=vmax, ax=axes[i, 0])
                plot_mel_spectrogram(recons[i].T, **self.spectrogram_params, vmin=vmin, vmax=vmax, ax=axes[i, 1])
            self.logger.experiment.log({ f"val/spectrogram_i": wandb.Image(fig) })
            specs = spec_trans.view(batch.x.size(0), -1, *spec_trans.size()[1:]).cpu().numpy()[:, :5]
            recons = recon_trans.view(batch.x.size(0), -1, *recon_trans.size()[1:]).cpu().numpy()[:, :5]
            nrows, ncols = specs.shape[0], specs.shape[1]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, figsize=(15, nrows * 2), sharey=True, sharex=True)
            for i in range(nrows):
                for j in range(ncols):
                    ax1, ax2 = axes[i, j], axes[i, j + ncols]
                    plot_mel_spectrogram(specs[i, j].T, **self.spectrogram_params, vmin=specs.min(), vmax=specs.max(), ax=ax1)
                    plot_mel_spectrogram(recons[i, j].T, **self.spectrogram_params, vmin=recons.min(), vmax=recons.max(), ax=ax2)
                    for ax in [ax1, ax2]:
                        ax.tick_params(axis="both", bottom=False, left=False, labelbottom=False, labelleft=False)
                        ax.set_ylabel("")
            self.logger.experiment.log({ f"val/spectrogram_j": wandb.Image(fig) })
            plt.close(fig)
        self.validation_step_outputs.clear()
