import copy
import enum
import itertools
import lightning as L
import logging
import hydra
import numpy as np
import seaborn as sns
import pathlib
import pandas as pd
import torch
import wandb

from dataclasses import dataclass
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions.normal import Normal
from torch.optim import Optimizer
from torchvision.transforms import functional as T
from typing import Any, Dict, Tuple, List, Iterator

from src.core.models.components import (
    Activation,
    NormType,
    init_cnn_feature_encoder,
    init_cnn_feature_decoder,
    init_mlp_content_encoder,
    init_mlp_content_decoder,
    init_alignment_encoder,
    init_alignment_encoder,
)
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior, autoregressive_mixture_kl_divergence
from src.core.transforms.log_mel_spectrogram import mel_filterbanks
from src.core.transforms.frame import unframe_fold as unframe, frame_fold as frame
from src.core.transforms.translation import translation
from src.core.utils.sketch import plot_mel_spectrogram, plot_latent_sequence_histogram, plot_latent_power_spectral_density_heatmap, plot_latent_time_series
from src.core.utils import soft_clip, linear_decay, nth_percentile, detach_values, prefix_keys, to_snake_case
from src.core.utils import Batch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["TCEMAVAE"]

class EMA:
    def __init__(self, decay=0.999):
        self.decay = decay

    @torch.no_grad()
    def update(self, student: torch.nn.Module, teacher: torch.nn.Module) -> None:
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(self.decay)
            t_param.data.add_((1 - self.decay) * s_param.data)

@dataclass(kw_only=True, unsafe_hash=True)
class LogMelSpectrogram(torch.nn.Module):
    sample_rate: int = 48_000
    n_fft: int = 512
    fft_window_length: int = 512
    fft_hop_length: int = 384
    num_mel_bins: int = 64
    mel_min_hertz: float | None = 0.0
    mel_max_hertz: float | None = None
    mel_scaling_factor: float | None = 4581.0
    mel_break_frequency: float | None = 1750.0

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        torch.nn.Module.__init__(obj)
        return obj

    def __post_init__(self):
        mel_basis = mel_filterbanks(
            num_mel_bins=self.num_mel_bins,
            mel_min_hertz=self.mel_min_hertz,
            mel_max_hertz=self.mel_max_hertz,
            linear_frequencies=torch.linspace(0.0, self.sample_rate / 2, (self.n_fft // 2) + 1),
            scaling_factor=self.mel_scaling_factor,
            break_frequency=self.mel_break_frequency,
        )
        self.register_buffer("mel_basis", torch.tensor(mel_basis, requires_grad=False), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # apply fourier transform
        window = torch.hann_window(self.fft_window_length).to(x.device)
        x = torch.stft(x, self.n_fft, window=window, **self.stft_params)
        # discard phase
        x = x.abs()
        # transpose time on inner axes
        x = x.transpose(-1, -2)
        # apply mel
        x = x @ self.mel_basis.t()
        # apply log
        x = torch.clamp(x, min=1e-6).log()
        # add a channel dimension
        x = x.unsqueeze(1)
        return x

    @property
    def stft_params(self):
        return dict(
            win_length=self.fft_window_length,
            hop_length=self.fft_hop_length,
            return_complex=True,
            pad_mode="constant",
        )

@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class TCEMAVAE(L.LightningModule):
    sample_rate: int = 48_000
    fft_window_length: int = 512
    fft_hop_length: int = 384
    mel_min_hertz: float | None = 0.0
    mel_max_hertz: float | None = None
    mel_scaling_factor: float | None = 4581.0
    mel_break_frequency: float | None = 1750.0
    frame_window_length: int = 192
    frame_hop_length: int | None = 192
    num_mel_bins: int = 64
    latent_dim: int = 128
    sigma_z_min: float = 0.0498
    weight_init_std: float = 1e-3
    cnn_block_width: int = 4
    cnn_block_depth: int = 3
    cnn_dropout_prob: float = 0.2
    cnn_padding_mode: str = "circular"
    cnn_activation: str = "LEAK"
    cnn_feature_reduction_factor: int = 4
    norm_type: str = "LN"
    mlp_activation: str = "LEAK"
    mlp_dropout_prob: float = 0.1
    mlp_reduction_factor: int = 4
    frame_padding_mode: str = "circular"
    sigma_x: float = 1.0 # TODO: vectorise?
    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1

    cross_decode_method: str = "soft"
    delta_sigma_step_start: int | None = None
    delta_sigma_step_end: int | None = None
    delta_sigma_min: float | None = None
    delta_sigma_max: float = 2.0
    delta_sigma_step_slope: float = 1.0
    alpha: float = 0.85
    num_kl_samples: int = 50

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        plt.switch_backend('agg')
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(
            self,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
            ckpt_path=config.get("ckpt_path")
        )

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self):
        self.save_hyperparameters()
        self.mel_max_hertz = self.mel_max_hertz or self.sample_rate / 2.0
        self.log_mel_spectrogram = LogMelSpectrogram(**self.log_mel_spectrogram_params)
        self.feature_encoder = init_cnn_feature_encoder(**self.cnn_encoder_params)
        self.content_encoder = init_mlp_content_encoder(**self.content_mlp_encoder_params)
        self.offset_encoder = init_alignment_encoder(**self.offset_mlp_params)
        self.feature_encoder_ema = copy.deepcopy(self.feature_encoder)
        self.feature_encoder_ema.eval()
        self.content_encoder_ema = copy.deepcopy(self.content_encoder)
        self.content_encoder_ema.eval()
        self.ema = EMA(decay=0.999)
        self.feature_decoder = init_cnn_feature_decoder(**self.cnn_decoder_params)
        self.content_decoder = init_mlp_content_decoder(**self.content_mlp_decoder_params)
        self._reset_cache()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # ------ pre-processing ------ #
        x_i = self.log_mel_spectrogram(x)
        x_i = T.center_crop(x_i, [(x_i.size(-2) - (x_i.size(-2) % self.frame_window_length)), x_i.size(-1)])
        sigma_delta = torch.tensor(self.bounded_sigmoid(self.trainer.global_step, **self.sigma_delta_params))
        x_i_framed = x_i.view(x_i.size(0), -1, 1, self.frame_window_length, x_i.size(-1)).flatten(end_dim=1)
        epsilon = torch.randn(x_i_framed.size(0), 1, 1, 1, device=x.device)
        x_j = translation(x_i_framed, epsilon * sigma_delta, padding_mode="circular")

        # ------ student forward ------ #
        k = 2 # number of cross-decoded samples
        q_z_i, delta_i = self.encode(x_i) # (bs, seq, ld)
        q_z_j, delta_j = self.encode(x_j) # (bs * seq, 1, ld)
        q_z = torch.stack([q_z_i, q_z_j.view(q_z_i.size())], dim=0) # (k, bs, seq, ld)
        delta = torch.stack([delta_i, delta_j.view(delta_i.size())], dim=0)
        z = self.cross_decode(q_z, method=self.cross_decode_method, k=k)
        # mask sharp features with zeros during pre-training
        if self.mask_sharp:
            z[:, :, :, self.sharp_feature_idx[0]] = 0
        U_hat = self.mlp_decode(z.flatten(end_dim=2)).unflatten(dim=0, sizes=(z.size(0), z.size(1), z.size(2))) # (k, bs, seq, ch, fr, fq)
        U_hat_j, U_hat_i = U_hat.chunk(k, dim=0)
        x_hat_i = self.cnn_decode(U_hat_j.flatten(end_dim=2), delta_i) # (bs, 1, fr * seq, fq)
        x_hat_j = self.cnn_decode(U_hat_i.flatten(end_dim=2), delta_j) # (bs * seq, 1, fr, fq)
        x_hat_i_framed = x_hat_i.view(x_hat_i.size(0), -1, 1, self.frame_window_length, x_hat_i.size(-1)).flatten(end_dim=1)
        x = torch.cat([x_i_framed, x_j], dim=0)
        x_hat = torch.cat([x_hat_i_framed, x_hat_j], dim=0)

        # ------ teacher forward ------ #
        with torch.no_grad():
            q_z_i_ema, q_z_j_ema = self.encode_ema(x_i), self.encode_ema(x_j)
            q_z_ema = torch.stack([q_z_i_ema, q_z_j_ema.view(q_z_i_ema.size())], dim=0)

        return dict(
            x=x, x_hat=x_hat,
            x_i=x_i, x_hat_i=x_hat_i,
            q_z=q_z, q_z_ema=q_z_ema,
            delta=delta, sigma_delta=sigma_delta,
        )

    def cross_decode(self, q_z: torch.Tensor, method: str = "soft", k: int = 2):
        q_z_i, q_z_j = q_z.chunk(k, dim=0)
        mu_z_i, log_sigma_sq_z_i = q_z_i.chunk(2, dim=-1)
        mu_z_j, log_sigma_sq_z_j = q_z_j.chunk(2, dim=-1)
        if method == "soft":
            # weighted sum of gaussians
            mu_z = 1/2 * mu_z_i + 1/2 * mu_z_j
            log_sigma_sq_z = (1/4 * log_sigma_sq_z_i.exp() + 1/4 * log_sigma_sq_z_j.exp()).log()
            z = Normal(mu_z, (1/2 * log_sigma_sq_z).exp()).rsample() # (bs, seq, ld)
            z = torch.cat([z, z], dim=0)
        elif method == "hard":
            # swapped samples
            z_i = Normal(mu_z_i, (1/2 * log_sigma_sq_z_i).exp()).rsample() # (bs, seq, ld)
            z_j = Normal(mu_z_j, (1/2 * log_sigma_sq_z_j).exp()).rsample() # (bs, seq, ld)
            z = torch.cat([z_j, z_i], dim=0)
        return z

    def k_way_cross_decode(self, q_z: torch.Tensor, method: str = "soft", k: int = 2):
        mu_zs, log_sigma_sq_zs = q_z.chunk(2, dim=-1)
        if method == "soft":
            # weighted average along shift dimension
            mu_z = (1 / k * mu_zs).sum(dim=0)
            log_sigma_sq_z = (1 / k**2 * log_sigma_sq_zs.exp()).sum(dim=0)).log()
            z = Normal(mu_z, (1/2 * log_sigma_sq_z).exp()).rsample()
            zs = torch.cat([z for _ in range(q_z.size(0))], dim=0)
        elif method == "hard":
            # randomly shuffle samples along shift dimension
            zs = Normal(mu_zs, (1/2 * log_sigma_sq_zs).exp()).rsample()
            zs = zs[torch.randperm(k)]
        return zs

    def encode(self, x: Tensor, hop_length: int | None = None) -> Tensor:
        x = self.cnn_encode(x, encoder=self.feature_encoder)
        hop_length = (hop_length or self.frame_hop_length) // 2**(self.cnn_layers)
        x = self.frame(x, window_length=self.latent_window_length, hop_length=hop_length, padding_mode=self.frame_padding_mode)
        delta = self.offset_encode(x, encoder=self.offset_encoder)
        q_z = self.content_encode(x, encoder=self.content_encoder)
        return q_z, delta

    def encode_ema(self, x: Tensor, hop_length: int | None = None) -> Tensor:
        x = self.cnn_encode(x, encoder=self.feature_encoder_ema)
        hop_length = (hop_length or self.frame_hop_length) // 2**(self.cnn_layers)
        x = self.frame(x, window_length=self.latent_window_length, hop_length=hop_length, padding_mode=self.frame_padding_mode)
        q_z = self.content_encode(x, encoder=self.content_encoder_ema)
        return q_z

    def cnn_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tensor:
        for i, block in enumerate(encoder):
            x = block(x)
        return x

    def content_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tuple[Tensor, Tensor]:
        q_z = encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        log_sigma_sq_z = soft_clip(log_sigma_sq_z, minimum=2*np.log(self.sigma_z_min))
        q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        return q_z

    def offset_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tuple[Tensor, Tensor]:
        delta = encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        return delta

    def decode(self, z: Tensor, delta: Tensor | None = None) -> Tensor:
        x = self.mlp_decode(z.flatten(end_dim=1))
        x = self.cnn_decode(x, delta)
        return x

    def mlp_decode(self, z: Tensor) -> Tensor:
        return self.content_decoder(z)

    def cnn_decode(self, U: Tensor, delta: Tensor) -> Tensor:
        for i, block in enumerate(self.feature_decoder):
            if i == len(self.feature_decoder) - 2:
                U = translation(U, delta.view(delta.size(0) * delta.size(1), 1, 1, 1), padding_mode="circular")
            if i == len(self.feature_decoder) - 1:
                U = U.unflatten(0, (delta.size(0), delta.size(1))).transpose(1, 2).flatten(start_dim=2, end_dim=3)
            U = block(U)
        return U

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        q_z: torch.Tensor,
        q_z_ema: torch.Tensor,
        delta: torch.Tensor,
        sigma_delta: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        # frame-wise reconstruction loss
        sigma_x = torch.tensor(self.sigma_x)
        nll_x = (1/2 * (2 * sigma_x.log() + ((x - x_hat) / sigma_x).pow(2))).flatten(start_dim=-3).sum(dim=-1).mean()

        # frame-wise shift loss
        nll_delta = (1/2 * (2 * sigma_delta.log() + (delta / sigma_delta).pow(2))).mean()
        # TODO: thoughts on delta diversity across samples,
        # for paired samples, repel values by ReLU(margin - |delta_i - delta_j|)
        # loss is minimal when difference is maximal
        # might be problematic given the interval, could set everything to -1, 1
        # might need a circular approach, map deltas to angles, then maximise
        # distance on the circle's circumference
        # the problem was across the sequence, i.e. it predicts the same value for every data point
        # we need diversity within each data-point, so we should incentivise diversity across the sequence?

        # frame-wise standard normal kl
        # TODO: switch between priors, only apply to sharp features after changeover
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        dkl = (-1/2 * (1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())).sum(dim=-1).mean()

        # TODO: only apply to smooth feature idx
        # TODO: switch between priors, only apply to smooth features after changeover
        # distilled temporal mixture kl (for subset of features)
        # sample from student posterior
        mu, log_sigma_sq = q_z.expand(self.num_kl_samples, *q_z.shape).flatten(end_dim=1).chunk(2, dim=-1)
        # (mu, _), (log_sigma_sq, _) = mu.chunk(2, dim=-1), log_sigma_sq.chunk(2, dim=-1)
        z = Normal(mu, (1/2 * log_sigma_sq).exp()).rsample()
        # teacher-derived prior is equal mixture across neighbouring timesteps
        mu, log_sigma_sq = q_z_ema.expand(self.num_kl_samples, *q_z_ema.shape).flatten(end_dim=1).chunk(2, dim=-1)
        # only apply to the first of latent space
        # (mu, _), (log_sigma_sq, _) = mu.chunk(2, dim=-1), log_sigma_sq.chunk(2, dim=-1)
        # log prob of normal at t, discounting t=0
        log_q = (-1/2 * (torch.tensor(2 * torch.pi).log() + log_sigma_sq[:, :, 1:] + ((z[:, :, 1:] - mu[:, :, 1:]).pow(2) / log_sigma_sq[:, :, 1:].exp())))
        # log prob of normal at t - 1
        log_p_prev = (-1/2 * (torch.tensor(2 * torch.pi).log() + log_sigma_sq[:, :, :-1] + ((z[:, :, 1:] - mu[:, :, :-1]).pow(2) / log_sigma_sq[:, :, :-1].exp())))
        # log prob of normal at t + 1
        log_p_next = (-1/2 * (torch.tensor(2 * torch.pi).log() + log_sigma_sq[:, :, 1:] + ((z[:, :, :-1] - mu[:, :, 1:]).pow(2) / log_sigma_sq[:, :, 1:].exp())))
        log_p = torch.logaddexp(log_p_prev, log_p_next) - torch.tensor(2).log()
        # expectation using samples
        temp_dkl = torch.clamp((log_q - log_p).sum(dim=-1).mean(dim=0), min=0).mean()

        # TODO: question: after deriving mixture between neighbours, use alpha to weight mixture between standard normal for smoothness degree?
        loss = nll_x + nll_delta + dkl + self.alpha * temp_dkl

        return dict(
            loss=loss,
            log_likelihood_x=-nll_x.detach(),
            log_likelihood_delta=-nll_delta.detach(),
            dkl=dkl.detach(),
            temp_dkl=temp_dkl.detach(),
            nll_x=nll_x.detach(),
            nll_delta=nll_delta.detach(),
        )

    def loss_2(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        q_z: torch.Tensor,
        q_z_ema: torch.Tensor,
        delta: torch.Tensor,
        sigma_delta: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        # frame-wise reconstruction loss
        sigma_x = torch.tensor(self.sigma_x)
        nll_x = (1/2 * (2 * sigma_x.log() + ((x - x_hat) / sigma_x).pow(2))).flatten(start_dim=-3).sum(dim=-1).mean()

        # frame-wise shift loss
        nll_delta = (1/2 * (2 * sigma_delta.log() + (delta / sigma_delta).pow(2))).mean()

        # distilled temporal mixture kl
        # sample from student posterior
        mu, log_sigma_sq = q_z.expand(self.num_kl_samples, *q_z.shape).flatten(end_dim=1).chunk(2, dim=-1)
        z = Normal(mu, (1/2 * log_sigma_sq).exp()).rsample()
        # prior is a weighted average between teacher at previous time-step and standard normal
        mu, log_sigma_sq = q_z_ema.expand(self.num_kl_samples, *q_z_ema.shape).flatten(end_dim=1).chunk(2, dim=-1)
        log_q = -1/2 * (torch.tensor(2 * torch.pi).log() + log_sigma_sq + ((z - mu).pow(2) / log_sigma_sq.exp()))
        # evaluate the log density of the prior at each timestep
        mu_prev, log_sigma_sq_prev  = torch.zeros_like(mu), torch.zeros_like(log_sigma_sq)
        # t=0 is (0, 1), t>0 is q(z_t-1)
        mu_prev[:, 1:], log_sigma_sq_prev[:, 1:] = mu[:, :-1], log_sigma_sq[:, :-1]
        # log prob under the posterior at t-1
        log_p_prev = -1/2 * (torch.tensor(2 * torch.pi).log() + log_sigma_sq_prev + ((z - mu_prev).pow(2) / log_sigma_sq_prev.exp()))
        # log prob under a standard normal distribution
        log_p_0 = -1/2 * (torch.tensor(2 * torch.pi).log() + z.pow(2))
        # log prob of mixture weighted by alpha
        alpha = torch.cat([torch.zeros_like(alpha).unsqueeze(0), alpha.expand(q_z.size(-2) - 1, *alpha.shape)])
        log_p = torch.logsumexp(torch.stack([alpha.log() + log_p_prev, (1 - alpha).log() + log_p_0], dim=0), dim=0)
        # expectation using samples
        dkl = torch.clamp((log_q - log_p).mean(dim=dim), min=0)

        loss = nll_x + nll_delta + dkl

        return dict(
            loss=loss,
            log_likelihood_x=-nll_x.detach(),
            log_likelihood_delta=-nll_delta.detach(),
            dkl=dkl.detach(),
            nll_x=nll_x.detach(),
            nll_delta=nll_delta.detach(),
        )

    # ------------------------------ LIGHTNING FUNCS --------------------------------- #

    def on_after_backward(self):
        self.ema.update(self.feature_encoder, self.feature_encoder_ema)
        self.ema.update(self.content_encoder, self.content_encoder_ema)

    def step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        stage: str,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        x, *_ = batch
        step_outputs = self(x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        metrics = self.metrics(**step_outputs)
        self.log_dict(prefix_keys(loss_outputs, stage), batch_size=x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, stage)))
        return loss_outputs, step_outputs

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "train")
        self.training_step_outputs.append(step_outputs)
        return loss_outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "val")
        self.validation_step_outputs.append(step_outputs)
        return loss_outputs

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "test")
        self.test_step_outputs.append(step_outputs)
        return loss_outputs

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self.training_step_outputs.clear()

    def on_validation_batch_end(
        self,
        outputs: Dict[str, Tensor],
        batch: Batch,
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        x, *_ = batch
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]

            specs = step_outputs["x_i"].squeeze().cpu().numpy()
            recons = step_outputs["x_hat_i"].squeeze().cpu().numpy()
            q_z = step_outputs["q_z"]
            q_z_i, q_z_j = q_z.chunk(2, dim=0)
            mu = q_z_i.squeeze(0).chunk(2, dim=-1)[0].cpu().numpy()
            num_samples = min(6, step_outputs["x_i"].size(0))

            for i in range(num_samples):
                fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12), width_ratios=[0.97, 0.03], constrained_layout=True)

                mesh = plot_mel_spectrogram(specs[i].T, **self.spectrogram_params, ax=axes[0, 0], vmin=specs.min(), vmax=specs.max())
                mesh = plot_mel_spectrogram(recons[i].T, **self.spectrogram_params, ax=axes[1, 0], vmin=specs.min(), vmax=specs.max())
                axes[0, 0].set_title("Original Mel Spectrogram")
                axes[1, 0].set_title("Reconstructed Mel Spectrogram")
                fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
                fig.colorbar(mesh, cax=axes[1, 1], orientation="vertical")

                mu_norm = ((mu[i] - mu[i].mean(axis=0)) / mu[i].std(axis=0))
                lc = plot_latent_time_series(mu_norm, **time_series_params, ax=axes[2, 0], lw=2, alpha=0.75)
                axes[2, 0].set_xlabel("Time ($t$)")
                axes[2, 0].set_ylabel(r"$\mathbf{z}_{\text{norm}}(t)$")
                axes[2, 0].set_title("Latent Time-series")
                cbar = plt.colorbar(lc, cax=axes[2, 1])
                cbar.set_label("Bandwidth ($h$)", rotation=90)

                im = plot_latent_power_spectral_density_heatmap(mu_norm, fft_length=mu[i].shape[0], **time_series_params, ax=axes[3, 0])
                axes[3, 0].set_title("Latent Power Spectral Density")
                fig.colorbar(im, cax=axes[3, 1], orientation="vertical")
                self.logger.experiment.log({ f"val/image": wandb.Image(fig) })
                plt.close(fig)
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def on_test_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self.test_step_outputs.clear()

    @torch.no_grad()
    def predict_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
        frame_hop_length: float | None = 192,
        **kwargs: Any
    ) -> pd.DataFrame:
        x, *_ = batch
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins])
        q_z, delta = self.encode(x, hop_length=frame_hop_length)
        bs, seq, *_ = q_z.size()
        sample_idx = batch.s.cpu().unsqueeze(0).repeat(seq, 1).t().flatten()
        seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq).cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().cpu()
        # seq start accounts for hop
        frame_hop_samples = self.fft_hop_length * frame_hop_length
        seq_start_samples = seq_idx * frame_hop_samples
        # seq end accounts for receptive field
        frame_duration_samples = self.fft_hop_length * self.frame_window_length
        seq_end_samples = seq_start_samples + frame_duration_samples
        # map to time in seconds
        seq_start_seconds = seq_start_samples / self.sample_rate
        seq_end_seconds = seq_end_samples / self.sample_rate
        ref_column_types = dict(
            file_i=int, dataloader_idx=int, timestep=int,
            t_start_samples=int, t_end_samples=int,
            t_start_seconds=float, t_end_seconds=float,
        )
        feat_column_types = dict(
            **{ f"z_mean_{d}": float  for d in range(q_z.size(-1)//2) },
            **{ f"z_log_var_{d}": float  for d in range(q_z.size(-1)//2) },
            **{ "delta": float },
        )
        column_types = (ref_column_types | feat_column_types)
        df = pd.DataFrame(
            data=dict(zip(column_types.keys(), [
                sample_idx, dl_idx, seq_idx,
                seq_start_samples, seq_end_samples,
                seq_start_seconds, seq_end_seconds,
                *q_z.flatten(end_dim=1).cpu().t(),
                *delta.flatten(end_dim=1).cpu().squeeze(-1),
            ])),
            columns=column_types.keys(),
        ).astype(dtype=column_types).set_index(list(ref_column_types.keys()))
        self.predict_step_outputs.append(df)
        return df

    def configure_optimizers(self) -> Optimizer:
        optimiser_config = DictConfig(dict(_target_=self.optimiser_cls, **(self.optimiser_config or {})))
        optimiser = hydra.utils.instantiate(optimiser_config, params=self.parameters(), lr=self.learning_rate)
        if self.scheduler_cls is not None:
            scheduler_config = DictConfig(dict(_target_=self.scheduler_cls, **(self.scheduler_config or {})))
            scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimiser)
            return [optimiser], [dict(
                scheduler=scheduler,
                interval=self.scheduler_interval,
                frequency=self.scheduler_frequency
            )]
        return optimiser

    def metrics(
        self,
        x: Tensor,
        x_hat: Tensor,
        q_z: Tensor,
        delta: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        mu_hist = np.histogram(mu_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[-5.0, 5.0])
        sigma_hist = np.histogram(sigma_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[0.0, 2.0])
        delta_hist = np.histogram(delta.flatten().cpu().numpy(), bins=64, range=[-5.0, 5.0])
        return dict(
            mu_z=wandb.Histogram(np_histogram=mu_hist),
            sigma_z=wandb.Histogram(np_histogram=sigma_hist),
            delta=wandb.Histogram(np_histogram=delta_hist),
        )

    def _reset_cache(self):
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

    # ------------------------------- GROUPED PROPS -------------------------------- #

    @property
    def log_mel_spectrogram_params(self):
        return dict(
            sample_rate=self.sample_rate,
            fft_window_length=self.fft_window_length,
            fft_hop_length=self.fft_hop_length,
            num_mel_bins=self.num_mel_bins,
            mel_min_hertz=self.mel_min_hertz,
            mel_max_hertz=self.mel_max_hertz,
            mel_scaling_factor=self.mel_scaling_factor,
            mel_break_frequency=self.mel_break_frequency,
        )

    @property
    def cnn_encoder_params(self):
        return dict(
            block_sizes=self.cnn_block_sizes,
            block_width=self.cnn_block_width,
            block_depth=self.cnn_block_depth,
            dropout_prob=self.cnn_dropout_prob,
            padding_mode=self.cnn_padding_mode,
            norm_fn=NormType[self.norm_type],
            activation_fn=Activation[self.cnn_activation],
            weight_init_std=self.weight_init_std,
    )

    @property
    def content_mlp_encoder_params(self):
        return dict(
            in_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
            out_channels=self.cnn_block_sizes[-1] * self.cnn_block_width // self.cnn_feature_reduction_factor,
            feature_height=self.latent_window_length,
            feature_width=self.latent_frequency_dim,
            mlp_reduction_factor=self.mlp_reduction_factor,
            activation_fn=Activation[self.mlp_activation],
            dropout_prob=self.mlp_dropout_prob,
            out_features=self.latent_dim * 2,
        )

    @property
    def offset_mlp_params(self):
        return dict(
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
    def cnn_decoder_params(self):
        return dict(
            block_sizes=list(reversed(self.cnn_block_sizes)),
            block_width=self.cnn_block_width,
            block_depth=self.cnn_block_depth,
            dropout_prob=self.cnn_dropout_prob,
            padding_mode=self.cnn_padding_mode,
            norm_fn=NormType[self.norm_type],
            activation_fn=Activation[self.cnn_activation],
        )

    @property
    def content_mlp_decoder_params(self):
        return dict(
            in_features=self.latent_dim,
            in_channels=self.cnn_block_sizes[-1] * self.cnn_block_width // self.cnn_feature_reduction_factor,
            out_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
            feature_height=self.latent_window_length,
            feature_width=self.latent_frequency_dim,
            mlp_reduction_factor=self.mlp_reduction_factor,
            activation_fn=Activation[self.mlp_activation],
            dropout_prob=self.mlp_dropout_prob,
    )

    @property
    def cnn_block_sizes(self):
        return [8, 16, 32, 64, 128]

    @property
    def cnn_layers(self):
        return len(self.cnn_block_sizes)

    @property
    def latent_frequency_dim(self) -> int:
        return self.num_mel_bins // 2**(self.cnn_layers - 1)

    @property
    def latent_window_length(self) -> int:
        return self.frame_window_length // 2**(self.cnn_layers)

    @property
    def latent_hop_length(self) -> int:
        return (
            self.frame_hop_length // 2**(self.cnn_layers)
            if self.frame_hop_length is not None
            else self.latent_window_length
        )

    @property
    def mask_sharp(self):
        return self.trainer.global_step < 12_500

    def sharp_mask(self):
        return torch.cat([torch.ones(self.latent_dim // 2), torch.zeros(self.latent_dim // 2, self.latent_dim)])

    @property
    def smooth_feature_idx(self):
        return (
            torch.arange(0, self.latent_dim // 2),
            torch.arange(self.latent_dim, self.latent_dim + latent_dim // 2),
        )

    @property
    def sharp_feature_idx(self):
        return (
            torch.arange(self.latent_dim // 2, self.latent_dim),
            torch.arange(self.latent_dim + self.latent_dim // 2, self.latent_dim * 2),
        )

    @property
    def frame_params(self):
        return dict(
            hop_length=self.frame_hop_length,
            window_length=self.frame_window_length,
            padding_mode=self.frame_padding_mode
        )

    @property
    def spectrogram_params(self):
        return dict(
            sample_rate=self.sample_rate,
            hop_length=self.fft_hop_length,
            window_length=self.fft_window_length,
            fft_length=self.n_fft,,
            mel_min_hertz=self.mel_min_hertz,
            mel_max_hertz=self.mel_max_hertz,
            mel_scaling_factor=self.mel_scaling_factor,
            mel_break_frequency=self.mel_break_frequency,
        )

    @property
    def sigma_delta_params(self):
        return dict(
            x_min=self.delta_sigma_step_start,
            x_max=self.delta_sigma_step_end,
            y_min=self.delta_sigma_min,
            y_max=self.delta_sigma_max,
            k=self.delta_sigma_step_slope or 1.0,
        )

    @property
    def time_series_params(self):
        return dict(
            audio_sample_rate=self.sample_rate,
            audio_fft_hop_length=self.fft_hop_length,
            audio_frame_length_hops=self.frame_window_length,
        )

    @staticmethod
    def bounded_sigmoid(x: float, x_min: float, x_max: float, y_min: float, y_max: float, k: float):
        s = np.floor(np.log10(np.abs(x_max)))
        z = k / 10**(s - 1)
        return y_min + (y_max - y_min) / (1 + np.exp(-z * (x - ((x_min + x_max) / 2))))

    @staticmethod
    def frame(x: Tensor, window_length: int, hop_length: int | None = None, padding_mode: str = "circular") -> Tensor:
        if hop_length != window_length:
            x = frame(x, window_length=window_length, hop_length=hop_length) if x.size(-2) > window_length else x.unsqueeze(1)
        else:
            x = x.view(x.size(0), x.size(1), x.size(2) // window_length, window_length, x.size(3)).transpose(1, 2)
        return x
