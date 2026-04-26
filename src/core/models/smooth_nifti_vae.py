import copy
import enum
import itertools
import lightning as L
import logging
import hydra
import math
import numpy as np
import pathlib
import pandas as pd
import seaborn as sns
import torch
import wandb

from dataclasses import dataclass
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions.normal import Normal
from torch.functional import F
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
)
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.transforms.frame import unframe_fold as unframe, frame_fold as frame
from src.core.transforms.translation import translation
from src.core.utils import Batch, soft_clip, detach_values, prefix_keys, bounded_sigmoid
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior, autoregressive_prior

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SmoothNiftiVAE"]

def central_finite_difference(x, padding_mode="circular"):
    x = F.pad(x, (1, 1), padding_mode)
    kernel = torch.tensor([[[-1.0, 0.0, 1.0]]]).to(x.device)
    dxdt = F.conv1d(x.unsqueeze(-2), kernel).squeeze(-2)
    return dxdt

@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class SmoothNiftiVAE(L.LightningModule):
    sample_rate: int = 48_000
    num_fft: int = 512
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
    sigma_x: float = 0.2
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
    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1
    delta_sigma_step_start: int | None = None
    delta_sigma_step_end: int | None = None
    delta_sigma_min: float | None = None
    delta_sigma_max: float = 2.0
    delta_sigma_step_slope: float = 1.0

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        plt.switch_backend('agg')
        log.info(f"Training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        checkpoint_path, resume = config.get("ckpt_path"), config.get("resume")
        if checkpoint_path is not None and resume:
            log.info(f"Resuming from {checkpoint_path}")
            trainer.fit(self, datamodule=data_module, ckpt_path=checkpoint_path)
        elif config.get("ckpt_path"):
            log.info(f"Loading state dict from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint["state_dict"], strict=False)
            trainer.fit(self, datamodule=data_module)
        else:
            trainer.fit(self, datamodule=data_module)
        log.info(f"Testing <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, dataloaders=data_module.predict_dataloader())

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self):
        self.mel_max_hertz = self.mel_max_hertz or self.sample_rate / 2.0
        self.save_hyperparameters()
        self.register_buffer("sigma_recon", torch.tensor(self.sigma_x, dtype=torch.float32, requires_grad=False))
        if self.sigma_z_min is not None:
            self.register_buffer("sigma_latent", torch.tensor(self.sigma_z_min, dtype=torch.float32, requires_grad=False))
        self.log_mel_spectrogram = LogMelSpectrogram(**self.log_mel_spectrogram_params)
        self.feature_encoder = init_cnn_feature_encoder(**self.cnn_encoder_params)
        self.content_encoder = init_mlp_content_encoder(**self.content_mlp_encoder_params)
        self.alignment_encoder = init_alignment_encoder(**self.alignment_encoder_params)
        self.feature_decoder = init_cnn_feature_decoder(**self.cnn_decoder_params)
        self.content_decoder = init_mlp_content_decoder(**self.content_mlp_decoder_params)
        self.strict_loading = False
        self._reset_cache()

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        # ensure x_i is a full sequence that can be divided into equal length frames
        x_i = self.log_mel_spectrogram(x)
        x_i = T.center_crop(x_i, [(x_i.size(-2) - (x_i.size(-2) % self.frame_window_length)), self.num_mel_bins])
        # encode posterior for full sequence
        q_z_i, delta_hat_i = self.encode(x_i) # (bs, seq, ld)
        mu_z_i, log_sigma_sq_z_i = q_z_i.chunk(2, dim=-1)
        # x_j is x_i chunked into independently translated frames
        x_i_framed = self.frame(x_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        epsilon = torch.randn(x_i_framed.size(0), 1, 1, 1).to(x_i.device)
        delta_sigma = self.delta_sigma_current(self.trainer.global_step)
        delta = epsilon * delta_sigma
        x_j = translation(x_i_framed, delta, padding_mode="circular")
        # encode posterior for translated frames separately
        q_z_j, delta_hat_j = self.encode(x_j) # (bs * seq, 1, ld)
        mu_z_j, log_sigma_sq_z_j = q_z_j.chunk(2, dim=-1)
        # soft cross-decoding averages the distributions
        # mu_k = (mu_i + mu_j) / 2, sigma^2_k = (sigma^2_i + sigma^2_j) / 2^2
        mu_z = torch.stack([mu_z_i.flatten(end_dim=1), mu_z_j.flatten(end_dim=1)], dim=1).mean(dim=1)
        log_sigma_sq_z = (torch.stack([log_sigma_sq_z_i.flatten(end_dim=1).exp(), log_sigma_sq_z_j.flatten(end_dim=1).exp()], dim=1).sum(dim=1) / 4).log()
        z = Normal(mu_z, (0.5 * log_sigma_sq_z).exp()).rsample()  # (bs, seq, ld)
        # stack q_z back together
        q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1).view(q_z_i.size())
        # decode to feature maps
        U_hat = self.content_decoder(z) # (bs * seq, ch, fr, fq)
        # reconstruct a contiguous sequence
        x_hat_i = self.cnn_decode(U_hat, delta_hat_i) # (bs, 1, fr * seq, fq)
        # and reconstruct independent translations
        x_hat_j = self.cnn_decode(U_hat, delta_hat_j) # (bs * seq, 1, fr, fq)
        # frame for frame-wise loss
        x_hat_i_framed = self.frame(x_hat_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        x = torch.cat([x_i_framed, x_j], dim=0)
        x_hat = torch.cat([x_hat_i_framed, x_hat_j], dim=0)
        return dict(
            x=x, x_i=x_i, x_j=x_j,
            x_hat=x_hat, x_hat_i=x_hat_i, x_hat_j=x_hat_j,
            q_z=q_z, q_z_i=q_z_i, q_z_j=q_z_j,
            delta=delta, delta_hat_i=delta_hat_i, delta_hat_j=delta_hat_j,
            delta_sigma=delta_sigma,
        )

    def predict(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        x = self.log_mel_spectrogram(x)
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])
        q_z, delta_hat = self.encode(x)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        z = torch.distributions.Normal(mu_z, (1/2 * log_sigma_sq_z).exp()).rsample()
        x_hat = self.cnn_decode(self.mlp_decode(z.flatten(end_dim=1)), delta_hat)
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_hop_length)
        x_hat_framed = self.frame(x_hat, window_length=self.frame_window_length, hop_length=self.frame_hop_length)
        return dict(
            x=x,
            x_hat=x_hat,
            x_framed=x_framed,
            x_hat_framed=x_hat_framed,
            q_z=q_z,
            delta_hat=delta_hat,
        )

    @torch.no_grad()
    def embed(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        dataloader_idx: int = 0,
        frame_hop_length: float | None = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        x, *_ = batch
        x = self.log_mel_spectrogram(x)
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])
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
        return df

    def encode(self, x: Tensor, hop_length: int | None = None) -> Tensor:
        x, u = self.cnn_encode(x)
        x = self.frame(x, window_length=self.latent_window_length, hop_length=(hop_length or self.frame_hop_length) // 2**(self.cnn_layers), padding_mode=self.frame_padding_mode)
        q_z = self.content_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        log_sigma_sq_z = soft_clip(log_sigma_sq_z, minimum=self.sigma_latent.pow(2).log())
        q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        delta_hat = self.alignment_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        return q_z, delta_hat

    def cnn_encode(self, x: Tensor) -> Tensor:
        U = []
        for i, block in enumerate(self.feature_encoder):
            x = block(x)
            U.append(x)
        return x, U[1]

    def decode(self, z: Tensor, delta: Tensor | None = None) -> Tensor:
        U = self.content_decoder(z.flatten(end_dim=1))
        x_hat = self.cnn_decode(U, delta)
        return x_hat

    def cnn_decode(self, U: Tensor, delta: Tensor) -> Tensor:
        for i, block in enumerate(self.feature_decoder):
            if i == len(self.feature_decoder) - 2:
                U = translation(U, delta.view(delta.size(0) * delta.size(1), 1, 1, 1), padding_mode="circular")
            if i == len(self.feature_decoder) - 1:
                U = U.unflatten(0, (delta.size(0), delta.size(1))).transpose(1, 2).flatten(start_dim=2, end_dim=3)
            U = block(U)
        return U

    @staticmethod
    def frame(x: Tensor, window_length: int, hop_length: int | None = None, padding_mode: str = "circular") -> Tensor:
        if x.size(-2) == window_length:
            return x.unsqueeze(1)
        if hop_length != window_length:
            return frame(x, window_length=window_length, hop_length=hop_length, padding_mode=padding_mode)
        return x.view(x.size(0), x.size(1), x.size(2) // window_length, window_length, x.size(3)).transpose(1, 2)

    def loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        q_z: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        delta_hat_i: Tensor,
        delta_hat_j: Tensor,
        delta_sigma: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # maximise likelihood p(x_i|z_j) framewise to ensure invariance to sequence length
        nll = negative_log_likelihood(x, x_hat, self.sigma_recon.pow(2).log()).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        outputs |= dict(log_likelihood_x=-nll.detach().mean(), )
        # MAP estimate of the alignment factor p(x|dt)p(dt)
        delta_hat = torch.cat([delta_hat_i.flatten(end_dim=1).unsqueeze(1), delta_hat_j], dim=0)
        nll_delta = negative_log_likelihood(delta_hat, torch.zeros(1).to(delta_hat.device), delta_sigma.pow(2).log())
        losses.append(nll_delta.mean())
        outputs |= dict(nll_delta=nll_delta.detach().mean())
        # standard normal dkl
        dkl = gaussian_kl_divergence_standard_prior(q_z).sum(dim=-1)
        losses.append(dkl.mean())
        outputs |= dict(dkl=dkl.detach().mean())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        return outputs

    def delta_sigma_current(self, t: int) -> Tensor:
        if self.delta_sigma_min is None: return self.delta_sigma_max
        return torch.tensor(bounded_sigmoid(t, **self.delta_sigma_params))

    @torch.no_grad()
    def metrics(
        self,
        x: Tensor,
        x_hat: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        delta: Tensor,
        delta_hat_i: Tensor,
        delta_hat_j: Tensor,
        delta_sigma: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        q_z = torch.cat([q_z_i, q_z_j.view(q_z_i.size())], dim=0)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        mu_hist = np.histogram(mu_z.flatten().cpu().numpy(), bins=32, range=[-5.0, 5.0])
        sigma_hist = np.histogram(sigma_z.flatten().cpu().numpy(), bins=32, range=[0.0, 2.0])
        mu_z_i, mu_z_j = q_z_i.chunk(2, dim=-1)[0], q_z_j.view(q_z_i.size()).chunk(2, dim=-1)[0]
        z_dist = (mu_z_j - mu_z_i).abs().mean()
        delta_hat = torch.cat([delta_hat_i, delta_hat_j.view(delta_hat_i.size())])
        delta_hat_hist = np.histogram(delta_hat.flatten().cpu().numpy(), bins=128, range=[-5.0, 5.0])
        delta_hat_mean = delta_hat.mean()
        delta_hat_var = delta_hat.var()
        delta_hat_seq_var = delta_hat.var(dim=1).mean()
        d, dh = delta.view(1, -1), delta_hat_j.view(1, -1)
        vx, vy = dh - dh.mean(), d - d.mean()
        delta_corr = ((vx * vy).sum(dim=-1) / (vx.norm(dim=-1) * vy.norm(dim=-1) + 1e-8)).mean().cpu().numpy()
        mae = (x_hat - x).abs().flatten(start_dim=-3).mean(dim=-1).mean()
        mse = (x_hat - x).pow(2).flatten(start_dim=-3).mean(dim=-1).mean()
        dkl_norm = ((-1/2 * (1 + log_sigma_sq_z - mu_z.pow(2) - log_sigma_sq_z.exp())).sum(dim=-1) / self.latent_dim).mean()
        return dict(
            mae=mae,
            mse=mse,
            mu_z=wandb.Histogram(np_histogram=mu_hist),
            sigma_z=wandb.Histogram(np_histogram=sigma_hist),
            z_dist=z_dist,
            delta_hat_hist=wandb.Histogram(np_histogram=delta_hat_hist),
            delta_hat_var=delta_hat_var,
            delta_hat_seq_var=delta_hat_seq_var,
            delta_corr=delta_corr,
            dkl_norm=dkl_norm,
            delta_sigma=delta_sigma,
        )

    # ------------------------------ LIGHTNING FUNCS --------------------------------- #

    def step(
        self,
        batch: Batch,
        batch_idx: int,
        stage: str,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        step_outputs = self(**batch, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        metrics = self.metrics(**step_outputs)
        self.log_dict(prefix_keys(loss_outputs, stage), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        if self.logger is not None and getattr(self.logger, "experiment") is not None:
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
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        return self.predict(**batch, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        pass

    def on_train_batch_end(self, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            step_outputs = self.training_step_outputs[0]
            self.on_batch_end(step_outputs, "train", min_num_samples=1)
        self.training_step_outputs.clear()

    def on_validation_batch_end(self, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            self.on_batch_end(step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_batch_end(self, step_outputs, stage: str, min_num_samples: int = 6):
        num_samples = min(min_num_samples, step_outputs["x"].size(0))
        num_frames = 6
        specs = step_outputs["x_i"].squeeze().cpu().numpy()
        recons = step_outputs["x_hat_i"].squeeze().cpu().numpy()
        for i in range(num_samples):
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), width_ratios=[0.97, 0.03], constrained_layout=True)
            mesh = self.log_mel_spectrogram.plot(specs[i].T, ax=axes[0, 0], vmin=specs.min(), vmax=specs.max())
            mesh = self.log_mel_spectrogram.plot(recons[i].T, ax=axes[1, 0], vmin=specs.min(), vmax=specs.max())
            axes[0, 0].set_title("Original")
            axes[1, 0].set_title("Reconstruction")
            fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
            fig.colorbar(mesh, cax=axes[1, 1], orientation="vertical")
            self.logger.experiment.log({f"{stage}/spectrogram": wandb.Image(fig)})
            plt.close(fig)
        # plot translated frames
        seq_len = step_outputs["q_z"].size(1)
        specs = step_outputs["x_j"].view(-1, seq_len, self.frame_window_length, self.num_mel_bins).cpu().numpy()
        recons = step_outputs["x_hat_j"].view(-1, seq_len, self.frame_window_length, self.num_mel_bins).cpu().numpy()
        for i in range(num_samples):
            fig, axes = plt.subplots(nrows=2, ncols=num_frames + 1, figsize=(10, 6), width_ratios=[*[0.97 / num_frames]*num_frames, 0.03], constrained_layout=True)
            for j in range(num_frames):
                mesh = self.log_mel_spectrogram.plot(specs[i, j].T, ax=axes[0, j], vmin=specs.min(), vmax=specs.max())
                mesh = self.log_mel_spectrogram.plot(recons[i, j].T, ax=axes[1, j], vmin=specs.min(), vmax=specs.max())
            axes[0, 0].set_title("Original")
            axes[1, 0].set_title("Reconstruction")
            fig.colorbar(mesh, cax=axes[0, -1], orientation="vertical")
            fig.colorbar(mesh, cax=axes[1, -1], orientation="vertical")
            self.logger.experiment.log({f"{stage}/frames/0": wandb.Image(fig)})
            plt.close(fig)

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
    def alignment_encoder_params(self):
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
    def delta_sigma_params(self):
        return dict(
            x_min=self.delta_sigma_step_start,
            x_max=self.delta_sigma_step_end,
            y_min=self.delta_sigma_min,
            y_max=self.delta_sigma_max,
            k=self.delta_sigma_step_slope,
        )

