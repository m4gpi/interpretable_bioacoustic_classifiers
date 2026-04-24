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
from src.core.transforms.translation import translation, circular_boundary
from src.core.utils import Batch, soft_clip, detach_values, prefix_keys, bounded_sigmoid, random_derange

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SIVAE"]

# class AlignmentEncoder(nn.Module):
#     def __init__(
#         self,
#         proj_dim: int = 16,
#         u_channels: int = 64,
#         x_channels: int = 512,
#         u_freq_dim: int = 32,
#         x_freq_dim: int = 4,
#     ) -> None:
#         super().__init__()
#         self.x_conv_freq = torch.nn.Conv2d(x_channels, x_channels, kernel_size=(1, x_freq_dim))
#         self.x_proj = nn.Linear(x_channels * 6, proj_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(proj_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 64),
#             nn.GELU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x: torch.Tensor, u: torch.Tensor):
#         bs, seq, ch1, fr1, fq1 = u.shape
#         _, _, ch2, fr2, fq2 = x.shape
#         x_feat = self.x_conv_freq(x.flatten(end_dim=1)).squeeze(-1).unflatten(0, (bs, seq))
#         h = self.x_proj(x_feat.flatten(start_dim=-2)) # (bs, seq, d)
#         # parametrise the shift
#         delta = self.mlp(h)
#         return delta

class AlignmentEncoder(nn.Module):
    def __init__(
        self,
        proj_dim: int = 16,
        x_channels: int = 512,
        x_freq_dim: int = 4,
        x_time_dim: int = 48,
        u_channels: int = 64,
        u_freq_dim: int = 32,
        u_time_dim: int = 6,
    ) -> None:
        super().__init__()
        self.x_conv_freq = torch.nn.Conv2d(x_channels, x_channels, kernel_size=(1, x_freq_dim))
        self.u_conv_freq = torch.nn.Conv2d(u_channels, u_channels, kernel_size=(1, u_freq_dim))
        self.x_proj = nn.Linear(x_channels * x_time_dim, proj_dim)
        self.u_proj = nn.Linear(u_channels * u_time_dim, proj_dim)
        in_features = proj_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.GELU(),
            nn.Linear(in_features * 2, in_features),
            nn.GELU(),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        bs, seq, *_ = u.shape
        x = self.x_conv_freq(x.flatten(end_dim=1)).squeeze(-1).unflatten(0, (bs, seq))
        u = self.u_conv_freq(u.flatten(end_dim=1)).squeeze(-1).unflatten(0, (bs, seq))
        x = self.x_proj(x.flatten(start_dim=-2))
        u = self.u_proj(u.flatten(start_dim=-2))
        h = torch.cat([x, u], dim=-1)  # (bs, seq, d)
        delta = self.mlp(h)
        return delta

def attention_entropy(attn: torch.Tensor, eps: float = 1e-8):
    attn = attn.clamp(min=eps)
    entropy = -(attn * attn.log()).sum(dim=-1)
    T = attn.size(-1)
    max_entropy = torch.log(torch.tensor(float(T), device=attn.device))
    norm_entropy = entropy / (max_entropy + eps)
    return entropy, entropy.mean(), norm_entropy

def sigma_schedule(
    step: int,
    sigma_min: float = 0.1,
    sigma_max: float = 4.0,
    warmup_steps: int = 10000,
    hold_steps: int = 10000,
) -> float:
    if step < hold_steps:
        return sigma_min
    if step >= hold_steps + warmup_steps:
        return sigma_max
    t = (step - hold_steps) / warmup_steps
    return sigma_min + (sigma_max - sigma_min) * t

@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class SIVAE(L.LightningModule):
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
    sigma_z_min: float = 0.0498
    sigma_x: float = 0.2
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

    cross_decode_method: str = "soft"
    k: int = 2
    contiguous_frames: bool = True
    translation_layer_idx: int = 3
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
        assert self.translation_layer_idx < self.cnn_layers
        self.save_hyperparameters()
        self.mel_max_hertz = self.mel_max_hertz or self.sample_rate / 2.0
        self.register_buffer("sigma_recon", torch.tensor(self.sigma_x, dtype=torch.float32))
        if self.sigma_z_min is not None:
            self.register_buffer("sigma_latent", torch.tensor(self.sigma_z_min, dtype=torch.float32))
        self.log_mel_spectrogram = LogMelSpectrogram(**self.log_mel_spectrogram_params)
        self.feature_encoder = init_cnn_feature_encoder(**self.cnn_encoder_params)
        self.content_encoder = init_mlp_content_encoder(**self.content_mlp_encoder_params)
        self.alignment_encoder = init_alignment_encoder(**self.alignment_encoder_params)
        # self.alignment_encoder = AlignmentEncoder(
        #     x_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
        #     x_freq_dim=self.num_mel_bins // 2**(self.cnn_layers-1),
        #     x_time_dim=self.frame_window_length // 2**self.cnn_layers,
        #     u_channels=self.cnn_block_sizes[1] * self.cnn_block_width,
        #     u_freq_dim=self.num_mel_bins // 2**1,
        #     u_time_dim=self.frame_window_length // 2**2,
        #     proj_dim=64,
        # )
        self.feature_decoder = init_cnn_feature_decoder(**self.cnn_decoder_params)
        self.content_decoder = init_mlp_content_decoder(**self.content_mlp_decoder_params)
        self.strict_loading = False
        self._reset_cache()

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        # ------ pre-processing ------ #
        x = self.log_mel_spectrogram(x)
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])

        # ------ shift samples------ #
        sigma_delta_params = dict(
            sigma_min=self.delta_sigma_min,
            sigma_max=self.delta_sigma_max,
            hold_steps=self.delta_sigma_step_start,
            warmup_steps=self.delta_sigma_step_end - self.delta_sigma_step_start,
        )
        sigma_delta = torch.tensor(sigma_schedule(self.trainer.global_step, **sigma_delta_params), device=x.device, dtype=torch.float32)
        # sigma_delta = torch.tensor(bounded_sigmoid(self.trainer.global_step, **self.sigma_delta_params))
        x_framed = x.view(x.size(0), -1, 1, self.frame_window_length, x.size(-1))
        delta = torch.randn(x_framed.size(0), x_framed.size(1), 1, 1, 1, device=x.device) * sigma_delta
        x_trans = translation(x_framed.flatten(end_dim=1), delta * sigma_delta).unflatten(0, (x_framed.size(0), x_framed.size(1)))
        # x_trans = self.k_way_translated_frames(x_framed, delta * sigma_delta, self.k - 1) # (k-1, bs, seq, 1, fr, fq)
        if not self.contiguous_frames:
            x = x_framed.flatten(end_dim=1)

        # ------ encode, reparam, cross-decode ------ #
        q_z, delta_hat = self.encode(x) # (bs, seq, ld)
        q_z_trans, delta_hat_trans = self.encode(x_trans.flatten(end_dim=1)) # (bs * seq, 1, ld)
        q_z_trans = q_z_trans.view(*q_z.size()) # (k-1, bs, seq, ld)
        delta_hat_trans = delta_hat_trans.view(*delta_hat.size()) # (k-1, bs, seq, 1)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        mu_z_trans, log_sigma_sq_z_trans = q_z_trans.chunk(2, dim=-1)
        mu_z = 1/2 * mu_z + 1/2 * mu_z_trans
        log_sigma_sq_z = ((log_sigma_sq_z.exp() + log_sigma_sq_z.exp()) / 2**2).log()
        z = torch.distributions.Normal(mu_z, (0.5 * log_sigma_sq_z).exp()).rsample()  # (bs, seq, ld)
        q_z_avg = torch.stack([mu_z, log_sigma_sq_z], dim=-1)
        q_z_stacked = torch.stack([q_z, q_z_trans], dim=1)
        U_hat = self.mlp_decode(torch.stack([z, z], dim=1).flatten(end_dim=2))
        U_hat, U_hat_trans = U_hat.unflatten(0, (z.size(0), 2, z.size(1))).chunk(2, dim=1)
        x_hat = self.cnn_decode(U_hat_trans.flatten(end_dim=2), delta_hat) # (bs, 1, fr * seq, fq)
        x_hat_framed = x_hat.view(*x_framed.size())
        x_hat_trans = self.cnn_decode(U_hat.flatten(end_dim=2), delta_hat_trans.view(-1, 1, 1))
        x_hat_trans = x_hat_trans.view(x_trans.size())

        x_framed_stacked = torch.stack([x_framed, x_trans], dim=1)
        x_hat_framed_stacked = torch.stack([x_hat_framed, x_hat_trans], dim=1)
        delta_hat_stacked = torch.stack([delta_hat, delta_hat_trans], dim=1)

        # q_z_trans, delta_hat_trans = self.encode(x_trans.flatten(end_dim=2)) # (k-1 * bs * seq, 1, ld)
        # q_z_trans = q_z_trans.view(-1, *q_z.size()) # (k-1, bs, seq, ld)
        # delta_hat_trans = delta_hat_trans.view(-1, *delta_hat.size()) # (k-1, bs, seq, 1)
        # q_z_stacked = torch.cat([q_z.unsqueeze(0), q_z_trans], dim=0) # (k, bs, seq, ld)
        # delta_hat_stacked = torch.cat([delta_hat.unsqueeze(0), delta_hat_trans], dim=0) # (k, bs, seq, 1)
        # z_stacked = self.k_way_cross_decode(q_z_stacked, method=self.cross_decode_method, k=self.k) # (k, bs, seq, ld)

        # # ------ decode & stack ------ #
        # U_hat = self.mlp_decode(z_stacked.flatten(end_dim=2))
        # U_hat = U_hat.unflatten(dim=0, sizes=(z_stacked.size(0), z_stacked.size(1), z_stacked.size(2))) # (k, bs, seq, ch, fr, fq)
        # U_hat, U_hat_trans = U_hat[0], U_hat[1:]
        # x_hat = self.cnn_decode(U_hat.flatten(end_dim=1), delta_hat) # (bs, 1, fr * seq, fq)
        # x_hat_framed = x_hat.view(*x_framed.size())
        # x_hat_trans = self.cnn_decode(U_hat_trans.flatten(end_dim=2), delta_hat_trans.flatten(end_dim=2).unsqueeze(1)) # (k-1 * bs * seq, 1, fr, fq)
        # x_hat_trans = x_hat_trans.view(x_trans.size()) # (k-1, bs, seq, 1, fr, fq)

        # x_framed_stacked = torch.cat([x_framed.unsqueeze(0), x_trans], dim=0)
        # x_hat_framed_stacked = torch.cat([x_hat_framed.unsqueeze(0), x_hat_trans], dim=0)

        return dict(
            x=x, x_framed=x_framed, x_framed_stacked=x_framed_stacked,
            x_hat=x_hat, x_hat_framed=x_hat_trans, x_hat_framed_stacked=x_hat_framed_stacked,
            q_z_avg=q_z_avg, q_z_stacked=q_z_stacked, q_z=q_z, q_z_trans=q_z_trans,
            # u_attn=u_attn, u_attn_trans=u_attn_trans,
            # x_attn=x_attn, x_attn_trans=x_attn_trans,
            delta=delta, delta_hat_stacked=delta_hat_stacked, delta_hat_trans=delta_hat_trans, delta_hat=delta_hat,
            sigma_delta=sigma_delta,
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
            delta=delta_hat,
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

    def k_way_translated_frames(self, x_framed: torch.Tensor, delta: torch.Tensor, k: int, mode: str = "bilinear") -> torch.Tensor:
        return torch.stack([
            translation(
                x_framed.flatten(end_dim=1).transpose(-1, -2),
                delta[i].flatten(end_dim=1),
                mode=mode,
                padding_mode="circular",
            ).transpose(-1, -2).unflatten(0, (x_framed.size(0), x_framed.size(1)))
            for i in range(k)
        ], dim=0)

    def k_way_cross_decode(self, q_z: torch.Tensor, method: str = "soft", k: int = 2):
        mu_zs, log_sigma_sq_zs = q_z.chunk(2, dim=-1)
        if method == "soft":
            # weighted average across translated representations
            mu_z = (1 / k * mu_zs).sum(dim=0, keepdims=True)
            log_sigma_sq_z = (1 / k**2 * log_sigma_sq_zs.exp()).sum(dim=0, keepdims=True).log()
            zs = Normal(mu_z, (1/2 * log_sigma_sq_z).exp()).rsample().expand(k, -1, -1, -1)
        elif method == "hard":
            # randomly derange (permute w/o original) translated representations
            zs = Normal(mu_zs, (1/2 * log_sigma_sq_zs).exp()).rsample()
            zs = zs[random_derange(k)]
        return zs

    def encode(self, x: Tensor, hop_length: int | None = None) -> Tensor:
        x, u = self.cnn_encode(x, encoder=self.feature_encoder)
        hop_length = (hop_length or self.frame_hop_length) // 2**(self.cnn_layers)
        u = self.frame(u, window_length=self.frame_window_length // 2**2, hop_length=self.frame_window_length // 2**2, padding_mode=self.frame_padding_mode)
        x = self.frame(x, window_length=self.latent_window_length, hop_length=hop_length, padding_mode=self.frame_padding_mode)
        delta = self.alignment_encode(x)
        q_z = self.content_encode(x, encoder=self.content_encoder)
        if self.sigma_z_min is not None:
            mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
            log_sigma_sq_z = log_sigma_sq_z.clamp(min=2*self.sigma_latent.log())
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        return q_z, delta

    def cnn_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tensor:
        U = []
        for i, block in enumerate(encoder):
            x = block(x)
            U.append(x)
        return x, U[1]

    def content_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tensor:
        q_z = encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        return q_z

    def alignment_encode(self, x: Tensor) -> Tensor:
        delta = self.alignment_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
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
                U = self.translate_cnn_features(U, delta.view(delta.size(0) * delta.size(1), 1, 1, 1))
            if i == len(self.feature_decoder) - 1:
                U = U.unflatten(0, (delta.size(0), delta.size(1))).transpose(1, 2).flatten(start_dim=2, end_dim=3)
            U = block(U)
        return U

    def translate_cnn_features(self, U: torch.Tensor, delta: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        return translation(U.transpose(-1, -2), delta, padding_mode="circular", mode=mode).transpose(-1, -2)

    @staticmethod
    def frame(x: Tensor, window_length: int, hop_length: int | None = None, padding_mode: str = "circular") -> Tensor:
        if x.size(-2) == window_length:
            return x.unsqueeze(1)
        if hop_length != window_length:
            return frame(x, window_length=window_length, hop_length=hop_length, padding_mode=padding_mode)
        return x.view(x.size(0), x.size(1), x.size(2) // window_length, window_length, x.size(3)).transpose(1, 2)

    def loss(
        self,
        x_framed_stacked: Tensor,
        x_hat_framed_stacked: Tensor,
        q_z_stacked: torch.Tensor,
        q_z_avg: torch.Tensor,
        delta_hat_stacked: torch.Tensor,
        sigma_delta: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # batch/sequence mean frame-wise sum reconstruction loss
        err = (x_framed_stacked - x_hat_framed_stacked).pow(2)
        var = self.sigma_recon.pow(2)
        nll = (1/2 * (var.log() + (err / var)))
        nll = nll.flatten(start_dim=-3).sum(dim=-1).mean()
        losses.append(nll)
        outputs |= dict(log_likelihood_x=-nll.detach().mean())
        # batch/sequence mean frame-wise shift loss
        delta_sigma_sq= sigma_delta.pow(2)
        nll_delta = (1/2 * (delta_sigma_sq.log() + (delta_hat_stacked.pow(2) / delta_sigma_sq)))
        nll_delta = nll_delta.mean()
        losses.append(nll_delta)
        outputs |= dict(log_likelihood_delta=-nll_delta.detach(), sigma_delta=sigma_delta)
        # batch/sequence mean frame-wise sum standard normal kl
        mu, log_sigma_sq = q_z_avg.chunk(2, dim=-1)
        dkl = (-1/2 * (1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())).sum(dim=-1).mean()
        losses.append(dkl)
        outputs |= dict(dkl=dkl.detach())
        # sum
        outputs |= dict(loss=sum(losses))
        return outputs

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
        self.log_dict(prefix_keys(loss_outputs, stage), batch_size=batch.x.size(0) * self.k, prog_bar=True, logger=False)
        if self.logger is not None and getattr(self.logger, "experiment") is not None:
            self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, stage)))
        return {**loss_outputs, **step_outputs}

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        outputs = self.step(batch, batch_idx, "train")
        self.training_step_outputs.append(outputs)
        return outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        outputs = self.step(batch, batch_idx, "val")
        self.validation_step_outputs.append(outputs)
        return outputs

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        return self.predict(**batch, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        pass

    def on_train_batch_end(self, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0 and len(self.training_step_outputs):
            step_outputs = self.training_step_outputs[0]
            self.on_batch_end(step_outputs, "train", min_num_samples=1)
        self.training_step_outputs.clear()

    def on_validation_batch_end(self, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            self.on_batch_end(step_outputs, "val")
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def on_test_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self.test_step_outputs.clear()

    def on_batch_end(self, step_outputs, stage: str, min_num_samples: int = 6):
        num_samples = min(min_num_samples, step_outputs["x"].size(0))
        if self.contiguous_frames:
            specs = step_outputs["x"].squeeze().cpu().numpy()
            recons = step_outputs["x_hat"].squeeze().cpu().numpy()
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
        # plot kth translated frames
        for k in range(2):
            specs = step_outputs["x_framed_stacked"][:, k].squeeze().cpu().numpy()
            recons = step_outputs["x_hat_framed_stacked"][:, k].squeeze().cpu().numpy()
            num_frames = 6
            for i in range(num_samples):
                fig, axes = plt.subplots(nrows=2, ncols=num_frames + 1, figsize=(10, 6), width_ratios=[*[0.97 / num_frames]*num_frames, 0.03], constrained_layout=True)
                for j in range(num_frames):
                    mesh = self.log_mel_spectrogram.plot(specs[i, j].T, ax=axes[0, j], vmin=specs.min(), vmax=specs.max())
                    mesh = self.log_mel_spectrogram.plot(recons[i, j].T, ax=axes[1, j], vmin=specs.min(), vmax=specs.max())
                axes[0, 0].set_title("Original")
                axes[1, 0].set_title("Reconstruction")
                fig.colorbar(mesh, cax=axes[0, -1], orientation="vertical")
                fig.colorbar(mesh, cax=axes[1, -1], orientation="vertical")
                self.logger.experiment.log({f"{stage}/frames/{k}": wandb.Image(fig)})
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

    def metrics(
        self,
        x_framed_stacked: Tensor,
        x_hat_framed_stacked: Tensor,
        q_z: torch.Tensor,
        q_z_trans: torch.Tensor,
        q_z_stacked: Tensor,
        delta: torch.Tensor,
        delta_hat_stacked: Tensor,
        # u_attn: torch.Tensor,
        # u_attn_trans: torch.Tensor,
        # x_attn: torch.Tensor,
        # x_attn_trans: torch.Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        mu_z, log_sigma_sq_z = q_z_stacked.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        mu_hist = np.histogram(mu_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[-5.0, 5.0])
        sigma_hist = np.histogram(sigma_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[0.0, 2.0])

        (mu_p, log_sigma_sq_p), (mu_q, log_sigma_sq_q), (mu_m, log_sigma_sq_m) = q_z.chunk(2, dim=-1), q_z_trans.chunk(2, dim=-1), q_z_avg.chunk(2, dim=-1)
        dkl_left = (-1/2 * (1 + log_sigma_sq_p - log_sigma_sq_m - (log_sigma_sq_p.exp() + (mu_p - mu_m).pow(2)) / log_sigma_sq_m.exp())).sum(dim=-1)
        dkl_right = (-1/2 * (1 + log_sigma_sq_q - log_sigma_sq_m - (log_sigma_sq_q.exp() + (mu_q - mu_m).pow(2)) / log_sigma_sq_m.exp())).sum(dim=-1)
        jsd = (1/2 * dkl_left + 1/2 * dkl_right).mean()
        delta_hat_mean = delta_hat_stacked.mean().cpu().numpy()
        delta_hat_var = delta_hat_stacked.var().cpu().numpy()
        delta_hat_seq_var = delta_hat_stacked.var(dim=2).mean().cpu().numpy()
        d, dh = delta.squeeze().view(1,-1), delta_hat_stacked.squeeze().view(2, -1)
        vx, vy = dh - dh.mean(), d - d.mean()
        delta_corr = ((vx * vy).sum(dim=-1) / (vx.norm(dim=-1) * vy.norm(dim=-1) + 1e-8)).mean().cpu().numpy()
        # u_entropy, u_entropy_mean, u_entropy_norm = attention_entropy(torch.cat([u_attn, u_attn_trans], dim=0))
        # u_entropy_hist = np.histogram(u_entropy.cpu().numpy(), bins=64, range=[0.0, np.log(u_attn.size(-1))])
        # x_entropy, x_entropy_mean, x_entropy_norm = attention_entropy(torch.cat([x_attn, x_attn_trans], dim=0))
        # x_entropy_hist = np.histogram(x_entropy.cpu().numpy(), bins=64, range=[0.0, np.log(x_attn.size(-1))])
        delta_hat_hist = np.histogram(circular_boundary(delta_hat_stacked).flatten().cpu().numpy(), bins=128, range=[-1.0, 1.0])
        mae = (x_hat_framed_stacked - x_framed_stacked).abs().flatten(start_dim=-3).mean(dim=-1).mean()
        mse = (x_hat_framed_stacked - x_framed_stacked).pow(2).flatten(start_dim=-3).mean(dim=-1).mean()
        dkl_norm = ((-1/2 * (1 + log_sigma_sq_z - mu_z.pow(2) - log_sigma_sq_z.exp())).sum(dim=-1) / self.latent_dim).mean()
        return dict(
            mae=mae,
            mse=mse,
            mu_z=wandb.Histogram(np_histogram=mu_hist),
            sigma_z=wandb.Histogram(np_histogram=sigma_hist),
            jsd=jsd,
            delta_hat_hist=wandb.Histogram(np_histogram=delta_hat_hist),
            # u_entropy=u_entropy_mean,
            # u_entropy_hist=wandb.Histogram(np_histogram=u_entropy_hist),
            # u_entropy_norm_std=u_entropy_norm.std(),
            # x_entropy=x_entropy_mean,
            # x_entropy_hist=wandb.Histogram(np_histogram=x_entropy_hist),
            # x_entropy_norm_std=x_entropy_norm.std(),
            delta_hat_var=delta_hat_var,
            delta_hat_seq_var=delta_hat_seq_var,
            delta_corr=delta_corr,
            dkl_norm=dkl_norm,
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
    def latent_hop_length(self) -> int:
        return (
            self.frame_hop_length // 2**(self.cnn_layers)
            if self.frame_hop_length is not None
            else self.latent_window_length
        )

    @property
    def frame_params(self):
        return dict(
            hop_length=self.frame_hop_length,
            window_length=self.frame_window_length,
            padding_mode=self.frame_padding_mode
        )

    @property
    def sigma_delta_params(self):
        return dict(
            x_min=self.delta_sigma_step_start,
            x_max=self.delta_sigma_step_end,
            y_min=self.delta_sigma_min,
            y_max=self.delta_sigma_max,
            k=self.delta_sigma_step_slope,
        )

