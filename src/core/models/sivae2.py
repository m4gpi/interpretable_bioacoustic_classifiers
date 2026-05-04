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
from torch.nn import functional as F
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
from src.core.utils import Batch, soft_clip, detach_values, prefix_keys, bounded_sigmoid, linear_schedule, linear_decay
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior, circular_variance

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SIVAE"]

class ArcTan2(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("identity", torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.identity
        dx, dy = x.chunk(2, dim=-1)
        # normalise to unit circle (S¹), detach so the model cannot simply reduce magnitude
        norm = (dx.pow(2) + dy.pow(2) + self.epsilon).sqrt().detach()
        dx, dy = dx / norm, dy / norm
        # calculate the angle
        theta = torch.atan2(dy, dx)
        return theta, dx, dy

# class AlignmentEncoder(nn.Module):
#     def __init__(
#         self,
#         proj_dim: int = 512,
#         x_channels: int = 512,
#         x_freq_dim: int = 4,
#         x_time_dim: int = 48,
#         u_channels: int = 64,
#         u_freq_dim: int = 32,
#         u_time_dim: int = 6,
#         weight_init_std: float = 1e-1,
#     ) -> None:
#         super().__init__()
#         self.u_norm = torch.nn.LayerNorm([u_channels, u_time_dim, u_freq_dim])
#         self.x_norm = torch.nn.LayerNorm([x_channels, x_time_dim, x_freq_dim])
#         self.x_conv_freq = torch.nn.Conv2d(x_channels, x_channels // 4, kernel_size=(1, x_freq_dim))
#         self.u_conv_freq = torch.nn.Conv2d(u_channels, u_channels // 4, kernel_size=(1, u_freq_dim))
#         self.x_proj = nn.Linear(x_channels // 4 * x_time_dim, proj_dim)
#         self.u_proj = nn.Linear(u_channels // 4 * u_time_dim, proj_dim)
#         self.u_proj.weight.data *= weight_init_std # try to reduce noise in u contributing early in training
#         in_features = proj_dim * 2
#         self.mlp = nn.Sequential(
#             nn.Linear(in_features, in_features),
#             nn.LeakyReLU(),
#             torch.nn.Linear(in_features, 2, bias=False),
#         )
#         self.atan2 = ArcTan2()

#     def forward(self, x: torch.Tensor, u: torch.Tensor, t: int | None = None):
#         bs, seq, *_ = u.shape
#         x = x.flatten(end_dim=1)
#         # x = self.x_norm(x)
#         x = self.x_conv_freq(x).squeeze(-1)
#         x = x.unflatten(0, (bs, seq)).flatten(start_dim=-2)
#         u = u.flatten(end_dim=1)
#         # u = self.u_norm(u)
#         u = self.u_conv_freq(u).squeeze(-1)
#         u = u.unflatten(0, (bs, seq)).flatten(start_dim=-2)
#         x = self.x_proj(x) # (bs, seq, d)
#         u = self.u_proj(u) # (bs, seq, d)
#         h = torch.cat([x, u], dim=-1)
#         h = self.mlp(h)
#         theta, dx, dy = self.atan2(h)
#         delta = theta / torch.pi
#         return delta, theta, dx, dy

class AlignmentEncoder(torch.nn.Module):
    def __init__(
        self,
        x_channels: int = 512,
        x_freq_dim: int = 4,
        x_time_dim: int = 6,
        weight_init_std: float = 1e-1,
    ) -> nn.Module:
        super().__init__()
        # self.x_norm = torch.nn.LayerNorm([x_channels, x_time_dim, x_freq_dim])
        self.x_conf_freq = torch.nn.Conv2d(x_channels, x_channels // 4, kernel_size=(1, x_freq_dim))
        in_features = x_channels // 4 * x_time_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features // 2, 2, bias=False)
        )
        # # bias the network to predict small angles to start with
        # self.mlp[-1].bias.data = torch.tensor([1.0, 0.0], requires_grad=True)
        self.atan2 = ArcTan2()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        bs, seq, *_ = x.shape
        x = x.flatten(end_dim=1)
        # x = self.x_norm(x)
        x = self.x_conf_freq(x).squeeze(-1)
        x = x.unflatten(0, (bs, seq)).flatten(start_dim=-2)
        dxdy = self.mlp(x)
        theta, dx, dy = self.atan2(dxdy)
        # map to translation co-ordinates [-1, 1]
        delta = theta / torch.pi
        return delta, theta, dx, dy

class SIVAE(L.LightningModule):
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

    def __init__(
        self,
        sample_rate: int = 48_000,
        num_fft: int = 512,
        fft_window_length: int = 512,
        fft_hop_length: int = 384,
        mel_min_hertz: float | None = 0.0,
        mel_max_hertz: float | None = None,
        mel_scaling_factor: float | None = 4581.0,
        mel_break_frequency: float | None = 1750.0,
        frame_window_length: int = 192,
        frame_hop_length: int | None = 192,
        num_mel_bins: int = 64,
        latent_dim: int = 128,
        beta: float = 0.2,
        gamma: float = 1.0,
        sigma_x: float = 1.0,
        sigma_z_min: float = 0.0498,
        sigma_delta: float = 0.01,
        weight_init_std: float = 1e-3,
        cnn_block_width: int = 4,
        cnn_block_depth: int = 3,
        cnn_dropout_prob: float = 0.2,
        cnn_padding_mode: str = "circular",
        cnn_activation: str = "LEAK",
        cnn_feature_reduction_factor: int = 4,
        norm_type: str = "LN",
        mlp_activation: str = "LEAK",
        mlp_dropout_prob: float = 0.1,
        mlp_reduction_factor: int = 4,
        frame_padding_mode: str = "circular",
        learning_rate: float = 4e-5,
        optimiser_cls: str = "torch.optim.AdamW",
        optimiser_config: DictConfig | None = None,
        scheduler_cls: str | None = None,
        scheduler_config: DictConfig | None = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        translation_mode: str = "bicubic",
        x_i_frame_prob: float = 0.0,
        translation_idx: int = 0,
        delta_weight_init_std: float = 1e-1,
        delta_prob_step_start: int = 0,
        delta_prob_step_end: int = 20000,
        delta_prob_min: float = 0.0,
        delta_prob_max: float = 0.9,
        delta_sigma_step_start: int | None = None,
        delta_sigma_step_end: int | None = None,
        delta_sigma_min: float | None = None,
        delta_sigma_max: float = 2.0,
        delta_sigma_step_slope: float = 1.0,
        cross_decode_method: str = "soft",
    ) -> None:
        super().__init__()
        # assert 1e-3 < delta_sigma_min <= 1, "delta sigma min cannot be greater than 1 or less than 1e-3"
        # assert 1e-3 < delta_sigma_max <= 1, "delta sigma max cannot be greater than 1 or less than 1e-3"
        self.save_hyperparameters()
        self.sample_rate = sample_rate
        self.num_fft = num_fft
        self.fft_window_length = fft_window_length
        self.fft_hop_length = fft_hop_length
        self.mel_min_hertz = mel_min_hertz
        self.mel_max_hertz = mel_max_hertz
        self.mel_scaling_factor = mel_scaling_factor
        self.mel_break_frequency = mel_break_frequency
        self.frame_window_length = frame_window_length
        self.frame_hop_length = frame_hop_length
        self.num_mel_bins = num_mel_bins
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.sigma_x = sigma_x
        self.sigma_z_min = sigma_z_min
        self.sigma_delta = sigma_delta
        self.weight_init_std = weight_init_std
        self.cnn_block_width = cnn_block_width
        self.cnn_block_depth = cnn_block_depth
        self.cnn_dropout_prob = cnn_dropout_prob
        self.cnn_padding_mode = cnn_padding_mode
        self.cnn_activation = cnn_activation
        self.cnn_feature_reduction_factor = cnn_feature_reduction_factor
        self.norm_type = norm_type
        self.mlp_activation = mlp_activation
        self.mlp_dropout_prob = mlp_dropout_prob
        self.mlp_reduction_factor = mlp_reduction_factor
        self.frame_padding_mode = frame_padding_mode
        self.learning_rate = learning_rate
        self.optimiser_cls = optimiser_cls
        self.optimiser_config = optimiser_config
        self.scheduler_cls = scheduler_cls
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.translation_mode = translation_mode
        self.x_i_frame_prob = x_i_frame_prob
        self.translation_idx = translation_idx
        self.delta_weight_init_std = delta_weight_init_std
        self.delta_prob_step_start = delta_prob_step_start
        self.delta_prob_step_end = delta_prob_step_end
        self.delta_prob_min = delta_prob_min
        self.delta_prob_max = delta_prob_max
        self.delta_sigma_step_slope = delta_sigma_step_slope
        self.delta_sigma_step_start = delta_sigma_step_start
        self.delta_sigma_step_end = delta_sigma_step_end
        self.delta_sigma_min = delta_sigma_min
        self.delta_sigma_max = delta_sigma_max
        self.delta_sigma_step_slope = delta_sigma_step_slope
        self.cross_decode_method = cross_decode_method
        self.mel_max_hertz = self.mel_max_hertz or self.sample_rate / 2.0
        self.log_mel_spectrogram = LogMelSpectrogram(**self.log_mel_spectrogram_params)
        self.feature_encoder = init_cnn_feature_encoder(**self.cnn_encoder_params)
        self.content_encoder = init_mlp_content_encoder(**self.content_mlp_encoder_params)
        # self.alignment_encoder = AlignmentEncoder(
        #     x_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
        #     x_freq_dim=self.num_mel_bins // 2**(self.cnn_layers-1),
        #     x_time_dim=self.frame_window_length // 2**self.cnn_layers,
        #     u_channels=self.cnn_block_sizes[1] * self.cnn_block_width,
        #     u_freq_dim=self.num_mel_bins // 2**1,
        #     u_time_dim=self.frame_window_length // 2**2,
        #     proj_dim=384,
        #     weight_init_std=self.delta_weight_init_std,
        # )
        self.alignment_encoder = AlignmentEncoder(
            x_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
            x_freq_dim=self.num_mel_bins // 2**(self.cnn_layers-1),
            x_time_dim=self.frame_window_length // 2**self.cnn_layers,
            weight_init_std=self.delta_weight_init_std,
        )
        self.feature_decoder = init_cnn_feature_decoder(**self.cnn_decoder_params)
        self.content_decoder = init_mlp_content_decoder(**self.content_mlp_decoder_params)
        self.strict_loading = False
        self._reset_cache()

    def forward(self, x: Tensor, *args: Any, t: int | None = None, **kwargs: Any) -> Dict[str, Tensor]:
        delta_sigma = self.delta_sigma_current(self.trainer.global_step)
        # log mel is our acoustic front-end, but doesnt have to be...
        x = self.log_mel_spectrogram(x)
        # ensure x is a full sequence that can be divided into equal length frames
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins])
        # framed spectorgrams are target signal
        x_i = x
        x_i_framed = self.frame(x_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length)
        seq_len = x_i_framed.size(1)
        # coin flip to determine whether to encode full or translated spectrograms for target 1 (x_i)
        if torch.bernoulli(torch.tensor(1), self.x_i_frame_prob):
            x_i_framed = x_i_framed.flatten(end_dim=1)
            delta_i, theta_i, dx_i, dy_i = self.sample_circle(x_i_framed.size(0), 1, 1, scaling_factor=delta_sigma, device=x.device)
            x_i_framed = self.translation(x_i_framed.transpose(-1, -2).contiguous(), delta_i, mode=self.translation_mode).transpose(-1, -2).contiguous()
            x_i = x_i_framed
            q_z_i, delta_hat_i, theta_hat_i, dx_hat_i, dy_hat_i = self.encode(x_i, t=t) # (bs * seq, 1, ld)
            mu_z_i, log_sigma_sq_z_i = q_z_i.chunk(2, dim=-1)
        else:
            # when full spectrograms are used, delta / theta is necessarily zero
            delta_i, theta_i = torch.zeros(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device), torch.zeros(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device)
            dx_i, dy_i = torch.ones(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device), torch.zeros(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device)
            x_i_framed = x_i_framed.flatten(end_dim=1)
            q_z_i, delta_hat_i, theta_hat_i, dx_hat_i, dy_hat_i = self.encode(x_i, t=t) # (bs, seq, ld)
            mu_z_i, log_sigma_sq_z_i = q_z_i.chunk(2, dim=-1)
        # always randomly translated frames for target 2 (x_j)
        delta_j, theta_j, dx_j, dy_j = self.sample_circle(x_i_framed.size(0), 1, 1, scaling_factor=delta_sigma, device=x.device)
        x_j = self.translation(x_i_framed.transpose(-1, -2).contiguous(), delta_j, mode=self.translation_mode).transpose(-1, -2).contiguous()
        # encode posterior for translated frames separately
        q_z_j, delta_hat_j, theta_hat_j, dx_hat_j, dy_hat_j = self.encode(x_j, t=t) # (bs * seq, 1, ld)
        mu_z_j, log_sigma_sq_z_j = q_z_j.chunk(2, dim=-1)
        # decode to feature maps
        if self.cross_decode_method == "soft":
            # soft cross-decoding averages the distributions
            # mu_k = (mu_i + mu_j) / 2, sigma^2_k = (sigma^2_i + sigma^2_j) / 2^2
            mu_z = torch.stack([mu_z_i.flatten(end_dim=1), mu_z_j.flatten(end_dim=1)], dim=1).mean(dim=1)
            log_sigma_sq_z = (torch.stack([log_sigma_sq_z_i.flatten(end_dim=1).exp(), log_sigma_sq_z_j.flatten(end_dim=1).exp()], dim=1).sum(dim=1) / 4).log()
            z = Normal(mu_z, (0.5 * log_sigma_sq_z).exp()).rsample()  # (bs, seq, ld)
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1).view(q_z_i.size())
            U_hat_i = self.content_decoder(z) # (bs * seq, ch, fr, fq)
            U_hat_j = U_hat_i
        elif self.cross_decode_method == "hard":
            z_i = Normal(mu_z_i, (0.5 * log_sigma_sq_z_i).exp()).rsample()  # (bs, seq, ld)
            z_j = Normal(mu_z_j, (0.5 * log_sigma_sq_z_j).exp()).rsample()  # (bs * seq, 1, ld)
            q_z = torch.cat([q_z_i, q_z_j.view(q_z_i.size())], dim=0)
            U_hat_i = self.content_decoder(z_i.flatten(end_dim=1))
            U_hat_j = self.content_decoder(z_j.flatten(end_dim=1))
        else:
            raise Exception("Cross decode method not specified, terminating")

        if self.training:
            # during training, occasionally aid the decoder by providing the true delta
            true_delta_prob = linear_decay(t, **self.delta_prob_params)
            mask_i = torch.bernoulli(torch.full((delta_i.size(0), delta_i.size(1), 1), true_delta_prob, device=delta_i.device))
            mask_j = torch.bernoulli(torch.full((delta_j.size(0), delta_j.size(1), 1), true_delta_prob, device=delta_i.device))
            delta_i_mixed = mask_i * delta_i + (1 - mask_i) * delta_hat_i
            delta_j_mixed = mask_j * delta_j + (1 - mask_j) * delta_hat_j
            # reconstruct a contiguous sequence
            x_hat_i = self.cnn_decode(U_hat_j, delta_i_mixed) # (bs, 1, fr * seq, fq)
            # reconstruct independent translations
            x_hat_j = self.cnn_decode(U_hat_i, delta_j_mixed) # (bs * seq, 1, fr, fq)
        else:
            # reconstruct a contiguous sequence
            x_hat_i = self.cnn_decode(U_hat_j, delta_hat_i) # (bs, 1, fr * seq, fq)
            # reconstruct independent translations
            x_hat_j = self.cnn_decode(U_hat_i, delta_hat_j) # (bs * seq, 1, fr, fq)
        # frame for frame-wise loss
        x_hat_i_framed = self.frame(x_hat_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        x_framed = torch.cat([x_i_framed, x_j], dim=0)
        x_hat_framed = torch.cat([x_hat_i_framed, x_hat_j], dim=0)
        return dict(
            x=x,
            x_framed=x_framed, x_i=x_i, x_j=x_j,
            x_hat_framed=x_hat_framed, x_hat_i=x_hat_i, x_hat_j=x_hat_j,
            q_z=q_z, q_z_i=q_z_i, q_z_j=q_z_j,
            delta_i=delta_i, delta_j=delta_j, delta_hat_i=delta_hat_i, delta_hat_j=delta_hat_j,
            theta_i=theta_i, theta_j=theta_j, theta_hat_i=theta_hat_i, theta_hat_j=theta_hat_j,
            dx_i=dx_i, dx_j=dx_j, dx_hat_i=dx_hat_i, dx_hat_j=dx_hat_j,
            dy_i=dy_i, dy_j=dy_j, dy_hat_i=dy_hat_i, dy_hat_j=dy_hat_j,
            delta_sigma=delta_sigma,
            seq_len=seq_len,
        )

    def predict(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        x = self.log_mel_spectrogram(x)
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])
        q_z, delta_hat = self.encode(x)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        z = torch.distributions.Normal(mu_z, (1/2 * log_sigma_sq_z).exp()).rsample()
        x_hat = self.cnn_decode(self.content_decoder(z.flatten(end_dim=1)), delta_hat)
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

    def sample_circle(self, *args: Any, scaling_factor: float = 1.0, **kwargs: Any):
        # sample points uniformly from an arc of the circle, set by scaling factor
        delta = scaling_factor * ((torch.rand(*args, **kwargs) * 2) - 1)
        theta = torch.pi * delta
        dx, dy = torch.cos(theta), torch.sin(theta)
        return delta, theta, dx, dy

    @torch.no_grad()
    def predict_delta(self, x: torch.Tensor, num_samples: int = 10):
        x = self.log_mel_spectrogram(x)
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins])
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_hop_length)
        bound = torch.ones(num_samples, x_framed.size(0), x_framed.size(1), device=x.device).permute(1, 2, 0)
        delta = torch.distributions.Uniform(low=-bound, high=bound).sample()
        x_framed = x_framed.expand(num_samples, -1, -1, -1, -1, -1).permute(1, 2, 0, 3, 4, 5)
        bs, seq, n, *_ = x_framed.size()
        x_trans = self.translation(
            x_framed.flatten(end_dim=2).transpose(-1, -2).contiguous(),
            delta.flatten(end_dim=2),
            mode=self.translation_mode
        ).transpose(-1, -2).contiguous().unflatten(0, (bs, seq, n))
        _, delta_hat = self.encode(x_trans.flatten(end_dim=2))
        delta_hat = delta_hat.unflatten(0, (bs, seq, n)).view(delta.size())
        return dict(x_trans=x_trans, delta=delta, delta_hat=delta_hat)

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

    def encode(self, x: Tensor, hop_length: int | None = None, t: int | None = None) -> Tensor:
        x, u = self.cnn_encode(x)
        x = self.frame(x, window_length=self.latent_window_length, hop_length=(hop_length or self.frame_hop_length) // 2**(self.cnn_layers), padding_mode=self.frame_padding_mode)
        u = self.frame(u, window_length=self.frame_window_length // 2**2, hop_length=self.frame_window_length // 2**2, padding_mode=self.frame_padding_mode)
        q_z = self.content_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        if self.sigma_z_min is not None:
            mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
            sigma_latent = torch.tensor(self.sigma_z_min, dtype=torch.float32, requires_grad=False, device=mu_z.device)
            log_sigma_sq_z = log_sigma_sq_z.clamp(min=2*sigma_latent.pow(2).log())
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        delta_hat, theta_hat, dx_hat, dy_hat = self.alignment_encoder(x, u)
        return q_z, delta_hat, theta_hat, dx_hat, dy_hat

    def cnn_encode(self, x: Tensor) -> Tensor:
        us = []
        for i, block in enumerate(self.feature_encoder):
            x = block(x)
            us.append(x)
        return x, us[1]

    def decode(self, z: Tensor, delta: Tensor | None = None) -> Tensor:
        U = self.content_decoder(z.flatten(end_dim=1))
        x_hat = self.cnn_decode(U, delta)
        return x_hat

    def cnn_decode(self, U: Tensor, delta: Tensor) -> Tensor:
        for i, block in enumerate(self.feature_decoder):
            if i == self.translation_idx:
                U = U.transpose(-1, -2).contiguous()
                U = self.translation(U, delta.view(delta.size(0) * delta.size(1)), mode=self.translation_mode)
                U = U.transpose(-1, -2).contiguous()
            if i == len(self.feature_decoder) - 1:
                U = U.unflatten(0, (delta.size(0), delta.size(1))).transpose(1, 2).flatten(start_dim=2, end_dim=3)
            U = block(U)
        return U

    @staticmethod
    def translation(x: torch.Tensor, delta: torch.Tensor, mode: str) -> torch.Tensor:
        bs, ch, fq, ts = x.shape
        x_flat = x.view(bs, ch * fq, 1, ts)
        xs = torch.linspace(-1, 1, ts, device=x.device)
        grid_x = xs.view(1, 1, ts).expand(bs, 1, ts)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        xx = grid[..., 0] + delta.view(bs, 1, 1)
        grid[..., 0] = circular_boundary(xx) # FIXME we dont need modulo under this parametrisation, its implicit in atan2
        x_tilde = F.grid_sample(x_flat, grid, mode=mode, padding_mode="zeros", align_corners=True)
        return x_tilde.view(bs, ch, fq, ts)

    @staticmethod
    def frame(x: Tensor, window_length: int, hop_length: int | None = None, padding_mode: str = "circular") -> Tensor:
        if x.size(-2) == window_length:
            return x.unsqueeze(1)
        if hop_length != window_length:
            return frame(x, window_length=window_length, hop_length=hop_length, padding_mode=padding_mode)
        return x.view(x.size(0), x.size(1), x.size(2) // window_length, window_length, x.size(3)).transpose(1, 2)

    def loss(
        self,
        x_framed: Tensor,
        x_hat_framed: Tensor,
        q_z: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        delta_i: Tensor,
        delta_j: Tensor,
        delta_hat_i: Tensor,
        delta_hat_j: Tensor,
        theta_i: Tensor,
        theta_j: Tensor,
        theta_hat_i: Tensor,
        theta_hat_j: Tensor,
        dx_i: Tensor,
        dx_j: Tensor,
        dx_hat_i: Tensor,
        dx_hat_j: Tensor,
        dy_i: Tensor,
        dy_j: Tensor,
        dy_hat_i: Tensor,
        dy_hat_j: Tensor,
        delta_sigma: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # maximise likelihood p(x_i|z_j) framewise to ensure invariance to sequence length
        sigma_recon = torch.tensor(self.sigma_x, dtype=torch.float32, requires_grad=False, device=x_framed.device)
        nll = negative_log_likelihood(x_framed, x_hat_framed, sigma_recon.pow(2).log()).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        outputs |= dict(log_likelihood_x=-nll.detach().mean())
        # zero-centered prior for delta_hat
        # delta_hat = torch.cat([delta_hat_i.flatten(end_dim=1).unsqueeze(1), delta_hat_j], dim=0)
        # nll_delta = negative_log_likelihood(delta_hat, torch.zeros(1).to(delta_hat.device), delta_sigma.pow(2).log())
        # losses.append(nll_delta.mean())
        # outputs |= dict(nll_delta=nll_delta.detach().mean())
        # sigma_trans = torch.tensor(self.sigma_delta, dtype=torch.float32, requires_grad=False, device=delta_j.device)
        # nll_delta_j = negative_log_likelihood(delta_hat_j, delta_j, sigma_trans.pow(2).log())
        # additional supervision signal for delta, minimise relative angular difference
        angular_error_i = 2 * (1 - (dx_hat_i * dx_i + dy_hat_i * dy_i))
        angular_error_j = 2 * (1 - (dx_hat_j * dx_j + dy_hat_j * dy_j))
        alignment_loss = self.gamma * torch.stack([angular_error_i, angular_error_j.view(angular_error_i.size())]).mean()
        losses.append(alignment_loss)
        outputs |= dict(alignment_loss=alignment_loss.detach())
        # standard normal dkl
        dkl = self.beta * gaussian_kl_divergence_standard_prior(q_z).sum(dim=-1)
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
        x_framed: Tensor,
        x_hat_framed: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        delta_i: torch.Tensor,
        delta_j: Tensor,
        delta_hat_i: Tensor,
        delta_hat_j: Tensor,
        delta_sigma: Tensor,
        theta_i: Tensor,
        theta_j: Tensor,
        theta_hat_i: Tensor,
        theta_hat_j: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        # distribution of z mean and varaince
        q_z = torch.cat([q_z_i, q_z_j.view(q_z_i.size())], dim=0)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        mu_hist = np.histogram(mu_z.flatten().cpu().numpy(), bins=32, range=[-5.0, 5.0])
        sigma_hist = np.histogram(sigma_z.flatten().cpu().numpy(), bins=32, range=[0.0, 2.0])
        mu_z_i, mu_z_j = q_z_i.chunk(2, dim=-1)[0], q_z_j.view(q_z_i.size()).chunk(2, dim=-1)[0]
        # mean distance between shifted embeddings
        z_dist = (mu_z_j - mu_z_i).abs().mean()
        # distribution of delta predictions
        delta = torch.cat([delta_i, delta_j.view(delta_i.size())])
        delta_hist = np.histogram(delta.flatten().cpu().numpy(), bins=128, range=[-1.0, 1.0])
        delta_hat = torch.cat([delta_hat_i, delta_hat_j.view(delta_hat_i.size())])
        delta_hat_hist = np.histogram(delta_hat.flatten().cpu().numpy(), bins=128, range=[-1.0, 1.0])
        # variance of predicted theta both independently and across sequence
        theta = torch.cat([theta_i, theta_j.view(theta_i.size())])
        theta_hat = torch.cat([theta_hat_i, theta_hat_j.view(theta_hat_i.size())])
        theta_hat_var = circular_variance(theta_hat, dim=None).mean()
        theta_hat_seq_var = circular_variance(theta_hat, dim=1).mean()
        # anglular error
        d_theta = theta_hat - theta
        angular_distance = torch.atan2(torch.sin(d_theta), torch.cos(d_theta))
        angular_error = 1 - torch.cos(d_theta)
        d_theta_hist = np.histogram(d_theta.flatten().cpu().numpy(), bins=128, range=[0.0, 2.0])
        # reconstruction error
        d_x = (x_hat_framed - x_framed).flatten(start_dim=-3)
        mae = d_x.abs().mean(dim=-1).mean()
        mse = d_x.pow(2).mean(dim=-1).mean()
        # normalised KL
        dkl_norm = ((-1/2 * (1 + log_sigma_sq_z - mu_z.pow(2) - log_sigma_sq_z.exp())).sum(dim=-1) / self.latent_dim).mean()
        # current probability of observing true shift
        true_delta_prob = linear_decay(self.trainer.global_step, **self.delta_prob_params)
        return dict(
            mae=mae,
            mse=mse,
            mu_z=wandb.Histogram(np_histogram=mu_hist),
            sigma_z=wandb.Histogram(np_histogram=sigma_hist),
            true_delta_prob=true_delta_prob,
            z_dist=z_dist,
            delta_hist=wandb.Histogram(np_histogram=delta_hist),
            delta_hat_hist=wandb.Histogram(np_histogram=delta_hat_hist),
            theta_hat_var=theta_hat_var,
            theta_hat_seq_var=theta_hat_seq_var,
            angular_distance_mean=angular_distance.mean(),
            angular_distance_std=angular_distance.std(),
            angular_error_mean=angular_error.mean(),
            angular_error_std=angular_error.std(),
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
        loss_outputs, step_outputs = self.step(batch, batch_idx, "train", t=self.trainer.global_step)
        self.training_step_outputs.append(step_outputs)
        return loss_outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "val", t=self.trainer.global_step)
        self.validation_step_outputs.append(step_outputs)
        return loss_outputs

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        return dict()
        return self.predict(**batch, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        return self.predict(**batch, **kwargs)

    def on_train_batch_end(self, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if self.trainer.log_every_n_steps is not None and self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            step_outputs = self.training_step_outputs[0]
            self.on_batch_end(step_outputs, "train", min_num_samples=3)
        self.training_step_outputs.clear()

    def on_validation_batch_end(self, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            self.on_batch_end(step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_batch_end(self, step_outputs, stage: str, min_num_samples: int = 6):
        num_samples = min(min_num_samples, step_outputs["x"].size(0))
        seq_len = step_outputs["seq_len"]
        num_frames = 6
        if step_outputs["q_z_i"].size(1) == seq_len:
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
        else:
            specs = step_outputs["x_i"].view(-1, seq_len, self.frame_window_length, self.num_mel_bins).cpu().numpy()
            recons = step_outputs["x_hat_i"].view(-1, seq_len, self.frame_window_length, self.num_mel_bins).cpu().numpy()
            for i in range(num_samples):
                fig, axes = plt.subplots(nrows=2, ncols=num_frames + 1, figsize=(10, 6), width_ratios=[*[0.97 / num_frames]*num_frames, 0.03], constrained_layout=True)
                for j in range(num_frames):
                    mesh = self.log_mel_spectrogram.plot(specs[i, j].T, ax=axes[0, j], vmin=specs.min(), vmax=specs.max())
                    mesh = self.log_mel_spectrogram.plot(recons[i, j].T, ax=axes[1, j], vmin=specs.min(), vmax=specs.max())
                axes[0, 0].set_title("Original")
                axes[1, 0].set_title("Reconstruction")
                fig.colorbar(mesh, cax=axes[0, -1], orientation="vertical")
                fig.colorbar(mesh, cax=axes[1, -1], orientation="vertical")
                self.logger.experiment.log({f"{stage}/frames/i": wandb.Image(fig)})
                plt.close(fig)
        # plot translated frames
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
            self.logger.experiment.log({f"{stage}/frames/j": wandb.Image(fig)})
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

    @property
    def delta_prob_params(self):
        return dict(
            minimum=self.delta_prob_min,
            maximum=self.delta_prob_max,
            t_start=self.delta_prob_step_start,
            t_end=self.delta_prob_step_end,
        )

