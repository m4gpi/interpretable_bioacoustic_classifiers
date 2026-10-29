import copy
import enum
import lightning as L
import hydra
import numpy as np
import pandas as pd
import pathlib
import omegaconf
import torch
import logging
import wandb

from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision.transforms import functional as T
from typing import Any, Dict, List, Tuple, Callable

from src.core.transforms.frame import frame_fold as frame
from src.core.utils import Batch, linear_decay, detach_values, prefix_keys, histogram_to_wandb
from src.core.utils.sketch import plot_latent_power_spectral_density_heatmap
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior

__all__ = ["VAE"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class EMA:
    def __init__(self, decay=0.999):
        self.decay = decay

    @torch.no_grad()
    def update(self, student: torch.nn.Module, teacher: torch.nn.Module) -> None:
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.lerp_(s_param, 1.0 - self.decay)

def gaussian_kernel(sigmas: torch.Tensor, mask_center: bool = False) -> torch.Tensor:
    sigmas = sigmas.float()
    s_max = sigmas.max()
    c = int(s_max / 0.3 + 1)
    k_size = 2 * c + 1
    x = torch.arange(k_size, device=sigmas.device) - c
    x = x.unsqueeze(0)
    sigmas = sigmas.unsqueeze(1)
    filt = torch.exp(-(x ** 2) / (2 * sigmas ** 2))
    if mask_center:
        filt[:, c] = 0.0
    filt = filt / filt.sum(dim=1, keepdim=True)
    return filt

def truncated_gaussian_kernel(sigmas: torch.Tensor, mask_center: bool = False) -> torch.Tensor:
    sigmas = sigmas.float()
    c = int(sigmas.max() / 0.3 + 1)
    k_size = c + 1
    # lag 0 is the current timestep; larger lags are further in the past
    lag = torch.arange(k_size, device=sigmas.device, dtype=sigmas.dtype)
    filt = torch.exp(-(lag.unsqueeze(0) ** 2) / (2 * sigmas.unsqueeze(1) ** 2))
    if mask_center:
        filt[:, 0] = 0.0
    filt = filt / filt.sum(dim=1, keepdim=True)
    # conv1d performs cross-correlation, so reverse the lag order:
    return filt.flip(dims=[1])

def laplace_kernel(sigmas: torch.Tensor, mask_center: bool = False) -> torch.Tensor:
    sigmas = sigmas.float()
    s_max = sigmas.max()
    c = int(s_max / 0.3 + 1)
    k_size = 2 * c + 1
    x = torch.arange(k_size, device=sigmas.device) - c
    x = x.unsqueeze(0)
    filt = torch.exp(-torch.abs(x) / sigmas.unsqueeze(1))
    if mask_center:
        filt[:, c] = 0.0
    filt = filt / filt.sum(dim=1, keepdim=True)
    return filt.unsqueeze(1)

def truncated_laplace_kernel(sigmas: torch.Tensor, mask_center: bool = False) -> torch.Tensor:
    sigmas = sigmas.float()
    c = int(sigmas.max() / 0.3 + 1)
    k_size = c + 1
    # lag 0 is the current timestep; larger lags are further in the past
    lag = torch.arange(k_size, device=sigmas.device, dtype=sigmas.dtype)
    filt = torch.exp(-torch.abs(lag.unsqueeze(0)) / sigmas.unsqueeze(1))
    if mask_center:
        filt[:, 0] = 0.0
    filt = filt / filt.sum(dim=1, keepdim=True)
    # conv1d performs cross-correlation, so reverse the lag order
    return filt.flip(dims=[1])

def causal_windows(x, kernels):
    L = kernels.size(-1)
    return F.pad(x, (L - 1, 0), value=0).unfold(-1, L, 1)

def bidirectional_windows(x, kernels):
    W = kernels.size(-1)
    c = W // 2
    return F.pad(x, (c, c), value=0).unfold(-1, W, 1)

class Kernel(enum.Enum):
    GAUSS = (gaussian_kernel, bidirectional_windows)
    GAUSS_TRUNC = (truncated_gaussian_kernel, causal_windows)
    LAPLACE = (laplace_kernel, bidirectional_windows)
    LAPLACE_TRUNC = (truncated_laplace_kernel, causal_windows)

    def __init__(self, init: Callable, window: Callable):
        self.init = init
        self.windows = window

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return cls.__members__.get(value)
        return None

def grad_stats(module):
    grads = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0, 0.0
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in grads]), 2)
    max_grad = max(g.abs().max() for g in grads)
    return norm, max_grad

class TCVAE(L.LightningModule):
    def __init__(
        self,
        front_end: omegaconf.DictConfig,
        feature_encoder: omegaconf.DictConfig,
        content_encoder: omegaconf.DictConfig,
        feature_decoder: omegaconf.DictConfig,
        content_decoder: omegaconf.DictConfig,
        latent_dim: int = 128,
        frequency_dim: int = 64,
        frame_window_length: int = 192,
        frame_padding_mode: int = "circular",
        sigma_x: float = 0.2,
        sigma_z_min: float = 1e-5,
        learning_rate: float = 1e-4,
        optimiser_cls: str = "torch.optim.AdamW",
        optimiser_config: omegaconf.DictConfig | None = {},
        scheduler_cls: str | None = None,
        scheduler_config: omegaconf.DictConfig | None = {},
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        smooth_prop: float = 0.75,
        num_temp_groups: int = 32,
        temp_log_sigma_min: float = -0.3,
        temp_log_sigma_max: float = 1.0,
        alpha: float | None = None,
        gamma: float = 1.0,
        sigma_bar_max: float = 0.5,
        sigma_bar_min: float = 0.2,
        sigma_bar_hold_steps: int | None = 15_000,
        sigma_bar_warmup_steps: int | None = 100_000,
        kernel_smoothing: bool = False,
        kernel_mode: str = "GAUSS_TRUNC",
        prior_smoothing: bool = True,
        ema_decay: float = 0.999,
        distilled: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.frequency_dim = frequency_dim
        self.frame_window_length = frame_window_length
        self.frame_padding_mode = frame_padding_mode
        self.sigma_x = sigma_x
        self.sigma_z_min = sigma_z_min
        self.smooth_prop = smooth_prop
        self.alpha = alpha
        self.gamma = gamma
        self.sigma_bar_max = sigma_bar_max
        self.sigma_bar_min = sigma_bar_min
        self.sigma_bar_hold_steps = sigma_bar_hold_steps
        self.sigma_bar_warmup_steps = sigma_bar_warmup_steps
        self.kernel_mode = kernel_mode
        self.num_temp_groups = num_temp_groups
        self.temp_log_sigma_min = temp_log_sigma_min
        self.temp_log_sigma_max = temp_log_sigma_max
        self.ema_decay = ema_decay
        self.kernel = Kernel(self.kernel_mode)

        self.learning_rate = learning_rate
        self.optimiser_cls = optimiser_cls
        self.optimiser_config = optimiser_config
        self.scheduler_cls = scheduler_cls
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.front_end = hydra.utils.instantiate(front_end)
        self.feature_encoder = hydra.utils.instantiate(feature_encoder)
        self.content_encoder = hydra.utils.instantiate(content_encoder)
        self.feature_decoder = hydra.utils.instantiate(feature_decoder)
        self.content_decoder = hydra.utils.instantiate(content_decoder)

        assert len(self.smooth_feature_idx) // self.num_temp_groups == len(self.smooth_feature_idx) / self.num_temp_groups

        # log sigma range, with warping factor gamma to tune spacing between kernel scales
        u = torch.linspace(0, 1, self.num_temp_groups)
        self.temporal_kernel_scales = torch.nn.Parameter(torch.stack([
            (self.temp_log_sigma_min + (self.temp_log_sigma_max - self.temp_log_sigma_min) * u**self.gamma).exp()
            for i in range(len(self.smooth_feature_idx) // self.num_temp_groups)
        ], dim=1).flatten().flip(dims=[0]), requires_grad=False)

        self.feature_encoder_ema = copy.deepcopy(self.feature_encoder)
        self.feature_encoder_ema.eval()
        self.feature_encoder_ema.requires_grad_(False)
        self.content_encoder_ema = copy.deepcopy(self.content_encoder)
        self.content_encoder_ema.eval()
        self.content_encoder_ema.requires_grad_(False)
        self.ema = EMA(decay=self.ema_decay)

    @property
    def smooth_feature_idx(self):
        return torch.arange(0, int(self.latent_dim * self.smooth_prop), device=self.device)

    @property
    def sharp_feature_idx(self):
        return torch.arange(int(self.latent_dim * self.smooth_prop), self.latent_dim, device=self.device)

    @property
    def sharp_features(self):
        mask = torch.zeros(self.latent_dim, device=self.device, dtype=int)
        mask[self.sharp_feature_idx] = 1
        return mask

    @property
    def smooth_features(self):
        mask = torch.zeros(self.latent_dim, device=self.device, dtype=int)
        mask[self.smooth_feature_idx] = 1
        return mask

    def random_intra_frame_shift(self, x: torch.Tensor) -> torch.Tensor:
        bs, _, ts, fq = x.shape
        l = ts - self.frame_window_length
        offset = torch.randint(low=0, high=self.frame_window_length, size=(bs,), device=x.device)
        idx = offset.unsqueeze(1) + torch.arange(l, device=x.device)
        batch_idx = torch.arange(bs, device=x.device).unsqueeze(1)
        x = x[batch_idx, 0, idx, :].unsqueeze(1)
        return x

    def pre_process(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.front_end(wav)
        x = x.transpose(-1, -2) # transpose time to inner axis
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])
        # apply a random temporal offset from the left and shorten the sequence by 1 frame during training
        if self.training:
            x = self.random_intra_frame_shift(x)
        return x

    def forward(self, x: torch.Tensor, *args: Any, t: int | None = None, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = {}
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        outputs |= dict(x=x, x_framed=x_framed)
        q_z, *_ = self.encode(x)
        outputs |= dict(q_z=q_z)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        z = torch.distributions.Normal(mu_z, (0.5 * log_sigma_sq_z).exp()).rsample()
        x_hat = self.decode(z).view(*x.size())
        x_hat_framed = self.frame(x_hat, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        outputs |= dict(x_hat=x_hat, x_hat_framed=x_hat_framed)
        with torch.no_grad():
            q_z_ema = self.encode_ema(x)
            q_z_bar = self.temporal_moment_matching(q_z_ema, self.smooth_feature_idx, self.temporal_kernel_scales)
        outputs |= dict(q_z_bar=q_z_bar, q_z_ema=q_z_ema)
        return outputs

    def predict(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.forward(x, *args, **kwargs)

    def encode(self, x: torch.Tensor, hop_length: int | None = None) -> Tuple[torch.Tensor]:
        x, *_ = self.feature_encoder(x)
        window_length = self.frame_window_length // 2**(self.feature_encoder.num_layers)
        hop_length = (hop_length or self.frame_window_length) // 2**(self.feature_encoder.num_layers)
        x = self.frame(x, window_length=window_length, hop_length=hop_length, padding_mode=self.frame_padding_mode)
        q_z = self.content_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        if self.sigma_z_min is not None:
            mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
            sigma_latent = torch.tensor(self.sigma_z_min, dtype=torch.float32, requires_grad=False, device=mu_z.device)
            log_sigma_sq_z = log_sigma_sq_z.clamp(min=2*sigma_latent.pow(2).log())
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        return q_z,

    def encode_ema(self, x: torch.Tensor, hop_length: int | None = None) -> Tuple[torch.Tensor]:
        x, *_ = self.feature_encoder_ema(x)
        window_length = self.frame_window_length // 2**(self.feature_encoder.num_layers)
        hop_length = (hop_length or self.frame_window_length) // 2**(self.feature_encoder.num_layers)
        x = self.frame(x, window_length=window_length, hop_length=hop_length, padding_mode=self.frame_padding_mode)
        q_z = self.content_encoder_ema(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        if self.training:
            mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
            sigma_current = linear_decay(self.trainer.global_step, **self.sigma_anneal_params)
            log_sigma_sq_z = log_sigma_sq_z.clamp(min=2*np.log(sigma_current))
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        return q_z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        bs, seq, *other_dims = z.size()
        U = self.content_decoder(z.flatten(end_dim=1))
        for i, block in enumerate(self.feature_decoder.blocks):
            if i == len(self.feature_decoder.blocks) - 1:
                U = U.unflatten(0, (bs, seq)).transpose(1, 2).flatten(start_dim=2, end_dim=3)
            U = block(U)
        return U

    @property
    def sigma_anneal_params(self):
        return dict(
            minimum=self.sigma_bar_min,
            maximum=self.sigma_bar_max,
            t_start=self.sigma_bar_hold_steps,
            t_end=self.sigma_bar_warmup_steps + self.sigma_bar_hold_steps
        )

    def temporal_moment_matching(
        self,
        q_z: torch.Tensor,
        feature_idx: torch.Tensor,
        kernel_scales: torch.Tensor,
        mask_center: bool = True,
        t: int | None = None,
    ) -> torch.Tensor:
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        # extract relevant features for smoothing
        mu = mu[:, :, feature_idx]
        log_sigma_sq = log_sigma_sq[:, :, feature_idx]
        sigma_sq = log_sigma_sq.exp()
        # permute time on the outer dim
        mu = mu.permute(0, 2, 1)
        sigma_sq = sigma_sq.permute(0, 2, 1)
        # init kernels and build windows
        kernels = self.kernel.init(kernel_scales, mask_center=mask_center)
        mu_windows = self.kernel.windows(mu, kernels)
        sigma_sq_windows = self.kernel.windows(sigma_sq, kernels)
        valid = self.kernel.windows(torch.ones_like(mu), kernels)
        # mask kernel weights that are invalid
        weights = kernels[None, :, None, :] * valid
        weight_sum = weights.sum(dim=-1, keepdim=True)
        has_valid = weight_sum > 0
        weights = weights / weight_sum.clamp_min(torch.finfo(weights.dtype).tiny)
        # weighted sum of gaussians (by convolution)
        # μ‾ = Σ wₜμₜ
        mu_bar = (weights * mu_windows).sum(dim=-1)
        # σ‾² = Σ wₜ(σ²ₜ+ μ²ₜ) - μ²‾ = Σ wₜσ²ₜ + Σ wₜ(μₜ - μ‾)²
        sigma_sq_within = (weights * sigma_sq_windows).sum(dim=-1)
        sigma_sq_between = (weights * (mu_windows - mu_bar.unsqueeze(-1)).pow(2)).sum(dim=-1)
        sigma_sq_bar = sigma_sq_within + sigma_sq_between
        # revert to standard normal prior at boundaries
        mu_bar = torch.where(has_valid.squeeze(-1), mu_bar, torch.zeros_like(mu_bar))
        sigma_sq_bar = torch.where(has_valid.squeeze(-1), sigma_sq_bar, torch.ones_like(sigma_sq_bar))
        # permute time to inner dim and concatenate
        mu_bar, sigma_sq_bar = mu_bar.permute(0, 2, 1), sigma_sq_bar.permute(0, 2, 1)
        q_z_bar = torch.cat([mu_bar, sigma_sq_bar.log()], dim=-1)
        return q_z_bar

    @staticmethod
    def frame(x: torch.Tensor, window_length: int, hop_length: int | None = None, padding_mode: str = "circular") -> torch.Tensor:
        if hop_length != window_length:
            return frame(x, window_length=window_length, hop_length=hop_length) if x.size(-2) > window_length else x.unsqueeze(1)
        return x.view(x.size(0), x.size(1), x.size(2) // window_length, window_length, x.size(3)).transpose(1, 2)

    def loss(
        self,
        x: torch.Tensor,
        x_framed: torch.Tensor,
        x_hat_framed: torch.Tensor,
        q_z: torch.Tensor,
        q_z_bar: torch.Tensor | None = None,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        outputs = dict()
        losses = []
        # batch/sequence mean frame-wise sum reconstruction loss
        sigma_recon = torch.tensor(self.sigma_x, dtype=torch.float32, requires_grad=False, device=x_framed.device)
        nll = negative_log_likelihood(x_framed, x_hat_framed, sigma_recon.pow(2).log()).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        outputs |= dict(log_likelihood_x=-nll.detach().mean())
        # frame-wise standard normal kl for 'sharp' features
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        dkl = (-1/2 * (1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp()))
        sharp_dkl = (self.sharp_features * dkl).sum(dim=-1).mean()
        losses.append(sharp_dkl)
        outputs |= dict(
            sharp_dkl=sharp_dkl.detach(),
            sharp_dkl_norm=sharp_dkl.detach() / len(self.sharp_feature_idx)
        )
        # alpha weighted standard normal kl for 'smooth' features
        smooth_dkl_uw = (self.smooth_features * dkl).sum(dim=-1).mean()
        smooth_dkl = self.alpha * smooth_dkl_uw
        outputs |= dict(
            smooth_dkl=smooth_dkl.detach(),
            smooth_dkl_uw=smooth_dkl_uw.detach(),
            smooth_dkl_norm_uw=smooth_dkl_uw.detach() / len(self.smooth_feature_idx)
        )
        losses.append(smooth_dkl)
        # temporal coherence kl
        mu_smooth, log_sigma_sq_smooth = mu[:, :, self.smooth_feature_idx], log_sigma_sq[:, :, self.smooth_feature_idx]
        mu_bar, log_sigma_sq_bar = q_z_bar.chunk(2, dim=-1)
        var_ratio = (log_sigma_sq_smooth - log_sigma_sq_bar).exp()
        sq_err = (mu_smooth - mu_bar).pow(2) * (-log_sigma_sq_bar).exp()
        temp_coh_dkl = -1/2 * (1 + log_sigma_sq_smooth - log_sigma_sq_bar - var_ratio - sq_err)
        temp_coh_dkl_uw = temp_coh_dkl.sum(dim=-1).mean()
        # (1 - alpha) weighted temporal coherence kl
        temp_coh_dkl = (1 - self.alpha) * temp_coh_dkl_uw
        losses.append(temp_coh_dkl)
        outputs |= dict(
            temp_coh_dkl=temp_coh_dkl.detach(),
            temp_coh_dkl_uw=temp_coh_dkl_uw.detach(),
            temp_coh_dkl_norm_uw=temp_coh_dkl_uw.detach() / len(self.smooth_feature_idx),
        )
        # log the sum of kls
        outputs |= dict(dkl=(sharp_dkl + smooth_dkl + temp_coh_dkl).detach())
        outputs |= dict(smooth_temp_dkl=(smooth_dkl + temp_coh_dkl).detach())
        # sum all loss components
        outputs |= dict(loss=sum(losses))
        return outputs

    @torch.no_grad()
    def metrics(
        self,
        x_framed: torch.Tensor,
        x_hat_framed: torch.Tensor,
        q_z: torch.Tensor,
        q_z_bar: torch.Tensor | None = None,
        include_hists: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        outputs = {}
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        if include_hists:
            mu_hist = np.histogram(mu_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[-5.0, 5.0])
            sigma_hist = np.histogram(sigma_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[0.0, 2.0])
            outputs |= dict(mu_z_hist=mu_hist, sigma_z_hist=sigma_hist)
        mae = (x_hat_framed - x_framed).abs().flatten(start_dim=-3).mean(dim=-1).mean()
        mse = (x_hat_framed - x_framed).pow(2).flatten(start_dim=-3).mean(dim=-1).mean()
        outputs |= dict(mse=mse, mae=mae)
        dkl_norm = ((-1/2 * (1 + log_sigma_sq_z - mu_z.pow(2) - log_sigma_sq_z.exp())).sum(dim=-1) / self.latent_dim).mean()
        outputs |= dict(dkl_norm=dkl_norm)
        delta_sharp = mu_z[:, 1:, self.sharp_feature_idx] - mu_z[:, :-1, self.sharp_feature_idx]
        delta_smooth = mu_z[:, 1:, self.smooth_feature_idx] - mu_z[:, :-1, self.smooth_feature_idx]
        outputs |= dict(
            sharp_neighbour_abs_err=delta_sharp.abs().mean(),
            sharp_neighbour_sq_err=delta_sharp.pow(2).mean(),
            sharp_neighbour_std=delta_sharp.std(),
            smooth_neighbour_abs_err=delta_smooth.abs().mean(),
            smooth_neighbour_sq_err=delta_smooth.pow(2).mean(),
            smooth_neighbour_std=delta_smooth.std(),
        )
        if q_z_bar is not None:
            mu_bar, log_sigma_sq_bar = q_z_bar.chunk(2, dim=-1)
            outputs |= dict(
                mu_bar_std=mu_bar.std(),
                mean_shift_abs_err=(mu_z[:, :, self.smooth_feature_idx] - mu_bar).abs().mean(),
            )
            if include_hists:
                mu_bar_hist = np.histogram(mu_bar.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[-5.0, 5.0])
                sigma_bar_hist = np.histogram((0.5 * log_sigma_sq_bar).exp().flatten(end_dim=-2).cpu().numpy(), bins=32, range=[0.0, 2.0])
                outputs |= dict(mu_bar_hist=mu_bar_hist, sigma_bar_hist=sigma_bar_hist)
        return outputs

    @torch.no_grad()
    def embed(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataloader_idx: int = 0,
        frame_hop_length: float | None = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        x, *_ = batch
        frame_hop_length = frame_hop_length or self.frame_window_length // 2
        q_z, *_ = self.encode(x, hop_length=frame_hop_length)
        bs, seq, *_ = q_z.size()
        sample_idx = batch.s.cpu().unsqueeze(0).repeat(seq, 1).t().flatten()
        seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq).cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().cpu()
        ref_column_types = dict(file_i=int, dataloader_idx=int, timestep=int)
        feat_column_types = dict(
            **{ f"z_mean_{d}": float  for d in range(q_z.size(-1)//2) },
            **{ f"z_log_var_{d}": float  for d in range(q_z.size(-1)//2) },
        )
        column_types = (ref_column_types | feat_column_types)
        return pd.DataFrame(
            data=dict(zip(column_types.keys(), [sample_idx, dl_idx, seq_idx, *q_z.flatten(end_dim=1).cpu().t()])),
            columns=column_types.keys(),
        ).astype(dtype=column_types).set_index(list(ref_column_types.keys()))

    @property
    def time_series_params(self):
        return dict(
            audio_sample_rate=self.front_end.sample_rate,
            audio_fft_hop_length=self.front_end.fft_hop_length,
            audio_frame_length_hops=self.frame_window_length,
        )

    def tracking_figures(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        q_z: torch.Tensor,
        figsize: Tuple[int, int] = (6, 6),
        dpi: int = 75,
        num_samples: int = 6,
        **kwargs: Any,
    ) -> List:
        figures = []
        num_samples = min(num_samples, x.size(0))
        xs = x.squeeze().cpu().numpy()
        x_hats = x_hat.squeeze().cpu().numpy()
        mu, _ = q_z.chunk(2, dim=-1)
        mu = mu.cpu().numpy()
        for i in range(num_samples):
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize, width_ratios=[0.97, 0.03], constrained_layout=True, dpi=dpi)
            mesh = self.front_end.plot(xs[i].T, ax=axes[0, 0], vmin=x.min(), vmax=x.max())
            mesh = self.front_end.plot(x_hats[i].T, ax=axes[1, 0], vmin=x.min(), vmax=x.max())
            axes[0, 0].set_title("Original")
            axes[1, 0].set_title("Reconstruction")
            fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
            fig.colorbar(mesh, cax=axes[1, 1], orientation="vertical")
            im = plot_latent_power_spectral_density_heatmap(mu[i], fft_length=mu[i].shape[0], **self.time_series_params, ax=axes[2, 0])
            axes[2, 0].set_title("Latent Power Spectral Density")
            fig.colorbar(im, cax=axes[2, 1], orientation="vertical")
            figures.append(("latent_spectra", fig))
        return figures

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        # run training
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
        # persist the model configuration
        checkpoint_dir = pathlib.Path(trainer.checkpoint_callback.dirpath)
        if checkpoint_dir.exists():
            config_path = checkpoint_dir / "config.yaml"
            log.info(f"Saving run config to {config_path}")
            omegaconf.OmegaConf.save(config, config_path)
        # running test
        log.info(f"Testing <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, dataloaders=data_module.predict_dataloader())

    def on_before_optimizer_step(self, optimizer):
        encoders = [("feature_encoder", self.feature_encoder), ("content_encoder", self.feature_encoder)]
        ema_encoders = [("feature_encoder_ema", self.feature_encoder_ema), ("content_encoder_ema", self.feature_encoder_ema)]
        decoders = [("feature_decoder", self.feature_decoder), ("content_decoder", self.feature_decoder)]
        for name, module in [*encoders, *decoders, *ema_encoders]:
            norm, max_grad = grad_stats(module)
            self.logger.experiment.log({"global_step": self.trainer.global_step, f"grad/{name}_norm": norm, f"grad/{name}_max": max_grad})

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update the student
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        # update the teacher
        self.ema.update(self.feature_encoder, self.feature_encoder_ema)
        self.ema.update(self.content_encoder, self.content_encoder_ema)

    def step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self.forward(**batch, t=self.trainer.global_step, **kwargs)
        loss_outputs = self.loss(**step_outputs, t=self.trainer.global_step)
        step_outputs = detach_values(step_outputs)
        return loss_outputs, step_outputs

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(loss_outputs, "train"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        metrics = histogram_to_wandb(self.metrics(**step_outputs))
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs | metrics, "train")))
        return loss_outputs | step_outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        metrics = histogram_to_wandb(self.metrics(**step_outputs))
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs | metrics, "val")))
        return loss_outputs | step_outputs

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predict(**batch, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predict(**batch, **kwargs)

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        x = self.pre_process(batch.x)
        return Batch(x=x, **{k: batch[k] for k in batch.keys() if k != "x"})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimiser_config = omegaconf.DictConfig(dict(_target_=self.optimiser_cls, **(self.optimiser_config or {})))
        optimiser = hydra.utils.instantiate(optimiser_config, params=self.parameters(), lr=self.learning_rate)
        if self.scheduler_cls is not None:
            scheduler_config = omegaconf.DictConfig(dict(_target_=self.scheduler_cls, **(self.scheduler_config or {})))
            scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimiser)
            return [optimiser], [dict(
                scheduler=scheduler,
                interval=self.scheduler_interval,
                frequency=self.scheduler_frequency
            )]
        return optimiser

# def temporal_smoothing(
#     self,
#     q_z: torch.Tensor,
#     kernel_scales: torch.Tensor,
#     mask_center: bool = False,
#     padding_mode: str = "zeros",
# ) -> torch.Tensor:
#     kernels = gaussian_kernel(kernel_scales, mask_center=mask_center)
#     K = (kernels.size(-1) - 1) // 2
#     mu, log_sigma_sq = q_z.chunk(2, dim=-1)
#     mu_bar = torch.zeros_like(mu)
#     # smooth over selected features by convolution
#     mu_bar[:, :, self.smooth_feature_idx] = F.conv1d(
#         F.pad(mu[:, :, self.smooth_feature_idx].permute(0, 2, 1), (K, K), mode=padding_mode),
#         kernels,
#         groups=kernel_scales.size(0)
#     ).permute(0, 2, 1)
#     # copy over remaining
#     mu_bar[:, :, self.sharp_feature_idx] = mu[:, :, self.sharp_feature_idx]
#     return torch.cat([mu_bar, log_sigma_sq], dim=-1)

# # apply hard smoothing (if specified)
# if self.kernel_smoothing:
#     q_z = self.temporal_smoothing(q_z, self.temporal_kernel_scales)

# if self.kernel_smoothing:
#     # Q: just use alpha?
#     # weight the KL inversely proportional to the kernel scale
#     # offsetting such that group with smallest scale doesn't have weight of 1
#     scales = torch.stack([
#         torch.linspace(
#             self.temp_log_sigma_min,
#             self.temp_log_sigma_max + ((self.temp_log_sigma_max - self.temp_log_sigma_min) / (self.num_temp_groups - 1)),
#             self.num_temp_groups + 1,
#             device=dkl.device,
#         ).exp()
#         for i in range(len(self.smooth_feature_idx) // self.num_temp_groups)
#     ], dim=1).flatten().flip(dims=[0])
#     inv_scales = 1 / scales
#     weight = (inv_scales / inv_scales.max())[:-3]
#     smooth_dkl = (weight * dkl[:, :, self.smooth_feature_idx]).sum(dim=-1).mean()
#     losses.append(smooth_dkl)
#     outputs |= dict(smooth_dkl=smooth_dkl.detach())

