import enum
import itertools
import lightning as L
import logging
import numpy as np
import hydra
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
from typing import Any, Dict, Tuple, List

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
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior, autoregressive_prior
from src.core.transforms.frame import unframe_fold as unframe, frame_fold as frame
from src.core.transforms.translation import translation
from src.core.utils.sketch import plot_mel_spectrogram
from src.core.utils import soft_clip, linear_decay, bounded_sigmoid, nth_percentile, detach_values, prefix_keys, to_snake_case

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.switch_backend('agg')

__all__ = ["SIVAE"]

@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class SIVAE(L.LightningModule):
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
    sigma_x_min: float = 0.0498
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
    sigma_z_max: float = 1.0
    sigma_z_min: float | None = None
    sigma_z_step_start: int | None = 0
    sigma_z_step_end: int | None = 1
    sigma_z_mode: str = "FIXED"
    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1

    p_dt_step_start: int | None = None
    p_dt_step_end: int | None = None
    p_dt_sigma_min: float | None = None
    p_dt_sigma_max: float = 2.0
    p_dt_step_slope: float = 1.0

    smooth_prop: float | None = None
    smooth_alpha_min: float | None = None
    smooth_alpha_max: float | None = None
    non_smooth_alpha: float | None = None

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(self, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        log.info(f"Beginning testing <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, dataloaders=data_module.test_dataloader())

    def evaluate(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any]):
        log.info(f"Encoding <{config.data.get('_target_')}> with <{config.model.get('_target_')}>")
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path is not None, f"No checkpoint found at {ckpt_path}"
        predictions = trainer.predict(self, data_module.predict_dataloader(), ckpt_path=config.get("ckpt_path"), return_predictions=True)
        df = pd.concat(list(itertools.chain(*predictions)), axis=0)
        save_path = pathlib.Path(ckpt_path).parent / "features.parquet"
        log.info(f"Saving predictions to {save_path}")
        df.to_parquet(save_path)

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self):
        self.mel_max_hertz = self.mel_max_hertz or self.sample_rate / 2.0
        self.sigma_z_min = self.sigma_z_min or self.sigma_z_max
        self.feature_encoder = init_cnn_feature_encoder(
            block_sizes=self.cnn_block_sizes,
            block_width=self.cnn_block_width,
            block_depth=self.cnn_block_depth,
            dropout_prob=self.cnn_dropout_prob,
            padding_mode=self.cnn_padding_mode,
            norm_fn=NormType[self.norm_type],
            activation_fn=Activation[self.cnn_activation],
            weight_init_std=self.weight_init_std,
        )
        self.content_encoder = init_mlp_content_encoder(
            in_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
            out_channels=self.cnn_block_sizes[-1] * self.cnn_block_width // self.cnn_feature_reduction_factor,
            feature_height=self.latent_window_length,
            feature_width=self.latent_frequency_dim,
            mlp_reduction_factor=self.mlp_reduction_factor,
            activation_fn=Activation[self.mlp_activation],
            dropout_prob=self.mlp_dropout_prob,
            out_features=self.latent_dim * 2,
        )
        self.feature_decoder = init_cnn_feature_decoder(
            block_sizes=list(reversed(self.cnn_block_sizes)),
            block_width=self.cnn_block_width,
            block_depth=self.cnn_block_depth,
            dropout_prob=self.cnn_dropout_prob,
            padding_mode=self.cnn_padding_mode,
            norm_fn=NormType[self.norm_type],
            activation_fn=Activation[self.cnn_activation],
        )
        self.content_decoder = init_mlp_content_decoder(
            in_features=self.latent_dim,
            in_channels=self.cnn_block_sizes[-1] * self.cnn_block_width // self.cnn_feature_reduction_factor,
            out_channels=self.cnn_block_sizes[-1] * self.cnn_block_width,
            feature_height=self.latent_window_length,
            feature_width=self.latent_frequency_dim,
            mlp_reduction_factor=self.mlp_reduction_factor,
            activation_fn=Activation[self.mlp_activation],
            dropout_prob=self.mlp_dropout_prob,
        )
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
        self._reset_cache()

    @property
    def cnn_block_sizes(self):
        return [8, 16, 32, 64, 128]

    @property
    def cnn_layers(self):
        return len(self.cnn_block_sizes)

    @property
    def latent_splits(self) -> Tuple[Tensor, ...]:
        return torch.arange(self.latent_dim),

    @property
    def latent_frequency_dim(self) -> int:
        return self.num_mel_bins // 2**(self.cnn_layers - 1)

    @property
    def latent_window_length(self) -> int:
        return self.frame_window_length // 2**(self.cnn_layers)

    @property
    def latent_hop_length(self) -> int:
        return self.frame_hop_length // 2**(self.cnn_layers) if self.frame_hop_length is not None else self.latent_window_length

    @property
    def frame_params(self):
        return dict(
            hop_length=self.frame_hop_length,
            window_length=self.frame_window_length,
            padding_mode=self.frame_padding_mode
        )

    @property
    def latent_frame_params(self):
        return dict(
            hop_length=self.latent_hop_length,
            window_length=self.latent_window_length,
            padding_mode=self.frame_padding_mode
        )

    @property
    def spectrogram_params(self):
        return dict(
            sample_rate=self.sample_rate,
            hop_length=self.fft_hop_length,
            window_length=self.fft_window_length,
            fft_length=int(np.power(2, np.ceil(np.log(self.fft_window_length) / np.log(2.0)))),
            mel_min_hertz=self.mel_min_hertz,
            mel_max_hertz=self.mel_max_hertz,
            mel_scaling_factor=self.mel_scaling_factor,
            mel_break_frequency=self.mel_break_frequency,
        )

    @property
    def latent_splits(self) -> Tuple[Tensor, Tensor]:
        return self.smooth_idx, self.non_smooth_idx

    @property
    def num_non_smooth(self):
        return self.latent_dim - self.num_smooth

    @property
    def num_smooth(self):
        return int(self.latent_dim * (self.smooth_prop or 0))

    @property
    def non_smooth_idx(self):
        return torch.arange(0, self.num_non_smooth)

    @property
    def smooth_idx(self):
        return torch.arange(self.num_non_smooth, self.num_non_smooth + self.num_smooth)

    @property
    def alpha(self) -> Tensor:
        return torch.cat([
            torch.ones(len(self.non_smooth_idx)) * self.non_smooth_alpha,
            torch.linspace(self.smooth_alpha_min, self.smooth_alpha_max, len(self.smooth_idx))
        ])[torch.cat([self.non_smooth_idx, self.smooth_idx])].to(list(self.parameters())[0].device)

    @property
    def p_dt_sigma_params(self):
        return dict(
            x_min=self.p_dt_step_start,
            x_max=self.p_dt_step_end,
            y_min=self.p_dt_sigma_min,
            y_max=self.p_dt_sigma_max,
            k=self.p_dt_step_slope or 1.0,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # ensure x_i is a full sequence that can be divided into equal length frames
        x_i = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins])
        # encode posterior for full sequence
        q_z_i, dt_i = self.encode(x_i) # (bs, seq, ld)
        mu_x_i, log_sigma_sq_x_i = q_z_i.chunk(2, dim=-1)
        # x_j is x_i chunked into independently translated frames
        x_i_framed = frame(x_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        epsilon = torch.randn(x_i_framed.size(0), 1, 1, 1).to(x_i.device)
        sigma_dt = self.p_dt_sigma_current(self.trainer.global_step)
        dt = epsilon * sigma_dt
        x_j = translation(x_i_framed, dt, padding_mode="circular")
        # encode posterior for translated frames separately
        q_z_j, dt_j = self.encode(x_j) # (bs * seq, 1, ld)
        mu_x_j, log_sigma_sq_x_j = q_z_j.chunk(2, dim=-1)
        # soft cross-decoding averages the distributions
        # mu_k = (mu_i + mu_j) / 2, sigma^2_k = (sigma^2_i + sigma^2_j) / 2^2
        mu_x = torch.stack([mu_x_i.flatten(end_dim=1), mu_x_j.flatten(end_dim=1)], dim=1).mean(dim=1)
        log_sigma_sq_x = (torch.stack([log_sigma_sq_x_i.flatten(end_dim=1).exp(), log_sigma_sq_x_j.flatten(end_dim=1).exp()], dim=1).sum(dim=1) / 4).log()
        z = Normal(mu_x, (0.5 * log_sigma_sq_x).exp()).rsample()  # (bs, seq, ld)
        # stack q_z back together
        q_z = torch.cat([mu_x, log_sigma_sq_x], dim=-1).view(q_z_i.size())
        # decode to feature maps
        U_hat = self.mlp_decode(z) # (bs * seq, ch, fr, fq)
        # reconstruct a contiguous sequence
        x_hat_i = self.cnn_decode(U_hat, dt_i) # (bs, 1, fr * seq, fq)
        # and reconstruct independent translations
        x_hat_j = self.cnn_decode(U_hat, dt_j) # (bs * seq, 1, fr, fq)
        # frame for frame-wise loss
        x_hat_i_framed = frame(x_hat_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        return dict(
            x_i=x_i, x_j=x_j,
            x_i_framed=x_i_framed,
            x_hat_i=x_hat_i, x_hat_j=x_hat_j,
            x_hat_i_framed=x_hat_i_framed,
            q_z=q_z,
            q_z_i=q_z_i, q_z_j=q_z_j,
            dt_i=dt_i, dt_j=dt_j
        )

    def encode(self, x: Tensor, hop_length: int | None = None) -> Tensor:
        x = self.cnn_encode(x)
        x = self.frame_encode(x, hop_length=hop_length)
        return self.mlp_encode(x)

    def cnn_encode(self, x: Tensor) -> Tensor:
        for i, block in enumerate(self.feature_encoder):
            x = block(x)
        return x

    def frame_encode(self, x: Tensor, hop_length: int | None = None) -> Tensor:
        frame_params = self.latent_frame_params
        if hop_length is not None:
            frame_params.update(dict(hop_length=hop_length // 2**(self.cnn_layers)))
        x = frame(x, **frame_params) if x.size(-2) > self.latent_window_length else x.unsqueeze(1)
        return x

    def mlp_encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q_z = self.content_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        mu_x, log_sigma_sq_x = q_z.chunk(2, dim=-1)
        log_sigma_sq_x = soft_clip(log_sigma_sq_x, minimum=np.log(self.sigma_x_min ** 2))
        q_z = torch.cat([mu_x, log_sigma_sq_x], dim=-1)
        dt = self.offset_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        return q_z, dt

    def decode(self, z: Tensor, dt: Tensor | None = None) -> Tensor:
        U = self.content_decoder(z.flatten(end_dim=1))
        for i, block in enumerate(self.feature_decoder):
            if dt is not None and i == len(self.feature_decoder) - 2:
                U = translation(U, dt.view(dt.size(0) * dt.size(1), 1, 1, 1), padding_mode="circular")
            if i == len(self.feature_decoder) - 1:
                num_timesteps = (self.frame_window_length * z.size(1)) // 2**(len(self.feature_decoder) - i)
                U = unframe(U.view(z.size(0), z.size(1), *U.size()[1:]), hop_length=U.size(-2), num_timesteps=num_timesteps)
            U = block(U)
        return U

    def mlp_decode(self, z: Tensor) -> Tensor:
        return self.content_decoder(z)

    def cnn_decode(self, U: Tensor, dt: Tensor) -> Tensor:
        for i, block in enumerate(self.feature_decoder):
            if i == len(self.feature_decoder) - 2:
                U = translation(U, dt.view(dt.size(0) * dt.size(1), 1, 1, 1), padding_mode="circular")
            if i == len(self.feature_decoder) - 1:
                num_timesteps = (self.frame_window_length * dt.size(1)) // 2**(len(self.feature_decoder) - i)
                U = unframe(U.view(dt.size(0), dt.size(1), *U.size()[1:]), hop_length=U.size(-2), num_timesteps=num_timesteps)
            U = block(U)
        return U

    def loss(
        self,
        x_i: Tensor,
        x_j: Tensor,
        x_i_framed: Tensor,
        x_hat_i: Tensor,
        x_hat_j: Tensor,
        x_hat_i_framed: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        dt_i: Tensor,
        dt_j: Tensor,
        q_z: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # maximise likelihood p(x_i|z_j) framewise to ensure invariance to sequence length
        x = torch.cat([x_i_framed, x_j], dim=0)
        x_hat = torch.cat([x_hat_i_framed, x_hat_j], dim=0)
        log_sigma_sq_z = torch.tensor(self.sigma_z_max).pow(2).log()
        nll = negative_log_likelihood(x, x_hat, log_sigma_sq_z).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        mae_frame = (x_hat - x).flatten(start_dim=-3).abs().sum(dim=-1)
        outputs |= dict(log_likelihood_x=-nll.detach().mean(), sigma_z=(0.5 * log_sigma_sq_z).exp().detach(), mae_frame=mae_frame.detach().mean())
        # MAP estimate of the alignment factor p(x|dt)p(dt)
        dt = torch.cat([dt_i.flatten(end_dim=1).unsqueeze(1), dt_j], dim=0)
        mu_dt_wrt_x = torch.zeros(1).to(dt.device)
        log_sigma_sq_dt_wrt_x= self.p_dt_sigma_current(self.trainer.global_step).pow(2).log()
        intra_frame_nll = negative_log_likelihood(dt, mu_dt_wrt_x, log_sigma_sq_dt_wrt_x)
        losses.append(intra_frame_nll.mean())
        outputs |= dict(log_likelihood_dt=-intra_frame_nll.detach().mean())
        # when applying the smoothness loss
        if self.smooth_prop is not None:
            # anchor q_z using an autoregressive prior, a mixture of gaussians between previous timestep and standard normal
            prior_dkl = torch.cat([
                gaussian_kl_divergence(q_z[:, t, :], p_z_t).unsqueeze(1)
                for t, p_z_t in autoregressive_prior(q_z[:, :, :], self.alpha)
            ], dim=1)
            smooth_dkl = prior_dkl[:, :, self.smooth_idx].sum(dim=-1)
            non_smooth_dkl = prior_dkl[:, :, self.non_smooth_idx].sum(dim=-1)
            prior_dkl = prior_dkl.sum(dim=-1)
            losses.append(prior_dkl.mean())
            outputs |= dict(
                prior_dkl=prior_dkl.detach().mean(),
                smooth_dkl=smooth_dkl.detach().mean(),
                non_smooth_dkl=non_smooth_dkl.detach().mean(),
            )
        else:
            prior_dkl = gaussian_kl_divergence_standard_prior(q_z).sum(dim=-1)
            losses.append(prior_dkl.mean())
            outputs |= dict(prior_dkl=prior_dkl.detach().mean())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        return outputs

    def p_dt_sigma_current(self, t: int) -> Tensor:
        if self.p_dt_sigma_min is None: return self.p_dt_sigma_max
        return torch.tensor(bounded_sigmoid(t, **self.p_dt_sigma_params))

    def metrics(
        self,
        x_i: Tensor,
        x_j: Tensor,
        x_i_framed: Tensor,
        x_hat_i: Tensor,
        x_hat_j: Tensor,
        x_hat_i_framed: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        dt_i: Tensor,
        dt_j: Tensor,
        q_z: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        x = torch.cat([x_i_framed, x_j], dim=0)
        x_hat = torch.cat([x_hat_i_framed, x_hat_j], dim=0)
        mae_frame = (x_hat - x).abs().flatten(start_dim=-3).mean(dim=-1).cpu()
        mu_x, log_sigma_sq_x = (t.flatten(end_dim=-2).cpu().numpy() for t in q_z.chunk(2, dim=-1))
        sigma_x = np.exp(0.5 * log_sigma_sq_x)
        mu_x_i, log_sigma_sq_x_i = (t.flatten(end_dim=-2).cpu().numpy() for t in q_z_i.chunk(2, dim=-1))
        sigma_x_i = np.exp(0.5 * log_sigma_sq_x_i)
        mu_x_j, log_sigma_sq_x_j = (t.flatten(end_dim=-2).cpu().numpy() for t in q_z_j.chunk(2, dim=-1))
        sigma_x_j = np.exp(0.5 * log_sigma_sq_x_j)
        dt = torch.cat([dt_i.flatten(end_dim=1).unsqueeze(1), dt_j])
        return dict(
            p_sigma_dt=self.p_dt_sigma_current(self.trainer.global_step),
            mae_frame=wandb.Histogram(np_histogram=np.histogram(mae_frame, range=[0, 5])),
            mu_x=wandb.Histogram(np_histogram=np.histogram(mu_x, range=[-5.0, 5.0])),
            sigma_x=wandb.Histogram(np_histogram=np.histogram(sigma_x, range=[-5.0, 5.0])),
            mu_x_i=wandb.Histogram(np_histogram=np.histogram(mu_x_i, range=[-5.0, 5.0])),
            sigma_x_i=wandb.Histogram(np_histogram=np.histogram(sigma_x_i, range=[0.0, 2.0])),
            mu_x_j=wandb.Histogram(np_histogram=np.histogram(mu_x_j, range=[-5.0, 5.0])),
            sigma_x_j=wandb.Histogram(np_histogram=np.histogram(sigma_x_j, range=[0.0, 2.0])),
            dt=wandb.Histogram(np_histogram=np.histogram(dt.flatten(end_dim=-2).cpu().numpy(), bins=64, range=[-5.0, 5.0])),
        )

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        x, *_ = batch
        step_outputs = self(x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.training_step_outputs.append(step_outputs)
        metrics = self.metrics(**step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "train"), batch_size=x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, "train")))
        return loss_outputs

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self.training_step_outputs.clear()

    @torch.no_grad()
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        x, *_ = batch
        step_outputs = self(x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.validation_step_outputs.append(step_outputs)
        metrics = self.metrics(**step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, "val")))
        return loss_outputs

    def on_validation_batch_end(
        self,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        x, *_ = batch
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            specs = step_outputs["x_i"].squeeze().cpu().numpy()
            recons = step_outputs["x_hat_i"].squeeze().cpu().numpy()
            nrows = step_outputs["x_i"].size(0)
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, nrows * 2))
            for i in range(nrows):
                mesh = plot_mel_spectrogram(specs[i].T, **self.spectrogram_params, vmin=specs.min(), vmax=specs.max(), ax=axes[i, 0])
                mesh = plot_mel_spectrogram(recons[i].T, **self.spectrogram_params, vmin=recons.min(), vmax=recons.max(), ax=axes[i, 1])
            self.logger.experiment.log({ f"val/spectrogram_i": wandb.Image(fig) })
            plt.close(fig)
            specs = step_outputs["x_j"].squeeze()
            specs = specs.view(x.size(0), -1, *specs.size()[1:]).cpu().numpy()[:, :5]
            recons = step_outputs["x_hat_j"].squeeze()
            recons = recons.view(x.size(0), -1, *recons.size()[1:]).cpu().numpy()[:, :5]
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

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        x, *_ = batch
        step_outputs = self(x, **kwargs)
        self.test_step_outputs.append(step_outputs)
        metrics = self.metrics(**step_outputs)
        self.logger.experiment.log(prefix_keys(metrics, "test"))

    @torch.no_grad()
    def on_test_batch_end(
        self,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        x, *_ = batch
        if batch_idx < 4 and len(self.test_step_outputs):
            step_outputs = self.test_step_outputs[0]
            specs = step_outputs["x_i"].squeeze().cpu().numpy()
            recons = step_outputs["x_hat_i"].squeeze().cpu().numpy()
            nrows = step_outputs["x_i"].size(0)
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, nrows * 2))
            for i in range(nrows):
                mesh = plot_mel_spectrogram(specs[i].T, **self.spectrogram_params, vmin=specs.min(), vmax=specs.max(), ax=axes[i, 0])
                mesh = plot_mel_spectrogram(recons[i].T, **self.spectrogram_params, vmin=recons.min(), vmax=recons.max(), ax=axes[i, 1])
            self.logger.experiment.log({ f"test/spectrogram_i": wandb.Image(fig) })
            plt.close(fig)
            specs = step_outputs["x_j"].squeeze()
            specs = specs.view(x.size(0), -1, *specs.size()[1:]).cpu().numpy()[:, :5]
            recons = step_outputs["x_hat_j"].squeeze()
            recons = recons.view(x.size(0), -1, *recons.size()[1:]).cpu().numpy()[:, :5]
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
            self.logger.experiment.log({ f"test/spectrogram_j": wandb.Image(fig) })
            plt.close(fig)
        self.test_step_outputs.clear()

    @torch.no_grad()
    def predict_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int,
        frame_hop_length: float | None = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        frame_hop_length = frame_hop_length or self.frame_hop_length
        x, *_ = batch
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins])
        q_z, dt = self.encode(x, hop_length=frame_hop_length)
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
                *dt.flatten(end_dim=1).cpu().squeeze(-1),
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

    def _reset_cache(self):
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []
