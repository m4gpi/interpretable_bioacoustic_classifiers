import os
import enum
import functools
import itertools
import lightning as L
import logging
import numpy as np
import pandas as pd
import torch
import wandb

from dataclasses import dataclass, field
from hydra.utils import instantiate
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.functional import F
from torchvision.transforms import functional as T
from torch.distributions.normal import Normal
from torch.optim import Optimizer
from typing import Any, Dict, Tuple

from src.core.transforms.frame import unframe_fold, frame_fold
from src.core.models.components import (
    Activation,
    NormType,
    init_cnn_feature_encoder,
    init_cnn_feature_decoder,
    init_mlp_content_encoder,
    init_mlp_content_decoder,
    init_alignment_encoder,
)
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence_standard_prior
from src.core.utils import soft_clip, linear_decay, exponential_decay, nth_percentile, bounded_sigmoid
from src.core.utils.sketch import plot_mel_spectrogram
from src.core.utils import to_snake_case, detach_values, prefix_keys, try_or

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["BaseVAE"]

class DecoderVarianceMode(enum.Enum):
    FIXED: str = "FIXED"
    LINEAR_DECAY: str = "LINEAR_DECAY"
    EXPONENTIAL_DECAY: str = "EXPONENTIAL_DECAY"
    LEARNED: str = "LEARNED"

@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class BaseVAE(L.LightningModule):
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
    # latent parameters
    latent_dim: int = 128
    sigma_x_min: float = 0.0498
    # CNN parameters
    weight_init_std: float = 1e-3
    cnn_block_width: int = 4
    cnn_block_depth: int = 3
    cnn_dropout_prob: float = 0.2
    # TODO: should be mixed padding, circular in time, reflective in frequency
    cnn_padding_mode: str = "circular"
    cnn_activation: str = "LEAK"
    cnn_feature_reduction_factor: int = 4
    norm_type: str = "LN"
    # MLP parameters
    mlp_activation: str = "LEAK"
    mlp_dropout_prob: float = 0.1
    mlp_reduction_factor: int = 4
    frame_padding_mode: str = "circular"
    # decoder parameters
    sigma_z_max: float = 1.0
    sigma_z_min: float | None = None
    sigma_z_step_start: int | None = 0
    sigma_z_step_end: int | None = 1
    sigma_z_mode: str = DecoderVarianceMode.FIXED.value
    # lightning parameters
    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(self, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self):
        self.decoder_variance = DecoderVarianceMode(self.sigma_z_mode)
        if self.decoder_variance == DecoderVarianceMode.LEARNED:
            self.log_sigma_sq_z = nn.Parameter(torch.tensor(self.sigma_z_max).pow(2).log().requires_grad_(True))
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

    def step(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward(*args, **kwargs)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins]).float()
        q_z, *_ = self.encode(x)
        mu_x, log_sigma_sq_x = q_z.chunk(2, dim=-1)
        z = Normal(mu_x, (0.5 * log_sigma_sq_x).exp()).rsample()
        x_hat = self.decode(z).view(*x.size())
        return dict(x=x, x_hat=x_hat, q_z=q_z)

    def encode(self, x: Tensor, hop_length: int | None = None) -> Tuple[Tensor, Tensor]:
        U = x
        for block in self.feature_encoder:
            U = block(U)
        frame_params = self.latent_frame_params
        if hop_length is not None: frame_params.update(dict(hop_length=hop_length // 2**(self.cnn_layers)))
        U = frame_fold(U, **frame_params) if x.size(-2) > self.frame_window_length else U.unsqueeze(1)
        mu_x, log_sigma_sq_x = self.content_encoder(U.flatten(end_dim=1)).unflatten(dim=0, sizes=(U.size(0), U.size(1))).chunk(2, dim=-1)
        log_sigma_sq_x = soft_clip(log_sigma_sq_x, minimum=np.log(self.sigma_x_min ** 2))
        return torch.cat([mu_x, log_sigma_sq_x], dim=-1), U

    def decode(self, z: Tensor) -> Tensor:
        bs, seq, *other_dims = z.size()
        U = self.content_decoder(z.flatten(end_dim=1))
        for i, block in enumerate(self.feature_decoder):
            if seq > 1 and i == len(self.feature_decoder) - 1:
                num_timesteps = (self.frame_window_length * seq) // 2**(len(self.feature_decoder) - i)
                U = unframe_fold(U.view(bs, seq, *U.size()[1:]), hop_length=U.size(-2), num_timesteps=num_timesteps)
            U = block(U)
        return U

    def predict(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        # remove extraneous added dimension by transform frame operation, flatten sequence and batch
        if len(x.size()) > 4:
            x = x.squeeze(1)
            step_outputs = self(x.flatten(end_dim=1), *args, **kwargs)
            step_outputs["x"] = step_outputs["x"].unflatten(0, (x.size(0), x.size(1)))
            step_outputs["x_hat"] = step_outputs["x_hat"].unflatten(0, (x.size(0), x.size(1)))
            step_outputs["q_z"] = step_outputs["q_z"].squeeze(1).unflatten(0, (x.size(0), x.size(1)))
        else:
            step_outputs = self(x, *args, **kwargs)
            # when a full sequence is provided, frame after pass through network for frame-wise MAE
            if step_outputs["q_z"].size(1) > 1:
                step_outputs["x"] = frame(step_outputs["x"], **self.frame_params)
                step_outputs["x_hat"] = frame(step_outputs["x_hat"], **self.frame_params)
            # when independent frames are provided, treat them as independent
            else:
                step_outputs["x"] = step_outputs["x"].unsqueeze(1)
                step_outputs["x_hat"] = step_outputs["x_hat"].unsqueeze(1)
        return step_outputs

    def loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        q_z: Tensor,
        **kwargs: Any
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # if passed a sequence, ensure invariance to sequence length by treating frames independently
        if x.size(-2) > self.frame_window_length:
            x = frame_fold(x, **self.frame_params)
            x_hat = frame_fold(x_hat, **self.frame_params)
        # frame-wise reconstruction loss
        log_sigma_sq_z = self.log_sigma_sq_z_current(self.trainer.global_step)
        nll = negative_log_likelihood(x, x_hat, log_sigma_sq_z).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        mae_frame = (x_hat - x).flatten(start_dim=-3).abs().sum(dim=-1)
        outputs |= dict(log_likelihood_x=-nll.detach().mean(), sigma_z=(0.5 * log_sigma_sq_z).exp().detach(), mae_frame=mae_frame.detach().mean())
        # frame-wise mean of the KL between posterior and prior
        prior_dkl = gaussian_kl_divergence_standard_prior(q_z).sum(dim=-1)
        losses.append(prior_dkl.mean())
        outputs |= dict(prior_dkl=prior_dkl.detach().mean())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        return outputs

    def log_sigma_sq_z_current(self, t: int) -> Tensor:
        if self.decoder_variance == DecoderVarianceMode.FIXED:
            return torch.tensor(self.sigma_z_max).pow(2).log()
        elif self.decoder_variance == DecoderVarianceMode.LINEAR_DECAY:
            bound_params = dict(t_start=self.sigma_z_step_start, t_end=self.sigma_z_step_end, maximum=self.sigma_z_max, minimum=self.sigma_z_min)
            return torch.tensor(linear_decay(t_current=t, **bound_params)).pow(2).log()
        elif self.decoder_variance == DecoderVarianceMode.EXPONENTIAL_DECAY:
            bound_params = dict(t_start=self.sigma_z_step_start, t_end=self.sigma_z_step_end, maximum=self.sigma_z_max, minimum=self.sigma_z_min)
            return torch.tensor(exponential_decay(t_current=t, **bound_params)).pow(2).log()
        elif self.decoder_variance == DecoderVarianceMode.LEARNED:
            return self.log_sigma_sq_z

    @torch.no_grad()
    def metrics(
        self,
        q_z: Tensor,
        **kwargs: Any,
     ) -> Dict[str, Any]:
        mu_x, log_sigma_sq_x = (t.flatten(end_dim=-2).cpu().numpy() for t in q_z.chunk(2, dim=-1))
        sigma_x = np.exp(0.5 * log_sigma_sq_x)
        return dict(
            mu_x=wandb.Histogram(np_histogram=np.histogram(mu_x, range=nth_percentile(mu_x, 1.28))),
            sigma_x=wandb.Histogram(np_histogram=np.histogram(sigma_x, range=nth_percentile(sigma_x, 1.28))),
        )

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        x, *_ = batch
        step_outputs = self.step(x, **kwargs)
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
        step_outputs = self.step(x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.validation_step_outputs.append(step_outputs)
        metrics = self.metrics(**step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, "val")))
        return loss_outputs

    def on_validation_batch_end(self, outputs: Dict[str, Tensor], batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, **kwargs: Any) -> None:
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            specs = step_outputs["x"].squeeze().cpu().numpy()
            recons = step_outputs["x_hat"].squeeze().cpu().numpy()
            nrows = step_outputs["x"].size(0)
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, nrows * 3))
            for i in range(nrows):
                vmin, vmax = min(recons[i].min(), specs[i].min()), max(recons[i].max(), specs[i].max())
                mesh = plot_mel_spectrogram(specs[i].T, **self.spectrogram_params, vmin=vmin, vmax=vmax, ax=axes[i, 0])
                plt.colorbar(mesh, ax=axes[i, 0], orientation="vertical")
                mesh = plot_mel_spectrogram(recons[i].T, **self.spectrogram_params, vmin=vmin, vmax=vmax, ax=axes[i, 1])
                plt.colorbar(mesh, ax=axes[i, 1], orientation="vertical")
            self.logger.experiment.log({ f"val/spectrogram": wandb.Image(fig) })
            plt.close(fig)
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, Tensor]:
        return detach_values(self(*batch, **kwargs))

    @torch.no_grad()
    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> None:
        x, *_ = batch
        _, x_hat, q_z, *_ = self.predict(x, **kwargs).values()
        bs, seq, *_ = q_z.size()
        sample_idx = batch.s.unsqueeze(0).repeat(seq, 1).t().flatten().unsqueeze(1).cpu()
        seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq, 1).cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().unsqueeze(1).cpu()
        mae = (x_hat - x).abs().flatten(start_dim=-3).mean(dim=-1).flatten(end_dim=1).unsqueeze(-1)
        q_z = q_z.flatten(end_dim=1)
        data = torch.cat([sample_idx.cpu(), seq_idx.cpu(), dl_idx.cpu(), q_z.cpu(), mae.cpu()], dim=-1)
        index_columns  = ["file_i", "timestep", "dataloader_idx"]
        z_mean_cols, z_log_var_cols = [f"z_mean_{d}" for d in range(q_z.size(-1)//2)], [f"z_log_var_{d}" for d in range(q_z.size(-1)//2)]
        data_columns = [z_mean_cols, z_log_var_cols, ["mae"]]
        dtypes = dict(**dict([(col, int) for col in index_columns]), **dict([(col, float) for columns in data_columns for col in columns]))
        df = pd.DataFrame(data=data, columns=[*index_columns, *sum(data_columns, [])]).astype(dtype=dtypes).set_index(index_columns)
        self.predict_step_outputs.append(df)
        return df

    def configure_optimizers(self) -> Optimizer:
        optimiser_config = DictConfig(dict(_target_=self.optimiser_cls, **(self.optimiser_config or {})))
        optimiser = instantiate(optimiser_config, params=self.parameters(), lr=self.learning_rate)
        if self.scheduler_cls is not None:
            scheduler_config = DictConfig(dict(_target_=self.scheduler_cls, **(self.scheduler_config or {})))
            scheduler = instantiate(scheduler_config, optimizer=optimiser)
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
