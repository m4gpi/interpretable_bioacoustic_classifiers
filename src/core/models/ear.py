import enum
import logging
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
import hydra

from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torchvision.transforms import functional as T
from torch.optim import Optimizer
from typing import Any, Dict, Tuple, List

from src.core.constants import Stage
from src.core.models.components import (
    Activation,
    NormType,
    init_cnn_feature_encoder,
    init_cnn_feature_decoder,
    init_mlp_content_encoder,
    init_mlp_content_decoder,
    init_alignment_encoder,
)
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior, autoregressive_prior, info_noise_constrastive_estimation
from src.core.transforms.frame import unframe_fold as unframe, frame_fold as frame
from src.core.transforms.translation import translation
from src.core.utils.sketch import plot_mel_spectrogram
from src.core.utils import soft_clip, linear_decay, bounded_sigmoid, nth_percentile, detach_values, prefix_keys, to_snake_case

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.switch_backend('agg')

__all__ = ["EAR"]

@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class VectorQuantiserEMA(torch.nn.Module):
    num_features: int
    num_clusters: int
    gamma: float = 0.99
    epsilon: float = 1e-5

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        torch.nn.Module.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        self.embedding = torch.nn.Embedding(self.num_features, self.num_clusters).requires_grad_(False)
        torch.nn.init.uniform_(self.embedding.weight, a=-(1 / self.num_clusters), b=(1 / self.num_clusters))
        self.ema_embedding = torch.nn.Embedding(self.num_features, self.num_clusters).requires_grad_(False)
        self.ema_embedding.weight.data = self.embedding.weight.data.clone()
        self.ema_cluster_size = torch.nn.Parameter(torch.zeros(self.num_clusters), requires_grad=False)

    def k_means(self, z_e: torch.Tensor, embedding: torch.Tensor):
        return z_e.pow(2).sum(1, keepdim=True) - (2 * z_e @ embedding) + embedding.pow(2).sum(0, keepdim=True)

    def forward(self, z_e: torch.Tensor):
        distances = self.k_means(z_e.flatten(end_dim=-2), self.embedding.weight)
        encoding_idx = distances.argmin(dim=-1, keepdim=True)
        encodings = torch.zeros(encoding_idx.size(0), self.num_clusters).to(z_e.device)
        encodings.scatter_(1, encoding_idx, 1)
        z_q = (encodings @ self.embedding.weight.t()).view(z_e.shape)
        return z_q, encoding_idx, encodings

    @torch.no_grad()
    def update(self, z_e: torch.Tensor, encodings: torch.Tensor) -> None:
        # update count of encoder hidden states using a running average
        self.ema_cluster_size.data = self.ema_cluster_size.data * self.gamma + (1 - self.gamma) * encodings.sum(dim=0)
        # apply laplace smoothing
        n = self.ema_cluster_size.sum()
        self.ema_cluster_size.data = (self.ema_cluster_size + self.epsilon) / n * (n + self.num_clusters * self.epsilon)
        # update the EMA
        self.ema_embedding.weight.data = self.ema_embedding.weight * self.gamma + (1 - self.gamma) * (encodings.t() @ z_e.flatten(end_dim=-2)).t()
        # update the codebook
        self.embedding.weight.data = self.ema_embedding.weight / self.ema_cluster_size.unsqueeze(0)

    @torch.no_grad()
    def perplexity(self, encodings: Tensor) -> Tensor:
        avg_probs = encodings.mean(dim=0)
        return (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()


@dataclass(unsafe_hash=True, kw_only=True, eq=False)
class EAR(L.LightningModule):
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
    sigma_z: float = 1.0
    beta: float = 0.25
    gamma: float = 0.999
    delta_sigma_min: float | None = None
    delta_sigma_max: float = 2.0
    delta_sigma_step_start: int | None = None
    delta_sigma_step_end: int | None = None
    delta_sigma_step_slope: float = 1.0
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
        self.mel_max_hertz = self.mel_max_hertz or self.sample_rate / 2.0
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
            out_features=self.latent_dim,
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
        self.quantise = VectorQuantiserEMA(num_clusters=512, num_features=8, gamma=self.gamma)
        self._reset_cache()

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
    def delta_sigma_params(self):
        return dict(
            x_min=self.delta_sigma_step_start,
            x_max=self.delta_sigma_step_end,
            y_min=self.delta_sigma_min,
            y_max=self.delta_sigma_max,
            k=self.delta_sigma_step_slope or 1.0,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # compute shifted versions
        x_i = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins]).float()
        x_i_framed = frame(x_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1)
        delta = torch.randn(x_i_framed.size(0), 1, 1, 1).to(x_i.device) * self.delta_sigma_current(self.trainer.global_step)
        x_j = translation(x_i_framed, delta, padding_mode="circular")
        # embed features
        z_e_i, delta_hat_i = self.encode(x_i) # (bs, seq, ld), (bs, seq, 1)
        z_e_j, delta_hat_j = self.encode(x_j) # (bs, seq, ld), (bs, seq, 1)
        # break up latent feature vector into quantisable chunks and quantise
        z_e_i_chunk = torch.stack(z_e_i.chunk(self.latent_dim // self.quantise.num_features, dim=-1), dim=-2) # (bs, seq, ld/ck, ck)
        z_q_i_chunk, encoding_idx_i, encodings_i = self.quantise(z_e_i_chunk) # (bs * seq * ld/ck, ck)
        z_q_i = z_q_i_chunk.reshape(*z_e_i_chunk.shape[:-2], -1) # (bs, seq, ld)
        z_q_i = z_e_i + (z_q_i - z_e_i).detach()
        z_e_j_chunk = torch.stack(z_e_j.chunk(self.latent_dim // self.quantise.num_features, dim=-1), dim=-2) # (bs, seq, ld/ck, ck)
        z_q_j_chunk, encoding_jdx_j, encodings_j = self.quantise(z_e_j_chunk) # (bs * seq * ld/ck, ck)
        z_q_j = z_q_j_chunk.reshape(*z_e_j_chunk.shape[:-2], -1) # (bs, seq, ld)
        z_q_j = z_e_j + (z_q_j - z_e_j).detach()
        # cross-decoded reconstructions, reconstruct x_i from delta_j and x_j from delta_i
        U_i = self.mlp_decode(z_q_i)
        U_j = self.mlp_decode(z_q_j)
        x_hat_i = self.cnn_decode(U_j, delta_hat_i) # (bs, 1, fr * seq, fq)
        x_hat_i_framed = frame(x_hat_i, window_length=self.frame_window_length, hop_length=self.frame_hop_length).flatten(end_dim=1) # (bs * seq, 1, fr, fq)
        x_hat_j = self.cnn_decode(U_i, delta_hat_j) # (bs * seq, 1, fr, fq)
        # update codebook EMA
        self.quantise.update(z_e_i_chunk, encodings_i)
        self.quantise.update(z_e_j_chunk, encodings_j)
        # quantify diversity of codebook usage
        perplexity_i = self.quantise.perplexity(encodings_i)
        perplexity_j = self.quantise.perplexity(encodings_j)
        return dict(
            x_i=x_i, x_j=x_j, x_i_framed=x_i_framed,
            x_hat_i=x_hat_i, x_hat_j=x_hat_j, x_hat_i_framed=x_hat_i_framed,
            z_q_i=z_q_i, z_e_i=z_e_i, z_e_i_chunk=z_e_i_chunk,
            z_q_j=z_q_j, z_e_j=z_e_j, z_e_j_chunk=z_e_j_chunk,
            perplexity_i=perplexity_i,
            perplexity_j=perplexity_j,
        )

    def step(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward(*args, **kwargs)

    def encode(self, x: Tensor) -> Tensor:
        x = self.cnn_encode(x) # (bs, ch, ts, fq)
        x = self.frame_encode(x) # (bs, seq, ch, ts, fq)
        return self.mlp_encode(x) # (bs, seq, ld), (bs, seq, 1)

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
        bs, seq, *_ = x.size()
        x = x.flatten(end_dim=1)
        x = self.content_encoder(x)
        delta_hat = self.offset_encoder(x)
        x = x.unflatten(dim=0, sizes=(bs, seq))
        delta_hat = delta_hat.unflatten(dim=0, sizes=(bs, seq))
        return x, delta_hat

    def decode(self, x: Tensor) -> Tensor:
        x = self.mlp_decode(x) # (bs, seq, ch, ts, fq)
        x = self.cnn_decode(x) # (bs, 1, ts, fq)
        return x

    def mlp_decode(self, x: Tensor) -> Tensor:
        bs, seq, *_ = x.size()
        x = x.flatten(end_dim=1)
        x = self.content_decoder(x)
        x = x.unflatten(dim=0, sizes=(bs, seq))
        return x

    def cnn_decode(self, x: Tensor, delta: Tensor) -> Tensor:
        for i, block in enumerate(self.feature_decoder):
            if i == len(self.feature_decoder) - 2:
                x = translation(x, delta.view(delta.size(0) * delta.size(1), 1, 1, 1), padding_mode="circular")
            if i == len(self.feature_decoder) - 1:
                num_timesteps = (self.frame_window_length * delta.size(1)) // 2**(len(self.feature_decoder) - i)
                x = unframe(x.view(delta.size(0), delta.size(1), *x.size()[1:]), hop_length=x.size(-2), num_timesteps=num_timesteps)
            x = block(x)
        return x

    def delta_sigma_current(self, t: int) -> Tensor:
        if self.delta_sigma_min is None: return self.delta_sigma_max
        return torch.tensor(bounded_sigmoid(t, **self.delta_sigma_params))

    def loss(
        self,
        x_i: Tensor,
        x_i_framed: Tensor,
        x_j: Tensor,
        x_hat_i: Tensor,
        x_hat_j: Tensor,
        x_hat_i_framed: Tensor,
        z_q_i: Tensor,
        z_q_j: Tensor,
        z_e_i: Tensor,
        z_e_j: Tensor,
        z_e_i_chunk: Tensor,
        z_e_j_chunk: Tensor,
        delta_i: Tensor,
        delta_j: Tensor,
        encoding_idx_i: Tensor,
        encoding_idx_j: Tensor,
        perplexity_i: Tensor,
        perplexity_j: Tensor,
        **kwargs: Any
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # maximise likelihood p(x_i|z_j) framewise
        x = torch.cat([x_i_framed, x_j], dim=0)
        x_hat = torch.cat([x_hat_i_framed, x_hat_j], dim=0)
        log_sigma_sq_z = torch.tensor(self.sigma_z).pow(2).log()
        nll = negative_log_likelihood(x, x_hat, log_sigma_sq_z).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        mae_frame = (x_hat - x).flatten(start_dim=-3).abs().sum(dim=-1)
        outputs |= dict(log_likelihood_x=-nll.detach().mean(), sigma_z=(0.5 * log_sigma_sq_z).exp().detach(), mae_frame=mae_frame.detach().mean())
        # MAP estimate of the alignment factor p(x|δ)p(δ) with a gaussian prior on a circular co-ordinate space
        dt = torch.cat([delta_i.flatten(end_dim=1).unsqueeze(1), delta_j], dim=0)
        mu_delta_wrt_x = torch.zeros(1).to(dt.device)
        log_sigma_sq_delta_wrt_x= self.delta_sigma_current(self.trainer.global_step).pow(2).log()
        intra_frame_nll = negative_log_likelihood(dt, mu_delta_wrt_x, log_sigma_sq_delta_wrt_x)
        losses.append(intra_frame_nll.mean())
        outputs |= dict(log_likelihood_dt=-intra_frame_nll.detach().mean())
        # regularise to encourage the encoder to commit to an embedding
        z_e = torch.cat([z_e_i, z_e_j], dim=0)
        z_q = torch.cat([z_q_i, z_q_j], dim=0)
        cml = (z_e - z_q.detach()).pow(2).sum(dim=-1)
        cml = self.beta * cml.mean()
        losses.append(cml)
        outputs |= dict(commitment_loss=cml.detach())
        # noise constrative estimation encourages codebook diversity
        # ince = info_noise_constrastive_estimation(z_e_chunk, self.quantise.embedding.weight, encoding_idx).sum(dim=-1)
        # ince = self.lamdba * ince.mean()
        # losses.append(ince)
        # outputs |= dict(info_nce=ince.detach())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        outputs |= dict(perplexity=torch.cat([perplexity_i, perplexity_j]).mean())
        return outputs

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        step_outputs = self.step(batch.x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.training_step_outputs.append(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "train"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs, "train")))
        return loss_outputs

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self.training_step_outputs.clear()

    @torch.no_grad()
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        step_outputs = self.step(batch.x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.validation_step_outputs.append(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs, "val")))
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

    # @torch.no_grad()
    # def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> None:
    #     x, *_ = batch
    #     _, x_hat, q_z, *_ = self.predict(x, **kwargs).values()
    #     bs, seq, *_ = q_z.size()
    #     sample_idx = batch.s.unsqueeze(0).repeat(seq, 1).t().flatten().unsqueeze(1).cpu()
    #     seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq, 1).cpu()
    #     dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().unsqueeze(1).cpu()
    #     mae = (x_hat - x).abs().flatten(start_dim=-3).mean(dim=-1).flatten(end_dim=1).unsqueeze(-1)
    #     q_z = q_z.flatten(end_dim=1)
    #     data = torch.cat([sample_idx.cpu(), seq_idx.cpu(), dl_idx.cpu(), q_z.cpu(), mae.cpu()], dim=-1)
    #     index_columns  = ["file_i", "timestep", "dataloader_idx"]
    #     z_mean_cols, z_log_var_cols = [f"z_mean_{d}" for d in range(q_z.size(-1)//2)], [f"z_log_var_{d}" for d in range(q_z.size(-1)//2)]
    #     data_columns = [z_mean_cols, z_log_var_cols, ["mae"]]
    #     dtypes = dict(**dict([(col, int) for col in index_columns]), **dict([(col, float) for columns in data_columns for col in columns]))
    #     df = pd.DataFrame(data=data, columns=[*index_columns, *sum(data_columns, [])]).astype(dtype=dtypes).set_index(index_columns)
    #     self.predict_step_outputs.append(df)
    #     return df

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
