import enum
import itertools
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

    def k_means(self, z_e: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return z_e.pow(2).sum(1, keepdim=True) - (2 * z_e @ embedding) + embedding.pow(2).sum(0, keepdim=True)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = self.k_means(z_e, self.embedding.weight)
        encoding_idx = distances.argmin(dim=-1, keepdim=True)
        z_q, encodings = self.quantise(encoding_idx)
        diff = (z_q.detach() - z_e).pow(2)
        z_q = z_e + (z_q - z_e).detach()
        return z_q, encoding_idx, encodings, diff

    def quantise(self, encoding_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encodings = torch.zeros(encoding_idx.size(0), self.num_clusters).to(encoding_idx.device)
        encodings.scatter_(1, encoding_idx, 1)
        return (encodings @ self.embedding.weight.t()), encodings

    @torch.no_grad()
    def update(self, z_e: torch.Tensor, encodings: torch.Tensor) -> None:
        """update the codebook via an exponential moving average of encoded values"""""
        # update count of encoder hidden states using a running average
        self.ema_cluster_size.data = self.ema_cluster_size.data * self.gamma + (1 - self.gamma) * encodings.sum(dim=0)
        # apply laplace smoothing
        n = self.ema_cluster_size.sum()
        self.ema_cluster_size.data = (self.ema_cluster_size + self.epsilon) / n * (n + self.num_clusters * self.epsilon)
        # update the EMA
        self.ema_embedding.weight.data = self.ema_embedding.weight * self.gamma + (1 - self.gamma) * (encodings.t() @ z_e).t()
        # update the codebook
        self.embedding.weight.data = self.ema_embedding.weight / self.ema_cluster_size.unsqueeze(0)

    def perplexity(self, encodings: Tensor) -> Tensor:
        """
        p = b*t*f: a unique code word for each encoding (feature magnitude)
        p = 1: the same code word for each embedding (feature magnitude)
        """
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
    num_mel_bins: int = 64
    frame_window_length: int = 192
    frame_hop_length: int | None = 192
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
    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(
            self,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
            ckpt_path=config.get("ckpt_path")
        )

    def evaluate(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Encoding <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        ckpt_path = pathlib.Path(config.get("ckpt_path"))
        pred_batch_dfs = trainer.predict(
            self,
            dataloaders=data_module.predict_dataloader(),
            ckpt_path=ckpt_path,
            return_predictions=True,
        )
        pred_df = pd.concat(list(itertools.chain(*pred_batch_dfs)), axis=0)
        save_path = ckpt_path.parent / f"{ckpt_path.name}_codes.parquet"
        pred_df.to_parquet(save_path)
        log.info(f"Saved to {save_path}")

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
        self.quantise_top = VectorQuantiserEMA(num_clusters=512, num_features=128, gamma=self.gamma)
        self.quantise_bottom = VectorQuantiserEMA(num_clusters=512, num_features=512, gamma=self.gamma)
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

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), self.num_mel_bins]).float()
        # embed features
        h_e = self.cnn_encode(x) # (bs, ch, ts, fq)
        h_e = self.frame_encode(h_e) # (bs, seq, ch, ts, fq)
        z_e_top = self.mlp_encode(h_e) # (bs, seq, ld)
        # average pool for a sequence level summary vector
        c_e = z_e_top.mean(dim=1, keepdims=True)
        # z_e describes time-step specific information (residual)
        z_e_res = (z_e_top + c_e).view(z_e_top.size(0) * z_e_top.size(1), -1)
        z_q_top, encoding_idx_top, encodings_top, diff_top = self.quantise_top(z_e_res)
        z_q_top = z_q_top.view(z_e_top.shape)
        # reconstruct
        h_q_top = self.mlp_decode(z_q_top)
        # h_e describes pixel-level information (residual)
        h_e_res = (h_q_top + h_e).permute(0, 1, 3, 4, 2).flatten(end_dim=-2)
        h_q_bottom, encoding_idx_bottom, encodings_bottom, diff_bottom = self.quantise_bottom(h_e_res)
        h_q_bottom = h_q_bottom.unflatten(0, (h_q_top.size(0), h_q_top.size(1), h_q_top.size(3), h_q_top.size(4))).permute(0, 1, 4, 2, 3)
        # decode spectrograms
        x_hat = self.cnn_decode(h_q_bottom)
        # update codebook EMA
        self.quantise_top.update(z_e_res, encodings_top)
        self.quantise_bottom.update(h_e_res, encodings_bottom)
        # quantify diversity of codebook usage
        perplexity_top = self.quantise_top.perplexity(encodings_top)
        perplexity_bottom = self.quantise_bottom.perplexity(encodings_bottom)
        return dict(
            x=x, x_hat=x_hat,
            perplexity_top=perplexity_top,
            encodings_top=encodings_top,
            encoding_idx_top=encoding_idx_top,
            perplexity_bottom=perplexity_bottom,
            encodings_bottom=encodings_bottom,
            encoding_idx_bottom=encoding_idx_bottom,
            diff_top=diff_top,
            diff_bottom=diff_bottom,
        )

    def step(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward(*args, **kwargs)

    def encode(self, x: Tensor) -> Tensor:
        x = self.cnn_encode(x) # (bs, ch, ts, fq)
        x = self.frame_encode(x) # (bs, seq, ch, ts, fq)
        x = self.mlp_encode(x) # (bs, seq, ld)
        return x

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
        x = x.unflatten(dim=0, sizes=(bs, seq))
        return x

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

    def cnn_decode(self, x: Tensor) -> Tensor:
        bs, seq, *_ = x.size()
        x = x.flatten(end_dim=1)
        for i, block in enumerate(self.feature_decoder):
            if seq > 1 and i == len(self.feature_decoder) - 1:
                num_timesteps = (self.frame_window_length * seq) // 2**(len(self.feature_decoder) - i)
                x = unframe(x.view(bs, seq, *x.size()[1:]), hop_length=x.size(-2), num_timesteps=num_timesteps)
            x = block(x)
        return x

    def loss(
        self,
        x: Tensor,
        x_hat: Tensor,
        diff_top: Tensor,
        diff_bottom: Tensor,
        perplexity_top: Tensor,
        perplexity_bottom: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # if passed a sequence, ensure invariance to sequence length by treating frames independently
        if x.size(-2) > self.frame_window_length:
            x = frame(x, **self.frame_params)
        if x_hat.size(-2) > self.frame_window_length:
            x_hat = frame(x_hat, **self.frame_params)
        # frame-wise reconstruction loss, sum over pixels, average over frames & samples
        log_sigma_sq_z = torch.tensor(self.sigma_z).pow(2).log()
        nll = negative_log_likelihood(x, x_hat, log_sigma_sq_z).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        mae_frame = (x_hat - x).flatten(start_dim=-3).abs().sum(dim=-1)
        outputs |= dict(log_likelihood_x=-nll.detach().mean(), sigma_z=(0.5 * log_sigma_sq_z).exp().detach(), mae_frame=mae_frame.detach().mean())
        # regularise to encourage the encoder to commit to an embedding
        commitment_bottom = self.quantise_top.num_features / self.quantise_bottom.num_features * diff_bottom.sum(dim=-1).mean()
        commitment_top = diff_top.sum(dim=-1).mean()
        cml = commitment_top + commitment_bottom
        losses.append(cml)
        outputs |= dict(commitment=cml.detach(), commitment_top=commitment_top.detach(), commitment_bottom=commitment_bottom.detach())
        # noise constrative estimation encourages codebook diversity
        # ince = info_noise_constrastive_estimation(z_e_chunk, self.quantise.embedding.weight, encoding_idx).sum(dim=-1)
        # ince = self.lamdba * ince.mean()
        # losses.append(ince)
        # outputs |= dict(info_nce=ince.detach())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        outputs |= dict(perplexity_top=perplexity_top)
        outputs |= dict(perplexity_bottom=perplexity_bottom)
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

    @torch.no_grad()
    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> None:
        _, _, _, _, z_e_chunk, z_q_chunk, _, encodings, encoding_idx = self.forward(batch.x, **kwargs).values()
        bs, seq, c_i = encoding_idx.shape
        sample_idx = batch.s.unsqueeze(0).repeat(seq, 1).t().flatten().unsqueeze(1).cpu()
        seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq, 1).cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().unsqueeze(1).cpu()
        data = torch.cat([sample_idx.cpu(), seq_idx.cpu(), dl_idx.cpu(), encoding_idx.flatten(end_dim=-2).cpu()], dim=-1)
        index_columns  = ["file_i", "timestep", "dataloader_idx"]
        data_columns = [[f"{i}" for i in range(c_i)]]
        dtypes = dict(**dict([(col, int) for col in index_columns]), **dict([(col, int) for columns in data_columns for col in columns]))
        df = pd.DataFrame(data=data, columns=[*index_columns, *sum(data_columns, [])]).astype(dtype=dtypes).set_index(index_columns)
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
