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
    init_alignment_encoder,
)
from src.core.transforms.log_mel_spectrogram import mel_filterbanks, hz_to_mel, mel_to_hz
from src.core.transforms.frame import unframe_fold as unframe, frame_fold as frame
from src.core.transforms.translation import translation
from src.core.utils import soft_clip, detach_values, prefix_keys, bounded_sigmoid

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SIVAE"]

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
        mel_basis = mel_filterbanks(**self.mel_params)
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

    @torch.no_grad()
    def plot(self, z: NDArray | torch.Tensor, vmin: float | None = None, vmax: float | None = None, cmap: str = "viridis", ax: None = None, **kwargs: Any):
        ax = ax if ax is not None else plt.gca()
        vmin = vmin if vmin is not None and kwargs.get("norm", None) is None else z.min()
        vmax = vmax if vmax is not None and kwargs.get("norm", None) is None else z.max()
        imshow_params = dict(vmin=vmin, vmax=vmax, origin="lower", aspect="auto", cmap=cmap, **kwargs)
        im = ax.imshow(z, **imshow_params)
        # TODO: allow parametrisation of tick duration
        duration_seconds = (z.shape[1] * self.fft_hop_length) // self.sample_rate
        time = np.linspace(0, duration_seconds, z.shape[1])
        x_tick_positions = np.linspace(0, duration_seconds, int(z.shape[1] // (self.sample_rate // self.fft_hop_length) / 5) + 1)
        x_tick_labels = [f"{np.format_float_positional(t, trim='-', precision=2)}" for t in x_tick_positions]
        x_tick_indices = [np.argmin(np.abs(time - t)) for t in x_tick_positions]
        ax.set_xticks(x_tick_indices, labels=x_tick_labels)
        ax.set_xlabel("Time (s)")
        # ticks for y-axis are on a log scale, find the nearest base 2 exponents for the ticks
        min_mel = hz_to_mel(self.mel_min_hertz, scaling_factor=self.mel_scaling_factor, break_frequency=self.mel_break_frequency)
        max_mel = hz_to_mel(self.mel_max_hertz, scaling_factor=self.mel_scaling_factor, break_frequency=self.mel_break_frequency)
        mels = np.linspace(min_mel, max_mel, z.shape[0])
        frequencies = mel_to_hz(mels, scaling_factor=self.mel_scaling_factor, break_frequency=self.mel_break_frequency)
        y_tick_positions = [2**i for i in range(max(9, int(np.ceil(np.log2(self.mel_min_hertz + 1e-8)))), int(np.floor(np.log2(self.mel_max_hertz))) + 1)]
        y_tick_labels = [f"{int(f)}" for f in y_tick_positions]
        y_tick_indices = [np.argmin(np.abs(frequencies - f)) for f in y_tick_positions]
        ax.set_yticks(y_tick_indices, labels=y_tick_labels)
        ax.set_ylabel("Frequency (Hz)")
        return im

    @property
    def stft_params(self):
        return dict(
            win_length=self.fft_window_length,
            hop_length=self.fft_hop_length,
            return_complex=True,
            pad_mode="constant",
        )

    @property
    def mel_params(self):
        return dict(
            num_mel_bins=self.num_mel_bins,
            mel_min_hertz=self.mel_min_hertz,
            mel_max_hertz=self.mel_max_hertz,
            linear_frequencies=torch.linspace(0.0, self.sample_rate / 2, (self.n_fft // 2) + 1),
            scaling_factor=self.mel_scaling_factor,
            break_frequency=self.mel_break_frequency,
        )


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
    sigma_z_min: float = 0.05
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

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        plt.switch_backend('agg')
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(self, datamodule=data_module, ckpt_path=config.get("ckpt_path"))

    def evaluate(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        save_dir = pathlib.Path(config["save_dir"])
        data = data_module.data

        predict_dfs = trainer.predict(self, datamodule=data_module, ckpt_path=config.get("ckpt_path"), return_predictions=True)
        df = pd.concat(list(itertools.chain(*predict_dfs)), axis=0)

        train_dir = pathlib.Path(save_dir) / "train"
        train_dir.mkdir(exist_ok=True, parents=True)
        train_features, train_labels = df[df.index.get_level_values("file_i").isin(data.train_idx.file_i)], data.train_labels
        train_features.to_parquet(train_dir / "features.parquet")
        train_labels.to_parquet(train_dir / "labels.parquet")

        test_dir = pathlib.Path(save_dir) / "test"
        test_dir.mkdir(exist_ok=True, parents=True)
        test_features, test_labels = df[df.index.get_level_values("file_i").isin(data.test_idx.file_i)], data.test_labels
        test_features.to_parquet(test_dir / "features.parquet")
        test_labels.to_parquet(test_dir / "labels.parquet")

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
        self.alignment_encoder = init_alignment_encoder(**self.alignment_mlp_params)
        self.feature_decoder = init_cnn_feature_decoder(**self.cnn_decoder_params)
        self.content_decoder = init_mlp_content_decoder(**self.content_mlp_decoder_params)
        self._reset_cache()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_i = self.log_mel_spectrogram(x)
        x_i = T.center_crop(x_i, [(x_i.size(-2) - (x_i.size(-2) % self.frame_window_length)), x_i.size(-1)])
        sigma_delta = torch.tensor(bounded_sigmoid(self.trainer.global_step, **self.sigma_delta_params))
        x_i_framed = x_i.view(x_i.size(0), -1, 1, self.frame_window_length, x_i.size(-1)).flatten(end_dim=1)
        epsilon = torch.randn(x_i_framed.size(0), 1, 1, 1, device=x.device)
        x_j = translation(x_i_framed, epsilon * sigma_delta, padding_mode="circular")

        k = 2 # number of cross-decoded samples
        q_z_i, delta_i = self.encode(x_i) # (bs, seq, ld)
        q_z_j, delta_j = self.encode(x_j) # (bs * seq, 1, ld)
        q_z = torch.stack([q_z_i, q_z_j.view(q_z_i.size())], dim=0) # (k, bs, seq, ld)
        delta = torch.stack([delta_i, delta_j.view(delta_i.size())], dim=0)
        z = self.cross_decode(q_z, method=self.cross_decode_method, k=k)

        U_hat = self.mlp_decode(z.flatten(end_dim=2)).unflatten(dim=0, sizes=(z.size(0), z.size(1), z.size(2))) # (k, bs, seq, ch, fr, fq)
        U_hat_j, U_hat_i = U_hat.chunk(k, dim=0)
        x_hat_i = self.cnn_decode(U_hat_j.flatten(end_dim=2), delta_i) # (bs, 1, fr * seq, fq)
        x_hat_j = self.cnn_decode(U_hat_i.flatten(end_dim=2), delta_j) # (bs * seq, 1, fr, fq)
        x_hat_i_framed = x_hat_i.view(x_hat_i.size(0), -1, 1, self.frame_window_length, x_hat_i.size(-1)).flatten(end_dim=1)
        x = torch.cat([x_i_framed, x_j], dim=0)
        x_hat = torch.cat([x_hat_i_framed, x_hat_j], dim=0)

        return dict(
            x=x, x_hat=x_hat,
            x_i=x_i, x_hat_i=x_hat_i,
            q_z=q_z,
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
            log_sigma_sq_z = (1 / k**2 * log_sigma_sq_zs.exp()).sum(dim=0).log()
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
        delta = self.alignment_encode(x, encoder=self.alignment_encoder)
        q_z = self.content_encode(x, encoder=self.content_encoder)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        log_sigma_sq_z = soft_clip(log_sigma_sq_z, minimum=2*np.log(self.sigma_z_min))
        q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        return q_z, delta

    def cnn_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tensor:
        for i, block in enumerate(encoder):
            x = block(x)
        return x

    def content_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tuple[Tensor, Tensor]:
        q_z = encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        return q_z

    def alignment_encode(self, x: Tensor, encoder: torch.nn.Module) -> Tuple[Tensor, Tensor]:
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

    @staticmethod
    def frame(x: Tensor, window_length: int, hop_length: int | None = None, padding_mode: str = "circular") -> Tensor:
        if hop_length != window_length:
            return frame(x, window_length=window_length, hop_length=hop_length) if x.size(-2) > window_length else x.unsqueeze(1)
        else:
            return x.view(x.size(0), x.size(1), x.size(2) // window_length, window_length, x.size(3)).transpose(1, 2)

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        q_z: torch.Tensor,
        delta: torch.Tensor,
        sigma_delta: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # frame-wise reconstruction loss
        sigma_x = torch.tensor(self.sigma_x)
        nll = (1/2 * (2 * sigma_x.log() + ((x - x_hat) / sigma_x).pow(2))).flatten(start_dim=-3).sum(dim=-1).mean()
        mae = (x_hat - x).abs().flatten(start_dim=-3).sum(dim=-1).mean()
        losses.append(nll)
        outputs |= dict(log_likelihood_x=-nll.detach().mean(), mae=mae.detach())
        # frame-wise shift loss
        nll_delta = (1/2 * (2 * sigma_delta.log() + (delta / sigma_delta).pow(2))).mean()
        losses.append(nll_delta)
        outputs |= dict(log_likelihood_delta=-nll_delta.detach(), sigma_delta=sigma_delta)
        # frame-wise standard normal kl
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        dkl = (-1/2 * (1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())).sum(dim=-1).mean()
        losses.append(dkl)
        outputs |= dict(dkl=dkl.detach())
        # sum
        outputs |= dict(loss=sum(losses))
        return outputs

    # ------------------------------ LIGHTNING FUNCS --------------------------------- #

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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "train")
        self.training_step_outputs.append(step_outputs)
        return loss_outputs

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "val")
        self.validation_step_outputs.append(step_outputs)
        return loss_outputs

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, **kwargs: Any) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "test")
        self.test_step_outputs.append(step_outputs)
        return loss_outputs

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self.training_step_outputs.clear()

    def on_validation_batch_end(
        self,
        outputs: Dict[str, Tensor],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6), width_ratios=[0.97, 0.03], constrained_layout=True)
                mesh = self.log_mel_spectrogram.plot(specs[i].T, ax=axes[0, 0], vmin=specs.min(), vmax=specs.max())
                mesh = self.log_mel_spectrogram.plot(recons[i].T, ax=axes[1, 0], vmin=specs.min(), vmax=specs.max())
                axes[0, 0].set_title("Original Mel Spectrogram")
                axes[1, 0].set_title("Reconstructed Mel Spectrogram")
                fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
                fig.colorbar(mesh, cax=axes[1, 1], orientation="vertical")
                self.logger.experiment.log({ f"val/spectrogram": wandb.Image(fig) })
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
        x = self.log_mel_spectrogram(x)
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
    def alignment_mlp_params(self):
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

