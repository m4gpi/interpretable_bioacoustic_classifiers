import copy
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
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import functional as T
from typing import Any, Dict, List, Tuple

from src.core.transforms.frame import frame_fold as frame
from src.core.utils import Batch, bounded_sigmoid, linear_schedule, linear_decay, detach_values, prefix_keys, histogram_to_wandb, gaussian_kernel, laplace_kernel
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior
from src.core.utils.sketch import plot_latent_power_spectral_density_heatmap

__all__ = ["TCVAE"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class EMA:
    def __init__(self, decay=0.999):
        self.decay = decay

    @torch.no_grad()
    def update(self, student: torch.nn.Module, teacher: torch.nn.Module) -> None:
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(self.decay)
            t_param.data.add_((1 - self.decay) * s_param.data)

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
        beta: float = 1.0,
        sigma_x: float = 0.2,
        sigma_z_min: float = 1e-5,
        ema_sigma_z_min: float = 1e-3,
        smooth_prop: float = 0.75,
        alpha_step_start: int | None = 15_000,
        alpha_step_end: int | None = 100_000,
        alpha_max: float = 1.0,
        alpha_min: float = 0.25,
        alpha_step_slope: float = 1.0,
        learning_rate: float = 1e-4,
        optimiser_cls: str = "torch.optim.AdamW",
        optimiser_config: omegaconf.DictConfig | None = {},
        scheduler_cls: str | None = None,
        scheduler_config: omegaconf.DictConfig | None = {},
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.frequency_dim = frequency_dim
        self.frame_window_length = frame_window_length
        self.frame_padding_mode = frame_padding_mode
        self.beta = beta
        self.sigma_x = sigma_x
        self.sigma_z_min = sigma_z_min
        self.ema_sigma_z_min = ema_sigma_z_min
        self.smooth_prop = smooth_prop
        self.alpha_step_start = alpha_step_start
        self.alpha_step_end = alpha_step_end
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_step_slope = alpha_step_slope

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
        self.feature_encoder_ema = copy.deepcopy(self.feature_encoder)
        self.feature_encoder_ema.eval()
        self.content_encoder_ema = copy.deepcopy(self.content_encoder)
        self.content_encoder_ema.eval()
        self.ema = EMA(decay=0.999)

    @property
    def smooth_feature_idx(self):
        return torch.arange(0, int(self.latent_dim * self.smooth_prop), device=self.device)

    @property
    def sharp_feature_idx(self):
        return torch.arange(int(self.latent_dim * self.smooth_prop), self.latent_dim),

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

    def pre_process(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.front_end(wav)
        x = x.transpose(-1, -2) # transpose time to inner axis
        return T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        # student forward
        q_z, *_ = self.encode(x)
        mu_x, log_sigma_sq_x = q_z.chunk(2, dim=-1)
        z = torch.distributions.Normal(mu_x, (0.5 * log_sigma_sq_x).exp()).rsample()
        x_hat = self.decode(z).view(*x.size())
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        x_hat_framed = self.frame(x_hat, window_length=self.frame_window_length, hop_length=self.frame_window_length)

        # teacher forward
        with torch.no_grad():
            q_z_ema, *_ = self.encode_ema(x)
            mu_ema, log_sigma_sq_ema = q_z_ema.chunk(2, dim=-1)
            # clip the EMA variance, drop sharp dims
            mu_ema = mu_ema[:, :, self.smooth_feature_idx]
            sigma_sq_ema = log_sigma_sq_ema[:, :, self.smooth_feature_idx].clamp(min=2*np.log(self.ema_sigma_z_min)).exp()
            # fit a gaussian over a window in temporal dimension, moment matching the mixture
            log_sigma_min, log_sigma_max = -0.3, 0.5
            num_groups = 32
            latent_dim = mu_ema.size(-1)
            assert self.latent_dim // num_groups == self.latent_dim / num_groups
            sigmas = torch.stack([torch.logspace(log_sigma_min, log_sigma_max, num_groups, device=mu_ema.device) for i in range(latent_dim // num_groups)], dim=1).flatten().flip(dims=[0])
            kernels = gaussian_kernel(sigmas, mask_center=True)
            K = (kernels.size(-1) - 1) // 2
            mu_ema = F.pad(mu_ema.permute(0, 2, 1), (K, K), mode='circular') # TODO: should truncate weights rather than pad
            sigma_sq_ema = F.pad(sigma_sq_ema.permute(0, 2, 1), (K, K), mode='circular')
            # weighted sum of gaussians by convolution, rearranged for stability
            # μ* = Σ wₜμₜ, σ²* = Σ wₜ(σ²ₜ+ μ²ₜ) - μ²* = Σ wₜσ²ₜ + Σ wₜ(μₜ - μ*)²
            mu_bar = F.conv1d(mu_ema, kernels, groups=sigmas.size(0))
            sigma_sq_within = F.conv1d((sigma_sq_ema), kernels, groups=sigmas.size(0))
            sigma_sq_between = (kernels * (mu_ema.unfold(dimension=-1, size=2*K+1, step=1) - mu_bar.unsqueeze(-1)).pow(2)).sum(dim=-1)
            log_sigma_sq_bar = (sigma_sq_within + sigma_sq_between).log()
            mu_bar, log_sigma_sq_bar = mu_bar.permute(0, 2, 1), log_sigma_sq_bar.permute(0, 2, 1)
            log_sigma_sq_bar = log_sigma_sq_bar.clamp(min=2*np.log(self.sigma_z_min)) # ?? this is the one that is used to teach, so THIS should be clamped to be higher, the teacher shouldnt punish the student too much
            q_z_bar = torch.cat([mu_bar, log_sigma_sq_bar], dim=-1)

        return dict(
            x=x, x_framed=x_framed,
            x_hat=x_hat, x_hat_framed=x_hat_framed,
            q_z=q_z, q_z_bar=q_z_bar,
        )

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
        if self.sigma_z_min is not None:
            mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
            sigma_latent = torch.tensor(self.sigma_z_min, dtype=torch.float32, requires_grad=False, device=mu_z.device)
            log_sigma_sq_z = log_sigma_sq_z.clamp(min=2*sigma_latent.pow(2).log())
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        return q_z,

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        bs, seq, *other_dims = z.size()
        U = self.content_decoder(z.flatten(end_dim=1))
        for i, block in enumerate(self.feature_decoder.blocks):
            if i == len(self.feature_decoder.blocks) - 1:
                U = U.unflatten(0, (bs, seq)).transpose(1, 2).flatten(start_dim=2, end_dim=3)
            U = block(U)
        return U

    @staticmethod
    def frame(x: Tensor, window_length: int, hop_length: int | None = None, padding_mode: str = "circular") -> Tensor:
        if hop_length != window_length:
            return frame(x, window_length=window_length, hop_length=hop_length) if x.size(-2) > window_length else x.unsqueeze(1)
        return x.view(x.size(0), x.size(1), x.size(2) // window_length, window_length, x.size(3)).transpose(1, 2)

    def loss(
        self,
        x_framed: Tensor,
        x_hat_framed: Tensor,
        q_z: Tensor,
        q_z_bar: torch.Tensor,
        **kwargs: Any
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # batch/sequence mean frame-wise sum reconstruction loss
        sigma_recon = torch.tensor(self.sigma_x, dtype=torch.float32, requires_grad=False, device=x_framed.device)
        nll = negative_log_likelihood(x_framed, x_hat_framed, sigma_recon.pow(2).log()).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        outputs |= dict(log_likelihood_x=-nll.detach().mean())
        # batch/sequence mean frame-wise sum standard normal kl
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        dkl = (-1/2 * (1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp()))
        sharp_dkl = (self.sharp_features * dkl)
        # smooth dkl describes how far 'smooth' feature is from a standard normal
        smooth_dkl = (self.smooth_features * dkl)
        # temp dkl describes how far 'smooth' feature subset is from the teacher's time-smoothed posterior
        temp_dkl = torch.zeros_like(sharp_dkl)
        mu_smooth, log_sigma_sq_smooth = mu[:, :, self.smooth_feature_idx], log_sigma_sq[:, :, self.smooth_feature_idx]
        mu_bar, log_sigma_sq_bar = q_z_bar.chunk(2, dim=-1)
        var_ratio = (log_sigma_sq_smooth - log_sigma_sq_bar).exp()
        sq_err = (mu_smooth - mu_bar).pow(2) * (-log_sigma_sq_bar).exp()
        temp_dkl[:, :, self.smooth_feature_idx] = -1/2 * (1 + log_sigma_sq_smooth - log_sigma_sq_bar - var_ratio - sq_err)
        # cross-weigh the standard normal prior on the student with smoothness prior on the teacher
        alpha = bounded_sigmoid(self.trainer.global_step, **self.alpha_params)
        smooth_dkl = alpha * smooth_dkl
        temp_dkl = (1 - alpha) * temp_dkl
        # sum all KL terms
        dkl = self.beta * (sharp_dkl + smooth_dkl + temp_dkl).sum(dim=-1).mean()
        losses.append(dkl)
        outputs |= dict(
            dkl=dkl.detach(),
            sharp_dkl=sharp_dkl.detach().sum(dim=-1).mean(),
            temp_dkl=temp_dkl.detach().sum(dim=-1).mean(),
            smooth_dkl=smooth_dkl.detach().sum(dim=-1).mean(),
        )
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        return outputs

    @torch.no_grad()
    def metrics(
        self,
        x_framed: Tensor,
        x_hat_framed: Tensor,
        q_z: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        mu_hist = np.histogram(mu_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[-5.0, 5.0])
        sigma_hist = np.histogram(sigma_z.flatten(end_dim=-2).cpu().numpy(), bins=32, range=[0.0, 2.0])
        mae = (x_hat_framed - x_framed).abs().flatten(start_dim=-3).mean(dim=-1).mean()
        mse = (x_hat_framed - x_framed).pow(2).flatten(start_dim=-3).mean(dim=-1).mean()
        dkl_norm = ((-1/2 * (1 + log_sigma_sq_z - mu_z.pow(2) - log_sigma_sq_z.exp())).sum(dim=-1) / self.latent_dim).mean()
        return dict(
            mae=mae,
            mse=mse,
            mu_z_hist=mu_hist,
            sigma_z_hist=sigma_hist,
            dkl_norm=dkl_norm,
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

    def tracking_figures(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        q_z: torch.Tensor,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        num_samples: int = 6,
        num_frames: int = 6,
        **kwargs: Any,
    ) -> List:
        figures = []
        num_samples = min(num_samples, x.size(0))
        xs = x.squeeze().cpu().numpy()
        x_hats = x_hat.squeeze().cpu().numpy()
        for i in range(num_samples):
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, width_ratios=[0.97, 0.03], constrained_layout=True, dpi=dpi)
            mesh = self.front_end.plot(xs[i].T, ax=axes[0, 0], vmin=x.min(), vmax=x.max())
            mesh = self.front_end.plot(x_hats[i].T, ax=axes[1, 0], vmin=x.min(), vmax=x.max())
            axes[0, 0].set_title("Original")
            axes[1, 0].set_title("Reconstruction")
            fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
            fig.colorbar(mesh, cax=axes[1, 1], orientation="vertical")
            figures.append(("spectrogram", fig))

        mu = q_z.chunk(2, dim=-1)[0].cpu().numpy()
        for i in range(num_samples):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), width_ratios=[0.97, 0.03], constrained_layout=True, dpi=dpi)
            mu_norm = ((mu[i] - mu[i].mean(axis=0)) / mu[i].std(axis=0))
            im = plot_latent_power_spectral_density_heatmap(mu_norm, fft_length=mu[i].shape[0], **self.time_series_params, ax=axes[0])
            axes[0].set_title("Latent Sequence PSD")
            cbar = fig.colorbar(im, cax=axes[1], orientation="vertical")
            cbar.set_label("Magnitude")
            figures.append((f"latent_sequence_psd", fig))
            plt.close(fig)

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

    @property
    def alpha_params(self):
        return dict(
            x_min=self.alpha_step_start,
            x_max=self.alpha_step_end,
            y_min=self.alpha_min,
            y_max=self.alpha_max,
            k=self.alpha_step_slope,
        )

    @property
    def time_series_params(self):
        return dict(
            audio_sample_rate=self.front_end.sample_rate,
            audio_fft_hop_length=self.front_end.fft_hop_length,
            audio_frame_length_hops=self.frame_window_length,
        )
