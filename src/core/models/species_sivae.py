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

from src.core.transforms.frame import unframe_fold as unframe, frame_fold as frame
from src.core.utils import Batch, TensorDict, bounded_sigmoid, linear_schedule, linear_decay, detach_values, prefix_keys, histogram_to_wandb
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior, circular_variance

__all__ = ["SpeciesSIVAE"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SpeciesSIVAE(L.LightningModule):
    def __init__(
        self,
        front_end: torch.nn.Module,
        feature_encoder: omegaconf.DictConfig,
        content_encoder: omegaconf.DictConfig,
        feature_decoder: omegaconf.DictConfig,
        content_decoder: omegaconf.DictConfig,
        classifier: omegaconf.DictConfig,
        alignment_encoder: omegaconf.DictConfig,
        target_names: List[str],
        target_counts: List[int],
        clf_checkpoint_path: str | None = None,
        latent_dim: int = 128,
        frequency_dim: int = 64,
        frame_window_length: int = 192,
        frame_padding_mode: int = "circular",
        translation_mode: str = "bicubic",
        translation_idx: int = 2,
        cross_decode_method: str = "soft",
        beta: float = 1.0,
        gamma: float = 1.0,
        sigma_x: float = 0.2,
        sigma_z_min: float = 1e-5,
        learning_rate: float = 1e-4,
        optimiser_cls: str = "torch.optim.AdamW",
        optimiser_config: omegaconf.DictConfig | None = None,
        scheduler_cls: str | None = None,
        scheduler_config: omegaconf.DictConfig | None = None,
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
        self.translation_mode = translation_mode
        self.translation_idx = translation_idx
        self.cross_decode_method = cross_decode_method
        self.beta = beta
        self.gamma = gamma
        self.sigma_x = sigma_x
        self.sigma_z_min = sigma_z_min
        self.x_i_frame_prob = x_i_frame_prob
        self.delta_prob_step_start = delta_prob_step_start
        self.delta_prob_step_end = delta_prob_step_end
        self.delta_prob_min = delta_prob_min
        self.delta_prob_max = delta_prob_max
        self.delta_sigma_min = delta_sigma_min
        self.delta_sigma_max = delta_sigma_max
        self.delta_sigma_step_slope = delta_sigma_step_slope
        self.delta_sigma_step_start = delta_sigma_step_start
        self.delta_sigma_step_end = delta_sigma_step_end
        self.align_only = align_only

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
        self.classifier = hydra.utils.instantiate(classifier, target_names, target_counts)
        if clf_checkpoint_path is not None:
            log.info("Loading classifier weights")
            ckpt = torch.load(clf_checkpoint_path, map_location="cuda")
            self.classifier.load_state_dict(ckpt["state_dict"])
        else:
            log.warning("No classifier weights provided!")

    def pre_process(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.front_end(wav)
        x = x.transpose(-1, -2) # transpose time to inner axis
        return T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])

    def forward(self, x: Tensor, y: torch.Tensor, s: torch.Tensor, *args: Any, t: int | None = None, **kwargs: Any) -> Dict[str, Tensor]:
        # shift by half for maximal coverage
        x_shifted = x.roll(shifts=(self.frame_window_length // 2,), dims=(-2,))
        x_i = torch.stack([x, x_shifted], dim=1)
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        x_shifted_framed = self.frame(x_shifted, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        x_i_framed = torch.stack([x_framed, x_shifted_framed], dim=1)
        seq_len = x_i.size(2)
        delta_i = torch.zeros(x_i_framed.size(0) * x_i_framed.size(1), x_i_framed.size(2), 1, device=x.device)
        x_i_framed = x_i_framed.flatten(end_dim=2)
        q_z_i = self.encode(x_i.flatten(end_dim=1), t=t) # (bs, seq, ld)
        mu_z_i, log_sigma_sq_z_i = q_z_i.chunk(2, dim=-1)
        # always randomly translated frames for target 2 (x_j)
        theta_j, *_ = self.sample_circle(x_i_framed.size(0), 1, 1, device=x.device)
        delta_j = theta_j / torch.pi
        x_j = self.translation(x_i_framed.transpose(-1, -2).contiguous(), delta_j, mode=self.translation_mode).transpose(-1, -2).contiguous()
        # encode posterior for translated frames separately
        q_z_j = self.encode(x_j, t=t) # (bs * seq, 1, ld)
        mu_z_j, log_sigma_sq_z_j = q_z_j.chunk(2, dim=-1)
        # stack together
        q_z = torch.cat([q_z_i, q_z_j.view(q_z_i.size())], dim=0)
        # decode to feature maps
        if self.cross_decode_method == "soft":
            # soft cross-decoding averages the distributions
            # mu_k = (mu_i + mu_j) / 2, sigma^2_k = (sigma^2_i + sigma^2_j) / 2^2
            mu_z = torch.stack([mu_z_i.flatten(end_dim=1), mu_z_j.flatten(end_dim=1)], dim=1).mean(dim=1)
            log_sigma_sq_z = (torch.stack([log_sigma_sq_z_i.flatten(end_dim=1).exp(), log_sigma_sq_z_j.flatten(end_dim=1).exp()], dim=1).sum(dim=1) / 4).log()
            z = torch.distributions.Normal(mu_z, (0.5 * log_sigma_sq_z).exp()).rsample()  # (bs, seq, ld)
            q_z_avg = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
            U_hat_i = self.content_decoder(z) # (bs * seq, ch, fr, fq)
            U_hat_j = U_hat_i
        elif self.cross_decode_method == "hard":
            z_i = torch.distributions.Normal(mu_z_i, (0.5 * log_sigma_sq_z_i).exp()).rsample()  # (bs, seq, ld)
            z_j = torch.distributions.Normal(mu_z_j, (0.5 * log_sigma_sq_z_j).exp()).rsample()  # (bs * seq, 1, ld)
            U_hat_i = self.content_decoder(z_i.flatten(end_dim=1))
            U_hat_j = self.content_decoder(z_j.flatten(end_dim=1))
        else:
            raise Exception("Cross decode method not specified, terminating")

        # reconstruct a contiguous sequence
        x_hat_i = self.cnn_decode(U_hat_j, delta_i) # (bs, 1, fr * seq, fq)
        # reconstruct independent translations
        x_hat_j = self.cnn_decode(U_hat_i, delta_j) # (bs * seq, 1, fr, fq)
        # frame for frame-wise loss
        x_hat_i_framed = self.frame(x_hat_i, window_length=self.frame_window_length, hop_length=self.frame_window_length).flatten(end_dim=1)
        x_framed = torch.cat([x_i_framed, x_j], dim=0)
        x_hat_framed = torch.cat([x_hat_i_framed, x_hat_j], dim=0)

        # fit classifiers using z
        q_z_avg = q_z_avg.view(x.size(0), q_z_i.size(1)*2, q_z_avg.size(-1))
        z = self.classifier.pre_process(q_z_avg)
        clf_outputs = self.classifier(z, y, s)

        return dict(
            x=x,
            x_framed=x_framed, x_i=x_i, x_j=x_j,
            x_hat_framed=x_hat_framed, x_hat_i=x_hat_i, x_hat_j=x_hat_j,
            q_z=q_z, q_z_i=q_z_i, q_z_j=q_z_j,
            seq_len=seq_len,
            **clf_outputs,
        )

    def predict(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        q_z, _ = self.encode(x)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        z = torch.distributions.Normal(mu_z, (1/2 * log_sigma_sq_z).exp()).rsample()
        clf_outputs = self.classifier(z)
        x_hat = self.cnn_decode(self.content_decoder(z.flatten(end_dim=1)), delta)
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        x_hat_framed = self.frame(x_hat, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        return dict(
            x=x,
            x_hat=x_hat,
            x_framed=x_framed,
            x_hat_framed=x_hat_framed,
            q_z=q_z,
            **clf_outputs,
        )

    def sample_circle(self, *args: Any, scaling_factor: float = 1.0, **kwargs: Any):
        delta = scaling_factor * ((torch.rand(*args, **kwargs) * 2) - 1)
        theta = torch.pi * delta
        dx, dy = torch.cos(theta), torch.sin(theta)
        return theta, dx, dy

    def encode(self, x: Tensor, hop_length: int | None = None, t: int | None = None) -> Tensor:
        # feature extraction
        x, us = self.feature_encoder(x)
        x_window_length = self.frame_window_length // 2**(self.feature_encoder.num_layers)
        x_hop_length = (hop_length or self.frame_window_length) // 2**(self.feature_encoder.num_layers)
        u_window_length = self.frame_window_length // 2**2
        u_hop_length = (hop_length or self.frame_window_length) // 2**2
        x = self.frame(x, window_length=x_window_length, hop_length=x_hop_length, padding_mode=self.frame_padding_mode)
        u = self.frame(us[1], window_length=u_window_length, hop_length=u_hop_length, padding_mode=self.frame_padding_mode)
        # content bottleneck
        q_z = self.content_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        if self.sigma_z_min is not None:
            mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
            sigma_latent = torch.tensor(self.sigma_z_min, dtype=torch.float32, requires_grad=False, device=mu_z.device)
            log_sigma_sq_z = log_sigma_sq_z.clamp(min=2*sigma_latent.pow(2).log())
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        return q_z

    def decode(self, z: Tensor, delta: Tensor | None = None) -> Tensor:
        U = self.content_decoder(z.flatten(end_dim=1))
        x_hat = self.cnn_decode(U, delta)
        return x_hat

    def cnn_decode(self, U: Tensor, delta: Tensor) -> Tensor:
        for i, block in enumerate(self.feature_decoder.blocks):
            if i == self.translation_idx:
                U = U.transpose(-1, -2).contiguous()
                U = self.translation(U, delta.view(delta.size(0) * delta.size(1)), mode=self.translation_mode)
                U = U.transpose(-1, -2).contiguous()
            if i == len(self.feature_decoder.blocks) - 1:
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
        grid[..., 0] = ((xx + 1) % 2) - 1
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
        y: float,
        y_probs: float,
        samples_per_class: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = TensorDict()
        losses = []
        # maximise likelihood p(x_i|z_j) framewise to ensure invariance to sequence length
        sigma_recon = torch.tensor(self.sigma_x, dtype=torch.float32, requires_grad=False, device=x_framed.device)
        nll = negative_log_likelihood(x_framed, x_hat_framed, sigma_recon.pow(2).log()).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        outputs |= dict(log_likelihood_x=-nll.detach().mean())
        # classification loss
        clf_loss_outputs = self.classifier.loss(y=y, y_probs=y_probs, samples_per_class=samples_per_class)
        # standard normal dkl
        dkl = self.beta * gaussian_kl_divergence_standard_prior(q_z).sum(dim=-1)
        losses.append(dkl.mean())
        outputs |= dict(dkl=dkl.detach().mean())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        return outputs + clf_loss_outputs

    @torch.no_grad()
    def metrics(
        self,
        x_framed: Tensor,
        x_hat_framed: Tensor,
        q_z: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        y: Tensor,
        y_probs: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        # distribution of z mean and varaince
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        mu_hist = np.histogram(mu_z.flatten().cpu().numpy(), bins=32, range=[-5.0, 5.0])
        sigma_hist = np.histogram(sigma_z.flatten().cpu().numpy(), bins=32, range=[0.0, 2.0])
        mu_z_i, mu_z_j = q_z_i.chunk(2, dim=-1)[0], q_z_j.view(q_z_i.size()).chunk(2, dim=-1)[0]
        # mean distance between shifted embeddings
        z_dist = (mu_z_j - mu_z_i).abs().mean()
        # distribution of delta predictions
        d_x = (x_hat_framed - x_framed).flatten(start_dim=-3)
        mae = d_x.abs().mean(dim=-1).mean()
        mse = d_x.pow(2).mean(dim=-1).mean()
        # normalised KL
        dkl_norm = ((-1/2 * (1 + log_sigma_sq_z - mu_z.pow(2) - log_sigma_sq_z.exp())).sum(dim=-1) / self.latent_dim).mean()
        return dict(
            mae=mae,
            mse=mse,
            mu_z_hist=mu_hist,
            sigma_z=sigma_hist,
            z_dist=z_dist,
            dkl_norm=dkl_norm,
        )

    @torch.no_grad()
    def tracking_figures(
        self,
        x: torch.Tensor,
        x_i: torch.Tensor,
        x_hat_i: torch.Tensor,
        x_j: torch.Tensor,
        x_hat_j: torch.Tensor,
        q_z_i: torch.Tensor,
        seq_len: int,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        num_samples: int = 6,
        num_frames: int = 6,
        **kwargs: Any,
    ) -> List:
        figures = []
        num_samples = min(num_samples, x.size(0))
        if q_z_i.size(1) == seq_len:
            specs = x_i.squeeze().cpu().numpy()
            recons = x_hat_i.squeeze().cpu().numpy()
            for i in range(num_samples):
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, width_ratios=[0.97, 0.03], constrained_layout=True, dpi=dpi)
                mesh = self.front_end.plot(specs[i].T, ax=axes[0, 0], vmin=specs.min(), vmax=specs.max())
                mesh = self.front_end.plot(recons[i].T, ax=axes[1, 0], vmin=specs.min(), vmax=specs.max())
                axes[0, 0].set_title("Original")
                axes[1, 0].set_title("Reconstruction")
                fig.colorbar(mesh, cax=axes[0, 1], orientation="vertical")
                fig.colorbar(mesh, cax=axes[1, 1], orientation="vertical")
                figures.append(("spectrogram", fig))
        else:
            specs = x_i.view(-1, seq_len, self.frame_window_length, self.frequency_dim).cpu().numpy()
            recons = x_hat_i.view(-1, seq_len, self.frame_window_length, self.frequency_dim).cpu().numpy()
            for i in range(num_samples):
                fig, axes = plt.subplots(nrows=2, ncols=num_frames + 1, figsize=figsize, width_ratios=[*[0.97 / num_frames]*num_frames, 0.03], constrained_layout=True, dpi=dpi)
                for j in range(num_frames):
                    mesh = self.front_end.plot(specs[i, j].T, ax=axes[0, j], vmin=specs.min(), vmax=specs.max())
                    mesh = self.front_end.plot(recons[i, j].T, ax=axes[1, j], vmin=specs.min(), vmax=specs.max())
                axes[0, 0].set_title("Original")
                axes[1, 0].set_title("Reconstruction")
                fig.colorbar(mesh, cax=axes[0, -1], orientation="vertical")
                fig.colorbar(mesh, cax=axes[1, -1], orientation="vertical")
                figures.append((f"frames/i", fig))
        # plot translated frames
        specs = x_j.view(-1, seq_len, self.frame_window_length, self.frequency_dim).cpu().numpy()
        recons = x_hat_j.view(-1, seq_len, self.frame_window_length, self.frequency_dim).cpu().numpy()
        for i in range(num_samples):
            fig, axes = plt.subplots(nrows=2, ncols=num_frames + 1, figsize=figsize, width_ratios=[*[0.97 / num_frames]*num_frames, 0.03], constrained_layout=True, dpi=dpi)
            for j in range(num_frames):
                mesh = self.front_end.plot(specs[i, j].T, ax=axes[0, j], vmin=specs.min(), vmax=specs.max())
                mesh = self.front_end.plot(recons[i, j].T, ax=axes[1, j], vmin=specs.min(), vmax=specs.max())
            axes[0, 0].set_title("Original")
            axes[1, 0].set_title("Reconstruction")
            fig.colorbar(mesh, cax=axes[0, -1], orientation="vertical")
            fig.colorbar(mesh, cax=axes[1, -1], orientation="vertical")
            figures.append((f"frames/j", fig))
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
            log.info(f"Saving model configuration to {config_path}")
            omegaconf.OmegaConf.save(config, config_path)
        # running test
        log.info(f"Testing <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, datamodule=data_module)

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
