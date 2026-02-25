import hydra
import lightning as L
import pathlib
import logging
import numpy as np
import torch
import wandb

from dataclasses import dataclass, field
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal
from typing import Any, Dict, Callable, List

from src.core.utils.metrics import gaussian_kl_divergence
from src.core.utils import detach_values, prefix_keys, Batch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["VAESoundscapeGenerator"]

# def timestep_interpolation(self, q_z: torch.Tensor, k: int):
#     ts = torch.arange(q_z.size(0) - 1, device=q_z.device)
#     ks = torch.linspace(0.0, 1.0 - 1 / (k + 1), k + 1, device=q_z.device)
#     # interpolate K points between each time-step by a weighted sum of gaussians
#     # encourage small steps in the latent space during training, during inference we throw these points away
#     mu, log_sigma_sq = q_z.chunk(2, dim=-1)
#     sigma_sq = log_sigma_sq.exp()
#     # weighted average of gaussians
#     mu_bar = torch.stack([
#         torch.stack([slerp(mu[t], mu[t + 1], k) for k in ks], dim=0)
#         for t in ts
#     ], dim=0) # (ts - 1, k, bs, ld)
#     # variance of weighted average of gaussians
#     # (1 - k)σ₁² + kσ₂² + k(1 - k)(μ₁ - μ₂)²
#     sigma_sq_bar = torch.stack([
#         torch.stack([(1 - k) * sigma_sq[t] + k * sigma_sq[t + 1] + (k * (1 - k) * (mu[t] - mu[t + 1]).pow(2)) for k in ks], dim=0)
#         for t in ts
#     ], dim=0) # (ts - 1, k, bs, ld)
#     mu_bar = torch.cat([mu_bar.flatten(start_dim=0, end_dim=1), mu[-1:]], dim=0)
#     sigma_sq_bar = torch.cat([sigma_sq_bar.flatten(start_dim=0, end_dim=1), sigma_sq[-1:]], dim=0)
#     q_z_bar = torch.cat([mu_bar, sigma_sq_bar.log()], dim=-1)
#     assert q_z_bar.size(0) == q_z.size(0) + (q_z.size(0) - 1) * k
#     return q_z_bar

# def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float | torch.Tensor, DOT_THRESHOLD: float = 0.9995):
#     assert v0.shape == v1.shape, "shapes of v0 and v1 must match"
#     # Normalize the vectors to get the directions and angles
#     v0_norm = torch.linalg.norm(v0, dim=-1)
#     v1_norm = torch.linalg.norm(v1, dim=-1)
#     v0_normed = v0 / v0_norm.unsqueeze(-1)
#     v1_normed = v1 / v1_norm.unsqueeze(-1)
#     # Dot product with the normalized vectors
#     dot = (v0_normed * v1_normed).sum(-1)
#     dot_mag = dot.abs()
#     # if dp is NaN, it's because the v0 or v1 row was filled with 0s
#     # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
#     gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
#     can_slerp = ~gotta_lerp
#     t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, torch.Tensor) else 0
#     t_batch_dims: torch.Size = t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
#     out = torch.zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))
#     # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
#     if gotta_lerp.any():
#         lerped = torch.lerp(v0, v1, t)
#         out = lerped.where(gotta_lerp.unsqueeze(-1), out)
#     # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
#     if can_slerp.any():
#         # Calculate initial angle between v0 and v1
#         theta_0 = dot.arccos().unsqueeze(-1)
#         sin_theta_0 = theta_0.sin()
#         # Angle at timestep t
#         theta_t = theta_0 * t
#         sin_theta_t = theta_t.sin()
#         # Finish the slerp algorithm
#         s0 = (theta_0 - theta_t).sin() / sin_theta_0
#         s1 = sin_theta_t / sin_theta_0
#         slerped = s0 * v0 + s1 * v1
#         out = slerped.where(can_slerp.unsqueeze(-1), out)
#     return out

@dataclass(unsafe_hash=True, kw_only=True)
class VAESoundscapeGenerator(L.LightningModule):
    num_features: int = 128
    num_samples: int = 1
    num_rnn_layers: int = 3
    num_init_frames: int = 10
    sigma_dzdt: float = 0.2

    teach_prob_start: float | None = 1.0
    teach_prob_end: float = 0.1
    teach_prob_step_start: int | None = None
    teach_prob_step_end: int | None = None
    teach_prob_slope: float | None = None

    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1

    mu_agg: torch.nn.Parameter | None = field(init=False)
    sigma_agg: torch.nn.Parameter | None = field(init=False)

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Calculating aggregate posterior over training samples <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        mu_agg, sigma_agg = data_module.train_data.aggregate_posterior_analytical()
        self.mu_agg = torch.nn.Parameter(mu_agg, requires_grad=False)
        self.sigma_agg = torch.nn.Parameter(sigma_agg, requires_grad=False)
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(self, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        trainer.test(self, dataloaders=data_module.test_dataloader())

    def evaluate(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Generating examples using <{config.model.get('_target_')}>")
        trainer.predict(self, dataloaders=data_module.predict_dataloader())

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        self.embedding = torch.nn.Linear(in_features=self.num_features, out_features=self.num_features)
        self.norm = torch.nn.LayerNorm(self.num_features)
        self.sequence_encoder = torch.nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.num_features,
            num_layers=self.num_rnn_layers,
        )
        self.sequence_decoder = torch.nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.num_features,
            num_layers=self.num_rnn_layers,
            batch_first=False,
        )
        self._reset_cache()

    def forward(self, q_z: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        # q_z = torch.cat([q_z[:, :, 64:128], q_z[:, :, 192:256]], dim=-1)
        q_z_bar = q_z.transpose(0, 1)
        # q_z_bar = self.timestep_interpolation(q_z, self.num_points)
        q_z_bar = q_z_bar.expand(self.num_samples, -1, -1, -1).transpose(0, 1).flatten(start_dim=1, end_dim=2) # (seq + k(seq-1), bs * samples, ld)
        mu, log_sigma_sq = q_z_bar.chunk(2, dim=-1)
        seq, bs, ld = mu.size()

        dt = 1
        z = Normal(mu, (0.5 * log_sigma_sq).exp()).rsample()
        dz_dt = z.diff(dim=0, n=1) / dt

        # _, (h, c) = self.sequence_encoder()

        if self.training:
            if self.with_curriculum:
                teach_prob = self.teach_prob_current(self.trainer.global_step)
                h = torch.zeros(self.num_rnn_layers, bs, ld, device=z.device)
                c = torch.zeros(self.num_rnn_layers, bs, ld, device=z.device)
                z_t = z[0].unsqueeze(0)
                z_hats = [z_t]
                dzhat_dts = []
                for t in range(seq - 1):
                    # predict Δẑₜ using ẑₜ (or zₜ)
                    dzhat_dt, (h, c) = self.sequence_decoder(self.norm(self.embedding(z_t)), (h, c))
                    # update ẑₜ₊₁ = ẑₜ+ Δẑₜ
                    z_t = z_t + dzhat_dt
                    dzhat_dts.append(dzhat_dt)
                    z_hats.append(z_t)
                    # optional: ẑₜ₊₁ = zₜ₊₁ (teacher forcing)
                    if np.random.choice([0, 1], p=[1 - teach_prob, teach_prob]):
                        z_t = z[t + 1].unsqueeze(0)
                dzhat_dt = torch.cat(dzhat_dts, dim=0)
                z_hat = torch.cat(z_hats, dim=0)
            else:
                # predict Δẑₜ
                dzhat_dt, (h, c) = self.sequence_decoder(self.norm(self.embedding(z[:-1])))
                # ẑₜ₊₁ = ẑₜ+ Δẑₜ
                z_hat = z[:-1] + dzhat_dt
        else:
            # initialize hidden states up to zₜ
            t_init = self.num_init_frames
            z_init = z[:t_init]
            dzhat_dts, (h, c) = self.sequence_decoder(self.norm(self.embedding(z_init)))
            # list of (1, bs, ld)
            z_hats = list(torch.cat([z_init[0].unsqueeze(0), z_init + dzhat_dts], dim=0).unsqueeze(1).unbind(dim=0))
            z_t = z_hats[t_init]
            dzhat_dts = []
            for t in range(t_init, seq - 1):
                # predict Δẑₜ using ẑₜ
                dzhat_dt, (h, c) = self.sequence_decoder(self.norm(self.embedding(z_t)), (h, c))
                # update ẑₜ₊₁ = ẑₜ+ Δẑₜ
                z_t = z_t + dzhat_dt
                dzhat_dts.append(dzhat_dt)
                z_hats.append(z_t)
            dz_dt = dz_dt[t_init:]
            dzhat_dt = torch.cat(dzhat_dts, dim=0)
            z_hat = torch.cat(z_hats, dim=0)
        return dict(
            dzhat_dt=dzhat_dt.transpose(0, 1),
            dz_dt=dz_dt.transpose(0, 1),
            z_hat=z_hat.transpose(0, 1),
            q_z=q_z_bar.transpose(0, 1),
            z=z.transpose(0, 1),
        )

    @property
    def with_curriculum(self) -> bool:
        return self.teach_prob_step_start is not None and self.teach_prob_step_end is not None

    @property
    def teach_prob_params(self) -> Dict[str, Any]:
        return dict(
            x_min=self.teach_prob_step_start,
            x_max=self.teach_prob_step_end,
            y_min=self.teach_prob_end,
            y_max=self.teach_prob_start,
            k=self.teach_prob_slope,
        )

    @staticmethod
    def bounded_sigmoid(x: float, x_min: float, x_max: float, y_min: float, y_max: float, k: float):
        s = np.floor(np.log10(np.abs(x_max)))
        z = k / 10**(s - 1)
        return y_min + (y_max - y_min) / (1 + np.exp(-z * (x - ((x_min + x_max) / 2))))

    def teach_prob_current(self, current_step: int) -> torch.Tensor:
        if not self.training:
            return 0.0
        elif self.teach_prob_start is None:
            return self.teach_prob_end
        return torch.tensor(self.bounded_sigmoid(current_step, **self.teach_prob_params))

    @torch.no_grad()
    def generate(
        self,
        seq: int,
        z0: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not z0:
            z0 = MultivariateNormal(self.mu_agg.expand(1, -1), self.sigma_agg.expand(1, -1, -1)).rsample().unsqueeze(0)
        if not h0:
            h0 = torch.zeros(self.num_rnn_layers, 1, self.num_features, device=z.device)
        if not c0:
            c0 = torch.zeros(self.num_rnn_layers, 1, self.num_features, device=z.device)
        # initial conditions for generation
        zs = []
        z, h, c = z0, h0, c0
        for i in range(seq):
            z, (h, c) = self.sequence_decoder(z, (h, c))
            zs.append(z)
        return torch.stack(zs, dim=0).transpose(0, 1)

    def loss(self, dzhat_dt: torch.Tensor, dz_dt: torch.Tensor, z_hat: torch.Tensor, q_z: torch.Tensor, z: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        bs, seq, ld = dz_dt.size()
        losses = []
        outputs = dict()
        log_sigma_sq_dzdt = torch.tensor(self.sigma_dzdt).pow(2).log()
        nll = 1/2 * (log_sigma_sq_dzdt + ((dz_dt - dzhat_dt).pow(2) / log_sigma_sq_dzdt.exp()))
        nll = nll.sum(dim=-1).mean()
        losses.append(nll)
        outputs |= dict(
            log_likelihood_dzdt=-nll.detach(),
            teach_prob=self.teach_prob_current(self.trainer.global_step),
            # **{f"nll_z{i}": nll[:, :, i].mean() for i in range(ld)},
        )
        alpha = torch.cat([
            torch.zeros(64, dtype=torch.float32, device=dzhat_dt.device),
            torch.linspace(0.5, 1.0, 64, device=dzhat_dt.device),
        ])
        dzdt = (alpha * dzhat_dt).abs().sum(dim=-1)
        losses.append(dzdt.mean())
        outputs |= dict(
            dzdt=dzdt.mean().detach(),
        )
        # # the negative log likelihood where our prior is the encoder's posterior
        # # is the mahanalobis distance between the predicted sample and the VAE encoders' posterior
        # mu, log_sigma_sq = q_z[:, 1:].chunk(2, dim=-1)
        # nll = (1/2 * (log_sigma_sq + ((z_hat - mu).pow(2) / log_sigma_sq.exp())))
        # loss = nll.sum(dim=-1).mean()
        # losses.append(loss)
        # outputs |= dict(
        #     log_likelihood_z=-loss.detach(),
        #     **{f"log_likelihood_z{i}": -nll[:, :, i].mean() for i in range(ld)},
        # )
        # # # an alternative approach we compute the KL divergence between p(z_t|z_<t) and q(z_t|x_t)
        # dkl = gaussian_kl_divergence(p_z, q_z)
        # regularise by minimising the absolute value of the first derivative of predicted features
        # dzhat_dt = z_hat.diff(dim=1, n=1).pow(2)
        # dz_dt = z.diff(dim=1, n=1).pow(2)
        # loss = dzhat_dt.sum(dim=-1).mean()
        # losses.append(self.gamma * loss)
        # outputs |= dict(
        #     dzhat_dt=loss.detach(),
        #     dz_dt=dz_dt.detach().sum(dim=-1).mean(),
        #     **{f"dzhat{i}_dt": dzhat_dt[:, :, i].mean() for i in range(ld)},
        #     **{f"dz{i}_dt": dz_dt[:, :, i].mean() for i in range(ld)},
        # )
        # backprop loss
        loss = sum(losses)
        outputs |= dict(loss=loss)
        return outputs

    def metrics(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = dict()
        if self.with_curriculum:
            outputs |= dict(teach_prob=self.teach_prob_current(self.trainer.global_step))
        return outputs

    def training_step(self, batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self(batch.x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        metrics = self.metrics(**step_outputs, **loss_outputs)
        self.training_step_outputs.append(step_outputs)
        self.log_dict(prefix_keys(loss_outputs | metrics, "train"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, "train")))
        return loss_outputs | step_outputs

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch, batch_idx: int) -> None:
        self.training_step_outputs.clear()

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self(batch.x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        metrics = self.metrics(**step_outputs)
        self.validation_step_outputs.append(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, "val")))
        return loss_outputs | step_outputs

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int) -> None:
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self(batch.x, **kwargs)

    @torch.no_grad()
    def on_test_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
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


# --------------------------------------------------------------------------------------- #

from src.core.models.backbones.transformer_encoder import TransformerEncoder

@dataclass(unsafe_hash=True, kw_only=True)
class VAESoundscapeGeneratorTransformer(L.LightningModule):
    num_features: int = 128
    num_attn_layers: int = 1
    num_attn_heads: int = 8
    mlp_ratio: int = 4
    num_samples: int = 1
    # num_points: int = 10
    gamma: float = 1.0

    teach_prob_start: float | None = 1.0
    teach_prob_end: float = 0.1
    teach_prob_step_start: int | None = None
    teach_prob_step_end: int | None = None
    teach_prob_slope: float | None = None

    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1

    mu_agg: torch.nn.Parameter | None = field(init=False)
    sigma_agg: torch.nn.Parameter | None = field(init=False)

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Calculating aggregate posterior over training samples <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        mu_agg, sigma_agg = data_module.train_data.aggregate_posterior_analytical()
        self.mu_agg = torch.nn.Parameter(mu_agg, requires_grad=False)
        self.sigma_agg = torch.nn.Parameter(sigma_agg, requires_grad=False)
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(self, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        trainer.test(self, dataloaders=data_module.test_dataloader())

    def evaluate(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Generating examples using <{config.model.get('_target_')}>")
        trainer.predict(self, dataloaders=data_module.predict_dataloader())

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        self.embed = torch.nn.Linear(in_features=self.num_features, out_features=self.num_features)
        self.sequence_decoder = TransformerEncoder(
            input_size=self.num_features,
            mlp_ratio=self.mlp_ratio,
            depth=self.num_attn_layers,
            num_heads=self.num_attn_heads,
        )
        self.project = torch.nn.Linear(in_features=self.num_features, out_features=self.num_features)

    def pre_process(self, q_z: torch.Tensor):
        k = self.num_points
        ts = torch.arange(q_z.size(0) - 1, device=q_z.device)
        ks = torch.linspace(0.0, 1.0 - 1 / (k + 1), k + 1, device=q_z.device)
        # interpolate K points between each time-step by a weighted sum of gaussians
        # encourage small steps in the latent space during training, during inference we throw these points away
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        sigma_sq = log_sigma_sq.exp()
        # mean of weighted mixture of gaussians
        # NB: just a linear interpolation at the moment, but a spherical linear interpolation may help generalise
        mu_bar = torch.stack([
            torch.stack([self.slerp(mu[t], mu[t + 1], k) for k in ks], dim=0)
            for t in ts
        ], dim=0) # (ts - 1, k, bs, ld)
        # variance of weighted sum of gaussians
        # (1 - k)σ₁² + kσ₂² + k(1 - k)(μ₁ - μ₂)²
        sigma_sq_bar = torch.stack([
            torch.stack([(1 - k) * sigma_sq[t] + k * sigma_sq[t + 1] + (k * (1 - k) * (mu[t] - mu[t + 1]).pow(2)) for k in ks], dim=0)
            for t in ts
        ], dim=0) # (ts - 1, k, bs, ld)
        mu_bar = torch.cat([mu_bar.flatten(start_dim=0, end_dim=1), mu[-1:]], dim=0)
        sigma_sq_bar = torch.cat([sigma_sq_bar.flatten(start_dim=0, end_dim=1), sigma_sq[-1:]], dim=0)
        q_z_bar = torch.cat([mu_bar, sigma_sq_bar.log()], dim=-1)
        assert q_z_bar.size(0) == q_z.size(0) + (q_z.size(0) - 1) * k
        return q_z_bar

    def forward(self, q_z: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        q_z = q_z.transpose(0, 1)
        q_z_bar = self.pre_process(q_z)
        q_z_bar = q_z_bar.expand(self.num_samples, -1, -1, -1).transpose(0, 1).flatten(start_dim=1, end_dim=2) # (seq + k(seq-1), bs * samples, ld)
        mu, log_sigma_sq = q_z_bar.chunk(2, dim=-1)
        seq, bs, ld = mu.size()

        z = Normal(mu[:-1], (0.5 * log_sigma_sq[:-1]).exp()).rsample()
        z_e = self.embed(z)
        z_e = z_e + self.positional_encoding(z_e.size(0), z_e.size(2)).to(z_e.device).view(z_e.size(0), 1, z_e.size(2))
        attn_mask = self.causal_mask(z_e.size(0)).to(z_e.device)

        if self.training:
            x, attn_w = self.sequence_decoder(z_e, attn_mask=attn_mask)
            z_hat = self.project(x)
        else:
            z_hat = torch.empty(0, *z[0].shape, device=z.device)
            for t in range(1, seq):
                z_e_t = self.embed(torch.cat([z[0].unsqueeze(0), z_hat], dim=0))
                z_e_t = z_e_t + self.positional_encoding(z_e_t.size(0), z_e_t.size(2)).to(z_e_t.device).view(z_e_t.size(0), 1, z_e_t.size(2))
                attn_mask = self.causal_mask(z_e_t.size(0)).to(z_e_t.device)
                x, attn_w = self.sequence_decoder(z_e_t, attn_mask=attn_mask)
                z_hat = self.project(x)

        return dict(
            z_hat=z_hat.transpose(0, 1),
            q_z=q_z_bar.transpose(0, 1),
        )

    @staticmethod
    def positional_encoding(sequence_len: int, embedding_dim: int) -> torch.Tensor:
        positional_embedding = torch.zeros(sequence_len, embedding_dim)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        return positional_embedding

    @staticmethod
    def causal_mask(sequence_len: int) -> torch.Tensor:
        neg_inf = torch.from_numpy(np.ones((sequence_len, sequence_len)) * -np.inf)
        return torch.triu(neg_inf, diagonal=1).float()

    @staticmethod
    def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float | torch.Tensor, DOT_THRESHOLD: float = 0.9995):
        assert v0.shape == v1.shape, "shapes of v0 and v1 must match"
        # Normalize the vectors to get the directions and angles
        v0_norm = torch.linalg.norm(v0, dim=-1)
        v1_norm = torch.linalg.norm(v1, dim=-1)
        v0_normed = v0 / v0_norm.unsqueeze(-1)
        v1_normed = v1 / v1_norm.unsqueeze(-1)
        # Dot product with the normalized vectors
        dot = (v0_normed * v1_normed).sum(-1)
        dot_mag = dot.abs()
        # if dp is NaN, it's because the v0 or v1 row was filled with 0s
        # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
        gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
        can_slerp = ~gotta_lerp
        t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, torch.Tensor) else 0
        t_batch_dims: torch.Size = t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
        out = torch.zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))
        # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
        if gotta_lerp.any():
            lerped = torch.lerp(v0, v1, t)
            out = lerped.where(gotta_lerp.unsqueeze(-1), out)
        # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
        if can_slerp.any():
            # Calculate initial angle between v0 and v1
            theta_0 = dot.arccos().unsqueeze(-1)
            sin_theta_0 = theta_0.sin()
            # Angle at timestep t
            theta_t = theta_0 * t
            sin_theta_t = theta_t.sin()
            # Finish the slerp algorithm
            s0 = (theta_0 - theta_t).sin() / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            slerped = s0 * v0 + s1 * v1
            out = slerped.where(can_slerp.unsqueeze(-1), out)
        return out

    @staticmethod
    def bounded_sigmoid(x: float, x_min: float, x_max: float, y_min: float, y_max: float, k: float):
        s = np.floor(np.log10(np.abs(x_max)))
        z = k / 10**(s - 1)
        return y_min + (y_max - y_min) / (1 + np.exp(-z * (x - ((x_min + x_max) / 2))))

    @property
    def teach_prob_params(self) -> Dict[str, Any]:
        return dict(
            x_min=self.teach_prob_step_start,
            x_max=self.teach_prob_step_end,
            y_min=self.teach_prob_start,
            y_max=self.teach_prob_end,
            k=self.teach_prob_slope,
        )

    def teach_prob_current(self, current_step: int) -> torch.Tensor:
        if not self.training:
            return 0.0
        elif self.teach_prob_start is None:
            return self.teach_prob_end
        return torch.tensor(self.bounded_sigmoid(current_step, **self.teach_prob_params))

    def loss(self, z_hat: torch.Tensor, q_z: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        losses = []
        outputs = dict()
        # the negative log likelihood where our prior is the encoder's posterior
        # is the mahanalobis distance between the predicted sample and the VAE encoders' posterior
        mu, log_sigma_sq = q_z[:, 1:].chunk(2, dim=-1)
        nll = (1/2 * (log_sigma_sq + ((z_hat - mu).pow(2) / log_sigma_sq.exp()))).sum(dim=-1).mean()
        losses.append(nll)
        outputs |= dict(log_likelihood_z=-nll.detach())
        # # an alternative approach we compute the KL divergence between p(z_t|z_<t) and q(z_t|x_t)
        # dkl = gaussian_kl_divergence(p_z, q_z)
        # regularise by minimising the absolute value of the first derivative of predicted features
        # encourage temporal smoothness between interpolated timesteps
        # maybe because we've smoothed out the space, this should be a squared error?
        # at the moment, this grows past the actual derivative, which means potentially its stepping around rather than smoothly transitioning?
        dzhdt = z_hat.diff(dim=1, n=1).pow(2).sum(dim=-1).mean()
        losses.append(self.gamma * dzhdt)
        outputs |= dict(
            dzhat_dt=dzhdt.detach(),
            dz_dt=mu.detach().diff(dim=1, n=1).abs().sum(dim=-1).mean(),
        )
        # backprop loss
        loss = sum(losses)
        outputs |= dict(loss=loss)
        return outputs

    def metrics(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return dict(
            teach_prob=self.teach_prob_current(self.trainer.global_step)
        )

    def step(self, batch, batch_idx: int, stage: str, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self(batch.x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        metrics = self.metrics(**step_outputs)
        self.log_dict(prefix_keys(loss_outputs | metrics, stage), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, stage)))
        return loss_outputs | step_outputs

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, "train")

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, "val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self(batch.x, **kwargs)

    @torch.no_grad()
    def configure_optimizers(self) -> torch.optim.Optimizer:
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
