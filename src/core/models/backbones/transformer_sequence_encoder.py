import einops
import lightning as L
import numpy as np
import torch
import wandb

from dataclasses import dataclass, field
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
from tempfile import TemporaryDirectory
from torch import Tensor
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    Dropout,
    Identity,
    init,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Parameter,
)
from torch.functional import F
from torch.distributions.normal import Normal
from torch.optim import Optimizer
from typing import (
    Any,
    Dict,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from temporal_vae.transformer import TransformerEncoder, TransformerDecoder
from temporal_vae import utils as U

__all__ = ["VAESequenceEncoder"]

@dataclass(unsafe_hash=True, kw_only=True)
class VAESequenceEncoder(L.LightningModule):
    latent_dim: int = 128
    num_smooth_in_time: int = 96
    latent_samples: int = 16
    alpha: float = 10.0
    k: int = 10
    Q: int  = 1
    log_epsilon_x: float = -6.0
    vae_artefact_id: Optional[str] = None

    attn_num_enc_heads: int = 8
    attn_num_dec_heads: int = 8
    attn_enc_depth: int = 1
    attn_dec_depth: int = 1
    attn_mlp_ratio: int = 4

    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: Optional[DictConfig] = None
    scheduler_cls: Optional[str] = None
    scheduler_config: Optional[DictConfig] = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1
    train_step_params: Optional[DictConfig] = None
    eval_step_params: Optional[DictConfig] = None

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        # initialise a learnable sequence summary token
        self.summary_token = Parameter(torch.zeros(1, 1, self.latent_dim))
        # apply dot-product weighted self-attention across time
        self.sequence_encoder = TransformerEncoder(
            input_size=self.latent_dim,
            mlp_ratio=self.attn_mlp_ratio,
            depth=self.attn_enc_depth,
            num_heads=self.attn_num_enc_heads,
        )
        # learn a token that initializes the auto-regressive process
        self.init_token = Parameter(torch.zeros(1, 1, self.latent_dim))
        # use encoded attentions to decode sequence autoregressively
        self.sequence_decoder = TransformerDecoder(
            input_size=self.latent_dim,
            mlp_ratio=self.attn_mlp_ratio,
            depth=self.attn_dec_depth,
            num_heads=self.attn_num_dec_heads,
        )
        # cache for each batch
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, q_z: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        q_z = self.pre_process(q_z)
        # extract mean and variance
        mu_x, log_sigma_x = q_z.chunk(2, dim=-1)
        # sample from VAE latent distribution
        z = Normal(mu_x, (0.5 * log_sigma_x).exp()).rsample()
        # encode contextualised embeddings
        c, context, encoder_attn = self.encode(z)
        # auto-regressively decode the time-series
        z_hat, decoder_attn = self.decode(z, c)
        # uncollapse samples and batch
        z, z_hat, q_z = self.post_process(z, z_hat, q_z)
        return dict(
            z=z,
            z_hat=z_hat,
            q_z=q_z,
            c=c,
            context=context.detach(),
            encoder_attn=encoder_attn,
            decoder_attn=decoder_attn,
        )

    def pre_process(self, q_z: Tensor) -> Tensor:
        # tranpose so sequence on outer dimension for attention
        q_z = q_z.squeeze(1).transpose(0, 1).float()
        # TODO: should interpolation just happen during training? Surely at inference we should also interpolate,
        # but drop interpolations in the output? The point of interpolations is to bootstrap smoothness
        if self.training:
        #     mu_x, log_sigma_x = q_z.chunk(2, dim=-1)
        #     sigma_x = log_sigma_x.exp()
        #     # interpolate k points between each timestep pair
        #     ks = torch.linspace(0, 1 - 1 / self.k, self.k + 1).to(self.device)
        #     ts = torch.arange(mu_x.size(0) - 1).to(self.device)
        #     # between each timestep expand q_z by k spherical interpolations of the mean
        #     mu_x = torch.stack([*[
        #         U.slerp(mu_x[t], mu_x[t + 1], k)
        #         for k in ks for t in ts
        #     ], mu_x[mu_x.size(0) - 1]], dim=0)
        #     # and k linear interpolations of the variance (variance of sum is sum of variances)
        #     log_sigma_x = torch.stack([*[
        #         torch.lerp(sigma_x[t], sigma_x[t + 1], k)
        #         for k in ks for t in ts
        #     ], sigma_x[sigma_x.size(0) - 1]], dim=0).log()
        #     # restack
        #     q_z = torch.cat([mu_x, log_sigma_x], dim=2)
            # expand sampling protocol to marginalise over the distribution
            q_z = einops.repeat(q_z, 'seq bs ld -> seq (bs n) ld', n=self.latent_samples)
        return q_z

    def encode(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # replicate summary token along batch and prepend to sequence
        z_e = torch.cat([self.summary_token.expand(-1, z.size(1), -1), z], dim=0)
        # encode attended representations and extract sequence summary representation
        c, encoder_attn = self.sequence_encoder(z_e)
        context = c[-1, 0, :, :].unsqueeze(0)
        return c, context, encoder_attn

    def decode(self, z: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        # replicate init token along batch and prepend to sequence, shifting the rest of the sequence right by 1
        # drop the final timestep because we cannot ground-truth the prediction
        z_d = torch.cat([self.init_token.expand(-1, z.size(1), -1), z[:-1]], dim=0)
        # given encoded timesteps c[0:t] and input at timestep z[t], predict z[t+1] using a causal mask
        attn_mask = self.causal_mask(z_d.size(0)).to(z_d.device)
        # drop the context for the final timestep because we cannot ground truth the prediction
        z_hat, decoder_attn = self.sequence_decoder(z_d, c[:, :-1, :, :], attn_mask=attn_mask)
        return z_hat, decoder_attn

    def post_process(self, z: Tensor, z_hat: Tensor, q_z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        op = 'seq (bs n) ld -> seq bs n ld'
        if self.training:
            z, z_hat, q_z = (einops.rearrange(x, op, n=self.latent_samples) for x in [z, z_hat, q_z])
        return z, z_hat, q_z


    # # TODO: given the daytime, what does this ecosystem / audio recorder sound like at night?
    # # select a daytime audio recording (A) from a habitat then sample B0 from the region
    # # of the latent space where that habitat's night-time recordings occur (or use a ground truth B0)
    def generate(self, sequence_len: int, z_0: Optional[Tensor] = None) -> Tensor:
        batch_size = 1 if z_0 is None else z_0.size(1)
        z_enc = torch.zeros(sequence_len + 1, 1, self.latent_dim)
        z_dec = torch.zeros(sequence_len + 1, 1, self.latent_dim)
        # set initial conditions
        z_enc[0, :, :] = self.summary_token.expand(-1, batch_size, -1)
        z_dec[0, :, :] = self.init_token.expand(-1, batch_size, -1)
        if z_0 is not None:
            z_enc[1:z_0.size(1) + 1, :, :] = z_0
            z_dec[1:z_0.size(1) + 1, :, :] = z_0
        # auto-regressively generate N timesteps
        for t in range(sequence_len - 1):
            # generate a causal mask up to timestep t
            attn_mask = self.causal_mask(sequence_len + 1)
            # encode context iteratively
            c, enc_attn_w = self.sequence_encoder(z_enc, attn_mask=attn_mask)
            # given internal state, decode next timestep
            output, dec_attn_w = self.sequence_decoder(z_dec, c, attn_mask=attn_mask)
            z_enc[t + 1, :, :] = output[t, :, :]
            z_dec[t + 1, :, :] = output[t, :, :]
        return z_dec[1:, :, :]

    # def congen(self, A: Tensor, B0: Tensor) -> Tensor:
    #     """given sequence A as context, generate sequence B"""
    #     z = torch.zeros(A.size(0), B.size(1), self.latent_dim)
    #     # set initial condition
    #     z[0, :, :] = B0
    #     # encode the whole of sequence A
    #     c, _ = self.sequence_encoder(A, attn_mask=attn_mask)
    #     # auto-regressively generate N-1 timesteps
    #     for t in range(1, sequence_len - 1):
    #         attn_mask = self.causal_mask(sequence_len)
    #         # given internal state, decode next timestep
    #         z[t, :, :], _ = self.sequence_decoder(z, c, attn_mask=attn_mask)
    #     return z

    def causal_mask(self, sequence_len: int) -> Tensor:
        # compute a matrix of negative infinity for softmax log probabilities
        neg_inf = torch.from_numpy(np.ones((sequence_len, sequence_len)) * -np.inf)
        # set lower triangle to zero for softmax log probabilities
        return torch.triu(neg_inf, diagonal=1).float()

    def loss(self, z_hat: Tensor, q_z: Tensor, **kwargs: Any) -> Dict[str, Tensor]:
        # minimise the mahalanobis distance between the posterior and the transformer output
        mahalanobis = self.mahalanobis_distance(z_hat, q_z).mean()
        # enforce smoothness on the output in time by minimising the size of derivatives over time
        sobolev = self.alpha * self.sobolev(z_hat[:-1]).sum(dim=1).mean()
        # sum loss components
        loss = mahalanobis + sobolev
        return dict(
            loss=loss,
            sobolev=sobolev.detach(),
            mahalanobis=mahalanobis.detach(),
        )

    def mahalanobis_distance(self, z_hat: Tensor, q_z: Tensor) -> Tensor:
        mu_x, log_sigma_x = q_z.chunk(2, dim=-1)
        log_sigma_x = self.log_epsilon_x + F.softplus(log_sigma_x - self.log_epsilon_x)
        mahalanobis = ((z_hat - mu_x).pow(2) / log_sigma_x.exp())
        mahalanobis = mahalanobis.transpose(0, 1)
        # compute average distance over batch
        return mahalanobis.flatten(start_dim=1).sum(dim=1).sqrt()

    def sobolev(self, z_hat: Tensor) -> Tensor:
        dzhdt = z_hat[:, :, :self.num_smooth_in_time].diff(dim=0, n=self.Q).abs()
        return dzhdt.transpose(0, 1).flatten(start_dim=1)

    def metrics(self, z: Tensor, z_hat: Tensor, q_z: Tensor, **kwargs) -> Dict[str, Tensor]:
        # baseline mahalanlobis between ground truth sample z_t and distribution q_z^{t+1} for comparison
        z_dist = self.mahalanobis_distance(z, q_z).mean()
        # MAE between predicted and actual time derivative dz/dt
        dzdt_mae = (z_hat.diff(0) - z.diff(0)).abs().transpose(0, 1).flatten(start_dim=1).mean()
        # Mean size of ground truth dz/dt and mean of predicted dz/dt
        dzdt = z.diff(dim=0, n=self.Q).abs().transpose(0, 1).flatten(start_dim=1).mean()
        dzhdt = z_hat.diff(dim=0, n=self.Q).abs().transpose(0, 1).flatten(start_dim=1).mean()
        return dict(
            z_dist=z_dist.detach(),
            dzdt_mae=dzdt_mae.detach(),
            dzhdt=dzhdt.detach(),
            dzdt=dzdt.detach(),
        )

    def latent_histogram(
        self,
        zs: Tensor,
        z_score: float,
        num_bins: int,
        epsilon: float = 1e-8
    ) -> Dict[str, Union[Tensor, float]]:
        # compute the percentile using specified z-score
        z_min = zs.mean() - z_score * zs.std()
        z_max = zs.mean() + z_score * zs.std()
        # sample and compute a histogram over samples
        batch_size, sequence_len, latent_dim = zs.shape
        hist = torch.zeros(batch_size, num_bins, latent_dim)
        bins = torch.linspace(z_min, z_max, num_bins + 1)
        for j in range(num_bins):
            hist[:, j, ...] = ((bins[j] < zs) & (zs < bins[j + 1])).sum(axis=1) / sequence_len
        hist = torch.softmax((hist + epsilon).log(), dim=1)
        return dict(histogram=hist.unsqueeze(1), z_min=z_min, z_max=z_max)

    def step(
        self,
        batch,
        batch_idx: int,
        stage,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        step_outputs = self(*batch, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        metric_outputs = self.metrics(**step_outputs)
        log_outputs = U.prefix_keys(loss_outputs | metric_outputs, stage)
        self.log_dict(log_outputs, batch_size=batch.x.size(0), prog_bar=True)
        step_outputs = U.detach_values(step_outputs)
        z_hist_outputs = self.latent_histogram(step_outputs["zs"])
        z_hat_hist_outputs = self.latent_histogram(step_outputs["z_hats"])
        step_outputs = step_outputs | dict(
            histogram=torch.stack([z_hist_outputs["histogram"], z_hat_hist_outputs["histogram"]], dim=1),
            z_max=torch.stack([z_max, z_hat_max], dim=1),
            z_min=torch.stack([z_min, z_hat_min], dim=1)
        )
        return loss_outputs, step_outputs

    def training_step(
        self,
        batch,
        batch_idx: int
    ) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "train", **(self.train_step_params or {}))
        self.train_step_outputs.append(step_outputs)
        return loss_outputs

    def on_train_batch_end(
        self,
        outputs: Dict[str, Tensor],
        batch,
        batch_idx: int,
    ) -> None:
        self.training_step_outputs.clear()

    @torch.no_grad()
    def validation_step(
        self,
        batch,
        batch_idx: int
    ) -> Dict[str, Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx, "val", **(self.train_step_params or {}))
        self.validation_step_outputs.append(step_outputs)
        return loss_outputs

    def on_validation_batch_end(
        self,
        outputs: Dict[str, Tensor],
        batch,
        batch_idx: int,
    ) -> None:
        self.validation_step_outputs.clear()

    def on_predict_start(self):
        if self.vae_artefact_id is not None:
            self.vae = VAE.load_from_wandb(self.vae_artefact_id, save_dir=Path.home() / "models")

    @torch.no_grad()
    def predict_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        step_outputs = self(*batch, **(self.eval_step_params or {}))
        return batch.s, torch.tensor(dataloader_idx).expand(batch.x.size(0)), step_outputs
        # TODO: for inference, SLERP across the VAE latent space
        # and autoregressively generate a batch of spectrograms using the `generate` function

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


