import numpy as np
import pandas as pd
import torch
import logging

from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import functional as T
from typing import Any, Dict, List, Tuple

from src.core.transforms.frame import unframe_fold as unframe, frame_fold as frame
from src.core.utils import Batch, bounded_sigmoid, linear_schedule, linear_decay
from src.core.utils.metrics import negative_log_likelihood, gaussian_kl_divergence, gaussian_kl_divergence_standard_prior, circular_variance

__all__ = ["SIVAE"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SIVAE(torch.nn.Module):
    def __init__(
        self,
        front_end: torch.nn.Module,
        feature_encoder: torch.nn.Module,
        content_encoder: torch.nn.Module,
        feature_decoder: torch.nn.Module,
        content_decoder: torch.nn.Module,
        alignment_encoder: torch.nn.Module | None = None,
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
        x_i_frame_prob: float = 0.75,
        delta_prob_step_start: int = 0,
        delta_prob_step_end: int = 0,
        delta_prob_min: int = 1.0,
        delta_prob_max: int = 1.0,
        delta_sigma_min: float = 1.0,
        delta_sigma_max: float = 1.0,
        delta_sigma_step_slope: float = 0.4,
        delta_sigma_step_start: int = 0,
        delta_sigma_step_end: int = 0,
        align_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
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

        self.front_end = front_end
        self.feature_encoder = feature_encoder
        self.content_encoder = content_encoder
        self.feature_decoder = feature_decoder
        self.content_decoder = content_decoder
        self.alignment_encoder = alignment_encoder

        if self.align_only:
            log.info("Freezing feature and content networks")
            params = list(self.feature_encoder.parameters()) + list(self.feature_decoder.parameters()) + \
                list(self.content_encoder.parameters()) + list(self.content_decoder.parameters())
            for param in params:
                param.requires_grad = False
        # if ground truth is always shown, freeze the alignment encoder
        if self.delta_prob_min == 1 and self.delta_prob_max == 1:
            log.info("Freezing alignment encoder")
            for param in self.alignment_encoder.parameters():
                param.requires_grad = False

    def pre_process(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.front_end(wav)
        x = x.transpose(-1, -2) # transpose time to inner axis
        return T.center_crop(x, [(x.size(-2) - (x.size(-2) % self.frame_window_length)), x.size(-1)])

    def forward(self, x: Tensor, *args: Any, t: int | None = None, **kwargs: Any) -> Dict[str, Tensor]:
        delta_sigma = self.delta_sigma_current(t)
        # framed spectorgrams are target signal
        x_i = x
        x_i_framed = self.frame(x_i, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        seq_len = x_i_framed.size(1)
        # coin flip to determine whether to encode full or translated spectrograms for target 1 (x_i)
        if torch.bernoulli(torch.tensor(1), self.x_i_frame_prob):
            x_i_framed = x_i_framed.flatten(end_dim=1)
            theta_i, dx_i, dy_i = self.sample_circle(x_i_framed.size(0), 1, 1, scaling_factor=delta_sigma, device=x.device)
            delta_i / torch.pi
            x_i_framed = self.translation(x_i_framed.transpose(-1, -2).contiguous(), delta_i, mode=self.translation_mode).transpose(-1, -2).contiguous()
            x_i = x_i_framed
            q_z_i, (theta_hat_i, dx_hat_i, dy_hat_i) = self.encode(x_i, t=t) # (bs * seq, 1, ld)
            delta_hat_i = theta_hat_i / torch.pi
            mu_z_i, log_sigma_sq_z_i = q_z_i.chunk(2, dim=-1)
        else:
            # when full spectrograms are used, delta / theta is necessarily zero
            theta_i = torch.zeros(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device)
            delta_i = torch.zeros(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device)
            dx_i = torch.ones(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device)
            dy_i = torch.zeros(x_i_framed.size(0), x_i_framed.size(1), 1, device=x.device)
            x_i_framed = x_i_framed.flatten(end_dim=1)
            q_z_i, (theta_hat_i, dx_hat_i, dy_hat_i) = self.encode(x_i, t=t) # (bs, seq, ld)
            delta_hat_i = theta_hat_i / torch.pi
            mu_z_i, log_sigma_sq_z_i = q_z_i.chunk(2, dim=-1)
        # always randomly translated frames for target 2 (x_j)
        theta_j, dx_j, dy_j = self.sample_circle(x_i_framed.size(0), 1, 1, scaling_factor=delta_sigma, device=x.device)
        delta_j = theta_j / torch.pi
        x_j = self.translation(x_i_framed.transpose(-1, -2).contiguous(), delta_j, mode=self.translation_mode).transpose(-1, -2).contiguous()
        # encode posterior for translated frames separately
        q_z_j, (theta_hat_j, dx_hat_j, dy_hat_j) = self.encode(x_j, t=t) # (bs * seq, 1, ld)
        delta_hat_j = theta_hat_j / torch.pi
        mu_z_j, log_sigma_sq_z_j = q_z_j.chunk(2, dim=-1)
        # decode to feature maps
        if self.cross_decode_method == "soft":
            # soft cross-decoding averages the distributions
            # mu_k = (mu_i + mu_j) / 2, sigma^2_k = (sigma^2_i + sigma^2_j) / 2^2
            mu_z = torch.stack([mu_z_i.flatten(end_dim=1), mu_z_j.flatten(end_dim=1)], dim=1).mean(dim=1)
            log_sigma_sq_z = (torch.stack([log_sigma_sq_z_i.flatten(end_dim=1).exp(), log_sigma_sq_z_j.flatten(end_dim=1).exp()], dim=1).sum(dim=1) / 4).log()
            z = torch.distributions.Normal(mu_z, (0.5 * log_sigma_sq_z).exp()).rsample()  # (bs, seq, ld)
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1).view(q_z_i.size())
            U_hat_i = self.content_decoder(z) # (bs * seq, ch, fr, fq)
            U_hat_j = U_hat_i
        elif self.cross_decode_method == "hard":
            z_i = torch.distributions.Normal(mu_z_i, (0.5 * log_sigma_sq_z_i).exp()).rsample()  # (bs, seq, ld)
            z_j = torch.distributions.Normal(mu_z_j, (0.5 * log_sigma_sq_z_j).exp()).rsample()  # (bs * seq, 1, ld)
            q_z = torch.cat([q_z_i, q_z_j.view(q_z_i.size())], dim=0)
            U_hat_i = self.content_decoder(z_i.flatten(end_dim=1))
            U_hat_j = self.content_decoder(z_j.flatten(end_dim=1))
        else:
            raise Exception("Cross decode method not specified, terminating")

        if self.training:
            # during training, occasionally aid the decoder by providing the true delta
            delta_prob = self.delta_prob_current(t)
            mask_i = torch.bernoulli(torch.full((delta_i.size(0), delta_i.size(1), 1), delta_prob, device=delta_i.device))
            mask_j = torch.bernoulli(torch.full((delta_j.size(0), delta_j.size(1), 1), delta_prob, device=delta_i.device))
            delta_i_mixed = mask_i * delta_i + (1 - mask_i) * delta_hat_i
            delta_j_mixed = mask_j * delta_j + (1 - mask_j) * delta_hat_j
            # reconstruct a contiguous sequence
            x_hat_i = self.cnn_decode(U_hat_j, delta_i_mixed) # (bs, 1, fr * seq, fq)
            # reconstruct independent translations
            x_hat_j = self.cnn_decode(U_hat_i, delta_j_mixed) # (bs * seq, 1, fr, fq)
        else:
            delta_prob = 0
            # reconstruct a contiguous sequence
            x_hat_i = self.cnn_decode(U_hat_j, delta_hat_i) # (bs, 1, fr * seq, fq)
            # reconstruct independent translations
            x_hat_j = self.cnn_decode(U_hat_i, delta_hat_j) # (bs * seq, 1, fr, fq)
        # frame for frame-wise loss
        x_hat_i_framed = self.frame(x_hat_i, window_length=self.frame_window_length, hop_length=self.frame_window_length).flatten(end_dim=1)
        x_framed = torch.cat([x_i_framed, x_j], dim=0)
        x_hat_framed = torch.cat([x_hat_i_framed, x_hat_j], dim=0)
        return dict(
            x=x,
            x_framed=x_framed, x_i=x_i, x_j=x_j,
            x_hat_framed=x_hat_framed, x_hat_i=x_hat_i, x_hat_j=x_hat_j,
            q_z=q_z, q_z_i=q_z_i, q_z_j=q_z_j,
            delta_i=delta_i, delta_j=delta_j, delta_hat_i=delta_hat_i, delta_hat_j=delta_hat_j,
            theta_i=theta_i, theta_j=theta_j, theta_hat_i=theta_hat_i, theta_hat_j=theta_hat_j,
            dx_i=dx_i, dx_j=dx_j, dx_hat_i=dx_hat_i, dx_hat_j=dx_hat_j,
            dy_i=dy_i, dy_j=dy_j, dy_hat_i=dy_hat_i, dy_hat_j=dy_hat_j,
            delta_sigma=delta_sigma,
            seq_len=seq_len,
            delta_prob=delta_prob,
        )

    def predict(self, x: Tensor, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        q_z, (theta, dx, dy) = self.encode(x)
        # ignore shift prediction network if its not been trained
        delta = torch.zeros_like(theta) if self.delta_prob_min == 1 and self.delta_prob_max == 1 else theta / torch.pi
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        z = torch.distributions.Normal(mu_z, (1/2 * log_sigma_sq_z).exp()).rsample()
        x_hat = self.cnn_decode(self.content_decoder(z.flatten(end_dim=1)), delta)
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        x_hat_framed = self.frame(x_hat, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        return dict(
            x=x,
            x_hat=x_hat,
            x_framed=x_framed,
            x_hat_framed=x_hat_framed,
            q_z=q_z,
            delta=delta,
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
        u = self.frame(us[0], window_length=u_window_length, hop_length=u_hop_length, padding_mode=self.frame_padding_mode)
        # content bottleneck
        q_z = self.content_encoder(x.flatten(end_dim=1)).unflatten(dim=0, sizes=(x.size(0), x.size(1)))
        if self.sigma_z_min is not None:
            mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
            sigma_latent = torch.tensor(self.sigma_z_min, dtype=torch.float32, requires_grad=False, device=mu_z.device)
            log_sigma_sq_z = log_sigma_sq_z.clamp(min=2*sigma_latent.pow(2).log())
            q_z = torch.cat([mu_z, log_sigma_sq_z], dim=-1)
        # alignment bottleneck
        delta = self.alignment_encoder(x, u)
        return q_z, delta

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
        dx_i: Tensor,
        dx_j: Tensor,
        dx_hat_i: Tensor,
        dx_hat_j: Tensor,
        dy_i: Tensor,
        dy_j: Tensor,
        dy_hat_i: Tensor,
        dy_hat_j: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        outputs = dict()
        losses = []
        # maximise likelihood p(x_i|z_j) framewise to ensure invariance to sequence length
        sigma_recon = torch.tensor(self.sigma_x, dtype=torch.float32, requires_grad=False, device=x_framed.device)
        nll = negative_log_likelihood(x_framed, x_hat_framed, sigma_recon.pow(2).log()).flatten(start_dim=-3).sum(dim=-1)
        losses.append(nll.mean())
        outputs |= dict(log_likelihood_x=-nll.detach().mean())
        # supervision signal for delta, minimise relative angular difference
        if self.delta_prob_min != 1 and self.delta_prob_max != 1:
            angular_error_i = 2 * (1 - (dx_hat_i * dx_i + dy_hat_i * dy_i))
            angular_error_j = 2 * (1 - (dx_hat_j * dx_j + dy_hat_j * dy_j))
            alignment_loss = self.gamma * torch.stack([angular_error_i, angular_error_j.view(angular_error_i.size())]).mean()
            losses.append(alignment_loss)
            outputs |= dict(alignment_loss=alignment_loss.detach())
        # standard normal dkl
        dkl = self.beta * gaussian_kl_divergence_standard_prior(q_z).sum(dim=-1)
        losses.append(dkl.mean())
        outputs |= dict(dkl=dkl.detach().mean())
        # sum the loss components
        outputs |= dict(loss=sum(losses))
        return outputs

    def delta_sigma_current(self, t: int) -> Tensor:
        if self.delta_sigma_min is None: return self.delta_sigma_max
        return torch.tensor(bounded_sigmoid(t, **self.delta_sigma_params))

    def delta_prob_current(self, t: int) -> float:
        if self.delta_prob_min is None: return self.delta_prob_max
        return linear_decay(t, **self.delta_prob_params)

    @torch.no_grad()
    def metrics(
        self,
        x_framed: Tensor,
        x_hat_framed: Tensor,
        q_z_i: Tensor,
        q_z_j: Tensor,
        delta_i: torch.Tensor,
        delta_j: Tensor,
        delta_hat_i: Tensor,
        delta_hat_j: Tensor,
        delta_sigma: Tensor,
        theta_i: Tensor,
        theta_j: Tensor,
        theta_hat_i: Tensor,
        theta_hat_j: Tensor,
        delta_prob: float,
        **kwargs: Any
    ) -> Dict[str, Any]:
        # distribution of z mean and varaince
        q_z = torch.cat([q_z_i, q_z_j.view(q_z_i.size())], dim=0)
        mu_z, log_sigma_sq_z = q_z.chunk(2, dim=-1)
        sigma_z = (0.5 * log_sigma_sq_z).exp()
        mu_hist = np.histogram(mu_z.flatten().cpu().numpy(), bins=32, range=[-5.0, 5.0])
        sigma_hist = np.histogram(sigma_z.flatten().cpu().numpy(), bins=32, range=[0.0, 2.0])
        mu_z_i, mu_z_j = q_z_i.chunk(2, dim=-1)[0], q_z_j.view(q_z_i.size()).chunk(2, dim=-1)[0]
        # mean distance between shifted embeddings
        z_dist = (mu_z_j - mu_z_i).abs().mean()
        # distribution of delta predictions
        delta = torch.cat([delta_i, delta_j.view(delta_i.size())])
        delta_hist = np.histogram(delta.flatten().cpu().numpy(), bins=128, range=[-1.0, 1.0])
        delta_hat = torch.cat([delta_hat_i, delta_hat_j.view(delta_hat_i.size())])
        delta_hat_hist = np.histogram(delta_hat.flatten().cpu().numpy(), bins=128, range=[-1.0, 1.0])
        # variance of predicted theta both independently and across sequence
        theta = torch.cat([theta_i, theta_j.view(theta_i.size())])
        theta_hat = torch.cat([theta_hat_i, theta_hat_j.view(theta_hat_i.size())])
        theta_hat_var = circular_variance(theta_hat, dim=None).mean()
        theta_hat_seq_var = circular_variance(theta_hat, dim=1).mean()
        # anglular error
        d_theta = theta_hat - theta
        angular_distance = torch.atan2(torch.sin(d_theta), torch.cos(d_theta))
        angular_error = 1 - torch.cos(d_theta)
        d_theta_hist = np.histogram(d_theta.flatten().cpu().numpy(), bins=128, range=[0.0, 2.0])
        # reconstruction error
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
            delta_hist=delta_hist,
            delta_hat_hist=delta_hat_hist,
            theta_hat_var=theta_hat_var,
            theta_hat_seq_var=theta_hat_seq_var,
            angular_distance_mean=angular_distance.mean(),
            angular_distance_std=angular_distance.std(),
            angular_error_mean=angular_error.mean(),
            angular_error_std=angular_error.std(),
            dkl_norm=dkl_norm,
            delta_sigma=delta_sigma,
            delta_prob=delta_prob,
        )

    @torch.no_grad()
    def predict_delta(self, x: torch.Tensor, num_samples: int = 10):
        x_framed = self.frame(x, window_length=self.frame_window_length, hop_length=self.frame_window_length)
        bound = torch.ones(num_samples, x_framed.size(0), x_framed.size(1), device=x.device).permute(1, 2, 0)
        delta = torch.distributions.Uniform(low=-bound, high=bound).sample()
        x_framed = x_framed.expand(num_samples, -1, -1, -1, -1, -1).permute(1, 2, 0, 3, 4, 5)
        bs, seq, n, *_ = x_framed.size()
        x_trans = self.translation(
            x_framed.flatten(end_dim=2).transpose(-1, -2).contiguous(),
            delta.flatten(end_dim=2),
            mode=self.translation_mode
        ).transpose(-1, -2).contiguous().unflatten(0, (bs, seq, n))
        _, (theta, dx, dy) = self.encode(x_trans.flatten(end_dim=2))
        delta_hat = theta / torch.pi
        delta_hat = delta_hat.unflatten(0, (bs, seq, n)).view(delta.size())
        return dict(x_trans=x_trans, delta=delta, delta_hat=delta_hat)

    @torch.no_grad()
    def embed(
        self,
        batch: Batch,
        dataloader_idx: int = 0,
        frame_hop_length: float | None = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        x, *_ = batch
        frame_hop_length = frame_hop_length or self.frame_window_length // 2
        q_z, delta = self.encode(x, hop_length=frame_hop_length)
        bs, seq, *_ = q_z.size()
        sample_idx = batch.s.cpu().unsqueeze(0).repeat(seq, 1).t().flatten()
        seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq).cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().cpu()
        ref_column_types = dict(file_i=int, dataloader_idx=int, timestep=int)
        feat_column_types = dict(
            **{ f"z_mean_{d}": float  for d in range(q_z.size(-1)//2) },
            **{ f"z_log_var_{d}": float  for d in range(q_z.size(-1)//2) },
            **{ "delta": float },
        )
        column_types = (ref_column_types | feat_column_types)
        return pd.DataFrame(
            data=dict(zip(column_types.keys(), [
                sample_idx, dl_idx, seq_idx,
                *q_z.flatten(end_dim=1).cpu().t(),
                *delta.flatten(end_dim=1).cpu().squeeze(-1),
            ])),
            columns=column_types.keys(),
        ).astype(dtype=column_types).set_index(list(ref_column_types.keys()))

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

    @property
    def delta_sigma_params(self):
        return dict(
            x_min=self.delta_sigma_step_start,
            x_max=self.delta_sigma_step_end,
            y_min=self.delta_sigma_min,
            y_max=self.delta_sigma_max,
            k=self.delta_sigma_step_slope,
        )

    @property
    def delta_prob_params(self):
        return dict(
            minimum=self.delta_prob_min,
            maximum=self.delta_prob_max,
            t_start=self.delta_prob_step_start,
            t_end=self.delta_prob_step_end,
        )
