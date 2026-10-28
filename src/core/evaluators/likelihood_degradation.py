import lightning as L
import logging
import pandas as pd
import pathlib
import torch
import numpy as np

from omegaconf import DictConfig
from typing import Any, Callable
from tqdm import tqdm

from src.core.transforms.frame import frame
from src.core.transforms.translation import translation_1d as translation
from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["LikelihoodDegradation"]

class LikelihoodDegradation(Evaluator):
    def __init__(
        self,
        save_path: str | pathlib.Path,
        num_samples: int = 6,
        latent_chunk_size: int = 16,
    ) -> None:
        self.num_samples = num_samples
        self.latent_chunk_size = latent_chunk_size
        self.save_path = pathlib.Path(save_path)

    @torch.inference_mode()
    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        log.info(f"Calculating likelhood degradation for <{data_module.__class__.__name__}> with <{model.__class__.__name__}> on validation dataset")

        records = []

        model = model.to("cuda").eval()
        dl = data_module.val_dataloader()
        pbar = tqdm(dl, total=len(dl))
        for batch in pbar:
            x = model.pre_process(batch.x.to("cuda", non_blocking=True))
            q_z, *_ = model.encode(x)
            mu_z, logvar_z = q_z.chunk(2, dim=-1)
            std_z = (0.5 * logvar_z).exp()
            z = mu_z + std_z * torch.randn_like(mu_z)
            B, D = mu_z.shape[0], mu_z.shape[-1]

            x_hat_base = model.decode(z)
            base_framed = model.frame(x_hat_base, window_length=model.frame_window_length, hop_length=model.frame_window_length)
            T = base_framed.shape[1]
            for j0 in range(0, D, self.latent_chunk_size):
                j1 = min(j0 + self.latent_chunk_size, D)
                J = j1 - j0

                base = z[:, None, None, ...]
                expand_shape = (B, J, self.num_samples, *z.shape[1:])
                z_intervened = base.expand(expand_shape).clone()
                # draw new samples from a standard normal prior
                prior_samples = torch.randn(B, J, self.num_samples, *z.shape[1:-1], device=z.device, dtype=z.dtype)
                # derive a mask to apply to the samples so we only change specified j's
                mask = torch.eye(D, device=z.device, dtype=z.dtype)[j0:j1].view(1, J, 1, *([1] * (z.ndim - 2)), D)
                # for non-masked, add the new value and subtract the existing value
                z_intervened = z_intervened + (prior_samples.unsqueeze(-1) - base) * mask
                # decode all interventions and samples together
                z_flat = z_intervened.flatten(0, 2)
                x_hat = model.decode(z_flat)
                x_hat = x_hat.reshape(B, J, self.num_samples, *x_hat.shape[1:])
                # frame all outputs
                x_flat = x_hat.flatten(0, 2)
                x_framed = model.frame(x_flat, window_length=model.frame_window_length, hop_length=model.frame_window_length)
                x_framed = x_framed.reshape(B, J, self.num_samples, *x_framed.shape[1:])
                # MSE with base, i.e. how much does the reconstruction error change for each frame and sample
                diff = x_framed - base_framed[:, None, None, ...]
                frame_mse = (diff ** 2).reshape(B, J, self.num_samples, T, -1).mean(dim=-1)
                # mean and standard deviation over samples
                degradation_mean = frame_mse.mean(dim=2)
                degradation_std = frame_mse.std(dim=2, unbiased=False)
                # move CPU only (once per chunk)
                means = degradation_mean.cpu().numpy()
                stds = degradation_std.cpu().numpy()
                files = list(batch.s)
                records.append(pd.DataFrame({
                    "degradation_mean": means.reshape(-1),
                    "degradation_std": stds.reshape(-1),
                    "alpha": model.alpha,
                    "file_i": np.repeat(files, J * T),
                    "latent_dim": np.tile(np.repeat(np.arange(j0, j1), T), B),
                    "timestep": np.tile(np.arange(T), B * J),
                }))
            pbar.set_postfix(batch_size=B, latent_dim=D)
        df2 = pd.concat(records, ignore_index=True)
        df2.to_parquet(save_dir / "results.parquet")
