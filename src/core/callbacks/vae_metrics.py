import lightning as L
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from typing import Any, Dict, List, Tuple

from src.core.utils import metrics

__all__ = ["VAEMetrics"]

class VAEMetrics(L.Callback):
    def __init__(self, save_path: str) -> None:
        super().__init__()
        self.data = []
        self.save_path = pathlib.Path(save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y, s = batch
        x_framed, x_hat_framed, q_z = outputs["x_framed"], outputs["x_hat_framed"], outputs["q_z"]
        bs, seq, *_ = x_framed.size()
        sample_idx = s.cpu().unsqueeze(0).repeat(seq, 1).t().flatten()
        seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq).cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().cpu()
        # frame-wise mean absolute error
        mae = (x_hat_framed - x_framed).abs().flatten(start_dim=-3).sum(dim=-1)
        # frame-wise kl divergence
        mu, log_sigma_sq = q_z.chunk(2, dim=-1)
        dkl = (-1/2 * (1 + log_sigma_sq - mu.pow(2) - log_sigma_sq.exp())).sum(dim=-1)
        # frame-wise full elbo
        sigma_x = torch.tensor(pl_module.sigma_x)
        nll = (1/2 * (2 * sigma_x.log() + ((x_framed - x_hat_framed) / sigma_x).pow(2))).flatten(start_dim=-3).sum(dim=-1)
        elbo = nll + dkl
        frame_hop_samples = pl_module.fft_hop_length * pl_module.frame_window_length
        seq_start_samples = seq_idx * frame_hop_samples
        # seq end accounts for receptive field
        frame_duration_samples = pl_module.fft_hop_length * pl_module.frame_window_length
        seq_end_samples = seq_start_samples + frame_duration_samples
        # map to time in seconds
        seq_start_seconds = seq_start_samples / pl_module.sample_rate
        seq_end_seconds = seq_end_samples / pl_module.sample_rate
        ref_column_types = dict(
            file_i=int, timestep=int, dataloader_idx=int,
            t_start_samples=int, t_end_samples=int,
            t_start_seconds=float, t_end_seconds=float,
        )
        feat_column_types = dict(mae=float, dkl=float, elbo=float)
        column_types = (ref_column_types | feat_column_types)
        df = pd.DataFrame(
            data=dict(zip(column_types.keys(), [
                sample_idx, seq_idx, dl_idx,
                seq_start_samples, seq_end_samples,
                seq_start_seconds, seq_end_seconds,
                mae.flatten(end_dim=1).cpu(),
                dkl.flatten(end_dim=1).cpu(),
                elbo.flatten(end_dim=1).cpu(),
            ])),
            columns=column_types.keys(),
        ).astype(dtype=column_types).set_index(list(ref_column_types.keys()))
        self.data.append(df)

    def on_test_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if len(self.data):
            df = pd.concat(self.data, axis=0).sort_index()
            df.to_parquet(self.save_path)
        self.data.clear()
