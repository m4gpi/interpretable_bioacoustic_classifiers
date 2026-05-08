import lightning as L
import logging
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from typing import Any, Dict, List, Tuple

from src.core.utils import metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SIVAEMetrics"]

def circular_mean_offset(y, y_hat, dim: int = -1):
    diff = y - y_hat
    sin_sum = torch.sin(diff).mean(dim=dim, keepdim=True)
    cos_sum = torch.cos(diff).mean(dim=dim, keepdim=True)
    offset = torch.atan2(sin_sum, cos_sum)
    y_hat_aligned = y_hat + offset
    return y_hat_aligned, offset

class SIVAEMetrics(L.Callback):
    def __init__(self, save_path: str, num_samples: int = 10) -> None:
        super().__init__()
        self.data = []
        self.save_path = pathlib.Path(save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.num_samples = num_samples

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        outputs = pl_module.predict_delta(batch.x, num_samples=self.num_samples)
        x_trans, delta, delta_hat = outputs["x_trans"], outputs["delta"], outputs["delta_hat"]
        bs, seq, n, *_ = x_trans.size()
        sample_idx = batch.s.cpu().unsqueeze(0).expand(seq, n, -1).permute(2, 0, 1).flatten().cpu()
        seq_idx = torch.arange(seq).expand(bs, n, -1).permute(0, 2, 1).flatten().cpu()
        shift_idx = torch.arange(n).expand(bs, seq, -1).flatten().cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs, seq, n).flatten().cpu()
        ref_column_types = dict(file_i=int, timestep=int, dataloader_idx=int, shift=float)
        feat_column_types = dict(
            delta=float,
            delta_hat=float,
        )
        column_types = (ref_column_types | feat_column_types)
        df = pd.DataFrame(
            data=dict(zip(column_types.keys(), [
                sample_idx, seq_idx, dl_idx, shift_idx,
                delta.flatten(end_dim=2).cpu(),
                delta_hat.flatten(end_dim=2).cpu(),
            ])),
            columns=column_types.keys(),
        ).astype(dtype=column_types).set_index(list(ref_column_types.keys()))
        df["model_name"] = pl_module.__class__.__name__
        df["latent_dim"] = pl_module.latent_dim
        df["sigma_x"] = pl_module.sigma_x
        df["learning_rate"] = pl_module.learning_rate
        self.data.append(df)

    def on_test_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if len(self.data):
            df = pd.concat(self.data, axis=0).sort_index().reset_index()
            df["stage"] = df["dataloader_idx"].map({0: "train", 1: "val", 2: "test"})
            log.info(f"Saved metrics to {self.save_path}")
            df.to_parquet(self.save_path)
            summary_stats = df.groupby(["stage", "latent_dim"])[["angular_error"]].agg(["mean", "std"]).reset_index()
            print(summary_stats.to_markdown())
        self.data.clear()
