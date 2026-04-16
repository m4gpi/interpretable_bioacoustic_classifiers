import lightning as L
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from torchvision.transforms import functional as T
from typing import Any, Dict, List, Tuple

from src.core.utils import metrics

__all__ = ["VAEEmbeddings"]

class VAEEmbeddings(L.Callback):
    def __init__(self, save_path: str, frame_hop_length: int | None = None) -> None:
        super().__init__()
        self.data = []
        self.save_path = pathlib.Path(save_path)
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        self.frame_hop_length = frame_hop_length
        self.data = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.frame_hop_length:
            frame_hop_length = pl_module.frame_window_length // 2
        x, *_ = batch
        x = pl_module.log_mel_spectrogram(x)
        x = T.center_crop(x, [(x.size(-2) - (x.size(-2) % pl_module.frame_window_length)), pl_module.num_mel_bins])
        # encode with a half-frame overlap
        q_z = pl_module.encode(x, hop_length=frame_hop_length)
        bs, seq, *_ = q_z.size()
        sample_idx = batch.s.cpu().unsqueeze(0).repeat(seq, 1).t().flatten()
        seq_idx = torch.arange(seq).repeat(bs, 1).view(bs * seq).cpu()
        dl_idx = torch.tensor(dataloader_idx).expand(bs).unsqueeze(0).repeat(seq, 1).t().flatten().cpu()
        # seq start accounts for hop
        frame_hop_samples = pl_module.fft_hop_length * frame_hop_length
        seq_start_samples = seq_idx * frame_hop_samples
        # seq end accounts for receptive field
        frame_duration_samples = pl_module.fft_hop_length * pl_module.frame_window_length
        seq_end_samples = seq_start_samples + frame_duration_samples
        # map to time in seconds
        seq_start_seconds = seq_start_samples / pl_module.sample_rate
        seq_end_seconds = seq_end_samples / pl_module.sample_rate
        ref_column_types = dict(
            file_i=int, dataloader_idx=int, timestep=int,
            t_start_samples=int, t_end_samples=int,
            t_start_seconds=float, t_end_seconds=float,
        )
        feat_column_types = dict(
            **{ f"z_mean_{d}": float  for d in range(q_z.size(-1)//2) },
            **{ f"z_log_var_{d}": float  for d in range(q_z.size(-1)//2) },
        )
        column_types = (ref_column_types | feat_column_types)
        df = pd.DataFrame(
            data=dict(zip(column_types.keys(), [
                sample_idx, dl_idx, seq_idx,
                seq_start_samples, seq_end_samples,
                seq_start_seconds, seq_end_seconds,
                *q_z.flatten(end_dim=1).cpu().t(),
            ])),
            columns=column_types.keys(),
        ).astype(dtype=column_types).set_index(list(ref_column_types.keys()))
        self.data.append(df)
        return df

    def on_test_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if len(self.data):
            df = pd.concat(self.data, axis=0).sort_index()
            df.to_parquet(self.save_path)
        self.data.clear()
