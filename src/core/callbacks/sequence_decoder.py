import lightning as L
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from matplotlib import pyplot as plt
from typing import Any, Dict, List, Tuple

from src.core.models.ear import EAR
from src.core.utils import metrics
from src.core.utils.sketch import plot_mel_spectrogram

plt.switch_backend('agg')

__all__ = ["SequenceDecoder"]

class SequenceDecoder(L.Callback):
    def __init__(
        self,
        ckpt_path: str,
        num_per_batch: int = 6,
    ) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self.num_per_batch = num_per_batch
        self.model = EAR.load_from_checkpoint(ckpt_path)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        zs = []
        for i in range(self.num_per_batch):
            encoding_idx = torch.multinomial(probs[i], num_samples=1)
            z_q, encodings = self.model.quantise.quantise(encoding_idx)
            z_q = z_q.view(-1, self.model.latent_dim // self.model.quantise.num_features, self.model.quantise.num_features)
            zs.append(z_q.view(-1, self.model.latent_dim))
        z_q = torch.stack(zs, dim=0)
        xs = self.model.decode(z_q).cpu().squeeze(1)
        fig, axes = plt.subplots(nrows=self.num_per_batch, ncols=1, figsize=(15, self.num_per_batch * 3))
        for i, (x, ax) in enumerate(zip(xs, axes)):
            mesh = plot_mel_spectrogram(
                x.T,
                **self.model.spectrogram_params,
                vmin=x.min(),
                vmax=x.max(),
                ax=ax
            )
            plt.colorbar(mesh, ax=ax, orientation="vertical")
        pl_module.logger.experiment.log({ f"val/spectrogram": wandb.Image(fig) })

