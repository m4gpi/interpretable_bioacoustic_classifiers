import lightning as L
import logging
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import torch
import wandb

from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision.transforms import functional as T
from typing import Any, Dict, List, Tuple

from src.core.transforms.translation import translation_1d as translation
from src.core.utils import metrics
from src.core.utils import Batch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["ShiftInvariancePlot"]

def central_finite_difference(x, padding_mode="circular"):
    x = F.pad(x, (1, 1), padding_mode)
    kernel = torch.tensor([[[-1.0, 0.0, 1.0]]]).to(x.device)
    dxdt = F.conv1d(x, kernel)
    return dxdt

class ShiftInvariancePlot(L.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx <= 4:
            x = pl_module.frame(batch.x[0:1], window_length=pl_module.frame_window_length, hop_length=pl_module.frame_window_length).flatten(end_dim=1)
            thetas = torch.linspace(-torch.pi, torch.pi, 50, device=x.device)
            x_trans = torch.stack([
                translation(x.transpose(-1, -2), (theta  / torch.pi).expand(x.size(0))).transpose(-1, -2)
                for theta in thetas
            ], dim=1)
            q_z, *_ = pl_module.encode(x_trans.flatten(end_dim=1))
            q_z = q_z.unflatten(0, (x_trans.size(0), x_trans.size(1)))
            mu_z, _ = q_z.chunk(2, dim=-1)

            # for each frame
            for i in range(mu_z.size(0)):
                dzdT = central_finite_difference(mu_z[i].transpose(-1, -3), "circular")
                fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
                im = ax.imshow(dzdT.squeeze().cpu(), vmin=-1, vmax=1, **self.imshow_params)
                cbar = fig.colorbar(im, ax=ax, orientation="vertical")
                cbar.set_label(r"$\frac{dz}{dT}$", rotation=0)
                ax.set_ylabel("Latent Dimension")
                ax.set_xlabel(r"Shift")
                if pl_module.logger is not None and getattr(pl_module.logger, "experiment") is not None:
                    pl_module.logger.experiment.log({f"val/dzdT": wandb.Image(fig)})
                plt.close(fig)

    @property
    def imshow_params(self):
        return dict(
            origin="lower",
            aspect="auto",
            cmap=sns.color_palette("vlag", as_cmap=True)
        )
