# TODO: fix this
import lightning as L
import numpy as np
import torch
import wandb

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Tuple, Iterator

from src.core.utils import Batch

__all__ = ["FigureLogger"]

class FigureLogger(L.Callback):
    def __init__(
        self,
        log_every_n_train_steps: int = 500,
        log_every_n_val_steps: int = 5,
        num_train_samples_per_batch: int = 1,
        num_val_samples_per_batch: int = 1,
        max_val_samples: int = 12,
        num_frames_per_sample: int = 6,
    ) -> None:
        super().__init__()
        self.log_every_n_train_steps = log_every_n_train_steps
        self.log_every_n_val_steps = log_every_n_val_steps
        self.num_train_samples_per_batch = num_train_samples_per_batch
        self.num_val_samples_per_batch = num_val_samples_per_batch
        self.num_frames_per_sample = num_frames_per_sample
        self.max_val_samples = max_val_samples
        self.val_step_count = 0
        plt.switch_backend('agg')

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Batch,
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        if trainer.global_step % self.log_every_n_train_steps == 0:
            self.on_batch_end(pl_module, outputs, "train", num_samples=self.num_train_samples_per_batch)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Batch,
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        if self.val_step_count < self.max_val_samples:
            self.on_batch_end(pl_module, outputs, "val", num_samples=self.num_val_samples_per_batch)
        self.val_step_count += 1

    def on_validation_epoch_end(self, *args: Any, **kwargs: Any):
        self.val_step_count = 0

    def on_batch_end(self, pl_module: L.LightningModule, step_outputs: Dict[str, torch.Tensor], stage: str, num_samples: int = 6):
        figures = pl_module.model.tracking_figures(**step_outputs, num_samples=num_samples)
        for fig_name, fig in figures:
            if pl_module.logger is not None and getattr(pl_module.logger, "experiment") is not None:
                pl_module.logger.experiment.log({f"{stage}/{fig_name}": wandb.Image(fig)})
            plt.close(fig)
