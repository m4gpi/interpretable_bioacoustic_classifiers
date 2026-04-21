import lightning as L
import logging
import hydra
import numpy as np
import pandas as pd
import torch
import wandb

from dataclasses import dataclass
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.optim import Optimizer
from typing import Any, Dict, Tuple, List

from src.core.utils import Batch, detach_values, prefix_keys, try_or

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["VAE"]

def process_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    for k, v in metrics.items():
        if k.endswith("hist"):
            v = wandb.Histogram(np_histogram=v)
        results[k] = v
    return results

class VAE(L.LightningModule):
    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any]):
        plt.switch_backend('agg')
        log.info(f"Training <{config.algorithm.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(self, datamodule=data_module, ckpt_path=config.get("ckpt_path"))
        log.info(f"Testing <{config.algorithm.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, dataloaders=data_module.predict_dataloader())

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        optimiser_cls: str,
        optimiser_config: DictConfig | None = None,
        scheduler_cls: str | None = None,
        scheduler_config: DictConfig | None = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.vae = model
        self.learning_rate = learning_rate
        self.optimiser_cls = optimiser_cls
        self.optimiser_config = optimiser_config
        self.scheduler_cls = scheduler_cls
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.strict_loading = False
        self._reset_cache()

    def forward(self, batch: Batch, batch_idx: int, stage: str, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs, batch_size = self.vae(**batch, global_step=self.trainer.global_step, **kwargs)
        loss_outputs = self.vae.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        metrics = process_metrics(self.vae.metrics(**step_outputs))
        self.log_dict(prefix_keys(loss_outputs, stage), batch_size=batch_size, prog_bar=True, logger=False)
        if self.logger is not None and getattr(self.logger, "experiment") is not None:
            self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, stage)))
        return {**loss_outputs, **step_outputs}

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch, batch_idx, "train")
        self.training_step_outputs.append(outputs)
        return outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch, batch_idx, "val")
        self.validation_step_outputs.append(outputs)
        return outputs

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predict(**batch, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        pass

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        self.training_step_outputs.clear()

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            self.on_batch_end(step_outputs, "val")
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def on_test_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self.test_step_outputs.clear()

    def on_batch_end(self, step_outputs, stage: str):
        figures = self.vae.tracking_figures(**step_outputs)
        for fig_name, fig in figures:
            self.logger.experiment.log({f"{stage}/{fig_name}": wandb.Image(fig)})
            plt.close(fig)

    def configure_optimizers(self) -> Optimizer:
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

