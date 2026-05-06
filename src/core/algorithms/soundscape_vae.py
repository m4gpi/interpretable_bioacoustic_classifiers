import hydra
import lightning as L
import torch
import logging
import omegaconf

from matplotlib import pyplot as plt
from torchvision.transforms import functional as T
from typing import Any, Dict, Tuple, List

from src.core.utils import Batch, detach_values, prefix_keys
from src.core.algorithms.base import Algorithm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SoundscapeVAE(Algorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        optimiser_cls: str,
        *args: Any,
        optimiser_config: omegaconf.DictConfig | None = None,
        scheduler_cls: str | None = None,
        scheduler_config: omegaconf.DictConfig | None = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        **kargs: Any,
    ) -> None:
        super().__init__(model)
        self.learning_rate = learning_rate
        self.optimiser_cls = optimiser_cls
        self.optimiser_config = optimiser_config
        self.scheduler_cls = scheduler_cls
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        x = self.model.pre_process(batch.x)
        return Batch(x=x, **{k: batch[k] for k in batch.keys() if k != "x"})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimiser_config = omegaconf.DictConfig(dict(_target_=self.hparams.optimiser_cls, **(self.hparams.get("optimiser_config") or {})))
        optimiser = hydra.utils.instantiate(optimiser_config, params=self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.get("scheduler_cls") is not None:
            scheduler_config = omegaconf.DictConfig(dict(_target_=self.hparams.scheduler_cls, **(self.hparams.get("scheduler_config") or {})))
            scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimiser)
            return [optimiser], [dict(
                scheduler=scheduler,
                interval=self.hparams.scheduler_interval,
                frequency=self.hparams.scheduler_frequency
            )]
        return optimiser
