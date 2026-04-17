import lightning as L
import logging

from omegaconf import DictConfig
from typing import Any, Callable

from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["LightningPredict", "LightningTest"]

class LightningTest(Evaluator):
    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path is not None, f"No checkpoint found at {ckpt_path}"
        log.info(f"Encoding <{config.data.get('_target_')}> with <{config.model.get('_target_')}>")
        trainer.test(model, data_module.predict_dataloader(), ckpt_path=config.get("ckpt_path"))

class LightningPredict(Evaluator):
    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path is not None, f"No checkpoint found at {ckpt_path}"
        log.info(f"Encoding <{config.data.get('_target_')}> with <{config.model.get('_target_')}>")
        trainer.predict(model, data_module.predict_dataloader(), ckpt_path=config.get("ckpt_path"))
