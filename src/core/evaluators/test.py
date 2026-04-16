import itertools
import lightning as L
import logging
import pathlib
import pandas as pd
import torch
import tqdm

from typing import Any

from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["Test"]

class Test(Evaluator):
    def __call__(self, trainer, model, data_module, config) -> None:
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path is not None, f"No checkpoint found at {ckpt_path}"
        log.info(f"Encoding <{config.data.get('_target_')}> with <{config.model.get('_target_')}>")
        trainer.test(model, data_module.predict_dataloader(), ckpt_path=config.get("ckpt_path"))
