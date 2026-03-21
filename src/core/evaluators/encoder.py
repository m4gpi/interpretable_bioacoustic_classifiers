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

__all__ = ["Encoder"]

class Encoder(Evaluator):
    def __init__(self, results_path: str | pathlib.Path) -> None:
        super().__init__()
        self.results_path = pathlib.Path(results_path)
        self.results_path.parent.mkdir(exist_ok=True, parents=True)

    def __call__(self, trainer, model, data_module, config) -> None:
        ckpt_path = config.get("ckpt_path")
        assert ckpt_path is not None, f"No checkpoint found at {ckpt_path}"
        log.info(f"Encoding <{config.data.get('_target_')}> with <{config.model.get('_target_')}>")
        predictions = trainer.predict(model, data_module.predict_dataloader(), ckpt_path=config.get("ckpt_path"), return_predictions=True)
        df = pd.concat(list(itertools.chain(*predictions)), axis=0)
        df.to_parquet(self.results_path)
