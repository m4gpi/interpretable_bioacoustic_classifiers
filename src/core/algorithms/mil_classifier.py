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

class MILClassifier(Algorithm):
    def __init__(
        self,
        model: torch.nn.Module,
        clf_learning_rate: float,
        attn_learning_rate: float,
        attn_weight_decay: float,
        *args: Any,
        train_sample_size: int | None = 1,
        eval_sample_size: int | None = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        self.clf_learning_rate = clf_learning_rate
        self.attn_learning_rate = attn_learning_rate
        self.attn_weight_decay = attn_weight_decay
        self.train_sample_size = train_sample_size
        self.eval_sample_size = eval_sample_size

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        num_samples = self.train_sample_size if dataloader_idx == 0 else self.eval_sample_size
        x = self.model.pre_process(batch.x, num_samples)
        return Batch(x=x, **{k: batch[k] for k in batch.keys() if k != "x"})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        groups = zip([self.clf_learning_rate, self.attn_learning_rate], [0.0, self.attn_weight_decay], self.model.param_groups)
        return torch.optim.Adam([
            dict(lr=learning_rate, params=params, weight_decay=weight_decay)
            for learning_rate, weight_decay, params in groups
        ])
