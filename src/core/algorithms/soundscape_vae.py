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
    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        x = self.model.pre_process(batch.x)
        return Batch(x=x, **{k: batch[k] for k in batch.keys() if k != "x"})
