import lightning as L
import logging
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from torchvision.transforms import functional as T
from typing import Any, Dict, List, Tuple

from src.core.utils import metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["VAEEmbeddings"]

class VAEEmbeddings(L.Callback):
    def __init__(self, save_path: str, frame_hop_length: int | None = None) -> None:
        super().__init__()
        self.save_path = pathlib.Path(save_path)
        self.frame_hop_length = frame_hop_length
        self.embeddings = []
        self.labels = []

    def on_predict_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        log.info(f"Target path for embeddings: {self.save_path.resolve()}")
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.train_save_path = (self.save_path / "train")
        self.val_save_path = (self.save_path / "val")
        self.test_save_path = (self.save_path / "test")
        self.train_save_path.mkdir(exist_ok=True, parents=True)
        self.val_save_path.mkdir(exist_ok=True, parents=True)
        self.test_save_path.mkdir(exist_ok=True, parents=True)

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, Any],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        embeddings_df = pl_module.embed(batch, dataloader_idx=dataloader_idx, frame_hop_length=self.frame_hop_length)
        labels_df = pd.DataFrame(data=batch.y.cpu().numpy(), columns=batch.metadata, index=batch.s.cpu().numpy())
        labels_df.index.name = "file_i"
        self.embeddings.append(embeddings_df)
        self.labels.append(labels_df)

    def on_predict_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        embeddings_df = pd.concat(self.embeddings, axis=0).sort_index()
        train_embeddings_df = embeddings_df[embeddings_df.index.get_level_values("dataloader_idx") == 0]
        val_embeddings_df = embeddings_df[embeddings_df.index.get_level_values("dataloader_idx") == 1]
        test_embeddings_df = embeddings_df[embeddings_df.index.get_level_values("dataloader_idx") == 2]
        train_embeddings_df.to_parquet(self.train_save_path / "features.parquet")
        log.info(f"Train embeddings saved to {(self.train_save_path  / 'features.parquet').resolve()}")
        val_embeddings_df.to_parquet(self.val_save_path / "features.parquet")
        log.info(f"Train embeddings saved to {(self.val_save_path  / 'features.parquet').resolve()}")
        test_embeddings_df.to_parquet(self.test_save_path / "features.parquet")
        log.info(f"Test embeddings saved to {(self.test_save_path / 'features.parquet').resolve()}")

        labels_df = pd.concat(self.labels, axis=0).sort_index()
        train_labels_df = labels_df[labels_df.index.isin(train_embeddings_df.index.get_level_values("file_i"))]
        val_labels_df = labels_df[labels_df.index.isin(val_embeddings_df.index.get_level_values("file_i"))]
        test_labels_df = labels_df[labels_df.index.isin(test_embeddings_df.index.get_level_values("file_i"))]
        train_labels_df.to_parquet(self.train_save_path / "labels.parquet")
        log.info(f"Train labels saved to {(self.train_save_path / 'labels.parquet').resolve()}")
        val_labels_df.to_parquet(self.val_save_path / "labels.parquet")
        log.info(f"Train labels saved to {(self.train_save_path / 'labels.parquet').resolve()}")
        test_labels_df.to_parquet(self.test_save_path / "labels.parquet")
        log.info(f"Test labels saved to {(self.test_save_path / 'labels.parquet').resolve() }")

