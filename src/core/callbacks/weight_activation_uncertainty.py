import itertools
import logging
import lightning as L
import numpy as np
import pandas as pd
import pathlib
import torch
import wandb

from typing import Any, Dict, List, Tuple

from src.core.utils import metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["WeightActivationUncertainty"]

class WeightActivationUncertainty(L.Callback):
    def __init__(
        self,
        save_dir: str,
        model_name: str,
        scope: str,
        seed: str,
        run_id: str,
    ) -> None:
        super().__init__()
        self.save_dir = pathlib.Path(save_dir).expanduser()
        self.model_name = model_name
        self.scope = scope
        self.seed = seed
        self.run_id = run_id

        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.train_save_path = (self.save_dir / "train.parquet")
        self.val_save_path = (self.save_dir / "val.parquet")
        self.test_save_path = (self.save_dir / "test.parquet")
        self.train_save_path.mkdir(exist_ok=True, parents=True)
        self.val_save_path.mkdir(exist_ok=True, parents=True)
        self.test_save_path.mkdir(exist_ok=True, parents=True)
        self.predictions = []

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        y_t_probs, mu_a, sigma_sq_a, attn_w = outputs["y_t_probs"], outputs["mu_a"], outputs["sigma_sq_a"], outputs["attn_w"]
        # cel = metrics.class_balanced_binary_cross_entropy(y, y_t_probs.mean(dim=1), samples_per_class=pl_module.target_counts, **self.cel_params).mean(dim=0)
        file_idx = batch.s.expand(y_t_probs.size(1), y_t_probs.size(2), y_t_probs.size(3), -1).permute(3, 0, 1, 2).flatten()
        _, i1, i2, i3 = torch.meshgrid(*[torch.arange(dim, device=y_t_probs.device) for dim in y_t_probs.size()], indexing="ij")
        table = torch.stack([file_idx, i1.flatten(), i2.flatten(), i3.flatten(), y_t_probs.flatten(), mu_a.flatten(), sigma_sq_a.flatten(), attn_w.flatten()], dim=1).cpu().numpy()
        df = pd.DataFrame(table, columns=["file_i", "sample_i", "t", "species_i", "y_t_prob", "mu_a", "sigma_sq_a", "attn_w"])
        df = df.astype(dtype={"file_i": int, "sample_i": int, "t": int, "species_i": int, "y_t_prob": float, "mu_a": float, "sigma_sq_a": float, "attn_w": float})
        df["dataloader_idx"] = dataloader_idx
        df = df.set_index(["dataloader_idx", "file_i", "sample_i", "t", "species_i"])
        self.predictions.append(df)

    def on_predict_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if len(self.predictions):
            results_df = pd.concat(self.predictions)
            train_results_df = results_df[results_df.index.get_level_values("dataloader_idx") == 0]
            val_results_df = results_df[results_df.index.get_level_values("dataloader_idx") == 1]
            test_results_df = results_df[results_df.index.get_level_values("dataloader_idx") == 2]
            train_results_df.to_parquet(self.train_save_path / f"{self.model_name}-{self.scope}.parquet")
            log.info(f"Train results saved to {(self.train_save_path  / f'{self.model_name}-{self.scope}.parquet').resolve()}")
            val_results_df.to_parquet(self.val_save_path / f"{self.model_name}-{self.scope}.parquet")
            log.info(f"Train results saved to {(self.val_save_path  / f'{self.model_name}-{self.scope}.parquet').resolve()}")
            test_results_df.to_parquet(self.test_save_path / f"{self.model_name}-{self.scope}.parquet")
            log.info(f"Test results saved to {(self.test_save_path / f'{self.model_name}-{self.scope}.parquet').resolve()}")
        self.predictions.clear()

