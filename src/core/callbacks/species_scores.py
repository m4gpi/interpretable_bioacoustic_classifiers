import lightning as L
import numpy as np
import pandas as pd
import pathlib
import torch
import sklearn
import wandb
import warnings

from typing import Any, Dict, List, Tuple

from src.core.utils import metrics

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

__all__ = ["SpeciesScores"]

class SpeciesScores(L.Callback):
    def __init__(self, run_id: str, save_dir: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        (self.save_dir / "val_scores.parquet").mkdir(exist_ok=True, parents=True)
        self.val_table = None
        self.train_predictions = []
        self.val_predictions = []
        self.test_predictions = []
        self.predict_predictions = []

    def score(self, results: pd.DataFrame) -> pd.DataFrame:
        scores = []
        for species_name in results.species_name.unique():
            y = results.loc[results.species_name == species_name, "label"].values
            y_prob = results.loc[results.species_name == species_name, "prob"].values
            if np.isnan(y_prob).any():
                prop_nans = np.isnan(y_prob).sum() / len(y_prob)
                log.warning(f"NaNs found in predicted probabilities for {species_name} with a proportional count of {prop_nans}")
                y_prob = np.nan_to_num(y_prob, nan=0.0)
            assert not np.isnan(y).any(), f"NaNs found in true labels for {species_name}"
            scores.append(dict(
                species_name=species_name,
                AP=metrics.average_precision(y, y_prob),
                auROC=sklearn.metrics.roc_auc_score(y, y_prob),
            ))
        return pd.DataFrame(data=scores).set_index("species_name")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        df = self._on_batch_end(outputs)
        self.train_predictions.append(df)

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if len(self.train_predictions):
            scores = self._on_epoch_end(self.train_predictions)
            pl_module.log_dict({f"train/{metric}": value for metric, value in scores.mean(axis=0).to_dict().items()}, prog_bar=True, on_epoch=True)
        self.train_predictions = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        df = self._on_batch_end(outputs)
        self.val_predictions.append(df)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        scores = self._on_epoch_end(self.val_predictions)
        pl_module.log_dict({f"val/{metric}": value for metric, value in scores.mean(axis=0).to_dict().items()}, prog_bar=True, on_epoch=True)
        freq_df = pd.DataFrame(data=zip(pl_module.target_counts), columns=["train_label_counts"], index=pl_module.target_names)
        scores = scores.join(freq_df, on="species_name")
        scores["run_id"] = self.run_id
        scores["epoch"] = pl_module.current_epoch
        for param, value in pl_module.hparams.items():
            if param not in ["target_counts", "target_names"]:
                scores[param] = value
        scores = scores.reset_index()
        if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
            pl_module.logger.experiment.log({"val_scores": self._update_table(self.val_table, scores)})
        scores.to_parquet(self.save_dir / "val_scores.parquet" / f"epoch={pl_module.current_epoch}.parquet")
        self.val_predictions = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: List[pd.DataFrame],
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        df = self._on_batch_end(outputs)
        self.test_predictions.append(df)

    def on_test_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        scores = self._on_epoch_end(self.test_predictions)
        freq_df = pd.DataFrame(data=zip(pl_module.target_counts), columns=["train_label_counts"], index=pl_module.target_names)
        scores = scores.join(freq_df, on="species_name")
        scores["run_id"] = self.run_id
        # save and log scores
        for param, value in pl_module.hparams.items():
            if param not in ["target_counts", "target_names"]:
                scores[param] = value
        scores = scores.reset_index()
        if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
            pl_module.logger.experiment.log({"test_scores": wandb.Table(dataframe=scores)})
        scores.to_parquet(self.save_dir / "test_scores.parquet")
        print(scores.to_markdown())
        # log summary stats
        summary_stats = scores.groupby("run_id").agg(
            auROC_mean=("auROC", "mean"),
            auROC_std=("auROC", "std"),
            AP_mean=("AP", "mean"),
            AP_std=("AP", "std"),
        ).reset_index()
        for param, value in pl_module.hparams.items():
            if param not in ["target_counts", "target_names"]:
                summary_stats[param] = value
        if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
            pl_module.logger.experiment.log({"test_scores_summary": wandb.Table(dataframe=summary_stats.T)})
        print(summary_stats.T.to_markdown())
        self.test_predictions = []

    def _on_batch_end(self, outputs: List[Dict[str, Any]]) -> pd.DataFrame:
        y, y_probs, s, target_names = outputs["y"], outputs["y_probs"], outputs["s"], outputs["target_names"]
        label_df = pd.DataFrame(data=y.detach().cpu(), columns=target_names, index=s.detach().cpu().tolist())
        probs_df = pd.DataFrame(data=y_probs.detach().cpu(), columns=target_names, index=s.detach().cpu().tolist())
        return (
            label_df
            .reset_index(names="file_i")
            .melt(id_vars="file_i", var_name="species_name", value_name="label")
            .merge(
                probs_df
                .reset_index(names="file_i")
                .melt(id_vars="file_i", var_name="species_name", value_name="prob"),
                on=["file_i", "species_name"],
                how="inner",
            )
        )

    def _on_epoch_end(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return self.score(pd.concat(results))

    def _update_table(self, table: wandb.Table, df: pd.DataFrame):
        if table is None:
            table = wandb.Table(dataframe=df, log_mode="INCREMENTAL")
        else:
            for _, row in scores.iterrows():
                table.add_data(*row.tolist())
        return table
