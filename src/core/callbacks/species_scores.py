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
    def __init__(
        self,
        save_dir: str,
        run_id: str,
        model: str,
        version: str,
        scope: str,
        fold_id: int | None = None,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.fold_id = fold_id
        self.model = model
        self.scope = scope
        self.version = version
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        (self.save_dir / "val_scores.parquet").mkdir(exist_ok=True, parents=True)
        (self.save_dir / "test_scores.parquet").mkdir(exist_ok=True, parents=True)
        (self.save_dir / "test_results.parquet").mkdir(exist_ok=True, parents=True)
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
            results = pd.concat(self.train_predictions)
            scores = self._on_epoch_end(results, pl_module)
            pl_module.log_dict({
                f"train/{metric}": value
                for metric, value in scores[["auROC", "AP"]].mean(axis=0).to_dict().items()
            }, prog_bar=True, on_epoch=True)
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
        if len(self.val_predictions):
            results = pd.concat(self.val_predictions)
            scores = self._on_epoch_end(results, pl_module)
            pl_module.log_dict({
                f"val/{metric}": value
                for metric, value in scores[["auROC", "AP"]].mean(axis=0).to_dict().items()
            }, prog_bar=True, on_epoch=True)
            scores["epoch"] = pl_module.current_epoch
            # if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
                # pl_module.logger.experiment.log({"val_scores": self._update_table(self.val_table, scores)})
            scores.to_parquet(self.save_dir / "val_scores.parquet" / f"run_id={self.run_id}_epoch={pl_module.current_epoch}.parquet")
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
        if len(self.test_predictions):
            results = pd.concat(self.test_predictions)
            scores = self._on_epoch_end(results, pl_module)
            # if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
                # pl_module.logger.experiment.log({"test_scores": wandb.Table(dataframe=scores)})
            scores.to_parquet(self.save_dir / "test_scores.parquet" / f"run_id={self.run_id}.parquet")
            results.to_parquet(self.save_dir / "test_results.parquet" / f"run_id={self.run_id}.parquet")
            # recall at k, proportion of species in the top K were predicted?
            print(scores.to_markdown())
            # log summary stats
            summary_stats = scores.groupby("run_id").agg(
                auROC_mean=("auROC", "mean"),
                auROC_std=("auROC", "std"),
                AP_mean=("AP", "mean"),
                AP_std=("AP", "std"),
            ).reset_index()
            results_pivot = results.pivot(columns="species_name", index="file_i")
            summary_stats["recall_at_k"] = metrics.recall_at_k(
                results_pivot["label"].to_numpy(),
                results_pivot["prob"].to_numpy(),
            )
            summary_stats = self._attach_hparams(summary_stats, pl_module.hparams)
            # if pl_module.logger is not None and hasattr(pl_module.logger, "experiment") and callable(pl_module.logger.experiment.log):
                # pl_module.logger.experiment.log({"test_scores_summary": wandb.Table(dataframe=summary_stats.T)})
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

    def _on_epoch_end(self, results: List[pd.DataFrame], pl_module: L.LightningModule) -> pd.DataFrame:
        df = self.score(results)
        df1 = self._freq_df(pl_module)
        df = df.join(df1, on="species_name").reset_index()
        df = self._attach_hparams(df, pl_module.hparams)
        return df

    def _update_table(self, table: wandb.Table, df: pd.DataFrame):
        if table is None:
            table = wandb.Table(dataframe=df, log_mode="INCREMENTAL")
        else:
            for _, row in scores.iterrows():
                table.add_data(*row.tolist())
        return table

    def _attach_hparams(self, df: pd.DataFrame, hparams: Dict[str, Any]):
        df["run_id"] = self.run_id
        df["model"] = self.model
        df["version"] = self.version
        df["scope"] = self.scope
        if self.fold_id is not None:
            df["fold_id"] = self.fold_id
        for param, value in hparams.items():
            if param not in ["target_counts", "target_names"]:
                df[param] = value
        return df

    def _freq_df(self, pl_module: L.LightningModule):
        return pd.DataFrame(
            data=zip(pl_module.target_counts),
            columns=["train_label_counts"],
            index=pl_module.target_names
        )
