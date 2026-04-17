import attrs
import itertools
import lightning as L
import logging
import pathlib
import pandas as pd
import torch
import tqdm

from omegaconf import DictConfig
from typing import Any, Callable

from src.core.evaluators.base import Evaluator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["BirdNETPredictor", "BirdNETEmbedder"]

class BirdNETPredictor(Evaluator):
    min_train_label_count: int = attrs.field(default=10)

    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        run_id = config.get("run_id")
        data_module.setup()

        data = data_module.test_data
        file_paths = data.test_metadata.file_path
        target_names = data.train_labels.loc[:, data.train_labels.sum(axis=0) > self.min_train_label_count].columns
        target_names = list(set(self.target_names).intersection(set(target_names)))

        probs = model(file_paths, target_names)

        results = (
            data.test_labels.reset_index()
            .melt(id_vars=["file_i", "file_name"], value_vars=target_names, value_name="label")
            .merge(probs, on=["file_name", "species_name"], how="inner")
        )
        results["run_id"] = run_id
        results["model"] = "birdnet"
        results["version"] = model.version
        results["scope"] = data_module.scope

        scores = metrics.score(results)
        scores["run_id"] = run_id
        scores["model"] = "birdnet"
        scores["version"] = model.version
        scores["scope"] = data_module.scope

        print(scores.to_markdown())

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
        print(summary_stats.to_markdown())

        save_dir = pathlib.Path(config.get("paths").get("results_dir")).expanduser()
        save_dir.mkdir(exist_ok=True, parents=True)
        results_dir = save_dir / "test_results.parquet"
        results_dir.mkdir(exist_ok=True, parents=True)
        scores_dir = save_dir / "test_scores.parquet"
        scores_dir.mkdir(exist_ok=True, parents=True)
        results.to_parquet(results_dir / f"run_id={run_id}.parquet")
        scores.to_parquet(scores_dir / f"run_id={run_id}.parquet")

class BirdNETEmbedder(Evaluator):
    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        data_module.setup()

        df = model(data_module)

        if (save_dir := config.get("paths").get("results_dir")):
            data = data_module.data
            save_dir = pathlib.Path(save_dir)
            train_dir = pathlib.Path(save_dir) / "train"
            train_dir.mkdir(exist_ok=True, parents=True)
            train_features, train_labels = df[df.index.get_level_values("file_i").isin(data.train_idx.file_i)], data.train_labels
            train_features.to_parquet(train_dir / "features.parquet")
            train_labels.to_parquet(train_dir / "labels.parquet")

            test_dir = pathlib.Path(save_dir) / "test"
            test_dir.mkdir(exist_ok=True, parents=True)
            test_features, test_labels = df[df.index.get_level_values("file_i").isin(data.test_idx.file_i)], data.test_labels
            test_features.to_parquet(test_dir / "features.parquet")
            test_labels.to_parquet(test_dir / "labels.parquet")
