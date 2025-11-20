import attrs
import birdnet
import pandas as pd
import pathlib
import lightning as L
import requests

from omegaconf import DictConfig
from typing import Any, List, Dict, Tuple

from src.core.utils import metrics

BIRDNET_LABEL_TXT_FILE = (
    "https://raw.githubusercontent.com/kahst/BirdNET-Analyzer"
    "/refs/tags/v1.5.0/birdnet_analyzer/checkpoints/V2.4/"
    "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
)

__all__ = ["BirdNET"]

@attrs.define()
class BirdNET:
    min_confidence: float = attrs.field(default=0.0)
    version: str = attrs.field(default="V2.4")

    def encode(self, file_names: List[str], target_names: List[str]) -> pd.DataFrame:
        results = []
        params = dict(min_confidence=self.min_confidence, species_filter=set(target_names))
        iterator = birdnet.predict_species_within_audio_files_mp(file_names, **params)
        for file_path, predictions in iterator:
            y_prob = pd.Series({target: 0.0 for target in target_names})
            for window, prediction in predictions.items():
                for target, prob in prediction.items():
                    y_prob[target] = max(prob, y_prob[target])
            for target in target_names:
                results.append({
                    "file_name": file_path.name,
                    "species_name": target,
                    "prob": y_prob[target],
                })
        return pd.DataFrame(results)

    def evaluate(self, trainer: None, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        run_id = config.get("run_id")

        data_module.setup()
        data = data_module.test_data
        birdnet_targets = [label for label in requests.get(BIRDNET_LABEL_TXT_FILE).text.split("\n")]
        target_names = list(set(birdnet_targets).intersection(set(data.target_names)))

        labels = data.labels.reset_index()
        file_names = (data.data_dir / labels.file_name).tolist()
        probs = self.encode(file_names, target_names)
        results = (
            labels
            .melt(id_vars=["file_i", "file_name"], value_vars=target_names, value_name="label")
            .merge(probs, on=["file_name", "species_name"], how="inner")
        )
        results["model"] = self.__class__.__name__
        results["version"] = self.version
        results["scope"] = data_module.scope

        scores = metrics.score(results)
        scores["run_id"] = run_id
        scores["model"] = self.__class__.__name__
        scores["version"] = self.version
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

        results_dir = pathlib.Path(config.get("results_dir")) / "test_results.parquet"
        results_dir.mkdir(exist_ok=True, parents=True)
        scores_dir = pathlib.Path(config.get("results_dir")) / "test_scores.parquet"
        scores_dir.mkdir(exist_ok=True, parents=True)

        results.to_parquet(results_dir / f"{run_id}.parquet")
        scores.to_parquet(scores_dir / f"{run_id}.parquet")
