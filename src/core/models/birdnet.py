import librosa
import contextlib
import os
import sys
import attrs
import functools
import multiprocessing as mp
import tqdm
import birdnetlib

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

__all__ = ["BirdNET", "BirdNETEmbeddings"]

_analyzer = None

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

@suppress_output()
def _fetch_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = birdnetlib.analyzer.Analyzer()
    return _analyzer

def chunked(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

@attrs.define()
class BirdNET:
    min_confidence: float = attrs.field(default=0.0)
    min_train_label_count: int = attrs.field(default=10)
    version: str = attrs.field(default="v2.4")

    @property
    def model_params(self):
        return dict()

    @property
    def target_names(self):
        return [label for label in requests.get(BIRDNET_LABEL_TXT_FILE).text.split("\n")]

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

        data_module.setup(stage="eval")
        data = data_module.test_data
        target_names = data.train_labels.loc[:, data.train_labels.sum(axis=0) > self.min_train_label_count].columns
        target_names = list(set(self.target_names).intersection(set(target_names)))

        labels = data.test_labels.reset_index()
        probs = self.encode(data.test_metadata.file_path, target_names)
        results = (
            labels
            .melt(id_vars=["file_i", "file_name"], value_vars=target_names, value_name="label")
            .merge(probs, on=["file_name", "species_name"], how="inner")
        )
        results["run_id"] = run_id
        results["model"] = "birdnet"
        results["version"] = self.version
        results["scope"] = data_module.scope

        scores = metrics.score(results)
        scores["run_id"] = run_id
        scores["model"] = "birdnet"
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

        out_dir = pathlib.Path(config.get("paths").get("results_dir")).expanduser()
        out_dir.mkdir(exist_ok=True, parents=True)
        results_dir = out_dir / "test_results.parquet"
        results_dir.mkdir(exist_ok=True, parents=True)
        scores_dir = out_dir / "test_scores.parquet"
        scores_dir.mkdir(exist_ok=True, parents=True)

        results.to_parquet(results_dir / f"run_id={run_id}.parquet")
        scores.to_parquet(scores_dir / f"run_id={run_id}.parquet")

@attrs.define()
class BirdNETEmbeddings:
    save_dir: str = attrs.field()
    version: str = attrs.field(default="v2.4")
    num_workers: int = attrs.field(default=32)
    batch_size: int = attrs.field(default=6)

    @property
    def model_params(self):
        return dict()

    def embed_file(self, file_i: int, file_path: str) -> pd.DataFrame:
        try:
            with suppress_output():
                analyzer = _fetch_analyzer()
                recording = birdnetlib.Recording(analyzer, str(file_path))
                recording.extract_embeddings()
                df = pd.DataFrame([
                    pd.concat([
                        pd.Series({str(dim): value for dim, value in enumerate(embedding_info["embeddings"])}),
                        pd.Series({k: v for k, v in embedding_info.items() if k != "embeddings"}),
                    ])
                    for embedding_info in recording.embeddings
                ])
                df = df.drop(["start_time", "end_time"], axis=1)
                df["file_i"] = file_i
                df = df.reset_index(names="timestep")
                df = df.set_index(["file_i", "timestep"])
                return df, None
        except:
            return pd.DataFrame(), file_path

    def embed_batch(self, inputs: List[Tuple[int, str]]) -> pd.DataFrame:
        batched, failed = [], []
        for input in inputs:
            df, file_path = self.embed_file(*input)
            batched.append(df)
            failed.append(file_path)
        return pd.concat(batched, axis=0), list(filter(None, failed))

    def evaluate(self, trainer: None, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        run_id = config.get("run_id")
        data_module.setup(stage="eval")
        data = data_module.data

        inputs = chunked(list(zip(data.metadata.index, data.metadata.file_path)), self.batch_size)
        dfs, failed = [], []
        with tqdm.tqdm(total=len(data.metadata)) as pbar:
            with mp.Pool(processes=self.num_workers, initializer=_fetch_analyzer) as pool:
                for df, fps in pool.imap(self.embed_batch, inputs):
                    dfs.append(df)
                    failed.extend(fps)
                    pbar.update(self.batch_size)
        df = pd.concat(dfs, axis=0)

        train_dir = pathlib.Path(self.save_dir) / "train"
        train_dir.mkdir(exist_ok=True, parents=True)
        train_features, train_labels = df[df.index.get_level_values("file_i").isin(data.train_idx.file_i)], data.train_labels
        train_features.to_parquet(train_dir / "features.parquet")
        train_labels.to_parquet(train_dir / "labels.parquet")

        test_dir = pathlib.Path(self.save_dir) / "test"
        test_dir.mkdir(exist_ok=True, parents=True)
        test_features, test_labels = df[df.index.get_level_values("file_i").isin(data.test_idx.file_i)], data.test_labels
        test_features.to_parquet(test_dir / "features.parquet")
        test_labels.to_parquet(test_dir / "labels.parquet")
