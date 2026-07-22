import attrs
import itertools
import functools
import lightning as L
import logging
import multiprocessing as mp
import os
import pathlib
import pandas as pd
import requests
import tempfile
import torch
import tqdm

from omegaconf import DictConfig
from typing import Any, Callable, List, Dict, Tuple

from src.core.evaluators.base import Evaluator
from src.core.models.birdnet import _fetch_analyzer
from src.core.utils import metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["BirdNETPredictor", "BirdNETEmbedder"]

BIRDNET_LABEL_TXT_FILE = (
    "https://raw.githubusercontent.com/kahst/BirdNET-Analyzer"
    "/refs/tags/v1.5.0/birdnet_analyzer/checkpoints/V2.4/"
    "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
)

def chunked(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

class BirdNETPredictor(Evaluator):
    def __init__(self, min_train_label_count: int = 10):
        self.min_train_label_count = min_train_label_count

    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        data_module.setup()
        data = data_module.test_data

        # persist species to tempfile for analyzer
        birdnet_species_list = [label for label in requests.get(BIRDNET_LABEL_TXT_FILE).text.split("\n")]
        species_list = list(set(birdnet_species_list).intersection(set(data.train_labels.loc[:, data.train_labels.sum(axis=0) > self.min_train_label_count].columns)))
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("\n".join(species_list))
            temp_path = temp_file.name

        # extract ground truth labels based on species list
        labels = data.test_labels.reset_index().melt(id_vars=["file_i"], value_vars=species_list, value_name="label")

        # process all files
        probs = []
        records = data.test_metadata.reset_index()[["file_i", "file_path"]].to_dict("records")
        fn, inputs = functools.partial(model, custom_species_list_path=temp_path), chunked(records, data_module.eval_batch_size)
        with tqdm.tqdm(total=len(records)) as pbar:
            with mp.Pool(processes=data_module.num_workers, initializer=_fetch_analyzer) as pool:
                for batch_probs, failed_files in pool.imap(fn, inputs):
                    probs.append(batch_probs)
                    pbar.update(data_module.eval_batch_size)
        probs = pd.concat(probs, axis=0)

        # cleanup species list
        os.remove(temp_path)

        # take the maximum probability for each species in each file
        probs = probs.groupby(["file_i", "species_name"])["prob"].max().reset_index()
        probs = probs[probs.species_name.isin(species_list)]
        # merge with ground truth labels, left join and fill missing species probs with 0
        results = labels.merge(probs, on=["file_i", "species_name"], how="left").fillna(0.0)

        results["model"] = "birdnet_native"
        results["scope"] = data_module.name
        # compute scores
        scores = metrics.score(results)
        scores["model"] = "birdnet"
        scores["scope"] = data_module.name

        print(scores.to_markdown())

        summary_stats = scores.groupby(["model", "scope"]).agg(
            auROC_mean=("auROC", "mean"),
            AP_mean=("AP", "mean"),
        ).reset_index()

        results_pivot = results.pivot(columns="species_name", index="file_i")
        summary_stats["recall_at_k"] = metrics.recall_at_k(results_pivot["label"].to_numpy(), results_pivot["prob"].to_numpy())
        print(summary_stats.to_markdown())

        if (save_dir := config.get("paths").get("results_dir")):
            save_dir = pathlib.Path(config.get("paths").get("results_dir")).expanduser()
            save_dir.mkdir(exist_ok=True, parents=True)
            results_dir = save_dir / "test_results.parquet"
            results_dir.mkdir(exist_ok=True, parents=True)
            scores_dir = save_dir / "test_scores.parquet"
            scores_dir.mkdir(exist_ok=True, parents=True)
            results.to_parquet(results_dir / f"model=birdnet_scope={data_module.name}.parquet")
            scores.to_parquet(scores_dir / f"model=birdnet_scope={data_module.name}.parquet")

class BirdNETEmbedder(Evaluator):
    def __call__(self, trainer: None, model: Callable, data_module: L.LightningDataModule, config: DictConfig, **kwargs: Any):
        data_module.setup()
        data = data_module.data

        results = []
        records = data_module.data.metadata.reset_index()[["file_i", "file_path"]].to_dict("records")
        fn, inputs = functools.partial(model), chunked(records, data_module.eval_batch_size)
        with tqdm.tqdm(total=len(records)) as pbar:
            with mp.Pool(processes=data_module.num_workers, initializer=_fetch_analyzer) as pool:
                for result, failed_files in pool.imap(fn, inputs):
                    results.append(result)
                    pbar.update(data_module.eval_batch_size)
        results = pd.concat(results, axis=0)

        if (save_dir := config.get("paths").get("results_dir")):
            data = data_module.data
            save_dir = pathlib.Path(save_dir)
            train_dir = pathlib.Path(save_dir) / "train"
            train_dir.mkdir(exist_ok=True, parents=True)
            train_features, train_labels = results[results.index.get_level_values("file_i").isin(data.train_idx.file_i)], data.train_labels
            train_features.to_parquet(train_dir / "features.parquet")
            train_labels.to_parquet(train_dir / "labels.parquet")

            test_dir = pathlib.Path(save_dir) / "test"
            test_dir.mkdir(exist_ok=True, parents=True)
            test_features, test_labels = results[results.index.get_level_values("file_i").isin(data.test_idx.file_i)], data.test_labels
            test_features.to_parquet(test_dir / "features.parquet")
            test_labels.to_parquet(test_dir / "labels.parquet")
