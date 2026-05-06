import librosa
import contextlib
import os
import sys
import functools
import multiprocessing as mp
import lightning as L
import tqdm
import birdnetlib
import birdnet
import pandas as pd
import pathlib
import requests

from collections import defaultdict
from omegaconf import DictConfig
from typing import Any, List, Dict, Tuple

from src.core.utils import metrics

BIRDNET_LABEL_TXT_FILE = (
    "https://raw.githubusercontent.com/kahst/BirdNET-Analyzer"
    "/refs/tags/v1.5.0/birdnet_analyzer/checkpoints/V2.4/"
    "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
)

__all__ = ["BirdNET"]

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

class BirdNET:
    def __init__(
        self,
        mode: str = "predict",
        min_confidence: float = 0.0,
        version: str = "v2.4",
    ) -> None:
        self.mode = mode
        self.min_confidence = min_confidence
        self.version = version
        self.target_names = [label for label in requests.get(BIRDNET_LABEL_TXT_FILE).text.split("\n")]

    def __call__(self, data_module: List[str], target_names: List[str]) -> pd.DataFrame:
        data = data_module.data
        inputs = chunked(list(zip(data.metadata.index, data.metadata.file_path)), data_module.eval_batch_size)
        if self.mode == "predict":
            return self.predict(inputs, target_names)
        elif self.mode == "embed":
            return self.embed(inputs)
        else:
            raise Exception(f"Selected mode {self.mode} for BirdNET is not valid, select from 'predict' or 'embed'")

    def predict(self, file_names: List[str], target_names: List[str]) -> pd.DataFrame:
        results = []
        for batch in birdnet.predict_species_within_audio_files_mp(file_names, min_confidence=self.min_confidence, species_filter=set(target_names)):
            df = self.step(*batch)
            results.append(df)
        df = pd.DataFrame(results)
        # TODO: pad missing species with zeros
        import code; code.interact(local=locals())
        return df

    def predict_step(self, file_path: str, predictions: Dict[float, Dict[str, float]]):
        results = []
        y_prob = defaultdict(lambda: 0.0)
        for window, prediction in predictions.items():
            for target, prob in prediction.items():
                y_prob[target] = max(prob, y_prob[target])
        for target, prob in y_prob.items():
            results.append({
                "file_name": file_path.name,
                "species_name": target,
                "prob": prob,
            })
        return pd.DataFrame(results)

    def embed(self, data_module: L.LightningDataModule):
        data = data_module.data
        inputs = chunked(list(zip(data.metadata.index, data.metadata.file_path)), data_module.eval_batch_size)
        dfs, failed = [], []
        with tqdm.tqdm(total=len(data.metadata)) as pbar:
            with mp.Pool(processes=data_module.num_workers, initializer=_fetch_analyzer) as pool:
                for df, fps in pool.imap(self.embed_step, inputs):
                    dfs.append(df)
                    failed.extend(fps)
                    pbar.update(data_module.eval_batch_size)
        return pd.concat(dfs, axis=0)

    def embed_step(self, batch: List[Tuple[int, str]]) -> pd.DataFrame:
        batched, failed = [], []
        for x in batch:
            df, file_path = self.embed_file(*x)
            batched.append(df)
            failed.append(file_path)
        return pd.concat(batched, axis=0), list(filter(None, failed))

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
