import attrs
import librosa
import contextlib
import os
import sys
import attrs
import functools
import tqdm
import birdnetlib

import attrs
import birdnet
import pandas as pd
import pathlib
import lightning as L
import requests

from collections import defaultdict
from omegaconf import DictConfig
from typing import Any, List, Dict, Tuple

from src.core.utils import metrics

__all__ = ["BirdNET", "BirdNETEmbeddings"]

_analyzer = None

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

@suppress_output()
def _fetch_analyzer(**kwargs: Any) -> birdnetlib.analyzer.Analyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = birdnetlib.analyzer.Analyzer(**kwargs)
    return _analyzer

class BirdNETPredictions:
    def __init__(self, min_confidence: float = 0.0):
        self.min_confidence = min_confidence

    def __call__(self, batch: List[Dict[str, Any]], **kwargs: Any) -> pd.DataFrame:
        return self.process_batch(batch, **kwargs)

    def process_batch(self, batch: List[Dict[str, Any]], **kwargs: Any) -> Any:
        batched, failed = [], []
        for audio_dict in batch:
            predictions, file_path = self.forward(**audio_dict, **kwargs)
            batched.append(predictions)
            failed.append(file_path)
        return pd.concat(batched, axis=0), list(filter(None, failed))

    @suppress_output()
    def forward(
        self,
        file_i: int,
        file_path: str,
        latitude: float | None = None,
        longitude: float | None = None,
        timestamp: float | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            analyzer = _fetch_analyzer(**kwargs)
            recording = birdnetlib.Recording(analyzer, file_path, min_conf=self.min_confidence)
            recording.analyze()
            if not len(recording.detections):
                return pd.DataFrame(), None
            df = pd.DataFrame(recording.detections)
            df = df.drop(["common_name", "scientific_name", "start_time", "end_time"], axis=1)
            df = df.rename(columns={"confidence": "prob", "label": "species_name"})
            df["file_i"] = file_i
            df["file_name"] = file_path.name
            return df, None
        except:
            return pd.DataFrame(), file_path

class BirdNETEmbeddings:
    def __call__(self, batch: List[Dict[str, Any]], **kwargs: Any) -> pd.DataFrame:
        return self.process_batch(batch, **kwargs)

    def process_batch(self, batch: List[Dict[str, Any]], **kwargs: Any) -> Any:
        batched, failed = [], []
        for audio_dict in batch:
            predictions, file_path = self.forward(**audio_dict, **kwargs)
            batched.append(predictions)
            failed.append(file_path)
        return pd.concat(batched, axis=0), list(filter(None, failed))

    @suppress_output()
    def forward(self, file_i: int, file_path: str, **kwargs: Any) -> pd.DataFrame:
        try:
            analyzer = _fetch_analyzer(**kwargs)
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

