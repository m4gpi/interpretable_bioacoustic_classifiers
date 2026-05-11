import attrs
import librosa
import lightning as L
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import ranzen
import ranzen.torch
import re
import requests
import shutil
import sklearn
import torch
import torchaudio

from torchvision import transforms as T
from typing import Any, Callable, ClassVar, Dict, Final, List, Tuple

from src.core.utils import Batch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = [
    "WABAD",
    "WABADDataModule",
]

class WABAD(torch.utils.data.Dataset):
    _DATA_DIR: ClassVar[str] = "Recordings"
    _AUDIO_SAMPLE_RATE: int = 32_000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.load_sample(self.x[idx]), self.y[idx], self.s[idx]

    def __len__(self):
        return len(self.x)

    def __init__(
        self,
        root: str,
        *,
        segment_len: float = 59.904,
        sample_rate: int = 48_000,
        reset_index: bool = False,
        test: bool | None = None,
        site_id: str | None = None,
        seed: int = 42,
        download: bool = False,
    ) -> None:
        self.base_dir = pathlib.Path(root).expanduser()
        self.data_dir = self.base_dir / self._DATA_DIR
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.num_frames_in_segment = int(self.segment_len * self.sample_rate)

        if download:
            self._download_files()
            self._build_metadata()
            self._split_testset()
        # load file info and labels
        self.metadata = pd.read_parquet(self.base_dir / "metadata.parquet")
        self.metadata["file_path"] = self.data_dir / self.metadata["file_name"]
        self.metadata = self.metadata[self.metadata.sample_rate >= sample_rate]
        if site_id is not None:
            self.metadata = self.metadata[self.metadata["site_id"] == site_id]
        self.metadata["site_id"] = self.metadata["site_id"].astype(pd.CategoricalDtype(categories=self.metadata["site_id"].unique()))
        # load occurrance labels
        self.labels = pd.read_parquet(self.base_dir / "labels.parquet").astype(bool).astype(int)
        # scope by train / test
        self.train_idx = pd.read_parquet(self.base_dir / "train_indices.parquet")
        self.test_idx = pd.read_parquet(self.base_dir / "test_indices.parquet")
        self.train_metadata = self.metadata[self.metadata.index.isin(self.train_idx.file_i)]
        self.test_metadata = self.metadata[self.metadata.index.isin(self.test_idx.file_i)]
        self.train_labels = self.labels.loc[self.labels.index.isin(self.train_idx.file_i)]
        self.test_labels = self.labels.loc[self.labels.index.isin(self.test_idx.file_i)]

        self.train_site_ids = self.train_metadata["site_id"].cat.codes
        self.test_site_ids = self.test_metadata["site_id"].cat.codes

        if test == True:
            self.x = self.test_metadata.file_path.to_numpy()
            self.y = torch.tensor(self.test_labels.to_numpy(), dtype=torch.float32)
            self.s = self.test_metadata.index
        elif test == False:
            self.x = self.train_metadata.file_path.to_numpy()
            self.y = torch.tensor(self.train_labels.to_numpy())
            self.s = self.train_metadata.index
        else:
            self.x = self.metadata.file_path.to_numpy()
            self.y = torch.tensor(self.labels.to_numpy())
            self.s = self.metadata.index

    @property
    def target_names(self) -> List[str]:
        return  self.labels.columns.tolist()

    @property
    def target_counts(self) -> List[int]:
        return self.labels.sum(axis=0).tolist()

    @property
    def model_params(self):
        return dict(
            target_names=self.target_names,
            target_counts=self.target_counts,
        )

    def load_sample(self, file_path: str) -> torch.Tensor:
        metadata = torchaudio.info(file_path)
        num_frames_segment = int(self.num_frames_in_segment / self.sample_rate * metadata.sample_rate)
        high = max(1, metadata.num_frames - num_frames_segment)
        frame_offset = torch.randint(low=0, high=high, size=(1,))
        waveform, _ = torchaudio.load(file_path, num_frames=num_frames_segment)
        return torchaudio.functional.resample(waveform[:1], orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

    def _check_files(self) -> None:
        """assert files exist and zip unpacked"""
        assert (self.base_dir / "train").exists() and (self.base_dir / "test").exists(), \
            f"audio not found at location {(self.base_dir / 'train').resolve()} or {(self.base_dir / 'test').resolve()}. Have you downloaded it?"

        assert (self.base_dir / "metadata.parquet").exists(), \
            f"'metadata.parquet' not found at location {self.base_dir.resolve()}. Have you downloaded it?"

        assert (self.base_dir / "labels.parquet").exists(), \
            f"'labels.parquet' not found at location {self.base_dir.resolve()}. Have you downloaded it?"

        assert (self.base_dir / "train_indices.parquet").exists() and (self.base_dir / "test_indices.parquet").exists(), \
            f"'train_indices.parquet' or 'test_indices.parquet' not found at location {self.base_dir.resolve()}. Pass 'reset_index=True seed=42' to rebuild the data split"

    def _build_metadata(self) -> None:
        df = pd.DataFrame(os.listdir(self.data_dir), columns=["file_name"])
        df["file_path"] = df["file_name"].map(lambda f: base_dir / "Recordings" / f)
        df["sample_rate"] = df["file_path"].map(librosa.get_samplerate)
        df["duration_seconds"] = df.apply(lambda row: librosa.get_duration(path=row.file_path, sr=row.sample_rate), axis=1)
        df = pd.concat([df, df["file_name"].str[:-4].str.split("_", expand=True).rename(columns={0: "site_id", 1: "date", 2: "time"})], axis=1)
        df["timestamp"] = df["date"] + "_" + df["time"]
        columns = ['Site ID', 'Study area', 'Recording location', 'Biome', 'Latitude', 'Longitude', 'Recorder (+ microphone)', 'Ominidirectional']
        new_columns = [to_snake_case(col.replace(" ", "_")) for col in columns]
        new_columns[6] = "recorder"
        new_columns[7] = "omnidirectional"
        metadata = pd.read_csv(base_dir / "Metadata.csv", engine="pyarrow")
        metadata = metadata.rename(columns=dict(zip(columns, new_columns)))
        df = df.merge(metadata[new_columns], on="site_id", how="left")
        df["omnidirectional"] = df["omnidirectional"].map({"No": False, "Yes": True})
        df = df.reset_index(names=["file_i"])
        df = df.drop("file_path", axis=1)
        df.to_parquet(self.base_dir / "metadata.parquet", engine="pyarrow")

    def _build_occurrence_labels(self) -> None:
        df = pd.read_csv(self.base_dir / "Pooled annotations.csv", engine="pyarrow")
        df["count"] = 1
        df = (
            df.drop(["Begin_Time_(s)", "End_Time_(s)", "Low_Freq_(Hz)", "High_Freq_(Hz)"], axis=1)
            .rename(columns={"Recording": "file_name", "Species": "species_name", "Site": "site_id"})
            .groupby(["species_name", "file_name", "site_id"])["count"]
            .sum()
            .reset_index()
            .pivot(index=["file_name", "site_id"], columns="species_name", values="count")
            .fillna(0.0)
            .reset_index()
        )
        df.to_parquet("labels.parquet", engine="pyarrow")

    def _split_testset(self):
        if (self.base_dir / f"train_indices.parquet").exists(): return
        metadata = pd.read_parquet(self.base_dir / "metadata.parquet")
        train_idx, test_idx = sklearn.model_selection.train_test_split(metadata.index.to_numpy(), test_size=0.2, random_state=self.seed)
        metadata.loc[train_idx, ["file_i", "file_name"]].to_parquet(self.base_dir / "train_indices.parquet")
        metadata.loc[test_idx, ["file_i", "file_name"]].to_parquet(self.base_dir / "test_indices.parquet")

@attrs.define(kw_only=True)
class WABADDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    segment_len: float = attrs.field(default=59.904)
    sample_rate: int = attrs.field(default=48_000)
    site_id: str | None = attrs.field(default=None)

    train_batch_size: int = attrs.field(default=6)
    eval_batch_size: int | None = attrs.field(default=None)
    val_prop: float = attrs.field(default=0.2)
    test_prop: float = attrs.field(default=0.2)
    num_workers: int = attrs.field(default=0)
    seed: int = attrs.field(default=42)
    persist_workers: bool = attrs.field(default=False)
    pin_memory: bool = attrs.field(default=True)
    training_mode: str = attrs.field(default="step")

    data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)
    train_data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)
    val_data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)
    test_data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)

    def _batch_converter(self, batch: Tuple):
        xs, ys, ss = zip(*batch)
        return Batch(x=torch.stack(xs, dim=0), y=torch.stack(ys, dim=0), s=torch.tensor(ss), metadata=self.data.target_names)

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)
        self.training_mode = ranzen.torch.TrainingMode[self.training_mode]

    def prepare_data(self):
        WABAD(root=self.root)
        return self

    def setup(self, *args: Any, **kwargs: Any):
        self.data = WABAD(self.root, test=False, **self.dataset_params)
        self.val_data, self.train_data = torch.utils.data.random_split(self.data, (self.val_prop, 1 - self.val_prop), generator=self.generator)
        self.test_data = WABAD(self.root, test=True, **self.dataset_params)
        return self

    @property
    def generator(self):
        return torch.Generator().manual_seed(self.seed)

    @property
    def dataset_params(self):
        return dict(
            segment_len=self.segment_len,
            sample_rate=self.sample_rate,
        )

    def train_dataloader_params(self, batch_size: int | None = None) -> Dict[str, Any]:
        if self.training_mode == ranzen.torch.TrainingMode.step:
            return dict(batch_size=batch_size, batch_sampler=self._default_train_sampler(batch_size))
        else:
            return dict(batch_size=batch_size, shuffle=True, generator=self.generator, drop_last=False)

    def train_dataloader(self, batch_size: int | None = None, batch_sampler: torch.utils.data.Sampler | None = None) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.train_data, **self.train_dataloader_params(self.train_batch_size))

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size)

    def predict_dataloader(self) -> List[torch.utils.data.DataLoader]:
        return [
            self._build_dataloader(self.train_data, batch_size=self.eval_batch_size),
            self.val_dataloader(),
            self.test_dataloader(),
        ]

    @property
    def dataloader_params(self) -> Dict[str, Any]:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persist_workers,
        )

    def _build_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        batch_sampler: torch.utils.data.Sampler | None = None,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size if batch_sampler is None else 1,
            batch_sampler=batch_sampler,
            collate_fn=self._batch_converter,
            **self.dataloader_params,
            **kwargs,
        )

    def _default_train_sampler(self, batch_size: int | None = None) -> torch.utils.data.Sampler:
        return ranzen.torch.SequentialBatchSampler(
            data_source=self.train_data,
            batch_size=batch_size or self.train_batch_size,
            shuffle=False,
            training_mode=self.training_mode,
            drop_last=False,
            generator=self.generator,
        )
        # return ranzen.torch.StratifiedBatchSampler(
        #     group_ids=self.data.train_metadata["site_id"].cat.codes,
        #     num_samples_per_group=batch_size / len(self.data.train_metadata["site_id"].cat.categories),
        #     shuffle=False,
        #     training_mode=self.training_mode,
        #     drop_last=False,
        #     generator=self.generator,
        # )
