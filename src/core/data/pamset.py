import attrs
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
    "PAMSet",
    "PAMSetDataModule",
]

class PAMSet(torch.utils.data.Dataset):
    _DATA_DIR: ClassVar[str] = "data"
    _MAX_AUDIO_LEN: Final[int] = 59.904
    _AUDIO_SAMPLE_RATE: int = 48_000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.load_sample(self.x[idx]), self.y[idx], self.s[idx]

    def __len__(self):
        return len(self.x)

    def __init__(
        self,
        root: str,
        *,
        country: str | None = None,
        segment_len: float = 59.904,
        sample_rate: int = 48_000,
        reset_index: bool = False,
        test: bool | None = None,
        seed: int = 42,
        with_labels: bool = False,
    ) -> None:
        self.base_dir = pathlib.Path(root).expanduser()
        self.data_dir = self.base_dir / self._DATA_DIR
        self.sample_rate = sample_rate
        self.segment_len = min(segment_len, self._MAX_AUDIO_LEN)
        self.num_frames_in_segment = int(self.segment_len * self.sample_rate)
        self.with_labels = with_labels

        # load file info and labels
        self.metadata = pd.read_parquet(self.base_dir / "metadata.parquet")
        self.metadata["file_path"] = self.data_dir / self.metadata["file_name"]
        self.metadata.index.name = "file_i"
        assert country in [None, *self.metadata.country.unique()], f"{country} is not a valid scope for country"
        if country is not None:
            self.metadata = self.metadata[self.metadata["country"] == country]
        self.metadata["country"] = self.metadata["country"].astype(
            pd.CategoricalDtype(categories=self.metadata["country"].unique())
        )

        self.train_idx = pd.read_parquet(self.base_dir / "train_indices.parquet")
        self.test_idx = pd.read_parquet(self.base_dir / "test_indices.parquet")
        self.train_metadata = self.metadata[self.metadata.index.isin(self.train_idx.file_i)]
        self.test_metadata = self.metadata[self.metadata.index.isin(self.test_idx.file_i)]

        if self.with_labels:
            self.labels = pd.read_parquet(self.base_dir / "labels.parquet")
            if country is not None:
                self.labels = self.labels[self.labels["country"] == country]
            self.labels = labels.pivot(index=["file_i", "file_name", "country"], columns="species_name", values="counts").fillna(0.0).astype(bool).astype(int)
            self.train_labels = self.labels.loc[self.labels.index.isin(self.train_idx.file_i)]
            self.test_labels = self.labels.loc[self.labels.index.isin(self.test_idx.file_i)]
        else:
            self.labels = pd.DataFrame(0, index=self.metadata.index, columns=["species_null"])
            self.train_labels = pd.DataFrame(0, index=self.train_metadata.index, columns=["species_null"])
            self.test_labels = pd.DataFrame(0, index=self.test_metadata.index, columns=["species_null"])

        self.train_country_ids = self.train_metadata["country"].cat.codes
        self.test_country_ids = self.test_metadata["country"].cat.codes

        if test == True:
            self.x = self.test_metadata.file_path.to_numpy()
            self.y = torch.tensor(self.test_labels.to_numpy())
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

    def load_sample(self, file_path: str) -> torch.Tensor:
        metadata = torchaudio.info(file_path)
        num_frames_segment = int(self.num_frames_in_segment / self.sample_rate * metadata.sample_rate)
        high = max(1, metadata.num_frames - num_frames_segment)
        frame_offset = torch.randint(low=0, high=high, size=(1,))
        waveform, _ = torchaudio.load(file_path, num_frames=num_frames_segment)
        return torchaudio.functional.resample(waveform, orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

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


@attrs.define(kw_only=True)
class PAMSetDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    segment_len: float = attrs.field(default=59.904)
    sample_rate: int = attrs.field(default=48_000)
    country: str | None = attrs.field(default=None)

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
        PAMSet(root=self.root)
        return self

    def setup(self, *args: Any, **kwargs: Any):
        self.data = PAMSet(self.root, test=False, **self.dataset_params)
        self.val_data, self.train_data = torch.utils.data.random_split(self.data, (self.val_prop, 1 - self.val_prop), generator=self.generator)
        self.test_data = PAMSet(self.root, test=True, **self.dataset_params)
        return self

    @property
    def model_params(self) -> Dict:
        return {}

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
        return ranzen.torch.StratifiedBatchSampler(
            group_ids=self.train_data.train_country_ids,
            num_samples_per_group=self.batch_size / len(self.train_data.train_country_ids),
            shuffle=False,
            training_mode=self.training_mode,
            drop_last=False,
            generator=self.generator,
        )

