import attrs
import lightning as L
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

__all__ = [
    "SoundingOutChorus",
    "SoundingOutChorusDataModule",
]

class SoundingOutChorus(torch.utils.data.Dataset):
    _DATA_DIR: ClassVar[str] = "data"
    num_frames_in_segment: int
    _MAX_AUDIO_LEN: Final[int] = 59.992458 # the smallest file is slighly under 60s
    _BITS_PER_BYTE: int = 8
    _AUDIO_SAMPLE_RATE: int = 48_000
    _BIT_RATE: int = 16

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.transforms(self.load_sample(self.x[idx])), self.y[idx], self.s[idx]

    def __len__(self):
        return len(self.x)

    def __init__(
        self,
        root: str,
        *,
        scope: str | None = None,
        transforms: Callable = lambda x: x,
        segment_len: float = 60.0,
        sample_rate: int = 48_000,
        download: bool = False,
        reset_index: bool = False,
        test: bool | None = None,
        seed: int = 42,
    ) -> None:
        self.base_dir = pathlib.Path(root).expanduser()
        self.sample_rate = sample_rate
        self.segment_len = min(segment_len, self._MAX_AUDIO_LEN)
        self.num_frames_in_segment = int(self.segment_len * self.sample_rate)
        self.transforms = transforms
        # fetch from remote and perform initial data split
        if download:
            self._download_files()
            self._split_testset()
        # rebuild the data split if specified
        if reset_index:
            self._split_testset()
        # load file info
        self.metadata = pd.read_parquet(self.base_dir / "metadata.parquet")
        self.metadata.index.name = "file_i"
        # load the labels, pivot so species are on columns, collapse point counts to presence/absence
        self.labels = (
            pd.read_parquet(self.base_dir / "birds.parquet")
            .reset_index()
            .pivot(index=["file_i", "file_name", "country"], columns="species_name", values="counts")
            .fillna(0.0)
            .astype(bool)
            .astype(int)
        )
        # scope by train / test
        train_idx = pd.read_parquet(self.base_dir / "train_indices.parquet")
        test_idx = pd.read_parquet(self.base_dir / "test_indices.parquet")
        self.train_metadata = self.metadata.loc[train_idx.file_i]
        self.train_labels = self.labels.loc[train_idx.file_i]
        self.test_metadata = self.metadata.loc[test_idx.file_i]
        self.test_labels = self.labels.loc[test_idx.file_i]
        # scope by country
        if scope is not None:
            idx, = np.where(self.metadata.country == scope.split("_")[-1])
            scope_idx = self.metadata.iloc[idx].index
            self.metadata = self.metadata.loc[scope_idx]
            self.train_metadata = self.train_metadata[self.train_metadata.index.isin(scope_idx)]
            self.test_metadata = self.test_metadata[self.test_metadata.index.isin(scope_idx)]
            self.labels = self.labels.loc[scope_idx]
            self.train_labels = self.train_labels[self.train_labels.index.get_level_values("file_i").isin(scope_idx)]
            self.test_labels = self.test_labels[self.test_labels.index.get_level_values("file_i").isin(scope_idx)]

        self.metadata["file_path"] = self.base_dir / self.metadata["stage"] / "data" / self.metadata["file_name"]
        self.train_metadata["file_path"] = self.base_dir / "train" / "data" / self.train_metadata["file_name"]
        self.test_metadata["file_path"] = self.base_dir / "test" / "data" / self.test_metadata["file_name"]

        if test == True:
            self.s = self.test_metadata.index
            self.x = self.test_metadata.file_path.to_numpy()
            self.y = self.test_labels.to_numpy()
        elif test == False:
            self.s = self.train_metadata.index
            self.x = self.train_metadata.file_path.to_numpy()
            self.y = self.train_labels.to_numpy()
        else:
            self.s = self.metadata.index
            self.x = self.metadata.file_path.to_numpy()
            self.y = self.labels.to_numpy()

    @property
    def target_names(self):
        return self.labels.columns.tolist()

    @property
    def model_params(self) -> Dict:
        return {}

    def load_sample(self, file_path: str) -> torch.Tensor:
        metadata = torchaudio.info(file_path)
        num_frames_segment = int(self.num_frames_in_segment / self.sample_rate * metadata.sample_rate)
        high = max(1, metadata.num_frames - num_frames_segment)
        frame_offset = torch.randint(low=0, high=high, size=(1,))
        waveform, _ = torchaudio.load(file_path, num_frames=num_frames_segment)
        return torchaudio.functional.resample(waveform, orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

    def _download_files(self):
        import requests
        import zipfile
        url = "https://sussex.box.com/s/h8f4zdfrhljlfm5615se7lsxwo3q72um"
        try:
            response = requests.get(url, stream=True)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
        zip_path = self.base_dir.parent / "sounding_out.zip"
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=512):
                f.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(path=self.base_dir)
        zip_path.unlink()

    def _split_testset(self):
        if (self.base_dir / f"train_indices.parquet").exists(): return
        metadata = pd.read_parquet(self.base_dir / "metadata.parquet", index=True)
        train_idx, test_idx = sklearn.model_selection.train_test_split(metadata.file_i.to_numpy(), test_size=0.2, random_state=self.seed)
        metadata[train_idx, ["file_i", "file_name"]].to_parquet(self.base_dir / "train_indices.parquet")
        metadata[test_idx, ["file_i", "file_name"]].to_parquet(self.base_dir / "test_indices.parquet")


@attrs.define(kw_only=True)
class SoundingOutChorusDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    segment_len: float = attrs.field(default=60.0)
    sample_rate: int = attrs.field(default=48_000)
    transforms: Callable = attrs.field(default=T.Compose([]))
    scope: str | None = attrs.field(default=None)

    train_batch_size: int = attrs.field(default=6)
    eval_batch_size: int | None = attrs.field(default=None)
    val_prop: float = attrs.field(default=0.2)
    test_prop: float = attrs.field(default=0.2)
    num_workers: int = attrs.field(default=0)
    seed: int = attrs.field(default=47)
    persist_workers: bool = attrs.field(default=False)
    pin_memory: bool = attrs.field(default=True)
    training_mode: str = attrs.field(default="step")

    data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)
    train_data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)
    val_data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)
    test_data: torch.utils.data.Dataset | None = attrs.field(default=None, init=False)
    sampler: Callable = attrs.field(default=None, init=False)

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)
        self.training_mode = ranzen.torch.TrainingMode[self.training_mode]

    def prepare_data(self):
        SoundingOutChorus(root=self.root, download=True)
        return self

    def setup(self, stage: str):
        self.data = SoundingOutChorus(self.root, test=False, download=False, **self.dataset_params)
        self.val_data, self.train_data = torch.utils.data.random_split(self.data, (self.val_prop, 1 - self.val_prop), generator=self.generator)
        self.test_data = SoundingOutChorus(self.root, test=True, download=False, **self.dataset_params)
        return self

    @property
    def generator(self):
        return torch.Generator().manual_seed(self.seed)

    @property
    def dataset_params(self):
        return dict(
            transforms=self.transforms,
            segment_len=self.segment_len,
            sample_rate=self.sample_rate,
            scope=self.scope,
        )

    def train_dataloader_params(self, batch_size: int | None = None) -> Dict[str, Any]:
        if self.training_mode == ranzen.torch.TrainingMode.step:
            return dict(batch_size=batch_size, batch_sampler=self._default_train_sampler(batch_size))
        else:
            return dict(batch_size=batch_size, shuffle=True, generator=self.generator, drop_last=False)

    def train_dataloader(self, batch_size: int | None = None, batch_sampler: torch.utils.data.Sampler | None = None) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.train_data, **self.train_dataloader_params(batch_size))

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size)

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
