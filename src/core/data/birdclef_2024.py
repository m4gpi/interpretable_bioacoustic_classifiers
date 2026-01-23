import attrs
import librosa
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

from src.core.utils import Batch

__all__ = [
    "BirdClef2024",
    "BirdClef2024DataModule",
]

class BirdClef2024Soundscapes(torch.utils.data.Dataset):
    _AUDIO_SAMPLE_RATE: int = 32_000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.metadata.iloc[idx]
        file_path = (self.base_dir / record.file_path)
        wav = self.load_sample(file_path, frame_offset=self.sample_rate * record.t_start)
        return self.transforms(wav), self.metadata.index[idx]

    def __len__(self):
        return len(self.metadata)

    def __init__(
        self,
        root: str,
        *,
        scope: str | None = None,
        sample_rate: int = 32_000,
        segment_len: float = 19.968,
        transforms: Callable = lambda x: x,
        download: bool = False,
        reset_index: bool = True,
        seed: int = 42,
    ) -> None:
        self.base_dir = pathlib.Path(root).expanduser()
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.num_frames_in_segment = int(self.segment_len * self.sample_rate)
        self.transforms = transforms
        # fetch from remote and perform initial data split
        if download:
            self._download_files()
        # rebuild the data split if specified
        self._extract_metadata()
        # load file index
        self.metadata = pd.read_parquet(self.base_dir / "soundscape_metadata.parquet")

    @property
    def target_names(self):
        return self.labels.columns.tolist()

    @property
    def model_params(self) -> Dict:
        return {}

    def load_sample(self, file_path: pathlib.Path, frame_offset: int = 0):
        metadata = torchaudio.info(file_path)
        num_frames = int(metadata.sample_rate * self.segment_len)
        waveform, _ = torchaudio.load(file_path, frame_offset=frame_offset, num_frames=num_frames)
        return torchaudio.functional.resample(waveform, orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

    def _download_files(self):
        import kagglehub
        kagglehub.competition_download("birdclef-2024", path="unlabeled_soundscapes")
        cached_dir = pathlib.Path.home() / ".cache" / "kagglehub" / "competitions" / "birdclef-2024"
        shutil.move(cached_dir, self.base_dir)

    def _extract_metadata(self):
        if (self.base_dir / f"soundscape_metadata.parquet").exists(): return
        data = []
        for file_path in (self.base_dir / "unlabeled_soundscapes").rglob('*.ogg'):
            data.append({
                "file_path": str(file_path.relative_to(self.base_dir)),
                "file_name": str(file_path.relative_to(self.base_dir / "unlabeled_soundscapes")),
                "duration_seconds": librosa.get_duration(path=self.base_dir / "unlabeled_soundscapes" / file_path),
            })
        metadata = pd.DataFrame(data=data).reset_index(names=["file_i"]).set_index("file_i")
        metadata["num_segments"] = (metadata["duration_seconds"] // self.segment_len).astype(int)
        metadata = metadata.loc[metadata.index.repeat(metadata["num_segments"])].reset_index(drop=True)
        metadata["t"] = metadata.groupby("file_name").cumcount()
        metadata["t_start"] = metadata["t"] * self.segment_len
        metadata["t_end"] = metadata["t_start"] + self.segment_len
        metadata["duration_seconds"] = self.segment_len
        metadata.to_parquet(self.base_dir / "soundscape_metadata.parquet")

    def _split_testset(self, seed: int):
        if (self.base_dir / f"train_indices.parquet").exists(): return
        metadata = pd.read_parquet(self.base_dir / "metadata.parquet")
        train_idx, test_idx = sklearn.model_selection.train_test_split(metadata.index.to_numpy(), test_size=0.2, random_state=seed)
        metadata.loc[train_idx, ["file_path"]].reset_index(names=["file_i"]).to_parquet(self.base_dir / "train_indices.parquet")
        metadata.loc[test_idx, ["file_path"]].reset_index(names=["file_i"]).to_parquet(self.base_dir / "test_indices.parquet")

class BirdClef2024(torch.utils.data.Dataset):
    _AUDIO_SAMPLE_RATE: int = 32_000
    _MAX_AUDIO_LEN: float = 240.0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.transforms(self.load_sample(self.x[idx])), self.s[idx]

    def __len__(self):
        return len(self.x)

    def __init__(
        self,
        root: str,
        *,
        scope: str | None = None,
        sample_rate: int = 32_000,
        segment_len: float = 240.0,
        transforms: Callable = lambda x: x,
        download: bool = False,
        reset_index: bool = True,
        unlabelled_only: bool = True,
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
        # rebuild the data split if specified
        self._extract_metadata()
        self._extract_labels()
        self._split_testset(seed=seed)
        # load file info
        self.metadata = pd.read_parquet(self.base_dir / "metadata.parquet")
        # load the labels, pivot so species are on columns, counts as presence/absence
        self.labels = pd.read_parquet(self.base_dir / "labels.parquet")
        self.labels["counts"] = 1.0
        index = ["file_i", "file_path", "file_name"]
        self.labels = (
            self.labels
            .pivot(index=index, columns="species_name", values="counts")
            .fillna(0.0)
            .astype(bool)
            .astype(int)
            .reset_index()
        )
        if include_unlabelled:
            # pad labels for the sake of loading but assume empty since we can't ground truth these files
            self.labels = self.labels.merge(self.metadata.reset_index(names=["file_i"])[index], on=index, how="outer")
        self.labels = self.labels.set_index(index).fillna(0.0).astype(bool).astype(int)
        # scope by train / test
        train_idx = pd.read_parquet(self.base_dir / "train_indices.parquet")
        test_idx = pd.read_parquet(self.base_dir / "test_indices.parquet")
        self.train_metadata = self.metadata.loc[self.metadata.index.isin(train_idx.file_i)]
        self.train_labels = self.labels.loc[train_idx.file_i]
        self.test_metadata = self.metadata.loc[self.metadata.index.isin(test_idx.file_i)]
        self.test_labels = self.labels.loc[test_idx.file_i]

        if test == True:
            self.x = (self.base_dir / self.test_metadata.file_path).to_numpy()
            self.y = self.test_labels.to_numpy()
            self.s = self.test_metadata.index.to_numpy()
        elif test == False:
            self.x = (self.base_dir / self.train_metadata.file_path).to_numpy()
            self.y = self.train_labels.to_numpy()
            self.s = self.train_metadata.index.to_numpy()
        else:
            self.x = (self.base_dir / self.metadata.file_path).to_numpy()
            self.y = self.labels.to_numpy()
            self.s = self.metadata.index.to_numpy()

    @property
    def target_names(self):
        return self.labels.columns.tolist()

    @property
    def model_params(self) -> Dict:
        return {}

    def load_sample(self, file_path: pathlib.Path):
        metadata = torchaudio.info(file_path)
        num_frames_segment = int(self.num_frames_in_segment / self.sample_rate * metadata.sample_rate)
        high = max(1, metadata.num_frames - num_frames_segment)
        frame_offset = torch.randint(low=0, high=high, size=(1,))
        waveform, _ = torchaudio.load(file_path, num_frames=num_frames_segment)
        return torchaudio.functional.resample(waveform, orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

    def _download_files(self):
        import kagglehub
        kagglehub.competition_download("birdclef-2024", path="train_audio")
        kagglehub.competition_download("birdclef-2024", path="unlabeled_soundscapes")
        kagglehub.competition_download("birdclef-2024", path="train_metadata.csv")
        cached_dir = pathlib.Path.home() / ".cache" / "kagglehub" / "competitions" / "birdclef-2024"
        shutil.move(cached_dir, self.base_dir)

    def _extract_metadata(self):
        if (self.base_dir / f"metadata.parquet").exists(): return
        labelled = []
        for file_path in (self.base_dir / "train_audio").rglob('*.*'):
            labelled.append({
                "file_path": str(file_path.relative_to(self.base_dir)),
                "file_name": str(file_path.relative_to(self.base_dir / "train_audio")),
                "duration_seconds": librosa.get_duration(path=self.base_dir / "train_audio" / file_path),
                "stage": "ANY"
            })
        unlabelled = []
        for file_path in (self.base_dir / "unlabeled_soundscapes").rglob('*.*'):
            unlabelled.append({
                "file_path": str(file_path.relative_to(self.base_dir)),
                "file_name": str(file_path.relative_to(self.base_dir / "unlabeled_soundscapes")),
                "duration_seconds": librosa.get_duration(path=self.base_dir / "unlabeled_soundscapes" / file_path),
                "stage": "PTO"
            })
        metadata = pd.DataFrame(data=[*labelled, *unlabelled])
        metadata = metadata.reset_index(names=["file_i"]).set_index("file_i")
        metadata.index.name = "file_i"
        df = pd.read_csv(self.base_dir / "train_metadata.csv")
        metadata = metadata.merge(df[["filename", "url", "latitude", "longitude"]], how="left", left_on="file_name", right_on="filename").drop("filename", axis=1)
        metadata.to_parquet(self.base_dir / "metadata.parquet")

    def _extract_labels(self):
        if (self.base_dir / f"labels.parquet").exists(): return
        labels = pd.read_csv(self.base_dir / "train_metadata.csv")
        labels["file_name"] = labels["filename"]
        labels["file_path"] = labels["file_name"].map(lambda file_name: f"train_audio/{file_name}")
        label_reference_df = pd.read_csv(self.base_dir / "eBird_Taxonomy_v2021.csv")
        label_reference_df["species_name"] = label_reference_df["SCI_NAME"] + "_" + label_reference_df["PRIMARY_COM_NAME"]
        labels = labels.merge(label_reference_df[["SPECIES_CODE", "species_name"]], left_on=["primary_label"], right_on="SPECIES_CODE", how="left")
        labels = labels[["file_name", "file_path", "species_name"]]
        metadata = pd.read_parquet(self.base_dir / "metadata.parquet").reset_index(names=["file_i"])
        labels = labels.merge(metadata[["file_i", "file_name"]], on="file_name", how="inner")
        labels.to_parquet(self.base_dir / "labels.parquet")

    def _split_testset(self, seed: int):
        if (self.base_dir / f"train_indices.parquet").exists(): return
        metadata = pd.read_parquet(self.base_dir / "metadata.parquet")
        train_idx, test_idx = sklearn.model_selection.train_test_split(metadata.index.to_numpy(), test_size=0.2, random_state=seed)
        metadata.loc[train_idx, ["file_path"]].reset_index(names=["file_i"]).to_parquet(self.base_dir / "train_indices.parquet")
        metadata.loc[test_idx, ["file_path"]].reset_index(names=["file_i"]).to_parquet(self.base_dir / "test_indices.parquet")


@attrs.define(kw_only=True)
class BirdClef2024DataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    sample_rate: int = 32_000
    segment_len: float = 19.968
    transforms: Callable = attrs.field(default=T.Compose([]))

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

    def _batch_converter(self, batch: Tuple):
        xs, ss = zip(*batch)
        return Batch(x=torch.stack(xs, dim=0), s=torch.tensor(ss))

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)
        self.training_mode = ranzen.torch.TrainingMode[self.training_mode]

    def prepare_data(self):
        BirdClef2024Soundscapes(root=self.root, download=True)
        return self

    def setup(self, stage: str = "fit"):
        self.data = BirdClef2024Soundscapes(self.root, download=False, transforms=self.transforms)
        self.val_data, self.train_data = torch.utils.data.random_split(self.data, (self.val_prop, 1 - self.val_prop), generator=self.generator)
        return self

    @property
    def generator(self):
        return torch.Generator().manual_seed(self.seed)

    def train_dataloader_params(self, batch_size: int | None = None) -> Dict[str, Any]:
        if self.training_mode == ranzen.torch.TrainingMode.step:
            return dict(batch_size=batch_size, batch_sampler=self._default_train_sampler(batch_size))
        else:
            return dict(batch_size=batch_size, shuffle=True, generator=self.generator, drop_last=False)

    def train_dataloader(self, batch_size: int | None = None, batch_sampler: torch.utils.data.Sampler | None = None) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.train_data, **self.train_dataloader_params(batch_size))

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

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
