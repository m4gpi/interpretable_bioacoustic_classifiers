import attrs
import lightning as L
import numpy as np
import os
import pandas as pd
import pathlib
import ranzen
import ranzen.torch
import re
import shutil
import sklearn
import torch
import torchaudio
import zipfile

from torchvision import transforms as T
from typing import Any, Callable, ClassVar, Dict, Final, List, Tuple

from src.core.utils import Batch

__all__ = [
    "RainforestConnection",
    "RainforestConnectionDataModule",
]

class RainforestConnection(torch.utils.data.Dataset):
    _LABEL_FILENAME: ClassVar[str] = "presences.parquet"
    _ADDITIONAL_LABEL_FILENAME: ClassVar[str] = "absences.parquet"
    _TRUE_POS_FILE: ClassVar[str] = "train_tp.csv"
    _FALSE_POS_FILE: ClassVar[str] = "train_fp.csv"
    _TRAIN_DATA_DIR: ClassVar[str] = "train"
    _TEST_DATA_DIR: ClassVar[str] = "test"
    _ENCODING: ClassVar[str] = ".flac"

    num_frames_in_segment: int
    _MAX_AUDIO_LEN: Final[int] = 60
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
        test: bool = False,
        seed: int = 42,
    ) -> None:
        self.base_dir = pathlib.Path(root).expanduser()
        self.data_dir = self.base_dir / self._TRAIN_DATA_DIR
        self.sample_rate = sample_rate
        self.segment_len = min(segment_len, self._MAX_AUDIO_LEN)
        self.num_frames_in_segment = int(self.segment_len * self.sample_rate)
        self.transforms = transforms
        # download through kaggle API
        if download:
            self._download_files()
            self._extract_metadata()
            self._extract_labels()
            self._split_testset()
        # rebuild the data split if specified
        if reset_index:
            self._reset_index()
        # check files are present and accounted for
        self._check_files()
        # load metadata and labels
        self.metadata = pd.read_parquet(self.base_dir / f"metadata.parquet")
        self.labels = pd.read_parquet(self.base_dir / f"labels.parquet")
        # scope the dataset by train / test
        self.test_metadata = pd.read_parquet(self.base_dir / f"test_metadata.parquet")
        self.test_metadata.index.name = "file_i"
        self.test_labels = pd.read_parquet(self.base_dir / f"test_labels.parquet")
        self.test_labels.drop("species_id", axis=1, inplace=True)
        self.train_metadata = pd.read_parquet(self.base_dir / f"train_metadata.parquet")
        self.train_metadata.index.name = "file_i"
        self.train_labels = pd.read_parquet(self.base_dir / f"train_labels.parquet")
        self.train_labels.drop("species_id", axis=1, inplace=True)
        # scope by taxa
        if scope is not None:
            self.train_labels = self.train_labels[self.train_labels.taxa == scope.split("_")[-1]]
            self.test_labels = self.test_labels[self.test_labels.taxa == scope.split("_")[-1]]

        self.train_labels = self.format_labels(self.train_metadata, self.train_labels)
        self.test_labels = self.format_labels(self.test_metadata, self.test_labels)

        self.metadata["file_path"] = self.data_dir / self.metadata["file_name"]
        self.train_metadata["file_path"] = self.data_dir / self.train_metadata["file_name"]
        self.test_metadata["file_path"] = self.data_dir / self.test_metadata["file_name"]

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
    def target_names(self):
        return self.labels.columns.tolist()

    @property
    def model_params(self) -> Dict:
        return {}

    def format_labels(self, metadata, labels):
        # count occurrences and drop duplicates
        labels = labels.groupby(["file_i", "file_name", "taxa", "species_name"]).size().reset_index(name='counts')
        # pad with missing file names
        indices = ["file_i", "file_name", "taxa", "species_name"]
        file_list = metadata[["file_i", "file_name"]].drop_duplicates()
        species_list = labels[["taxa", "species_name"]].drop_duplicates()
        labels = file_list.merge(species_list, how="cross").merge(labels, on=indices, how="left").fillna(0.0)
        # expand species across columns
        labels = labels.pivot(index=["file_i", "file_name", "taxa"], columns="species_name", values="counts")
        # labels are binary presence/absence for each species
        labels = labels.fillna(0.0).astype(bool).astype(int)
        # sort indices so they align as matrices
        metadata = metadata.sort_index()
        labels = labels.sort_index()
        return labels

    def load_sample(self, file_name: str) -> torch.Tensor:
        file_path = self.data_dir / file_name
        metadata = torchaudio.info(str(file_path))
        num_frames_segment = int(self.num_frames_in_segment / self.sample_rate * metadata.sample_rate)
        high = max(1, metadata.num_frames - num_frames_segment)
        frame_offset = torch.randint(low=0, high=high, size=(1,))
        waveform, _ = torchaudio.load(str(file_path), num_frames=num_frames_segment)
        return torchaudio.functional.resample(waveform, orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

    def _check_files(self) -> None:
        """assert files exist and zip unpacked"""
        assert self.data_dir.exists(), \
            f"data not found at location {path.resolve()}. have you downloaded it?"
        assert (self.data_dir / "metadata.parquet").exists(), \
            f"metadata.parquet not found at location {path.resolve()}. have you downloaded it?"
        assert (self.data_dir / "labels.parquet").exists(), \
            f"labels.parquet not found at location {path.resolve()}. have you downloaded it?"
        assert (self.data_dir / "train_labels.parquet").exists() and (self.data_dir / "test_labels.parquet").exists() \
            f"'train_labels.parquet' or 'test_labels.parquet' not found at location {path.resolve()}. Pass 'reset_index=True seed=42' to rebuild the data split"

    def _reset_index(self):
        for dataset in ["train", "test"]:
            (self.base_dir / f"{dataset}_metadata.parquet").unlink(missing_ok=True)
        self._split_testset()

    def _download_files(self):
        import kagglehub
        kagglehub.competition_download("rfcx-species-audio-detection", path="train")
        kagglehub.competition_download("rfcx-species-audio-detection", path="train_tp.csv")
        cached_dir = pathlib.Path.home() / ".cache" / "kagglehub" / "competitions" / "rfcx-species-audio-detection"
        shutil.move(cached_dir, self.base_dir)

    def _extract_metadata(self):
        if (self.base_dir / f"metadata.parquet").exists(): return
        file_list = data_dir.glob("*.flac")
        file_ids = np.arange(file_list)
        metadata = pd.DataFrame(data=zip(file_ids, file_list), columns=["file_i", "file_name"]).set_index("file_i")
        metadata.to_parquet(data_dir / "metadata.parquet", index=True)

    def _split_testset(self):
        if (self.base_dir / f"train_metadata.parquet").exists(): return
        metadata = pd.read_parquet(self.base_dir / "metadata.parquet", index=True)
        train_idx, test_idx = sklearn.model_selection.train_test_split(metadata.index, test_size=0.2, random_state=self.seed)
        train_metadata, test_metadata = metadata.loc[train_idx], metadata.loc[test_idx]
        train_metadata.to_parquet(data_dir / "train_metadata.parquet", index=True)
        test_metadata.to_parquet(data_dir / "test_metadata.parquet", index=True)

    def _extract_labels(self):
        if (self.base_dir / f"train_labels.parquet").exists(): return
        # align data annotations
        metadata = pd.read_parquet(data_dir / "metadata.parquet").set_index("file_name")
        labels = pd.read_csv(data_dir / "labels" / "train_tp.csv")
        labels["species_name"] = labels.species_id.map(lambda species_id: self.species_key.loc[species_id].species_name)
        labels["taxa"] = labels.species_id.map(lambda species_id: self.species_key.loc[species_id].taxon.lower())
        labels["file_name"] = labels.recording_id + ".flac"
        labels["file_i"] = labels.file_name.map(lambda f: metadata.loc[f].file_i)
        labels = labels.set_index("file_i")
        labels = labels.drop(["recording_id", "species_id"], axis=1)
        train_metadata = pd.read_parquet(data_dir / "train_metadata.parquet").set_index("file_i")
        test_metadata = pd.read_parquet(data_dir / "test_metadata.parquet").set_index("file_i")
        labels[labels.index.isin(train_metadata.index)].to_parquet(data_dir / "train_labels.parquet", index=True)
        labels[labels.index.isin(test_metadata.index)].to_parquet(data_dir / "test_labels.parquet", index=True)

    def species_key(self) -> pd.DataFrame:
        return pd.DataFrame([
            {'id': 18, 'species_name': 'Eleutherodactylus unicolor_Dwarf coquí', 'code': 'ELUN', 'taxon': 'frog'},
            {'id': 1, 'species_name': 'Eleutherodactylus brittoni_Grass coquí', 'code': 'ELBR', 'taxon': 'frog'},
            {'id': 21, 'species_name': 'Eleutherodactylus wightmanae_Melodius coquí', 'code': 'ELWI', 'taxon': 'frog'},
            {'id': 3, 'species_name': 'Eleutherodactylus coqui_Common coquí', 'code': 'ELCO', 'taxon': 'frog'},
            {'id': 4, 'species_name': "Eleutherodactylus hedricki_Hedrick's coquí", 'code': 'ELHE', 'taxon': 'frog'},
            {'id': 0, 'species_name': 'Eleutherodactylus gryllus_Cricket coquí', 'code': 'ELGR', 'taxon': 'frog'},
            {'id': 14, 'species_name': 'Eleutherodactylus richmondi_Bronze coquí', 'code': 'ELRI', 'taxon': 'frog'},
            {'id': 12, 'species_name': 'Eleutherodactylus portoricensis_Forest Coqui', 'code': 'ELPO', 'taxon': 'frog'},
            {'id': 8, 'species_name': 'Eleutherodactylus locustus_Locust coquí', 'code': 'ELLO', 'taxon': 'frog'},
            {'id': 16, 'species_name': 'Eleutherodactylus antillensis_Red-eyed coquí', 'code': 'ELAN', 'taxon': 'frog'},
            {'id': 2, 'species_name': 'Leptodactylus albilabris_Caribbean White-lipped Frog', 'code': 'LEAL', 'taxon': 'frog'},
            {'id': 11, 'species_name': 'Vireo altiloquus_Black-whiskered Vireo', 'code': 'VIAL', 'taxon': 'bird'},
            {'id': 10, 'species_name': 'Melopyrrha portoricensis_Puerto Rican Bullfinch', 'code': 'LOPO', 'taxon': 'bird'},
            {'id': 15, 'species_name': 'Patagioenas squamosa_Scaly-naped Pigeon', 'code': 'PASQ', 'taxon': 'bird'},
            {'id': 23, 'species_name': 'Spindalis portoricensis_Puerto Rican Spindalis', 'code': 'SPPO', 'taxon': 'bird'},
            {'id': 22, 'species_name': 'Nesospingus speculiferus_Puerto Rican Tanager', 'code': 'NEES', 'taxon': 'bird'},
            {'id': 13, 'species_name': 'Gymnasio nudipes_Puerto Rican Owl', 'code': 'MENU', 'taxon': 'bird'},
            {'id': 9, 'species_name': 'Margarops fuscatus_Pearly-eyed Thrasher', 'code': 'MAFU', 'taxon': 'bird'},
            {'id': 5, 'species_name': 'Setophaga angelae_Elfin Woods Warbler', 'code': 'SEAN', 'taxon': 'bird'},
            {'id': 17, 'species_name': 'Turdus plumbeus_Red-legged Thrush', 'code': 'TUPL', 'taxon': 'bird'},
            {'id': 6, 'species_name': 'Melanerpes portoricensis_Puerto Rican Woodpecker', 'code': 'MEPO', 'taxon': 'bird'},
            {'id': 20, 'species_name': 'Todus mexicanus_Puerto Rican Tody', 'code': 'TOME', 'taxon': 'bird'},
            {'id': 7, 'species_name': 'Coereba flaveola_Bananaquit', 'code': 'COFL', 'taxon': 'bird'},
            {'id': 19, 'species_name': 'Coccyzus vieilloti_Puerto Rican Lizard-Cuckoo', 'code': 'COVI', 'taxon': 'bird'},
        ]).set_index("id")


@attrs.define(kw_only=True)
class RainforestConnectionDataModule(L.LightningDataModule):
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

    def _batch_converter(self, batch: Tuple):
        xs, ys, ss = zip(*batch)
        return Batch(x=torch.stack(xs, dim=0), y=torch.stack(ys, dim=0), s=torch.tensor(ss))

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)
        self.training_mode = ranzen.torch.TrainingMode[self.training_mode]

    def prepare_data(self):
        RainforestConnection(root=self.root, download=True)
        return self

    def setup(self, stage: str):
        self.data = RainforestConnection(self.root, test=False, download=False, **self.dataset_params)
        self.val_data, self.train_data = torch.utils.data.random_split(self.data, (self.val_prop, 1 - self.val_prop), generator=self.generator)
        self.test_data = RainforestConnection(self.root, test=True, download=False, **self.dataset_params)
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
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

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

