import attrs
import pathlib
import lightning as L
import numpy as np
import pandas as pd
import sklearn
import torch

from typing import Any, Callable, Dict, List, Tuple

from src.core.utils import Batch

__all__ = [
    "SoundscapeCodes",
    "SoundscapeCodesDataModule",
]

@attrs.define(kw_only=True)
class SoundscapeCodes(torch.utils.data.Dataset):
    num_classes: int = 512
    features: pd.DataFrame = attrs.field()
    download: bool = attrs.field(default=False)

    x: torch.Tensor = attrs.field(init=False)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (self.x[idx], self.s[idx])

    def __attrs_post_init__(self):
        if self.download:
            self._download_files()
        indices = torch.tensor(self.features[self.index_columns].values, dtype=torch.long)
        self.x = indices.reshape(-1, self.seq_len, *indices.shape[1:])
        self.s = self.features["file_i"].unique()

    @property
    def index_columns(self):
        return [str(i) for i in range(16)]

    @property
    def seq_len(self):
        return self.features.timestep.max() + 1

    @property
    def model_params(self):
        return dict()

    # def _download_files(self):
    #     import requests
    #     import zipfile
    #     url = "https://sussex.box.com/s/1ob205h3t6wce8igt60vl360gycqv37o"
    #     try:
    #         response = requests.get(url, stream=True)
    #     except requests.exceptions.RequestException as e:
    #         raise SystemExit(e)
    #     zip_path = self.base_dir.parent / "soundscape_embeddings.zip"
    #     with open(zip_path, 'wb') as f:
    #         for chunk in response.iter_content(chunk_size=512):
    #             f.write(chunk)
    #     with zipfile.ZipFile(zip_path) as zf:
    #         zf.extractall(path=self.base_dir)
    #     zip_path.unlink()

@attrs.define(kw_only=True)
class SoundscapeCodesDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    num_classes: int = 512
    transforms: Callable | None = None

    train_batch_size: int | None = attrs.field(default=None)
    eval_batch_size: int | None = attrs.field(default=None)
    val_prop: float = attrs.field(default=0.2, validator=attrs.validators.instance_of(float))

    seed: int = attrs.field(default=None)
    num_workers: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    persist_workers: bool | None = attrs.field(default=None)
    pin_memory: bool = attrs.field(default=True, validator=attrs.validators.instance_of(bool))

    generator: torch.Generator = attrs.field(init=False)

    data: torch.utils.data.Dataset = attrs.field(init=False)
    train_data: torch.utils.data.Subset = attrs.field(init=False)
    val_data: torch.utils.data.Subset = attrs.field(init=False)
    test_data: torch.utils.data.Dataset = attrs.field(init=False)

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)

    @property
    def dataloader_params(self) -> Dict[str, Any]:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persist_workers,
        )

    def setup(self, stage: str | None = None) -> None:
        self.features = pd.read_parquet(self.root / "features.parquet").reset_index()
        self.data = SoundscapeCodes(features=self.features, num_classes=self.num_classes)
        self.train_data = SoundscapeCodes(features=self.features[self.features["dataloader_idx"] == 0], num_classes=self.num_classes)
        self.val_data = SoundscapeCodes(features=self.features[self.features["dataloader_idx"] == 1], num_classes=self.num_classes)
        self.test_data = SoundscapeCodes(features=self.features[self.features["dataloader_idx"] == 2], num_classes=self.num_classes)
        return self

    def train_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.train_data, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

    def predict_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        batch_size = batch_size or self.eval_batch_size or len(self.test_data)
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

    def batch_converter(self, batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        xs, ss = zip(*batch)
        return Batch(x=torch.stack(xs), s=torch.tensor(ss))

    def _build_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or len(dataset),
            collate_fn=self.batch_converter,
            **self.dataloader_params,
            **kwargs
        )
