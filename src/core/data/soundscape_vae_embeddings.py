import attrs
import pathlib
import lightning as L
import numpy as np
import pandas as pd
import torch

from typing import Any, Dict, List, Tuple

__all__ = [
    "VAESoundscapeEmbeddings",
    "VAESoundscapeEmbeddingsDataModule",
]

@attrs.define(kw_only=True)
class SoundscapeVAEEmbeddings(torch.utils.data.Dataset):
    features: pd.DataFrame = attrs.field()
    labels: pd.DataFrame = attrs.field()
    index: List[int] = attrs.field()
    num_samples: int = attrs.field(default=1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        q_z = self.x[idx]
        mean, log_var = q_z.chunk(2, dim=-1)
        mean = mean.unsqueeze(1).expand(-1, self.num_samples, -1, -1)
        log_var = log_var.unsqueeze(1).expand(-1, self.num_samples, -1, -1)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        return (z, self.y[idx], self.index[idx])

    def __attrs_post_init__(self):
        self.x = torch.tensor(self.features.values.reshape(self.labels.values.shape[0], -1, self.features.values.shape[-1]), dtype=torch.float32)
        self.y = torch.tensor(self.labels.values, dtype=torch.int64)
        self.y_freq = zip(dict(self.labels.columns, [self.labels[y].sum() for y in self.labels.columns]))

@attrs.define(kw_only=True)
class SoundscapeVAEEmbeddingsDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    train_batch_size: int | None = attrs.field(default=None)
    eval_batch_size: int | None = attrs.field(default=None)
    train_sample_size: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    eval_sample_size: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    val_prop: float = attrs.field(default=0.0, validator=attrs.validators.instance_of(float))
    seed: int = attrs.field(default=8, validator=attrs.validators.instance_of(int))
    num_workers: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    persist_workers: bool | None = attrs.field(default=None)
    pin_memory: bool = attrs.field(default=True, validator=attrs.validators.instance_of(bool))
    drop_last: bool = attrs.field(default=False, validator=attrs.validators.instance_of(bool))

    @root.validator
    def check_features_and_labels_exist(self, attribute, value) -> None:
        assert pathlib.Path(self.train_features_path).exists(), f"'{self.train_features_path}' does not exist"
        assert pathlib.Path(self.train_labels_path).exists(), f"'{self.train_labels_path}' does not exist"
        assert pathlib.Path(self.test_features_path).exists(), f"'{self.test_features_path}' does not exist"
        assert pathlib.Path(self.test_labels_path).exists(), f"'{self.test_labels_path}' does not exist"

    def __attrs_post_init__(self):
        self.generator = torch.Generator().manual_seed(self.seed)

    @property
    def train_features_path(self):
        return self.root / "train" / "features.parquet"

    @property
    def test_features_path(self):
        return self.root / "test" / "features.parquet"

    @property
    def train_labels_path(self):
        return self.root / "train" / "labels.parquet"

    @property
    def test_labels_path(self):
        return self.root / "test" / "labels.parquet"

    @property
    def dataloader_params(self) -> Dict[str, Any]:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persist_workers,
        )

    def setup(self):
        train_features = pd.read_parquet(self.train_features_path)
        train_labels = pd.read_parquet(self.train_labels_path)
        test_features = pd.read_parquet(self.test_features_path)
        test_labels = pd.read_parquet(self.test_labels_path)
        labels = set(train_labels.columns).intersection(set(test_labels.columns))
        train_data = SoundscapeVAEEmbeddings(
            features=train_features,
            labels=train_labels[labels],
            index=train_labels.index.get_level_values(0),
            num_samples=self.train_sample_size,
        )
        self.train_data, self.val_data = torch.utils.data.random_split(train_data, (1 - self.val_prop, self.val_prop), generator=self.generator)
        self.test_data = SoundscapeVAEEmbeddings(
            features=test_features,
            labels=test_labels[labels],
            index=test_labels.index.get_level_values(0),
            num_samples=self.eval_sample_size,
        )

    def train_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        batch_size = batch_size or self.train_batch_size or len(self.train_data)
        return torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, collate_fn=self.batch_converter, **self.dataloader_params, **kwargs)

    def val_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        batch_size = batch_size or self.eval_batch_size or len(self.val_data)
        return torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, collate_fn=self.batch_converter, **self.dataloader_params, **kwargs)

    def test_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        batch_size = batch_size or self.eval_batch_size or len(self.test_data)
        return torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, collate_fn=self.batch_converter, **self.dataloader_params, **kwargs)

    @staticmethod
    def batch_converter(batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        xs, ys, idx = zip(*batch)
        return (torch.stack(xs), torch.stack(ys), torch.stack(idx), self.train_data.dataset.y_freq)
