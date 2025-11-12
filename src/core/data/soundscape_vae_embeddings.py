import attrs
import pathlib
import lightning as L
import numpy as np
import pandas as pd
import sklearn
import torch

from typing import Any, Dict, List, Tuple

from src.core.utils import tree

__all__ = [
    "SoundscapeVAEEmbeddings",
    "SoundscapeVAEEmbeddingsDataModule",
]

@attrs.define(kw_only=True)
class SoundscapeVAEEmbeddings(torch.utils.data.Dataset):
    features: pd.DataFrame = attrs.field()
    labels: pd.DataFrame = attrs.field()
    index: List[int] = attrs.field()
    num_samples: int = attrs.field(default=1)

    x: torch.Tensor = attrs.field(init=False)
    y: torch.Tensor = attrs.field(init=False)
    y_freq: torch.Tensor = attrs.field(init=False)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        q_z = self.x[idx]
        mean, log_var = q_z.chunk(2, dim=-1)
        mean = mean.unsqueeze(0).expand(self.num_samples, -1, -1)
        log_var = log_var.unsqueeze(0).expand(self.num_samples, -1, -1)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        return (z, self.y[idx], self.index[idx])

    def __attrs_post_init__(self):
        self.x = torch.tensor(self.features.values.reshape(self.labels.values.shape[0], -1, self.features.values.shape[-1]), dtype=torch.float32)
        self.y = torch.tensor(self.labels.values, dtype=torch.int64)
        self.y_freq = dict(zip(self.labels.columns, torch.tensor([self.labels[y].sum() for y in self.labels.columns])))

@attrs.define(kw_only=True)
class SoundscapeVAEEmbeddingsDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    model: str = attrs.field(default=None)
    scope: str = attrs.field(default=None)
    version: str = attrs.field(default=None)

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

    generator: torch.Generator = attrs.field(init=False)

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)
        self.generator = torch.Generator().manual_seed(self.seed)

    @model.validator
    def check_model_is_valid_if_not_none(self, attribute, value):
        return value in ["base_vae", "smooth_nifti_vae", "nifti_vae"] if value is not None else True

    @scope.validator
    def check_scope_is_valid_if_not_none(self, attribute, value):
        return value in ["EC", "UK", "bird", "frog"] if value is not None else True

    @version.validator
    def check_version_is_valid_if_not_none(self, attribute, value):
        return (value[0] == "v" and int(value[1:])) if value is not None else True

    @property
    def train_features_path(self):
        return self._build_subset_path(self.model, self.version, self.scope) / "train" / "features.parquet"

    @property
    def test_features_path(self):
        return self._build_subset_path(self.model, self.version, self.scope) / "test" / "features.parquet"

    @property
    def train_labels_path(self):
        return self._build_subset_path(self.model, self.version, self.scope) / "train" / "labels.parquet"

    @property
    def test_labels_path(self):
        return self._build_subset_path(self.model, self.version, self.scope) / "test" / "labels.parquet"

    @property
    def dataloader_params(self) -> Dict[str, Any]:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persist_workers,
        )

    def setup(self, stage: str) -> None:
        assert pathlib.Path(self.train_features_path).exists(), f"'{self.train_features_path}' does not exist"
        assert pathlib.Path(self.train_labels_path).exists(), f"'{self.train_labels_path}' does not exist"
        assert pathlib.Path(self.test_features_path).exists(), f"'{self.test_features_path}' does not exist"
        assert pathlib.Path(self.test_labels_path).exists(), f"'{self.test_labels_path}' does not exist"

        train_labels = pd.read_parquet(self.train_labels_path)
        test_labels = pd.read_parquet(self.test_labels_path)
        labels = list(set(train_labels.columns).intersection(set(test_labels.columns)))

        self.data = SoundscapeVAEEmbeddings(
            features=pd.read_parquet(self.train_features_path),
            labels=train_labels[labels],
            index=train_labels.index.get_level_values(0),
            num_samples=self.train_sample_size,
        )
        self.test_data = SoundscapeVAEEmbeddings(
            features=pd.read_parquet(self.test_features_path),
            labels=test_labels[labels],
            index=test_labels.index.get_level_values(0),
            num_samples=self.eval_sample_size,
        )
        self.train_data, self.val_data = torch.utils.data.random_split(
            self.data,
            (1 - self.val_prop, self.val_prop),
            generator=self.generator
        )
        return self

    def cross_validation_setup(self, k_folds: int) -> None:
        self.cross_val_dataloaders = tree()
        model_index = pd.read_parquet(self.root / "index.parquet")
        for model_name in model_index["model_name"].unique():
            for subset in model_index.loc[model_index["model_name"] == model_name, "dataset"].unique():
                for scope in model_index.loc[model_index["dataset"] == dataset, "scope"].unique():
                    train_features_path = self._build_subset_path(model, version, "/".join([subset, scope])) / "train" / "features.parquet"
                    train_labels_path = self._build_subset_path(model, version, "/".join([subset, scope])) / "train" / "labels.parquet"
                    test_features_path = self._build_subset_path(model, version, "/".join([subset, scope])) / "test" / "features.parquet"
                    test_labels_path = self._build_subset_path(model, version, "/".join([subset, scope])) / "test" / "labels.parquet"
                    train_labels = pd.read_parquet(train_labels_path)
                    test_labels = pd.read_parquet(test_labels_path)
                    labels = list(set(train_labels.columns).intersection(set(test_labels.columns)))
                    data = SoundscapeVAEEmbeddings(
                        features=pd.read_parquet(train_features_path),
                        labels=train_labels[labels],
                        index=train_labels.index.get_level_values(0),
                        num_samples=self.train_sample_size,
                    )
                    test_data = SoundscapeVAEEmbeddings(
                        features=pd.read_parquet(test_features_path),
                        labels=test_labels[labels],
                        index=test_labels.index.get_level_values(0),
                        num_samples=self.eval_sample_size,
                    )
                    test_dl = self._build_dataloader(self.test_data, batch_size-self.eval_batch_size, shuffle=False)
                    self.cross_val_dataloaders[model_name][dataset][scope]["test"] = test_dl
                    for k, (train_idx, val_idx) in enumerate(kf.split(range(len(self.data)))):
                        train_dl = self._build_dataloader(self.data[train_idx], batch_size=self.train_batch_size, shuffle=True)
                        val_dl = self._build_dataloader(self.data[val_idx], batch_size=self.eval_batch_size, shuffle=False)
                        self.cross_val_dataloaders[model_name][dataset][scope]["k"] = k
                        self.cross_val_dataloaders[model_name][dataset][scope]["train"] = train_dl
                        self.cross_val_dataloaders[model_name][dataset][scope]["val"] = val_dl
        return self.cross_val_dataloaders

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
        xs, ys, idx = zip(*batch)
        return (torch.stack(xs), torch.stack(ys), torch.tensor(idx), self.train_data.dataset.y_freq)

    def _build_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or len(dataset),
            collate_fn=self.batch_converter,
            **self.dataloader_params,
            **kwargs
        )

    def _build_subset_path(self, model: str | None, version: int | None, scope: str | None):
        assert model is not None, f"'model' is not specified"
        assert version is not None, f"'version' is not specified"
        assert scope is not None, f"'scope' is not specified"
        return self.root / (model + ".pt:" + version) / scope
