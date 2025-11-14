import attrs
import pathlib
import lightning as L
import logging
import numpy as np
import pandas as pd
import sklearn
import torch

from typing import Any, Dict, List, Tuple

from src.core.utils import tree

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
        q_z = q_z.unsqueeze(0) if q_z.dim() == 2 else q_z
        mean, log_var = q_z.chunk(2, dim=-1)
        mean = mean.unsqueeze(1).expand(-1, self.num_samples, -1, -1)
        log_var = log_var.unsqueeze(1).expand(-1, self.num_samples, -1, -1)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        return (z.squeeze(0), self.y[idx], self.index[idx])

    def __attrs_post_init__(self):
        self.x = torch.tensor(self.features.values.reshape(self.labels.values.shape[0], -1, self.features.values.shape[-1]), dtype=torch.float32)
        self.y = torch.tensor(self.labels.values, dtype=torch.int64)
        self.y_freq = dict(zip(self.labels.columns, torch.tensor([self.labels[y].sum() for y in self.labels.columns])))

    @property
    def model_params(self):
        return dict(
            target_names=self.labels.columns.tolist(),
            target_counts=list(self.y_freq.values())
        )

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
    # data: torch.utils.data.Dataset = attrs.field(init=False)
    # train_data: torch.utils.data.Subset = attrs.field(init=False)
    # val_data: torch.utils.data.Subset = attrs.field(init=False)
    # test_data: torch.utils.data.Dataset = attrs.field(init=False)

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)
        self.generator = torch.Generator().manual_seed(self.seed)

    @model.validator
    def check_model_is_valid_if_not_none(self, attribute, value):
        return value in ["base_vae", "smooth_nifti_vae", "nifti_vae"] if value is not None else True

    @scope.validator
    def check_scope_is_valid_if_not_none(self, attribute, value):
        return value in ["SO_EC", "SO_UK", "RFCX_bird", "RFCX_frog"] if value is not None else True

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
        self._validate_features_and_labels_present(self.train_features_path, self.train_labels_path)
        self._validate_features_and_labels_present(self.test_features_path, self.test_labels_path)

        train_labels = pd.read_parquet(self.train_labels_path)
        test_labels = pd.read_parquet(self.test_labels_path)
        target_names = list(set(train_labels.columns).intersection(set(test_labels.columns)))

        self.data = SoundscapeVAEEmbeddings(
            features=pd.read_parquet(self.train_features_path),
            labels=train_labels[target_names],
            index=train_labels.index.get_level_values(0),
            num_samples=self.train_sample_size,
        )
        self.test_data = SoundscapeVAEEmbeddings(
            features=pd.read_parquet(self.test_features_path),
            labels=test_labels[target_names],
            index=test_labels.index.get_level_values(0),
            num_samples=self.eval_sample_size,
        )
        self.train_data, self.val_data = torch.utils.data.random_split(
            self.data,
            (1 - self.val_prop, self.val_prop),
            generator=self.generator
        )
        return self

    def cross_validation_setup(self, num_folds: int) -> None:
        datasets = []
        index = pd.read_parquet(self.root / "index.parquet")
        for (model_name, scope), df in index.groupby(["model_name", "scope"]):
            for i, row in df.iterrows():
                train_features_path = self._build_subset_path(model_name, row.version, scope) / "train" / "features.parquet"
                train_labels_path = self._build_subset_path(model_name, row.version, scope) / "train" / "labels.parquet"
                test_features_path = self._build_subset_path(model_name, row.version, scope) / "test" / "features.parquet"
                test_labels_path = self._build_subset_path(model_name, row.version, scope) / "test" / "labels.parquet"
                self._validate_features_and_labels_present(train_features_path, train_labels_path)
                self._validate_features_and_labels_present(test_features_path, test_labels_path)

                log.info(f"Setting up dataset {model_name, row.version, scope}")

                train_labels = pd.read_parquet(train_labels_path)
                test_labels = pd.read_parquet(test_labels_path)
                target_names = list(set(train_labels.columns).intersection(set(test_labels.columns)))

                data = SoundscapeVAEEmbeddings(
                    features=pd.read_parquet(train_features_path),
                    labels=train_labels[target_names],
                    index=train_labels.index.get_level_values(0),
                    num_samples=self.train_sample_size,
                )
                test_data = SoundscapeVAEEmbeddings(
                    features=pd.read_parquet(test_features_path),
                    labels=test_labels[target_names],
                    index=test_labels.index.get_level_values(0),
                    num_samples=self.eval_sample_size,
                )
                folder = sklearn.model_selection.KFold(n_splits=num_folds, random_state=self.seed, shuffle=True)
                for fold_id, (train_idx, val_idx) in enumerate(folder.split(range(len(data)))):
                    train_data = torch.utils.data.Subset(data, train_idx)
                    val_data = torch.utils.data.Subset(data, val_idx)
                    datasets.append(dict(
                        fold_id=fold_id,
                        model_params=data.model_params,
                        train_dataloader_params=dict(
                            dataset=train_data,
                            batch_size=self.train_batch_size or len(train_data),
                            shuffle=True,
                            collate_fn=self.batch_converter,
                            **self.dataloader_params,
                        ),
                        val_dataloader_params=dict(
                            dataset=val_data,
                            batch_size=self.eval_batch_size or len(val_data),
                            shuffle=False,
                            collate_fn=self.batch_converter,
                            **self.dataloader_params,
                        ),
                        test_dataloader_params=dict(
                            dataset=test_data,
                            batch_size=self.eval_batch_size or len(test_data),
                            shuffle=False,
                            collate_fn=self.batch_converter,
                            **self.dataloader_params,
                        ),
                    ))
        return datasets

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
        return (torch.stack(xs), torch.stack(ys), torch.tensor(idx))

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

    def _validate_features_and_labels_present(self, features_path, labels_path):
        assert pathlib.Path(features_path).exists(), f"'{features_path}' does not exist"
        assert pathlib.Path(labels_path).exists(), f"'{labels_path}' does not exist"

