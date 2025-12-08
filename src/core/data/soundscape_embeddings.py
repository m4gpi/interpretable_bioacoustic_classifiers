import attrs
import pathlib
import lightning as L
import numpy as np
import pandas as pd
import sklearn
import torch

from typing import Any, Callable, Dict, List, Tuple

from src.core.utils import tree

__all__ = [
    "SoundscapeEmbeddings",
    "SoundscapeEmbeddingsDataModule",
]


@attrs.define(kw_only=True)
class SoundscapeEmbeddings(torch.utils.data.Dataset):
    features: pd.DataFrame = attrs.field()
    labels: pd.DataFrame = attrs.field()
    index: List[int] = attrs.field()
    seed: int = attrs.field(default=None)

    x: torch.Tensor = attrs.field(init=False)
    y: torch.Tensor = attrs.field(init=False)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (self.x[idx], self.y[idx], self.index[idx])

    def __attrs_post_init__(self):
        self.x = torch.tensor(self.features.values.reshape(self.labels.values.shape[0], -1, self.features.values.shape[-1]), dtype=torch.float32)
        self.y = torch.tensor(self.labels.values, dtype=torch.int64)

    @property
    def target_names(self) -> List[str]:
        return self.labels.columns.tolist()

    @property
    def target_counts(self) -> List[int]:
        return self.labels.sum(axis=0).tolist()

    @property
    def model_params(self):
        return dict(
            target_names=self.target_names,
            target_counts=self.target_counts,
            seed=self.seed,
        )

@attrs.define(kw_only=True)
class SoundscapeEmbeddingsDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    model: str = attrs.field(default=None)
    scope: str = attrs.field(default=None)
    version: str = attrs.field(default=None)

    transforms: List[Callable] = attrs.field(default=None)
    train_batch_size: int | None = attrs.field(default=None)
    eval_batch_size: int | None = attrs.field(default=None)
    val_prop: float = attrs.field(default=0.2, validator=attrs.validators.instance_of(float))
    min_train_label_count: int = attrs.field(default=10, validator=attrs.validators.instance_of(int))

    seed: int = attrs.field(default=None)
    num_workers: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    persist_workers: bool | None = attrs.field(default=None)
    pin_memory: bool = attrs.field(default=True, validator=attrs.validators.instance_of(bool))

    generator: torch.Generator = attrs.field(init=False)
    fold_id: int = attrs.field(default=None)
    num_folds: int = attrs.field(default=None)

    data: torch.utils.data.Dataset = attrs.field(init=False)
    train_data: torch.utils.data.Subset = attrs.field(init=False)
    val_data: torch.utils.data.Subset = attrs.field(init=False)
    test_data: torch.utils.data.Dataset = attrs.field(init=False)

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)

    @num_folds.validator
    def check_fold_is_integer_if_not_none(self, attribute, value):
        return isinstance(value, int) if value is not None else True

    @fold_id.validator
    def check_fold_is_integer_if_not_none(self, attribute, value):
        return isinstance(value, int) if value is not None else True

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
            persistent_workers=self.persist_workers,
        )

    # FIXME
    # def prepare_data(self):
        # MODELS[self.model].load_from_checkpoint(self.root / "model.pt")

    def setup(self, stage: str | None = None) -> None:
        self.index = pd.read_parquet(self.root / "index.parquet")
        query = (self.index["model_name"] == self.model) & (self.index["version"] == self.version) & (self.index["scope"] == self.scope)
        assert query.any(), f"Data does not exist for {self.model} {self.version} {self.scope}"
        # reuse the seed from pre-training
        record = self.index[query].iloc[0]
        self.seed = record.seed
        L.seed_everything(self.seed)

        self._validate_features_and_labels_present(self.train_features_path, self.train_labels_path)
        self._validate_features_and_labels_present(self.test_features_path, self.test_labels_path)
        # load features
        train_features = pd.read_parquet(self.train_features_path)
        test_features = pd.read_parquet(self.test_features_path)
        # align train and test label columns
        train_labels = pd.read_parquet(self.train_labels_path)
        test_labels = pd.read_parquet(self.test_labels_path)
        train_labels = train_labels.loc[:, train_labels.columns[train_labels.sum(axis=0) > self.min_train_label_count]]
        target_names = list(set(train_labels.columns).intersection(set(test_labels.columns)))

        self.data = SoundscapeEmbeddings(
            features=train_features,
            labels=train_labels[target_names],
            index=train_labels.index.get_level_values(0),
        )
        self.test_data = SoundscapeEmbeddings(
            features=test_features,
            labels=test_labels[target_names],
            index=test_labels.index.get_level_values(0),
        )

        if self.num_folds is None and self.val_prop == 0.0:
            self.train_data = self.data
            return self

        if self.num_folds is not None and self.fold_id is not None:
            # same seed across runs ensures we get consistent splits
            # allows for indexing splits by their fold_id
            folder = sklearn.model_selection.KFold(
                n_splits=self.num_folds,
                random_state=self.seed,
                shuffle=True,
            )
            folds = list(folder.split(range(len(self.data))))
            train_idx, val_idx = folds[self.fold_id]
        else:
            train_idx, val_idx = sklearn.model_selection.train_test_split(
                range(len(self.data)),
                test_size=self.val_prop,
                random_state=self.seed,
                shuffle=True,
            )
        index = train_features.index.get_level_values("file_i").unique()
        self.train_data = SoundscapeEmbeddings(
            features=train_features.loc[index[train_idx]],
            labels=train_labels.loc[index[train_idx], target_names],
            index=index[train_idx],
            seed=self.seed,
        )
        self.val_data = SoundscapeEmbeddings(
            features=train_features.loc[index[val_idx]],
            labels=train_labels.loc[index[val_idx], target_names],
            index=index[val_idx],
            seed=self.seed,
        )
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

