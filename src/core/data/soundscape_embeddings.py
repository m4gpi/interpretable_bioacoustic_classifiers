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
    "SoundscapeEmbeddings",
    "SoundscapeEmbeddingsDataModule",
]

@attrs.define(kw_only=True)
class SoundscapeEmbeddings(torch.utils.data.Dataset):
    features: pd.DataFrame = attrs.field()
    labels: pd.DataFrame = attrs.field()
    index: List[int] = attrs.field()
    seed: int = attrs.field(default=None)
    download: bool = attrs.field(default=False)
    chunked: bool = attrs.field(default=True)
    num_samples: int = attrs.field(default=1)

    x: torch.Tensor = attrs.field(init=False)
    y: torch.Tensor = attrs.field(init=False)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.x[idx], self.y[idx], self.index[idx]

    def __attrs_post_init__(self):
        if self.download:
            self._download_files()
        self.labels = self.labels[self.labels.index.get_level_values("file_i").isin(self.features.index.get_level_values("file_i"))]
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
            in_features=self.x.shape[-1] // 2 if self.chunked else self.x.shape[-1],
            target_names=self.target_names,
            target_counts=self.target_counts,
            seed=self.seed,
        )

    def _download_files(self):
        import requests
        import zipfile
        url = "https://sussex.box.com/s/1ob205h3t6wce8igt60vl360gycqv37o"
        try:
            response = requests.get(url, stream=True)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
        zip_path = self.base_dir.parent / "soundscape_embeddings.zip"
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=512):
                f.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(path=self.base_dir)
        zip_path.unlink()

@attrs.define(kw_only=True)
class SoundscapeEmbeddingsDataModule(L.LightningDataModule):
    root: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    transforms: List[Callable] = attrs.field(default=None)
    train_batch_size: int | None = attrs.field(default=None)
    eval_batch_size: int | None = attrs.field(default=None)
    train_sample_size: int = attrs.field(default=1)
    eval_sample_size: int = attrs.field(default=1)
    val_prop: float = attrs.field(default=0.2, validator=attrs.validators.instance_of(float))
    min_train_label_count: int = attrs.field(default=10, validator=attrs.validators.instance_of(int))
    chunked: bool = attrs.field(default=True)

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

    def pre_process(self, x: torch.Tensor, num_samples: int):
        mean, log_var = x.chunk(2, dim=-1)
        mean = mean.unsqueeze(1).expand(-1, num_samples, -1, -1)
        log_var = log_var.unsqueeze(1).expand(-1, num_samples, -1, -1)
        return mean + torch.randn_like(mean) * (0.5 * log_var).exp()

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        if not self.chunked: return batch
        num_samples = {0: self.train_sample_size, 1: self.eval_sample_size, 2: self.eval_sample_size}[dataloader_idx]
        x = self.pre_process(batch.x, num_samples)
        return Batch(x=x, **{k: batch[k] for k in batch.keys() if k != "x"})

    def setup(self, stage: str | None = None) -> None:
        self._validate_features_and_labels_present(self.train_features_path, self.train_labels_path)
        self._validate_features_and_labels_present(self.test_features_path, self.test_labels_path)

        # load features and labels
        train_features = pd.read_parquet(self.train_features_path)
        if self.val_features_path.exists():
            val_features = pd.read_parquet(self.val_features_path)
        test_features = pd.read_parquet(self.test_features_path)
        train_labels = pd.read_parquet(self.train_labels_path)
        if self.val_labels_path.exists():
            val_labels = pd.read_parquet(self.val_labels_path)
        test_labels = pd.read_parquet(self.test_labels_path)

        # align train, val and test label columns so we don't train on labels we can't make predictions about
        train_labels = train_labels.loc[:, train_labels.columns[train_labels.sum(axis=0) > self.min_train_label_count]]
        target_names = set(train_labels.columns).intersection(set(test_labels.columns))
        if self.val_labels_path.exists():
            target_names = target_names.intersection(set(val_labels.columns))
        target_names = list(target_names)

        # test set is fixed across all training regimens
        self.test_data = SoundscapeEmbeddings(
            features=test_features,
            labels=test_labels[target_names],
            index=test_labels.index.get_level_values(0),
            num_samples=self.eval_sample_size,
            chunked=self.chunked,
        )

        # train on everything when no validation is specified
        if self.val_prop == 0.0:
            features = pd.concat([train_features, val_features]) if self.val_features_path.exists() else train_features
            labels = pd.concat([train_labels, val_labels]) if self.val_labels_path.exists() else train_labels
            self.train_data = SoundscapeEmbeddings(
                features=features,
                labels=labels[target_names],
                index=labels.index.get_level_values(0),
                num_samples=self.train_sample_size,
                chunked=self.chunked,
            )
            return self

        # during cross-validation, recombine the original train/val splits before folding
        if self.num_folds is not None and self.fold_id is not None:
            features = pd.concat([train_features, val_features]) if self.val_features_path.exists() else train_features
            labels = pd.concat([train_labels, val_labels]) if self.val_labels_path.exists() else train_labels
            # same seed across runs ensures we get consistent splits
            # allows for indexing splits by their fold_id
            folder = sklearn.model_selection.KFold(
                n_splits=self.num_folds,
                random_state=self.seed,
                shuffle=True,
            )
            folds = list(folder.split(range(len(self.train_data))))
            train_idx, val_idx = folds[self.fold_id]
            index = features.index.get_level_values("file_i").unique()
            self.train_data = SoundscapeEmbeddings(
                features=features.loc[index[train_idx]],
                labels=labels.loc[index[train_idx], target_names],
                index=index[train_idx],
                seed=self.seed,
                num_samples=self.train_sample_size,
                chunked=self.chunked,
            )
            self.val_data = SoundscapeEmbeddings(
                features=features.loc[index[val_idx]],
                labels=labels.loc[index[val_idx], target_names],
                index=index[val_idx],
                seed=self.seed,
                num_samples=self.eval_sample_size,
                chunked=self.chunked,
            )
            return self

        if self.val_features_path.exists():
            # under a normal training regime, keep the split from pretraining
            self.data = self.train_data = SoundscapeEmbeddings(
                features=train_features,
                labels=train_labels[target_names],
                index=train_labels.index.get_level_values(0),
                num_samples=self.train_sample_size,
                chunked=self.chunked,
            )
            self.val_data = SoundscapeEmbeddings(
                features=val_features,
                labels=val_labels[target_names],
                index=val_labels.index.get_level_values(0),
                num_samples=self.eval_sample_size,
                chunked=self.chunked,
            )
        else:
            index = features.index.get_level_values("file_i").unique()
            train_idx, val_idx = sklearn.model_selection.train_test_split(
                list(range(len(index))),
                test_size=self.val_prop,
                shuffle=True,
                random_state=self.seed
            )
            self.train_data = SoundscapeEmbeddings(
                features=features.loc[index[train_idx]],
                labels=labels.loc[index[train_idx], target_names],
                index=index[train_idx],
                seed=self.seed,
                num_samples=self.train_sample_size,
                chunked=self.chunked,
            )
            self.val_data = SoundscapeEmbeddings(
                features=features.loc[index[val_idx]],
                labels=labels.loc[index[val_idx], target_names],
                index=index[val_idx],
                seed=self.seed,
                num_samples=self.eval_sample_size,
                chunked=self.chunked,
            )
        return self


    def train_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.train_data, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

    def predict_dataloader(self) -> List[torch.utils.data.DataLoader]:
        return [
            self._build_dataloader(self.train_data, batch_size=self.eval_batch_size),
            self.val_dataloader(),
            self.test_dataloader(),
        ]

    def batch_converter(self, batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        xs, ys, ss = zip(*batch)
        return Batch(x=torch.stack(xs), y=torch.stack(ys), s=torch.tensor(ss))

    def _build_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or len(dataset),
            collate_fn=self.batch_converter,
            **self.dataloader_params,
            **kwargs
        )

    def _validate_features_and_labels_present(self, features_path, labels_path):
        assert pathlib.Path(features_path).exists(), f"'{features_path}' does not exist"
        assert pathlib.Path(labels_path).exists(), f"'{labels_path}' does not exist"

    def features_path(self, stage: str):
        return self.root / stage / "features.parquet"

    @property
    def train_features_path(self):
        return self.features_path("train")

    @property
    def val_features_path(self):
        return self.features_path("val")

    @property
    def test_features_path(self):
        return self.features_path("test")

    def labels_path(self, stage: str):
        return self.root / stage / "labels.parquet"

    @property
    def train_labels_path(self):
        return self.labels_path("train")

    @property
    def val_labels_path(self):
        return self.labels_path("val")

    @property
    def test_labels_path(self):
        return self.labels_path("test")

    @property
    def dataloader_params(self) -> Dict[str, Any]:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persist_workers,
        )

