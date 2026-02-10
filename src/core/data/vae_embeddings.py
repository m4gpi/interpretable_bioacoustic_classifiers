import attrs
import pathlib
import lightning as L
import numpy as np
import pandas as pd
import sklearn
import torch
import ranzen
import ranzen.torch

from typing import Any, Callable, Dict, List, Tuple

from src.core.utils import Batch

__all__ = [
    "VAEEmbeddings",
    "VAEEmbeddingsDataModule",
]

@attrs.define(kw_only=True)
class VAEEmbeddings(torch.utils.data.Dataset):
    features: pd.DataFrame = attrs.field()
    download: bool = attrs.field(default=False)

    x: torch.Tensor = attrs.field(init=False)

    def __len__(self) -> int:
        return len(self.q_z)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (self.q_z[idx], self.delta[idx], self.s[idx])

    def __attrs_post_init__(self):
        if self.download:
            self._download_files()
        self.q_z = torch.tensor(self.features[self.posterior_columns].values.reshape(-1, self.seq_len, 256), dtype=torch.float32)
        self.delta = torch.tensor(self.features[self.shift_columns].values.reshape(-1, self.seq_len, 1), dtype=torch.float32)
        self.s = self.features["file_i"].unique()

    def aggregate_posterior_analytical(self):
        mu, log_sigma_sq = self.q_z.flatten(end_dim=1).chunk(2, dim=-1)
        sigma_sq = log_sigma_sq.exp()
        mu_bar = mu.mean(dim=0)
        sigma_bar = torch.diag(sigma_sq.mean(dim=0)) + ((mu - mu_bar).t() @ (mu - mu_bar)) / mu.size(0)
        return mu_bar, sigma_bar

    def aggregate_posterior_samples(self):
        num_samples = 100
        mu, log_sigma_sq = self.q_z.expand(num_samples, -1, -1, -1).flatten(end_dim=2).chunk(2, dim=-1)
        z = Normal(mu, (0.5 * log_sigma_sq).exp()).rsample()
        mu_bar = z.mean(dim=0)
        sigma_bar = ((z - mu_bar).t() @ (z - mu_bar)) / mu.size(0)
        return mu_bar, sigma_bar

    @property
    def posterior_columns(self):
        return [*self.posterior_mean_columns, *self.posterior_log_variance_columns]

    @property
    def posterior_mean_columns(self):
        return [f"z_mean_{d}" for d in range(128)]

    @property
    def posterior_log_variance_columns(self):
        return [f"z_log_var_{d}" for d in range(128)]

    @property
    def shift_columns(self):
        return ["delta"]

    @property
    def seq_len(self):
        return self.features.timestep.max() + 1

    @property
    def model_params(self):
        return dict()

@attrs.define(kw_only=True)
class VAEEmbeddingsDataModule(L.LightningDataModule):
    data_path: str | pathlib.Path = attrs.field(converter=pathlib.Path)
    transforms: Callable | None = None

    train_batch_size: int | None = attrs.field(default=None)
    eval_batch_size: int | None = attrs.field(default=None)
    val_prop: float = attrs.field(default=0.2, validator=attrs.validators.instance_of(float))

    seed: int = attrs.field(default=None)
    num_workers: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    persist_workers: bool | None = attrs.field(default=None)
    pin_memory: bool = attrs.field(default=True, validator=attrs.validators.instance_of(bool))
    training_mode: str = attrs.field(default="step")

    _generator: torch.Generator = attrs.field(init=False)

    data: torch.utils.data.Dataset = attrs.field(init=False)
    train_data: torch.utils.data.Subset = attrs.field(init=False)
    val_data: torch.utils.data.Subset = attrs.field(init=False)
    test_data: torch.utils.data.Dataset = attrs.field(init=False)

    def __attrs_post_init__(self):
        L.LightningDataModule.__init__(self)
        self._generator = self.generator()

    @property
    def dataloader_params(self) -> Dict[str, Any]:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persist_workers,
        )

    def generator(self):
        return torch.Generator().manual_seed(self.seed)

    def setup(self, stage: str | None = None) -> None:
        self.features = pd.read_parquet(self.data_path).reset_index()
        self.data = VAEEmbeddings(features=self.features)
        self.train_data = VAEEmbeddings(features=self.features[self.features["dataloader_idx"] == 0])
        self.val_data = VAEEmbeddings(features=self.features[self.features["dataloader_idx"] == 1])
        self.test_data = VAEEmbeddings(features=self.features[self.features["dataloader_idx"] == 2])
        return self

    def train_dataloader_params(self, batch_size: int | None = None) -> Dict[str, Any]:
        if self.training_mode == "step":
            return dict(batch_size=batch_size, batch_sampler=self._default_train_sampler(batch_size))
        else:
            return dict(batch_size=batch_size, shuffle=True, generator=self._generator, drop_last=False)

    def train_dataloader(self, batch_size: int | None = None, batch_sampler: torch.utils.data.Sampler | None = None) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.train_data, **self.train_dataloader_params(batch_size))

    def val_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

    def predict_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        batch_size = batch_size or self.eval_batch_size or len(self.test_data)
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

    def batch_converter(self, batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        xs, ys, ss = zip(*batch)
        return Batch(x=torch.stack(xs), y=torch.stack(ys), s=torch.tensor(ss))

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
            collate_fn=self.batch_converter,
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
            generator=self._generator,
        )

