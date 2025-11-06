import pytest
import numpy as np
import pandas as pd
import pathlib
import tempfile
import torch

from src.core.data.soundscape_vae_embeddings import SoundscapeVAEEmbeddings, SoundscapeVAEEmbeddingsDataModule
from test import utils

@pytest.fixture()
def num_train_samples():
    return 100

@pytest.fixture()
def num_test_samples(num_train_samples):
    return num_train_samples // 4

@pytest.fixture()
def seq_len():
    return 20

@pytest.fixture()
def latent_dim():
    return 128

@pytest.fixture()
def num_classes():
    return 64

@pytest.fixture(autouse=True)
def root(num_train_samples, num_test_samples, seq_len, latent_dim, num_classes):
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        (root / "train").mkdir(exist_ok=True, parents=True)
        (root / "test").mkdir(exist_ok=True, parents=True)
        label_names = [utils.random_string(6) for i in range(num_classes + 10)]
        train_label_names = np.random.choice(label_names, size=num_classes, replace=False)
        test_label_names = np.random.choice(label_names, size=num_classes, replace=False)
        # generate some fake data
        pd.DataFrame(
            data=np.concatenate((
                np.random.randn(num_train_samples * seq_len, latent_dim),
                np.log(np.random.randn(num_train_samples * seq_len, latent_dim) ** 2),
            ), axis=-1),
            columns=list(map(str, range(latent_dim * 2)))
        ).to_parquet(root / "train" / "features.parquet")
        pd.DataFrame(
            data=(np.random.rand(num_train_samples, num_classes) > 0.5).astype(int),
            columns=train_label_names,
            index=list(range(num_train_samples)),
        ).to_parquet(root / "train" / "labels.parquet")
        pd.DataFrame(
            data=np.concatenate((
                np.random.randn(num_test_samples * seq_len, latent_dim),
                np.log(np.random.randn(num_test_samples * seq_len, latent_dim) ** 2),
            ), axis=-1),
            columns=list(map(str, range(latent_dim * 2)))
        ).to_parquet(root / "test" / "features.parquet")
        pd.DataFrame(
            data=(np.random.rand(num_test_samples, num_classes) > 0.5).astype(int),
            columns=test_label_names,
            index=list(range(num_test_samples)),
        ).to_parquet(root / "test" / "labels.parquet")

        yield root

def test_soundscape_vae_embeddings(root, num_train_samples, seq_len, latent_dim, num_classes):
    num_samples = 1
    data = SoundscapeVAEEmbeddings(
        features=pd.read_parquet(root / "train" / "features.parquet"),
        labels=pd.read_parquet(root / "train" / "labels.parquet"),
        index=list(range(num_train_samples)),
        num_samples=num_samples,
    )
    sample = data[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    assert sample[0].shape == (num_samples, seq_len, latent_dim)
    assert sample[1].shape == (num_classes,)
    assert isinstance(sample[2], int)

def test_soundscape_vae_embedding_data_module(root, num_train_samples, seq_len, latent_dim):
    dm = SoundscapeVAEEmbeddingsDataModule(root=root)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, list)
    assert len(batch) == 4
    assert batch[0].shape == (num_train_samples, 1, seq_len, latent_dim)
    assert batch[1].shape == (num_train_samples, len(dm.train_data.dataset.labels.columns))
    assert batch[2].shape == (num_train_samples,)
    assert isinstance(batch[3], dict) and all(isinstance(k, str) and isinstance(v, torch.LongTensor) for k, v in batch[3].items())
