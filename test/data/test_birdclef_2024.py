import pytest
import numpy as np
import pandas as pd
import pathlib
import tempfile
import torch

from src.core.data.birdclef_2024 import BirdClef2024DataModule
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from test import utils

def test_birdclef_2024_batch_converter():
    transforms = LogMelSpectrogram(sample_rate=32000, hop_length=256, window_length=341)
    dm = BirdClef2024DataModule(root="/srv/thetis2/kag25/data/birdclef2024", transforms=transforms)
    dm.setup()
    next(iter(dm.train_dataloader()))

