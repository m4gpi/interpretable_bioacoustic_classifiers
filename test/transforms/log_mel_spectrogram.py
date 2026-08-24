import pytest

from src.core.data.sounding_out_chorus import SoundingOutChorus
from src.core.models.vae import LogMelSpectrogram as TorchSpec
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram as LibrosaSpec

def test_equivalency():
    data = SoundingOutChorus(root="~/data/sounding_out")
    s1 = TorchSpec(
        sample_rate=48_000,
        n_fft=512,
        fft_window_length=512,
        fft_hop_length=384,
        num_mel_bins=64,
        mel_min_hertz=150,
        mel_max_hertz=15_000,
        mel_scaling_factor=4581.0,
        mel_break_frequency=1750.0,
    )
    s2 = LibrosaSpec(
        sample_rate=48_000,
        window_length=512,
        hop_length=384,
        num_mel_bins=64,
        mel_min_hertz=150,
        mel_max_hertz=15_000,
        mel_scaling_factor=4581.0,
        mel_break_frequency=1750.0,
    )
    i = 0
    for i in range(len(data)):
        x, *_ = data[i]
        torch.testing.assert_allclose(s1(x.unsqueeze(0)).squeeze(), s2(x).squeeze(), atol=1e-3, rtol=1e-2)
