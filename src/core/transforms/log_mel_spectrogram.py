import enum
import functools
import librosa
import numpy as np
import torch

from dataclasses import dataclass
from functools import cached_property
from numpy.typing import NDArray
from scipy import signal
from torch import Tensor, nn
from typing import Any, Dict

__all__ = ["LogMelSpectrogram", "hz_to_mel", "mel_to_hz", "mel_filterbanks"]

def hz_to_mel(
    frequencies: NDArray | float,
    scaling_factor: float = 4581.0,
    break_frequency: float = 1750.0
) -> NDArray | float:
    return scaling_factor * np.log10(1 + frequencies / break_frequency)

def mel_to_hz(
    mels: NDArray | float,
    scaling_factor: float = 4581.0,
    break_frequency: float = 1750.0
) -> NDArray | float:
    return break_frequency * (10.0 ** (mels / scaling_factor) - 1.0)

def mel_filterbanks(
    num_mel_bins: int,
    mel_min_hertz: float,
    mel_max_hertz: float,
    linear_frequencies: NDArray,
    scaling_factor: float = 4581.0,
    break_frequency: float = 1750.0,
    norm: str | None = None,
) -> NDArray:
    # find minimum & maximum frequencies in mel basis
    linear_range =  np.array([mel_min_hertz, mel_max_hertz])
    min_mel, max_mel = hz_to_mel(linear_range, scaling_factor=scaling_factor, break_frequency=break_frequency)
    mel_points = hz_to_mel(linear_frequencies)
    # The i'th mel band (starting from i = 1) has center frequency
    # banks_ends[i], lower edge banks_ends[i - 1], and higher edge
    # banks_ends[i + 1]. Thus, we need num_mel_bins + 2 values in
    # the mel_bands array
    banks_ends = np.linspace(min_mel, max_mel, num_mel_bins + 2)
    filterbank = np.empty((len(linear_frequencies), num_mel_bins), dtype=np.float32)
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = banks_ends[i:i + 3]
        # calculate lower and upper slopes for every spectrogram bin
        # line segments are linear in the mel domain, not hertz
        lower_slope = ((mel_points - lower_edge_mel) / (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - mel_points) / (upper_edge_mel - center_mel))
        # then intersect them with each other and zero
        filterbank[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
        # HTK excludes the spectrogram DC bin; make sure it always gets a zero coefficient
        filterbank[0, :] = 0.0
    # Scale and normalize, so that all the triangles do not have same height and the gain gets adjusted appropriately.
    if norm == "slaney":
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)
    # return the filterbank
    return filterbank.T

class LogMelSpectrogram(object):
    def __init__(
        self,
        sample_rate: int = 48_000,
        window_length: int = 512,
        hop_length: int = 384,
        num_mel_bins: int = 64,
        mel_min_hertz: float = 150.0,
        mel_max_hertz: float = 15_000.0,
        mel_scaling_factor: float = 4581.0,
        mel_break_frequency: float = 1750.0,
        griffin_lim_iterations: int = 32,
        griffin_lim_momentum: float = 0.99,
        mel_filterbank_norm: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.num_mel_bins = num_mel_bins
        self.mel_min_hertz = mel_min_hertz
        self.mel_max_hertz = mel_max_hertz
        self.mel_scaling_factor = mel_scaling_factor
        self.mel_break_frequency = mel_break_frequency
        self.griffin_lim_iterations = griffin_lim_iterations
        self.griffin_lim_momentum = griffin_lim_momentum
        self.mel_filterbank_norm = mel_filterbank_norm

    def forward(self, wav: NDArray | Tensor) -> Tensor:
        return_type = type(wav)
        if return_type == Tensor:
            device = wav.device
            wav = wav.numpy()
        spec = librosa.stft(wav.squeeze(), **self.fft_params)
        spec, phase = librosa.magphase(spec, power=1)
        mel = (spec.T @ self.mel_filterbanks.T).T
        log_mel = np.log(np.maximum(1e-6, mel))
        if return_type == Tensor:
            return torch.as_tensor(log_mel.T, device=device, dtype=torch.float32).unsqueeze(0)
        return log_mel

    def backward(self, log_mel: Tensor) -> Tensor:
        mel = log_mel.exp().numpy().squeeze()
        mel = librosa.util.nnls(self.mel_filterbanks, mel.T)
        wav = librosa.griffinlim(mel, **self.griffin_lim_params)
        wav = self.bandpass_filter(wav)
        return torch.as_tensor(wav.copy(), device=log_mel.device).unsqueeze(0)

    def seconds_to_hops(self, seconds: float) -> int:
        hops_per_second = self.sample_rate / self.hop_length
        return int(seconds * hops_per_second)

    def hops_to_seconds(self, hops: float) -> int:
        hops_per_second = self.sample_rate / self.hop_length
        return 1 / hops_per_second * hops

    def hz_to_mel_bin(self, frequency: float) -> int:
        return np.abs(self.mel_range - hz_to_mel(frequency)).argmin()

    def mel_bin_to_hz(self, mel_bin: int) -> float:
        return mel_to_hz(
            self.mel_range,
            scaling_factor=self.mel_scaling_factor,
            break_frequency=self.mel_break_frequency,
        )[mel_bin]

    def __call__(self, wav: Tensor) -> Tensor:
        return self.forward(wav)

    @cached_property
    def mel_filterbanks(self) -> NDArray:
        return mel_filterbanks(**self.mel_filterbank_params)

    @property
    def bandpass_filter(self) -> NDArray:
        return functools.partial(signal.sosfiltfilt, signal.butter(**self.butter_params))

    @property
    def frequency_range(self):
        return np.array([self.mel_min_hertz, self.mel_max_hertz])

    @property
    def fft_length(self):
        return int(np.power(2, np.ceil(np.log(self.window_length) / np.log(2.0))))

    @property
    def stft_hop_length_seconds(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def stft_window_length_seconds(self) -> float:
        return self.window_length / self.sample_rate

    @property
    def overlap_length_seconds(self) -> float:
        return self.stft_window_length_seconds - self.stft_hop_length_seconds

    @property
    def overlap_length(self) -> int:
        return self.window_length - self.hop_length

    @property
    def overlap_prop(self) -> float:
        return 1 - (self.hop_length / self.window_length)

    @property
    def nyquist_frequency(self) -> float:
        return self.sample_rate / 2.0

    @property
    def linear_frequency_range(self) -> NDArray:
        return np.linspace(0.0, self.nyquist_frequency, (self.fft_length // 2) + 1)

    @property
    def min_mel(self) -> float:
        return hz_to_mel(
            self.mel_min_hertz,
            scaling_factor=self.mel_scaling_factor,
            break_frequency=self.mel_break_frequency,
        )

    @property
    def max_mel(self) -> float:
        return hz_to_mel(
            self.mel_max_hertz,
            scaling_factor=self.mel_scaling_factor,
            break_frequency=self.mel_break_frequency,
        )

    @property
    def mel_range(self) -> float:
        return np.linspace(self.min_mel, self.max_mel, self.num_mel_bins)

    @property
    def mel_filterbank_params(self) -> Dict[str, Any]:
        return dict(
            num_mel_bins=self.num_mel_bins,
            mel_min_hertz=self.mel_min_hertz,
            mel_max_hertz=self.mel_max_hertz,
            linear_frequencies=self.linear_frequency_range,
            scaling_factor=self.mel_scaling_factor,
            break_frequency=self.mel_break_frequency,
            norm=self.mel_filterbank_norm,
        )

    @property
    def fft_params(self) -> Dict[str, Any]:
        return dict(
            n_fft=self.fft_length,
            win_length=self.window_length,
            hop_length=self.hop_length,
            window='hann',
        )

    @property
    def butter_params(self) -> Dict[str, Any]:
        return dict(
            N=4,
            Wn=self.frequency_range / self.nyquist_frequency,
            btype='bandpass',
            output='sos'
        )

    @property
    def griffin_lim_params(self) -> Dict[str, Any]:
        return dict(
            n_iter=self.griffin_lim_iterations,
            momentum=self.griffin_lim_momentum,
            init='random',
            **self.fft_params,
        )
