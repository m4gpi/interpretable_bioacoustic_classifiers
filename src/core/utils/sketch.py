import librosa
import numpy as np
import torch
import seaborn as sns
import pandas as pd

from torch import Tensor
from matplotlib import animation
from matplotlib import ticker
from matplotlib import patches
from matplotlib import lines
from matplotlib import gridspec as gs
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh
from matplotlib import cm
from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple

from src.core.transforms.log_mel_spectrogram import hz_to_mel, mel_to_hz

__all__ = [
    "plot_mel_spectrogram",
]

def plot_mel_spectrogram(
    z: NDArray,
    sample_rate: int,
    hop_length: int,
    mel_min_hertz: float,
    mel_max_hertz: float,
    mel_scaling_factor: float,
    mel_break_frequency: float,
    window_length: int | None = None,
    fft_length: int | None = None,
    num_mel_bins: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap = sns.color_palette("viridis", as_cmap=True),
    ax: Axes | None = None,
    **kwargs: Any,
) -> AxesImage:
    ax = ax if ax is not None else plt.gca()
    imshow_params = dict(
        vmin=vmin if vmin is not None and kwargs.get("norm", None) is None else z.min(),
        vmax=vmax if vmax is not None and kwargs.get("norm", None) is None else z.max(),
        origin="lower", aspect="auto", cmap=cmap,
        **kwargs,
    )
    im = ax.imshow(z, **imshow_params)
    # TODO: allow parametrisation of tick duration
    duration_seconds = (z.shape[1] * hop_length) // sample_rate
    time = np.linspace(0, duration_seconds, z.shape[1])
    x_tick_positions = np.linspace(0, duration_seconds, int(z.shape[1] // (sample_rate // hop_length) / 5) + 1)
    x_tick_labels = [f"{np.format_float_positional(t, trim='-', precision=2)}" for t in x_tick_positions]
    x_tick_indices = [np.argmin(np.abs(time - t)) for t in x_tick_positions]
    ax.set_xticks(x_tick_indices, labels=x_tick_labels)
    ax.set_xlabel("Time (s)")
    # ticks for y-axis are on a log scale, so we find the nearest base 2 exponents for the ticks
    min_mel = hz_to_mel(mel_min_hertz, scaling_factor=mel_scaling_factor, break_frequency=mel_break_frequency)
    max_mel = hz_to_mel(mel_max_hertz, scaling_factor=mel_scaling_factor, break_frequency=mel_break_frequency)
    mels = np.linspace(min_mel, max_mel, z.shape[0])
    frequencies = mel_to_hz(mels, scaling_factor=mel_scaling_factor, break_frequency=mel_break_frequency)
    y_tick_positions = [2**i for i in range(max(9, int(np.ceil(np.log2(mel_min_hertz)))), int(np.floor(np.log2(mel_max_hertz))) + 1)]
    y_tick_labels = [f"{int(f)}" for f in y_tick_positions]
    y_tick_indices = [np.argmin(np.abs(frequencies - f)) for f in y_tick_positions]
    ax.set_yticks(y_tick_indices, labels=y_tick_labels)
    ax.set_ylabel("Frequency (Hz)")
    return im

def plot_mel_filterbanks(x: NDArray, y: NDArray, f: NDArray, ax: Axes | None = None, cmap: str | Colormap = "inferno") -> AxesImage:
    ax = ax if ax is not None else plt.gca()
    img = ax.imshow(f, origin='lower', aspect="auto", cmap=cmap)
    ax.set_xlabel("Linear Frequencies")
    ax.set_xticks(range(0, len(x), 16), labels=[np.format_float_positional(x[i], precision=0) for i in range(0, len(x), 16)], rotation=60)
    ax.set_ylabel("Mel Frequencies")
    ax.set_yticks(range(0, len(y), 8), labels=[np.format_float_positional(y[i], precision=0) for i in range(0, len(y), 8)])
    ax.set_title('Mel filter bank')
    return img

def make_ax_invisible(ax: Axes) -> None:
    from matplotlib import ticker
    # set background to white
    ax.set_facecolor('white')
    # make spines invisible
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # set grid to white
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    # remove all ticks
    ax.set_yticks([])
    ax.set_xticks([])

