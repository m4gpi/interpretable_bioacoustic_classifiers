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

def plot_species_proba_spectrogram(
    x: NDArray,
    y_probs: NDArray,
    frame_length: int,
    sample_rate: int = 48_000,
    hop_length: int = 384,
    bounding_boxes: NDArray | None = None,
    ax: Axes = None,
    spectrogram_cmap = "Grays",
    frame_prob_colour = "#42BC71",
    **kwargs: Any
) -> AxesImage:
    hops_per_second = sample_rate / hop_length
    hop_duration = 1 / hops_per_second
    duration = x.shape[-2] * hop_duration
    frame_duration = hop_duration * frame_length

    im = plot_mel_spectrogram(x.T, ax=ax, cmap="Grays", sample_rate=sample_rate, hop_length=hop_length, **kwargs)

    x_tick_labels = [np.format_float_positional(t, precision=3) for t in np.arange(0, duration, frame_duration * 5)]
    ax.set_xticks(np.arange(0, x.shape[-2], frame_length * 5), x_tick_labels)

    if bounding_boxes is not None:
        for (t_start, t_end, f_start, f_end) in bounding_boxes:
            rect = patches.Rectangle([t_start, f_start], t_end - t_start, f_end - f_start, linewidth=3, edgecolor="white", facecolor='none', zorder=10)
            ax.add_patch(rect)

    for t in np.arange(0, int(duration / frame_duration)):
        ax.axvline(x=(t * frame_length), ymin=0, ymax=1, color="white", linestyle="dashed", alpha=0.75)

    if threshold is not None:
        ax2 = ax.twinx()
        for i in range(y_probs.shape[0]):
            x_start, x_end = i * frame_length, i * frame_length + frame_length
            xs = np.arange(x_start, x_end)
            ys = y_probs[i].repeat(frame_length)
            # define a binary confusion matrix to track the prediction
            flags = np.zeros((2, 2))
            for t_start, t_end, _, _ in bounding_boxes:
                ts = np.arange(t_start, t_end)
                # if there's an intersection, we know theres a call here
                overlap = set(xs).intersection(ts)
                if len(overlap):
                    if y_probs[i] > threshold: # true positive
                        flags[0, 0] = 1
                    else: # false negative
                        flags[1, 0] = 1 
                else:
                    # then we're looking either at a bounding box that shouldn't be being considered
                    # we're missing a condition...
                    if y_probs[i] > threshold: # false positive
                        flags[0, 1] = 1
            if flags[0, 0] == 1:
                colour = true_pos_colour
            elif flags[1, 0] == 1:
                colour = false_neg_colour
            elif flags[0, 1] == 1:
                colour = false_pos_colour
            else:
                colour = true_neg_colour
            ax2.fill_between(xs, ys, step="pre", alpha=0.25, color=colour)
            ax2.plot(xs, ys, drawstyle="steps", color=colour)

        ax2.axhline(y=threshold, xmin=0, xmax=x.shape[-2], color=thresh_colour, linewidth=2.0, linestyle="dotted")
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(0, 1)
        ax2.set_ylabel(r"$p(y_s|\mathbf{z}^t)$")

    frames_line = lines.Line2D([], [], color="black", linestyle="--", linewidth=2, label='Frames')
    bounding_box_patch = patches.Patch(edgecolor="black", facecolor='none', linewidth=2, label='Species Call')
    thresh_line = lines.Line2D([], [], color=thresh_colour, linestyle='dotted', linewidth=2, label='Threshold')
    spec_handles = [frames_line, thresh_line, bounding_box_patch]

    tp_line = lines.Line2D([], [], color=true_pos_colour, linewidth=2, label='True Positive')
    fp_line = lines.Line2D([], [], color=false_pos_colour, linewidth=2, label='False Positive')
    fn_line = lines.Line2D([], [], color=false_neg_colour, linewidth=2, label='False Negative')
    tn_line = lines.Line2D([], [], color="#c4c4c4", linewidth=2, label='True Negative')
    pred_handles = [tp_line, fp_line, fn_line, tn_line]

    return im, spec_handles, pred_handles, ax2
