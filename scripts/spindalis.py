files = ["2e698aba5.flac"]

import argparse
import numpy as np
import pathlib
import pandas as pd
import torch
import logging
import seaborn as sns
import hydra
import rootutils
import yaml

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import lines
from pathlib import Path
from torchvision import transforms as T
from tqdm.notebook import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.data.rainforest_connection import RainforestConnection, RainforestConnectionDataModule
from src.core.data.soundscape_embeddings import SoundscapeEmbeddingsDataModule
from src.core.models.mil_species_detector import MILSpeciesDetector
from src.core.models.sivae import SIVAE
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.cli.utils.instantiators import instantiate_transforms

device_id = 0
device = f"cuda:{device_id}"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.titlesize': 10,
    'legend.fontsize': 6,
})

def plot_frame_probs_spectrogram(
    x, y_probs, attn_weights,
    frame_length, bounding_boxes,
    sample_rate = 48_000,
    hop_length = 384,
    frame_prob_colour = "#42BC71",
    frame_attn_colour = "#FCA311",
    spectrogram_cmap = "Grays",
    ax = None,
    **kwargs,
):
    if not ax:
        ax = plt.gca()
    hops_per_second = sample_rate / hop_length
    hop_duration = 1 / hops_per_second
    duration = x.shape[-2] * hop_duration
    frame_duration = hop_duration * frame_length

    im = plot_mel_spectrogram(x.T, ax=ax, cmap=spectrogram_cmap, sample_rate=sample_rate, hop_length=hop_length, **kwargs)

    x_tick_labels = [np.format_float_positional(t, precision=3) for t in np.arange(0, duration, frame_duration * 5)]
    ax.set_xticks(np.arange(0, x.shape[-2], frame_length * 5), x_tick_labels)

    if len(bounding_boxes):
        for (t_start, t_end, f_start, f_end) in bounding_boxes:
            rect = patches.Rectangle([t_start, f_start], t_end - t_start, f_end - f_start, linewidth=1, edgecolor="white", facecolor='none', zorder=10)
            ax.add_patch(rect)

    for t in np.arange(0, int(duration / frame_duration)):
        ax.axvline(x=(t * frame_length), ymin=0, ymax=1, color="white", linestyle="dashed", linewidth=0.25, alpha=0.75)

    ax2 = ax.twinx()
    for i in range(attn_weights.shape[0]):
        x_start, x_end = i * frame_length, i * frame_length + frame_length
        xs = np.arange(x_start, x_end)
        ys = attn_weights[i].repeat(frame_length)
        ax2.plot(xs, ys, drawstyle="steps", color=frame_attn_colour)
    ax2.set_yticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    ax2.set_ylim([0.0, 1.0])
    ax2.set_ylabel(rf"p(y)")

    ax3 = ax.twinx()
    for i in range(y_probs.shape[0]):
        x_start, x_end = i * frame_length, i * frame_length + frame_length
        xs = np.arange(x_start, x_end)
        ys = y_probs[i].repeat(frame_length)
        ax3.fill_between(xs, y1=0, y2=ys, step="pre", alpha=0.25, color=frame_prob_colour)
        ax3.plot(xs, ys, drawstyle="steps", color=frame_prob_colour)
    ax3.set_yticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    ax3.set_ylim([0.0, 1.0])

    frames_line = lines.Line2D([], [], color="black", linestyle="--", linewidth=1.0, label="Frames")
    bounding_box_patch = patches.Patch(edgecolor="black", facecolor='none', linewidth=1.0, label="Species Call")
    fp_line = lines.Line2D([], [], color=frame_prob_colour, linewidth=1, label="Frame Prob")
    aw_line = lines.Line2D([], [], color=frame_attn_colour, linewidth=1, label="Frame Weight")
    handles = [frames_line, bounding_box_patch, fp_line, aw_line]

    return im, handles

@torch.no_grad()
def main(
    embedding_dir: pathlib.Path,
    audio_dir: pathlib.Path,
    results_dir: Path,
    save_dir: pathlib.Path,
    device_id: int = 0,
) -> None:
    device = f"cuda:{device_id}" if device_id is not None else "cpu"

    scope = "RFCX_bird"
    model_name = "sivae"
    version = "earthy-virgo"

    results_df = pd.read_parquet(results_dir / "test_results.parquet" / "model_earthy-virgo_run_id=dazzling-albert.parquet")
    scores_df = pd.read_parquet(results_dir / "test_scores.parquet" / "model_earthy-virgo_run_id=dazzling-albert.parquet")
    scores_df = scores_df.reset_index().set_index("species_name")

    vae = SIVAE.load_from_checkpoint("/its/home/kag25/models/v4/sivae/earthy-virgo/step=180000.ckpt", map_location="cuda")
    clf = MILSpeciesDetector.load_from_checkpoint("/its/home/kag25/models/v4/species_detectors/earthy-virgo_RFCX_bird.ckpt", map_location="cuda")

    dm = SoundscapeEmbeddingsDataModule(root=embedding_dir / "earthy-virgo" / "RFCX_bird")
    dm.setup()
    embedding_data = dm.data
    audio_data = RainforestConnection(audio_dir, test=True, scope="bird")

    spectrogram = vae.front_end
    train_embeddings = pd.read_parquet(dm.train_features_path)
    z0 = train_embeddings.iloc[:, :128].mean(axis=0).to_numpy()
    z0 = torch.tensor(z0.reshape(1, 1, -1), dtype=torch.float32, device=device)

    hops_per_second = spectrogram.sample_rate / spectrogram.fft_hop_length
    frame_length_seconds = vae.frame_window_length / hops_per_second
    frame_length_hops = vae.frame_window_length

    test_labels = pd.read_parquet(audio_data.base_dir / "test_labels.parquet")

    # first we look at positive examples, when we correctly predicted the outcome and it overlaps with the bounding box
    file_names = ["2e698aba5.flac", "e393a4c21.flac", "6c032e356.flac"]
    species_name = "Spindalis portoricensis_Puerto Rican Spindalis"
    deltas = [30, 30, 30]
    dts = [-1.0, -1.0, -1.0]
    frame_start_idx = [192*2, 0, 192+192//2]
    dts = [torch.ones(1, 1, 1, device=vae.device) * dt for dt in dts]

    test_labels = test_labels[test_labels["file_name"].isin(file_names) & (test_labels.species_name == species_name)].copy()
    test_labels["t_min_hops"] = test_labels["t_min"].map(spectrogram.seconds_to_hops)
    test_labels["t_max_hops"] = test_labels["t_max"].map(spectrogram.seconds_to_hops)
    test_labels["f_min_bin"] = test_labels["f_min"].map(spectrogram.hz_to_mel_bin)
    test_labels["f_max_bin"] = test_labels["f_max"].map(spectrogram.hz_to_mel_bin)

    scores = scores_df.loc[species_name, ["auROC", "AP"]]
    score_str = " ".join([f"{k}: {np.format_float_positional(v, precision=2)}" for k, v in scores.to_dict().items()])
    title = ", ".join([species_name.split("_")[-1], score_str])

    nrows = len(file_names)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(8.3, 1.5 * nrows), width_ratios=[0.5, 0.25, 0.25], constrained_layout=True)
    palette = sns.color_palette("colorblind", 6)
    for j, (file_name, delta, dt, start_idx) in enumerate(zip(file_names, deltas, dts, frame_start_idx)):
        record = test_labels[(test_labels.file_name == file_name) & (test_labels.species_name == species_name)]
        bounding_boxes = record[["t_min_hops", "t_max_hops", "f_min_bin", "f_max_bin"]].to_numpy()

        x = vae.pre_process(audio_data.load_sample(file_name).to(device).unsqueeze(0))
        q, *_ = vae.encode(x)
        z, _ = q.chunk(2, dim=-1)
        frame_probs, attn_weights = clf.target_frame_probs(z.unsqueeze(0), species_name)

        # plot the region around the bounding box
        t_start, t_end, _, _ = bounding_boxes[0]
        t_mid = t_end - int((t_end - t_start) / 2)
        t_start = t_mid - 192 * 2
        t_end = t_mid + 192 * 2
        ts = np.arange(t_start, t_end)
        ts = ts[start_idx:start_idx+192*2]
        x_t = x.squeeze().exp().cpu().numpy()[ts]
        im = vae.front_end.plot(
            20 * np.log10(x_t.T),
            vmin=-80.0,
            vmax=10.0,
            cmap="Greys",
            ax=axes[j, 0],
        )
        tsc = (ts * 1/hops_per_second)
        axes[j, 0].set_xticks(np.arange(0, len(ts), 192/2), tsc[np.arange(0, len(ts), 192//2)])
        if j == 0:
            axes[j, 0].set_title("Example Call")
        if j != nrows - 1:
            axes[j, 0].set_xlabel("")

        # plot the maximal frame
        frame_probs = frame_probs.squeeze().cpu()
        seq_idx = frame_probs.argmax().item()
        p_y_t = frame_probs[seq_idx].item()
        t_start, t_end = seq_idx * frame_length_hops, (seq_idx + 1) * frame_length_hops
        x_t = x.squeeze().exp().cpu().numpy()[t_start:t_end]
        im = vae.front_end.plot(
            20 * np.log10(x_t.T),
            vmin=-80.0,
            vmax=10.0,
            cmap="Greys",
            ax=axes[j, 1],
        )
        if j == 0:
            axes[j, 1].set_title("Predicted Frame")
        axes[j, 1].set_xlabel(rf"$p(y_{{t={seq_idx}}}) = {np.format_float_positional(p_y_t, precision=2)}$")
        axes[j, 1].set_xticks([0, 191], labels=[t_start * 1/hops_per_second, t_end * 1/hops_per_second])
        axes[j, 1].tick_params(labelleft=False, left=False)
        axes[j, 1].set_ylabel("")

        if j == 0:
            W = clf.species_weights(species_name)
            norm = torch.linalg.norm(W)
            z_tilde = z0 + ((z0 @ W / norm) + delta) * (W / norm)
            x_tilde = vae.decode(z_tilde, dt)
            x_tilde = x_tilde.squeeze().exp().cpu()
            im = vae.front_end.plot(
                20 * np.log10(x_tilde.T),
                vmin=-80.0,
                vmax=10.0,
                cmap="Greys",
                ax=axes[j, 2],
            )
            axes[j, 2].set_xticks([0, 191], [0.0, 1.536])
            if j == 0:
                axes[j, 2].set_title("Prediction Basis")
            if j != nrows - 1:
                axes[j, 2].set_xlabel("")
                axes[j, 2].tick_params(bottom=False, labelbottom=False)
            axes[j, 2].tick_params(labelleft=False, left=False)
            axes[j, 2].set_ylabel("")
        else:
            make_ax_invisible(axes[j, 2])

    fig.suptitle(title)
    if save_dir is not None:
        save_file = save_dir / f"spindalis.pdf"
        print(save_file)
        fig.savefig(save_file, format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--audio-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--results-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/test_results.parquet",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/saved/",
    )
    args = parser.parse_args()
    main(**vars(args))
