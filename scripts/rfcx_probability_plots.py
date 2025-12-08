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
from src.core.models.species_detector import SpeciesDetector
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

def get_transforms():
    with open(rootutils.find_root() / "config" / "transforms" / "cropped_log_mel_spectrogram.yaml", "r") as f:
        transform_conf = yaml.safe_load(f.read())
        transforms = instantiate_transforms(transform_conf)
        log_mel_spectrogram_params = transform_conf["log_mel_spectrogram"]
        del log_mel_spectrogram_params["_target_"]
    return transforms

def get_vae(model_dict):
    with open(rootutils.find_root() / "config" / "model" / f"{model_dict['model_name']}.yaml", "r") as f:
        model_conf = yaml.safe_load(f.read())
        vae = hydra.utils.instantiate(model_conf)
    checkpoint = torch.load(model_dict["vae_checkpoint_path"], map_location="cpu")
    vae.load_state_dict(checkpoint["model_state_dict"])
    log.info(f"Loaded {model_dict['model_name']} from {model_dict['vae_checkpoint_path']}")
    return vae

def get_clf(model_dict):
    log.info(f"Loaded {model_dict['clf_checkpoint_path']}")
    checkpoint = torch.load(model_dict["clf_checkpoint_path"], map_location=device)
    clf = SpeciesDetector(**checkpoint["hyper_parameters"])
    clf.load_state_dict(checkpoint["state_dict"])
    return clf

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
    save_dir.mkdir(exist_ok=True, parents=True)

    scope = "RFCX_bird"
    model_name = "base_vae"
    version = "v8"

    index = pd.read_parquet(embedding_dir / "index.parquet")
    df = index[(index["scope"] == scope) & (index["model_name"] == model_name) & (index["version"] == version)].copy()
    results_df = pd.read_parquet(results_dir / "test_results.parquet", columns=["model", "version", "scope", "file_i", "species_name", "prob", "label"])
    results_df = results_df[(results_df["scope"] == scope) & (results_df["model"] == model_name) & (results_df["version"] == version)].drop(["scope", "model", "version"], axis=1).copy()
    scores_df = pd.read_parquet(results_dir / "test_scores.parquet")
    scores_df = scores_df[(scores_df["scope"] == scope) & (scores_df["model"] == model_name)].groupby(["scope", "model", "species_name"])[["auROC", "AP"]].mean()
    scores_df = scores_df.reset_index().set_index("species_name")

    model_dict = df.iloc[0].to_dict()
    vae = get_vae(model_dict).to(device)
    clf = get_clf(model_dict).to(device)

    dm = SoundscapeEmbeddingsDataModule(root=embedding_dir, model="nifti_vae", version="v14", scope="RFCX_bird")
    dm.setup()
    embedding_data = dm.data
    audio_data = RainforestConnection(audio_dir, test=True, scope="RFCX_bird")
    transforms = get_transforms()
    spectrogram = transforms.transforms[0]
    spectrogram_params = dict(
        hop_length=spectrogram.hop_length,
        sample_rate=spectrogram.sample_rate,
        mel_min_hertz=spectrogram.mel_min_hertz,
        mel_max_hertz=spectrogram.mel_max_hertz,
        mel_scaling_factor=spectrogram.mel_scaling_factor,
        mel_break_frequency=spectrogram.mel_break_frequency,
    )

    train_embeddings = pd.read_parquet(dm.train_features_path)
    z0 = train_embeddings.iloc[:, :128].mean(axis=0).to_numpy()
    z0 = torch.tensor(z0.reshape(1, 1, -1), dtype=torch.float32, device=device)

    hops_per_second = spectrogram.sample_rate / spectrogram.hop_length
    frame_length_seconds = vae.frame_hop_length / hops_per_second
    frame_length_hops = vae.frame_hop_length

    test_labels = pd.read_parquet(audio_data.base_dir / "test_labels.parquet")

    # first we look at positive examples, when we correctly predicted the outcome and it overlaps with the bounding box
    file_names = ["4a7fe1b31.flac", "b9a6e6ce4.flac", "21e2f2977.flac", "745171bf2.flac"]
    species_names = ["Turdus plumbeus_Red-legged Thrush", "Setophaga angelae_Elfin Woods Warbler", "Patagioenas squamosa_Scaly-naped Pigeon", "Melanerpes portoricensis_Puerto Rican Woodpecker"]
    deltas = [30, 30, 20, 30]
    dts = [-0.65, 0.0, 0.1, -0.65]
    dts = [torch.ones(1, 1, 1, device=vae.device) * dt for dt in dts]

    success_test_labels = test_labels[test_labels["file_name"].isin(file_names) & test_labels.species_name.isin(clf.target_names)].copy()
    success_test_labels["t_min_hops"] = success_test_labels["t_min"].map(spectrogram.seconds_to_hops)
    success_test_labels["t_max_hops"] = success_test_labels["t_max"].map(spectrogram.seconds_to_hops)
    success_test_labels["f_min_bin"] = success_test_labels["f_min"].map(spectrogram.hz_to_mel_bin)
    success_test_labels["f_max_bin"] = success_test_labels["f_max"].map(spectrogram.hz_to_mel_bin)

    nrows = len(file_names)
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(8.3, 1.5 * nrows), width_ratios=[0.61, 0.13, 0.13, 0.13], constrained_layout=True)
    palette = sns.color_palette("colorblind", 6)
    for j, (file_name, species_name, delta, dt) in enumerate(zip(file_names, species_names, deltas, dts)):
        record = success_test_labels[(success_test_labels.file_name == file_name) & (success_test_labels.species_name == species_name)]
        bounding_boxes = record[["t_min_hops", "t_max_hops", "f_min_bin", "f_max_bin"]].to_numpy()

        x = transforms(audio_data.load_sample(file_name)).to(device).unsqueeze(0)
        q, *_ = vae.encode(x)
        frame_probs, weighted_frame_probs, attn_weights = clf.species_frame_probs(q, species_name)

        # plot the species name and scores as a row title
        scores = scores_df.loc[species_name, ["auROC", "AP"]]
        score_str = " ".join([f"{k}: {np.format_float_positional(v, precision=2)}" for k, v in scores.to_dict().items()])
        title = ", ".join([species_name.split("_")[-1], score_str])
        axes[j, 0].set_title(title)

        # plot the spectrogram
        ax_i = 0
        im, handles = plot_frame_probs_spectrogram(
            x=20 * np.log10(x.squeeze().exp().cpu().numpy()),
            y_probs=frame_probs.squeeze().cpu().numpy(),
            attn_weights=attn_weights.squeeze().cpu().numpy(),
            frame_length=vae.frame_hop_length,
            bounding_boxes=bounding_boxes,
            **spectrogram_params,
            ax=axes[j, ax_i],
            vmin=-80.0,
            vmax=10.0,
        )

        if j == 0:
            legend_params = dict(frameon=True, facecolor='white', edgecolor='black', fontsize=10)
            legend_1 = axes[j, ax_i].legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.25), ncol=2, **legend_params)
        if j != nrows - 1:
            axes[j, ax_i].set_xlabel("")

        # plot the bounding box
        ax_i = 1
        if len(bounding_boxes):
            t_start, t_end, _, _ = bounding_boxes[0]
            x_t = x.squeeze().exp().cpu().numpy()[t_start:t_end]
            im = plot_mel_spectrogram(
                20 * np.log10(x_t.T),
                **spectrogram_params,
                vmin=-80.0,
                vmax=10.0,
                cmap="Greys",
                ax=axes[j, ax_i],
            )
            if j == 0:
                axes[j, ax_i].set_title("Bounding Box")
            if j != nrows - 1:
                axes[j, ax_i].set_xlabel("")
            axes[j, ax_i].set_xticks([0, t_end - t_start], labels=[t_start * 1/hops_per_second, t_end * 1/hops_per_second])
            axes[j, ax_i].tick_params(labelleft=False, left=False)
            axes[j, ax_i].set_ylabel("")
        else:
            make_ax_invisible(axes[j, ax_i])

        # plot the maximal frame
        ax_i = 2
        frame_probs = frame_probs.squeeze().cpu()
        seq_idx = frame_probs.argmax().item()
        p_y_t = frame_probs[seq_idx].item()
        t_start, t_end = seq_idx * frame_length_hops, (seq_idx + 1) * frame_length_hops
        x_t = x.squeeze().exp().cpu().numpy()[t_start:t_end]
        im = plot_mel_spectrogram(
            20 * np.log10(x_t.T),
            **spectrogram_params,
            vmin=-80.0,
            vmax=10.0,
            cmap="Greys",
            ax=axes[j, ax_i],
        )
        axes[j, ax_i].set_title(rf"$p(y_{{t={seq_idx}}}) = {np.format_float_positional(p_y_t, precision=2)}$")
        if j != nrows - 1:
            axes[j, ax_i].set_xlabel("")
        axes[j, ax_i].set_xticks([0, 191], labels=[t_start * 1/hops_per_second, t_end * 1/hops_per_second])
        axes[j, ax_i].tick_params(labelleft=False, left=False)
        axes[j, ax_i].set_ylabel("")

        # plot the generated prototype
        ax_i = 3
        W = clf.classifier_weights(species_name)
        norm = torch.linalg.norm(W)
        z_tilde = z0 + ((z0 @ W.T / norm) + delta) * (W / norm)
        if model_name == "base_vae":
            x_tilde = vae.decode(z_tilde)
        else:
            x_tilde = vae.decode(z_tilde, dt)
        x_tilde = x_tilde.squeeze().exp().cpu()
        im = plot_mel_spectrogram(
            20 * np.log10(x_tilde.T),
            **spectrogram_params,
            vmin=-80.0,
            vmax=10.0,
            cmap="Greys",
            ax=axes[j, ax_i],
        )
        axes[j, ax_i].set_xticks([0, 191], [0.0, 1.536])
        if j == 0:
            axes[j, ax_i].set_title("Basis for Prediction")
        if j != nrows - 1:
            axes[j, ax_i].set_xlabel("")
            axes[j, ax_i].tick_params(bottom=False, labelbottom=False)
        axes[j, ax_i].tick_params(labelleft=False, left=False)
        axes[j, ax_i].set_ylabel("")

    save_file = save_dir / f"rfcx_frame_probs_success.pdf"
    print(save_file)
    fig.savefig(save_file, format="pdf")

    file_names = ["a0af97b46.flac", "6c032e356.flac", "1702d35a0.flac", "e23e84994.flac"]
    species_names = ["Coereba flaveola_Bananaquit", "Spindalis portoricensis_Puerto Rican Spindalis", "Nesospingus speculiferus_Puerto Rican Tanager", "Setophaga angelae_Elfin Woods Warbler"]
    deltas = [20, 20, 40, 10]
    dts = [0.5, -0.5, -0.75, -0.75]
    dts = [torch.ones(1, 1, 1, device=vae.device) * dt for dt in dts]

    fail_test_labels = test_labels[test_labels["file_name"].isin(file_names) & test_labels.species_name.isin(clf.target_names)].copy()
    fail_test_labels["t_min_hops"] = fail_test_labels["t_min"].map(spectrogram.seconds_to_hops)
    fail_test_labels["t_max_hops"] = fail_test_labels["t_max"].map(spectrogram.seconds_to_hops)
    fail_test_labels["f_min_bin"] = fail_test_labels["f_min"].map(spectrogram.hz_to_mel_bin)
    fail_test_labels["f_max_bin"] = fail_test_labels["f_max"].map(spectrogram.hz_to_mel_bin)

    nrows = len(file_names)
    palette = sns.color_palette("colorblind", 6)
    for j, (file_name, species_name, delta, dt) in enumerate(zip(file_names, species_names, deltas, dts)):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8.1, 1.5 + 0.5), width_ratios=[0.61, 0.13, 0.13, 0.13], constrained_layout=True)

        record = fail_test_labels[(fail_test_labels.file_name == file_name) & (fail_test_labels.species_name == species_name)]
        bounding_boxes = record[["t_min_hops", "t_max_hops", "f_min_bin", "f_max_bin"]].to_numpy()

        x = transforms(audio_data.load_sample(file_name)).to(device).unsqueeze(0)
        q, *_ = vae.encode(x)
        frame_probs, weighted_frame_probs, attn_weights = clf.species_frame_probs(q, species_name)

        # plot the species name and scores as a row title
        scores = scores_df.loc[species_name, ["auROC", "AP"]]
        score_str = " ".join([f"{k}: {np.format_float_positional(v, precision=2)}" for k, v in scores.to_dict().items()])
        row_label = " ".join([species_name.split("_")[-1], score_str])
        axes[0].set_title(row_label)

        # plot the spectrogram
        ax_i = 0
        im, handles = plot_frame_probs_spectrogram(
            x=20 * np.log10(x.squeeze().exp().cpu().numpy()),
            y_probs=frame_probs.squeeze().cpu().numpy(),
            attn_weights=attn_weights.squeeze().cpu().numpy(),
            frame_length=vae.frame_hop_length,
            bounding_boxes=bounding_boxes,
            **spectrogram_params,
            ax=axes[ax_i],
            vmin=-80.0,
            vmax=10.0,
        )

        legend_params = dict(frameon=True, facecolor='white', edgecolor='black', fontsize=10)
        legend_1 = axes[ax_i].legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.15), ncol=2, **legend_params)

        # plot the bounding box
        ax_i = 1
        if len(bounding_boxes):
            t_start, t_end, _, _ = bounding_boxes[0]
            x_t = x.squeeze().exp().cpu().numpy()[t_start:t_end]
            im = plot_mel_spectrogram(
                20 * np.log10(x_t.T),
                **spectrogram_params,
                vmin=-80.0,
                vmax=10.0,
                cmap="Greys",
                ax=axes[ax_i],
            )
            axes[ax_i].set_title("Bounding Box")
            axes[ax_i].set_xticks([0, t_end - t_start], labels=[t_start * 1/hops_per_second, t_end * 1/hops_per_second])
            axes[ax_i].tick_params(labelleft=False, left=False)
            axes[ax_i].set_ylabel("")
        else:
            make_ax_invisible(axes[ax_i])

        # plot the maximal frame
        ax_i = 2
        frame_probs = frame_probs.squeeze().cpu()
        seq_idx = frame_probs.argmax().item()
        p_y_t = frame_probs[seq_idx].item()
        t_start, t_end = seq_idx * frame_length_hops, (seq_idx + 1) * frame_length_hops
        x_t = x.squeeze().exp().cpu().numpy()[t_start:t_end]
        im = plot_mel_spectrogram(
            20 * np.log10(x_t.T),
            **spectrogram_params,
            vmin=-80.0,
            vmax=10.0,
            cmap="Greys",
            ax=axes[ax_i],
        )
        axes[ax_i].set_title(rf"$p(y_{{t={seq_idx}}}) = {np.format_float_positional(p_y_t, precision=2)}$")
        axes[ax_i].set_xticks([0, 191], labels=[t_start * 1/hops_per_second, t_end * 1/hops_per_second])
        axes[ax_i].tick_params(labelleft=False, left=False)
        axes[ax_i].set_ylabel("")

        # plot the generated prototype
        ax_i = 3
        W = clf.classifier_weights(species_name)
        norm = torch.linalg.norm(W)
        z_tilde = z0 + ((z0 @ W.T / norm) + delta) * (W / norm)
        if model_name == "base_vae":
            x_tilde = vae.decode(z_tilde)
        else:
            x_tilde = vae.decode(z_tilde, dt)
        x_tilde = x_tilde.squeeze().exp().cpu()
        im = plot_mel_spectrogram(
            20 * np.log10(x_tilde.T),
            **spectrogram_params,
            vmin=-80.0,
            vmax=10.0,
            cmap="Greys",
            ax=axes[ax_i],
        )
        axes[ax_i].set_xticks([0, 191], [0.0, 1.536])
        axes[ax_i].set_title("Basis for Detection")
        axes[ax_i].tick_params(labelleft=False, left=False)
        axes[ax_i].set_ylabel("")

        save_file = save_dir / f"rfcx_frame_probs_fail_{species_name}.pdf"
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
