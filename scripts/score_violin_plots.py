import argparse
import pandas as pd
import numpy as np
import re
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path

plt.rcParams.update({
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13,
})

def flatten_label(label):
    return label.replace('\\\\', ' ').replace('\n', ' ').strip()

def darken_color(color, amount=0.7):
    return tuple(amount * channel for channel in mcolors.to_rgb(color))

def lighten_color(color, amount=0.7):
    return tuple(1 - (1 - channel) * amount for channel in mcolors.to_rgb(color))

def main(
    vae_clf_scores_path: Path,
    birdnet_scores_path: Path,
    label_frequency_path: Path,
    save_dir: Path,
    frequency_threshold: int = 25,
) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)
    df1 = pd.read_parquet(birdnet_scores_path) # .drop("frequency", axis=1).reset_index(drop=True)
    df1["model"] = "BirdNET"
    df2 = pd.read_parquet(vae_clf_scores_path).reset_index()
    df2["model"] = "VAE CLF"
    s = pd.concat([df1, df2]).reset_index(drop=True)
    df3 = pd.read_parquet(label_frequency_path).reset_index()
    s = s.merge(df3, on=["species_name", "dataset", "scope"], how="inner")
    s = s[s.frequency >= frequency_threshold]
    s["dataset_name"] = s["dataset"] + " " + s["scope"].str.split("==", expand=True)[1].str.replace("'", "")

    name_map = {
        "Base": "Classic VAE",
        "Base\n(occlusions only)": "Classic VAE\n(occlusions only)",
        "NIFTI": "Shift Invariant VAE",
        "NIFTI\n(occlusions only)": "Shift Invariant VAE\n(occlusions only)",
        "Smooth NIFTI": "Shift Invariant \&\nTemporally Smooth\nVAE",
        "Smooth NIFTI\n(occlusions only)": "Shift Invariant \&\nTemporally Smooth VAE\n(occlusions only)",
        "BirdNET V2.4": "BirdNET V2.4",
        "BirdNET V2.4\n(occlusions only)": "BirdNET V2.4\n(occlusions only)",
    }
    s["model_class"] = s["model_class"].map(name_map)

    # ---- SO ---- #
    so_results = s[s["dataset_name"].isin(['SO UK', 'SO EC'])]

    palette = np.stack([
        (darken_color(color, 0.8), lighten_color(color, 0.6))
        for color in sns.color_palette("husl", 4)
    ]).reshape(8, 3)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 6), constrained_layout=True)

    sns.violinplot(
        data=so_results,
        x="dataset_name",
        y="auROC",
        hue="model_class",
        hue_order=list(name_map.values()),
        palette=palette,
        split=True,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        legend=True,
        ax=ax1,
    )
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel("")
    ax1.set_ylabel("Area under\nROC curve (auROC)")
    ax1.set_xticks([])
    sns.violinplot(
        data=so_results,
        x="dataset_name",
        y="mAP",
        hue="model_class",
        hue_order=list(name_map.values()),
        palette=palette,
        split=True,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        ax=ax2,
        legend=False,
    )
    ax2.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Average Precision (AP)")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend_.remove()
    ax1.legend(handles, list(map(flatten_label, labels)), loc='upper left', bbox_to_anchor=(1.01, 1.04), ncols=1, title="")

    plt.savefig(save_dir / "so_violin.pdf", format="pdf", bbox_inches="tight")

    # ---- RFCX ---- #
    name_map = {
        "Base": "Classic VAE",
        "NIFTI": "Shift Invariant VAE",
        "Smooth NIFTI": "Shift Invariant \&\nTemporally Smooth\nVAE",
        "BirdNET V2.4": "BirdNET V2.4",
    }
    rfcx_results = s[s["dataset_name"].isin(['RFCX bird', 'RFCX frog'])]

    palette = sns.color_palette("husl", 4)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18.5, 6), constrained_layout=True)

    sns.violinplot(
        data=rfcx_results,
        x="dataset_name",
        y="auROC",
        hue="model_class",
        hue_order=list(name_map.values()),
        palette=palette,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        legend=True,
        ax=ax1,
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Area under\nROC curve (auROC)")
    ax1.set_xticks([])
    sns.violinplot(
        data=rfcx_results,
        x="dataset_name",
        y="mAP",
        hue="model_class",
        hue_order=list(name_map.values()),
        palette=palette,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        ax=ax2,
        legend=False,
    )
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Average Precision (AP)")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend_.remove()
    ax1.legend(handles, list(map(flatten_label, labels)), loc='upper left', bbox_to_anchor=(1.01, 1.04), ncols=1, title="")

    plt.savefig(save_dir / "rfcx_violin.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vae-clf-scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/scores.parquet",
    )
    parser.add_argument(
        "--birdnet-scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/scores.parquet",
    )
    parser.add_argument(
        "--label-frequency-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/train_label_frequency.parquet",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )

    args = parser.parse_args()
    main(**vars(args))

