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

def main(
    vae_clf_scores_path: Path,
    birdnet_scores_path: Path,
    label_frequency_path: Path,
    save_dir: Path,
    frequency_threshold: int = 25,
):
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
        "NIFTI": "Shift Invariant\nVAE",
        "Smooth_NIFTI": "Shift Invariant \&\nTemporally Smooth\nVAE",
        "BirdNET V2.4": "BirdNET V2.4",
    }
    s = s[s["model_class"].isin(list(name_map.keys()))]
    s["model_class"] = s["model_class"].map(name_map)

    order = (s[s.model == "VAE CLF"].groupby("species_name")["mAP"].max().sort_values(ascending=False)).index
    palette = np.stack([color for color in sns.color_palette("husl", 4)]).reshape(4, 3)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(
        data=s,
        x="species_name",
        y="mAP",
        hue="model_class",
        hue_order=list(name_map.values()),
        order=order,
        ax=ax,
        palette=palette,
    )
    ax.tick_params("x", rotation=90, labelsize=8)
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Average Precision (AP)")
    ax.set_xlabel("Species")
    sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncols=4, title="")
    fig.savefig(save_dir / "species_ap_box_plot.pdf", format="pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(
        data=s,
        x="species_name",
        y="auROC",
        hue="model_class",
        hue_order=list(name_map.values()),
        order=order,
        ax=ax,
        palette=palette,

    )
    ax.tick_params("x", rotation=90, labelsize=8)
    ax.set_ylim([0.5, 1.0])
    ax.set_ylabel("auROC")
    ax.set_xlabel("Species")
    sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncols=4, title="")
    fig.savefig(save_dir / "species_auroc_box_plot.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vae-clf-scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/vae_clf_scores.parquet",
    )
    parser.add_argument(
        "--birdnet-scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/birdnet_scores.parquet",
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
        default="./assets",
        help="/path/to/saved/plots",
    )
    args = parser.parse_args()
    main(**vars(args))
