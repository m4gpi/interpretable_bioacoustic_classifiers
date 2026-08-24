import argparse
import pandas as pd
import pathlib
import numpy as np
import rootutils
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.utils import metrics

plt.rcParams.update({
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 7,
})

def flatten_label(label):
    return label.replace('\\\\', ' ').replace('\n', ' ').strip()

def darken_color(color, amount=0.7):
    return tuple(amount * channel for channel in mcolors.to_rgb(color))

def lighten_color(color, amount=0.7):
    return tuple(1 - (1 - channel) * amount for channel in mcolors.to_rgb(color))

def main(
    results_dir: pathlib.Path,
    occlusions_path: pathlib.Path,
    save_dir: pathlib.Path,
) -> None:
    df = pd.read_parquet(results_dir / "test_scores.parquet")

    # switch around colours for consistency across the paper
    palette = list(sns.color_palette("colorblind", 5))
    palette = [palette[0], palette[4], palette[1], palette[2]]
    hue_order = ["BirdNET V2.4 (OOB)", "BirdNET V2.4 (FT)", "VAE", "SIVAE"]

    model_map = {
        "birdnet_native": "BirdNET V2.4 (OOB)",
        "birdnet_8": "BirdNET V2.4 (FT)",
        "birdnet_16": "BirdNET V2.4 (FT)",
        "birdnet_24": "BirdNET V2.4 (FT)",
        "just-drum": "SIVAE",
        "dynamic-malta": "SIVAE",
        "daring-system": "SIVAE",
        "earthy-virgo": "SIVAE",
        "part-armor": "SIVAE",
        "secluded-montana": "SIVAE",
        "lumpy-gibson": "VAE",
        "slow-partner": "VAE",
        "unique-tiger": "VAE",
        "jumpy-engine": "VAE",
        "quaint-pilot": "VAE",
        "numb-chef": "VAE",
    }

    # ---- SO ---- #
    df["model_class"] = df["model"].map(model_map)
    df["dataset_name"] = df["scope"].str.replace("_", " ")
    so_df = df[df["dataset_name"].isin(['SO UK', 'SO EC'])]

    plot_params = {}
    if occlusions_path is not None:
        # name_map = {
        #     "birdnet": "BirdNET V2.4",
        #     "birdnet_occ": "BirdNET V2.4 (Occlusions only)",
        #     "base_vae": "VAE",
        #     "base_vae_occ": "VAE (Occlusions only)",
        #     "nifti_vae": "SIVAE",
        #     "nifti_vae_occ": "SIVAE (Occlusions only)",
        #     # "smooth_nifti_vae": "TSSIVAE",
        #     # "smooth_nifti_vae_occ": "TSSIVAE (Occlusions only)",
        # }
        probs_df = pd.read_parquet(results_dir / "test_results.parquet")
        occ_df = (
            probs_df[probs_df.file_i.isin(pd.read_csv(occlusions_path)["file_i"]) & (probs_df["scope"].isin(["SO_EC", "SO_UK"]))]
            .groupby(["model", "scope"])
            .apply(lambda df: metrics.score(df))
            .reset_index()
            .set_index("species_name")
        )
        occ_df["model_class"] = occ_df["model"].map(model_map)
        occ_df["model_class"] = occ_df["model_class"] + " (Occlusions only)"
        occ_df["dataset_name"] = occ_df["scope"].str.replace("_", " ")
        so_df = pd.concat([so_df, occ_df])
        palette = np.stack([(darken_color(color, 0.8), lighten_color(color, 0.6)) for color in palette]).reshape(8, 3).tolist()
        plot_params["split"] = True
        hue_order = ["BirdNET V2.4 (OOB)", "BirdNET V2.4 (OOB) (Occlusions only)", "BirdNET V2.4 (FT)", "BirdNET V2.4 (FT) (Occlusions only)", "VAE", "VAE (Occlusions only)", "SIVAE", "SIVAE (Occlusions only)"]

    so_df = so_df.reset_index()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(8.1, 6), constrained_layout=True)

    sns.violinplot(
        data=so_df,
        x="dataset_name",
        y="auROC",
        hue="model_class",
        hue_order=hue_order,
        palette=palette,
        **plot_params,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        legend=True,
        ax=ax1,
        width=0.9,
    )
    ax1.margins(y=0.00)
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel("")
    ax1.set_ylabel("auROC")
    ax1.set_xticks([])

    sns.violinplot(
        data=so_df,
        x="dataset_name",
        y="AP",
        hue="model_class",
        hue_order=hue_order,
        palette=palette,
        **plot_params,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        ax=ax2,
        legend=False,
        width=0.9,
    )
    ax2.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("")
    ax2.set_ylabel("AP")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend_.remove()
    ax1.legend(handles, list(map(flatten_label, labels)), loc="lower center", bbox_to_anchor=(0.5, 1.1), ncols=4, title="")

    rfcx_df = df[df["dataset_name"].isin(['RFCX bird', 'RFCX frog'])]
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8.1, 4), constrained_layout=True)
    palette = list(sns.color_palette("colorblind", 5))
    palette = [darken_color(palette[0], 0.8), darken_color(palette[4], 0.8),darken_color(palette[1], 0.8), darken_color(palette[2], 0.8)]
    hue_order = ["BirdNET V2.4 (OOB)", "BirdNET V2.4 (FT)", "VAE", "SIVAE"]

    sns.violinplot(
        data=rfcx_df,
        x="dataset_name",
        y="auROC",
        hue="model_class",
        hue_order=hue_order,
        palette=palette,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        legend=False,
        ax=ax3,
        width=0.9,
    )
    ax3.margins(y=0.00)
    ax3.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax3.set_ylim([0.0, 1.0])
    ax3.set_xlabel("")
    ax3.set_ylabel("auROC")
    ax3.set_xticks([])

    sns.violinplot(
        data=rfcx_df,
        x="dataset_name",
        y="AP",
        hue="model_class",
        hue_order=hue_order,
        palette=palette,
        cut=0,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        ax=ax4,
        legend=False,
        width=0.9,
    )
    ax4.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax4.set_ylim([0.0, 1.0])
    ax4.set_xlabel("Dataset")
    ax4.set_ylabel("AP")
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend_.remove()
    # ax1.legend(handles, list(map(flatten_label, labels)), loc="lower center", bbox_to_anchor=(0.5, 1.1), ncols=4, title="")

    if save_dir is not None:
        plt.savefig(save_dir / "scores_violin.pdf", format="pdf", bbox_inches="tight")
        print(f"Saved: {(save_dir / 'scores_violin.pdf').expanduser()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=lambda p: pathlib.Path(p).expanduser(),
        required=True,
        help="/path/to/results/dir",
    )
    parser.add_argument(
        "--occlusions-path",
        type=lambda p: pathlib.Path(p).expanduser(),
        required=False,
        help="/path/to/occlusions.csv",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: pathlib.Path(p).expanduser(),
        required=False,
        help="/path/to/saved/",
    )

    args = parser.parse_args()
    main(**vars(args))

