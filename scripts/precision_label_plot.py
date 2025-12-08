import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import ticker
from pathlib import Path

from src.core.utils.sketch import make_ax_invisible

def main(
    results_dir: Path,
    save_dir: Path,
) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)

    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
    })

    s = pd.read_parquet(results_dir / "test_scores.parquet")
    models = ["base_vae", "nifti_vae"]
    scopes = ["SO_UK", "SO_EC", "RFCX_bird", "RFCX_frog"]
    name_map = lambda x: {
        "SO_UK": "SO UK",
        "SO_EC": "SO EC",
        "RFCX_bird": "RFCX Birds",
        "RFCX_frog": "RFCX Frogs",
    }[x]

    palette = sns.color_palette("colorblind", len(scopes))
    fig = plt.figure(figsize=(8.1, 3), constrained_layout=True)
    grid_spec = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[0.999, 0.01], hspace=0.01)
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1])
    groups = [[("SO_UK", palette[0], ax1), ("SO_EC", palette[1], ax1)], [("RFCX_bird", palette[2], ax2), ("RFCX_frog", palette[3], ax2)]]
    for group in groups:
        for scope, colour, ax in group:
            sns.regplot(
                data=s[(s["scope"] == scope) & (s["model"].isin(models))],
                x="train_label_counts",
                y="AP",
                marker="o",
                color=colour,
                scatter_kws=dict(alpha=0.1),
                ax=ax
            )
        dataset_name = scope.split("_")[0]
        ax.set_ylim([0.0, 1.0])
        ax.set_title(dataset_name)
        ax.set_xlabel("")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2.legend(
        handles=[mpatches.Patch(color=color, label=name) for name, color in zip(list(map(name_map, scopes)), palette)],
        bbox_to_anchor=(1.01, 1.04),
        loc='upper left'
    )
    ax1.set_ylabel("Average Precision (AP)")
    ax2.set_ylabel("")
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax2.tick_params(left=False, labelleft=False)
    ax = fig.add_subplot(grid_spec[1, :])
    make_ax_invisible(ax)
    ax.set_xlabel("Count of Presence Labels")
    save_path = save_dir / "label_precision_by_dataset.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(save_path)

    # frequency by accuracy split on count threshold
    label_threshold = 50
    palette = sns.color_palette("colorblind", len(scopes))
    fig = plt.figure(figsize=(8.1, 3), constrained_layout=True)
    grid_spec = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[0.999, 0.01], hspace=0.01)
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1])
    groups = [[("SO_UK", palette[0], ax1), ("SO_EC", palette[1], ax1)], [("RFCX_bird", palette[2], ax2), ("RFCX_frog", palette[3], ax2)]]
    for group in groups:
        for scope, colour, ax in group:
            sns.regplot(
                data=s[(s["scope"] == scope) & (s["model"].isin(models)) & (s["train_label_counts"] < label_threshold)],
                x="train_label_counts",
                y="AP",
                marker="o",
                color=colour,
                scatter_kws=dict(alpha=0.1),
                ax=ax1
            )
        ax1.set_ylim([0.0, 1.0])
        ax1.set_title(rf"Train Label Count $<$ {label_threshold}")
        ax1.set_xlabel("")
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        for scope, colour, ax in group:
            sns.regplot(
                data=s[(s["scope"] == scope) & (s["model"].isin(models)) & (s["train_label_counts"] >= label_threshold)],
                x="train_label_counts",
                y="AP",
                marker="o",
                color=colour,
                scatter_kws=dict(alpha=0.1),
                ax=ax2
            )
        ax2.set_ylim([0.0, 1.0])
        ax2.set_title(rf"Train Label Count $>$ {label_threshold}")
        ax2.set_xlabel("")
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2.legend(
        handles=[mpatches.Patch(color=color, label=name) for name, color in zip(list(map(name_map, scopes)), palette)],
        bbox_to_anchor=(1.01, 1.04),
        loc='upper left'

    )
    ax1.set_ylabel("Average Precision (AP)")
    ax2.set_ylabel("")
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax2.tick_params(left=False, labelleft=False)
    ax = fig.add_subplot(grid_spec[1, :])
    make_ax_invisible(ax)
    ax.set_xlabel("Count of Presence Labels")


    save_path = save_dir / "label_precision_by_count.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(save_path)

    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 6,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
    })

    counts = s[["scope", "species_name", "train_label_counts"]].groupby(["scope", "species_name"]).first().reset_index().sort_values(by="train_label_counts", ascending=False)
    for dataset in ["SO", "RFCX"]:
        df = counts[counts["scope"].str.startswith(dataset)].copy()
        df["species"] = df["species_name"].str.split("_", expand=True)[1]
        for scope in df["scope"].unique():
            fig, ax = plt.subplots(figsize=(8.1, 3), constrained_layout=True)
            sns.barplot(
                data=df[df["scope"] == scope],
                x="species",
                y="train_label_counts",
                fill=True,
                ax=ax,
            )
            ax.set_xlabel("Species")
            ax.set_ylabel("Number of Labels")
            ax.tick_params(axis="x", rotation=90)
            save_path = save_dir / f"{scope}_label_frequency.pdf"
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            print(save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )

    args = parser.parse_args()
    main(**vars(args))
