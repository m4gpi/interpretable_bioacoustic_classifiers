import argparse
import pandas as pd
import pathlib
import numpy as np
import rootutils
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import lines as mlines

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

plt.rcParams.update({
    'axes.labelsize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
})

def main(
    results_dir: pathlib.Path,
    save_dir: pathlib.Path,
):
    df = pd.read_parquet(results_dir / "test_scores.parquet", columns=["species_name", "AP", "auROC", "model", "scope", "version", "train_label_counts"])
    name_map = {
        "birdnet": "BirdNET V2.4",
        "base_vae": "VAE",
        "nifti_vae": "SIVAE",
        # "smooth_nifti_vae": "TSSIVAE",
    }
    df["model_class"] = df["model"].map(name_map)
    df["dataset_name"] = df["scope"].str.replace("_", " ")
    df["species_name"] = df["species_name"].map(lambda s: s.split("_")[-1])

    palette = np.stack([color for color in sns.color_palette("colorblind", 4)]).reshape(4, 3).tolist()
    df = df.reset_index()

    # show the top 6 from each dataset
    vae_df = df[df["model"].isin(["base_vae", "nifti_vae"])]
    order = vae_df.groupby("species_name")["AP"].max().sort_values(ascending=False).index[:24]
    fig, (ax1, ax2) = plt.subplots(figsize=(8.1, 6), nrows=2)
    sns.boxplot(
        data=vae_df,
        x="species_name",
        y="auROC",
        hue="model_class",
        hue_order=list(name_map.values())[1:],
        order=order,
        ax=ax1,
        palette=palette[1:-1],
        legend=True,
    )
    ax1.set_ylim([0.75, 1.0])
    ax1.set_ylabel("auROC")
    ax1.set_xticklabels([])
    ax1.set_xlabel("")
    sns.move_legend(ax1, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncols=2, title="")
    sns.boxplot(
        data=vae_df,
        x="species_name",
        y="AP",
        hue="model_class",
        hue_order=list(name_map.values())[1:],
        order=order,
        ax=ax2,
        palette=palette[1:-1],
        legend=False,
    )
    # ax2.tick_params("x", ha='right', rotation_mode='anchor', rotation=60)
    for label in ax2.get_xticklabels():
        label.set_rotation(60)
        label.set_horizontalalignment('right')
        label.set_rotation_mode('anchor')
    ax2.set_ylim([0.2, 1.0])
    ax2.set_ylabel("AP")
    ax2.set_xlabel("Species")
    fig.savefig(save_dir / f"species_box_plot.pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {(save_dir / f'species_box_plot.pdf').expanduser()}")

    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 6,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
    })

    for scope in df["scope"].unique():
        vae_df = df[df["model"].isin(["base_vae", "nifti_vae"]) & (df["scope"] == scope)]
        order = vae_df.groupby("species_name")["train_label_counts"].first().sort_values(ascending=False).index
        fig, (ax1, ax2) = plt.subplots(figsize=(8.1, 3.5), nrows=2)
        sns.boxplot(
            data=vae_df,
            x="species_name",
            y="auROC",
            hue="model_class",
            hue_order=list(name_map.values())[1:],
            order=order,
            ax=ax1,
            palette=palette[1:-1],
            legend=True,
        )
        ax1.set_ylim([0.5, 1.0])
        ax1.set_ylabel("auROC")
        ax1.set_xticklabels([])
        ax1.set_xlabel("")
        sns.move_legend(ax1, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncols=2, title="")
        sns.boxplot(
            data=vae_df,
            x="species_name",
            y="AP",
            hue="model_class",
            hue_order=list(name_map.values())[1:],
            order=order,
            ax=ax2,
            palette=palette[1:-1],
            legend=False,
        )
        for label in ax2.get_xticklabels():
            label.set_rotation(60)
            label.set_horizontalalignment('right')
            label.set_rotation_mode('anchor')
        ax2.set_ylim([0.0, 1.0])
        ax2.set_ylabel("AP")
        ax2.set_xlabel("Species")
        fig.savefig(save_dir / f"{scope}_box_plot.pdf", format="pdf", bbox_inches="tight")
        print(f"Saved: {(save_dir / f'{scope}_box_plot.pdf').expanduser()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=lambda p: pathlib.Path(p).expanduser(),
        required=True,
        help="/path/to/results/dir",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: pathlib.Path(p).expanduser(),
        required=False,
        help="/path/to/saved/",
    )
    args = parser.parse_args()
    main(**vars(args))
