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
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13,
})

def main(
    results_dir: pathlib.Path,
    save_dir: pathlib.Path,
):
    df = pd.read_parquet(results_dir / "test_scores.parquet", columns=["species_name", "AP", "auROC", "model", "scope", "version"])
    name_map = {
        "birdnet": "BirdNET V2.4",
        "base_vae": "VAE",
        "nifti_vae": "SIVAE",
        "smooth_nifti_vae": "TSSIVAE",
    }
    df["model_class"] = df["model"].map(name_map)
    df["dataset_name"] = df["scope"].str.replace("_", " ")

    order = (df[df["model"].isin(["base_vae", "nifti_vae", "smooth_nifti_vae"])].groupby("species_name")["AP"].max().sort_values(ascending=False)).index
    palette = np.stack([color for color in sns.color_palette("husl", 4)]).reshape(4, 3).tolist()

    df = df.reset_index()

    vae_df = df[df["model"].isin(["base_vae", "nifti_vae", "smooth_nifti_vae"])]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(
        data=vae_df,
        x="species_name",
        y="AP",
        hue="model_class",
        hue_order=list(name_map.values())[1:],
        order=order,
        ax=ax,
        palette=palette[1:],
    )
    ax.tick_params("x", rotation=90, labelsize=8)
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Average Precision (AP)")
    ax.set_xlabel("Species")

    x_positions = dict([(text.get_text(), text.get_position()[0]) for text in ax.get_xticklabels()])
    for _, row in df[df["model"] == "birdnet"].iterrows():
        if row.species_name in vae_df.species_name.unique():
            plt.scatter(
                x_positions[row["species_name"]],
                row["AP"],
                color=palette[0],
                marker='D',
                s=2,
                zorder=10,
            )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mlines.Line2D(
        [], [],
        color=palette[0],
        marker='D',
        markersize=8,
        label=name_map["birdnet"],
    ))
    labels.append(name_map["birdnet"])
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncols=4, title="")
    ax.margins(x=0.005)
    fig.savefig(save_dir / "species_ap_box_plot.pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {(save_dir / 'species_ap_box_plot.pdf').expanduser()}")

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(
        data=vae_df,
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
    x_positions = dict([(text.get_text(), text.get_position()[0]) for text in ax.get_xticklabels()])
    for _, row in df[df["model"] == "birdnet"].iterrows():
        if row.species_name in vae_df.species_name.unique():
            plt.scatter(
                x_positions[row["species_name"]],
                row["AP"],
                color=palette[0],
                marker='D',
                s=2,
                zorder=10,
            )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mlines.Line2D(
        [], [],
        color=palette[0],
        marker='D',
        markersize=8,
        label=name_map["birdnet"],
    ))
    labels.append(name_map["birdnet"])
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncols=4, title="")
    ax.margins(x=0.005)
    fig.savefig(save_dir / "species_auroc_box_plot.pdf", format="pdf", bbox_inches="tight")
    print(f"Saved: {(save_dir / 'species_auroc_box_plot.pdf').expanduser()}")

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
