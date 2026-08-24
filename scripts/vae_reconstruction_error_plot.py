import argparse
import duckdb
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
})

def main(results_dir, save_dir):
    z_score = 2.326
    if not save_dir:
        log.warn("save_dir not assigned, will not persist results")
    else:
        save_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_parquet(results_dir).reset_index()
    df["stage"] = df["dataloader_idx"].map({0: "Train", 2: "Test"})
    df["Model"] = df["model_name"]
    df["Dataset"] = df["dataset"]
    df = df.sort_values(by=["model_name", "dataset"])
    df["dkl_norm"] = df["dkl"] / df["latent_dim"]
    summary_stats = df.groupby(["dataset", "model_name", "stage"])[["mae", "mse", "dkl_norm"]].agg(["mean", "std"])
    print(summary_stats)

    palette = list(sns.color_palette("colorblind", 3))[1:]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3), constrained_layout=True)
    sns.boxplot(
        data=df,
        x="Dataset",
        y="mse",
        hue="Model",
        palette=palette,
        legend=False,
        gap=.1,
        width=0.9,
        showfliers=False,
        whis=2.0,
        ax=ax1,
    )
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("MSE")
    sns.boxplot(
        data=df,
        x="Dataset",
        y="dkl_norm",
        hue="Model",
        palette=palette,
        legend=True,
        gap=.1,
        width=0.9,
        showfliers=False,
        whis=2.0,
        ax=ax2,
    )
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("DKL / d")
    sns.move_legend(
        ax2,
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncols=2,
        title="",
    )
    if save_dir:
        file_name = (save_dir / "metrics.pdf").expanduser()
        plt.savefig(file_name, format="pdf", bbox_inches="tight")
        log.info(f"Saved: {file_name}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/results.parquet/",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/save/dir/",
    )
    args = parser.parse_args()
    main(**vars(args))

