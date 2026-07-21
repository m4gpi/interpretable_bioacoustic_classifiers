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
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
})

def main(results_dir, save_dir):
    z_score = 2.326
    if not save_dir:
        log.warn("save_dir not assigned, will not persist results")
    else:
        save_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_parquet(results_dir).reset_index()
    df["stage"] = df["dataloader_idx"].map({0: "Train", 1: "Validation", 2: "Test"})
    test_idx = df["dataloader_idx"] == 2
    df = df.sort_values(by=["sigma_x", "latent_dim"])
    df["group"] = df["model_name"]
    df["dataset_name"] = df["dataset_name"].map({"SoundingOutChorusDataModule": "SO", "RainforestConnectionDataModule": "RFCX"})

    summary_stats = df.groupby(["dataset_name", "model_name", "stage", "latent_dim", "sigma_x"])[["mae", "mse", "dkl_norm", "elbo"]].agg(["mean", "std"])
    print(summary_stats)

    palette = list(sns.color_palette("colorblind", len(df["group"].unique())))

    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    g = sns.catplot(
        data=df,
        kind="boxen",
        x="model_name",
        y="mae",
        col="stage",
        row="dataset_name",
        hue="group",
        hue_order=["VAE", "SIVAE"],
        sharey="row",
        palette=palette,
        log_scale=True,
        legend=True,
        gap=.1,
        width=0.9,
        showfliers=False,
        # common_norm=True,
        # density_norm="area",
        # gap=.1,
        # bw_adjust=0.75,
        # width=0.9,
    )
    # y_max = df["mae"].mean() + z_score * df["mae"].std()
    # y_min = df["mae"].mean() - z_score * df["mae"].std()
    # g.set(ylim=(y_min, y_max))
    g.set_axis_labels("Model", "MAE")
    if save_dir:
        file_name = (save_dir / "mae.pdf").expanduser()
        plt.savefig(file_name, format="pdf", bbox_inches="tight")
        log.info(f"Saved: {file_name}")

    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    g = sns.catplot(
        data=df,
        kind="boxen",
        x="model_name",
        y="mse",
        col="stage",
        row="dataset_name",
        hue="group",
        hue_order=["VAE", "SIVAE"],
        log_scale=True,
        sharey="row",
        palette=palette,
        legend=True,
        gap=.1,
        width=0.9,
        showfliers=False,
        # common_norm=True,
        # density_norm="area",
        # gap=.1,
        # bw_adjust=0.75,
        # width=0.9,
    )
    # y_max = df["mse"].mean() + z_score * df["mse"].std()
    # y_min = df["mse"].mean() - z_score * df["mse"].std()
    # g.set(ylim=(y_min, y_max))
    g.set_axis_labels("Model", "MSE")
    if save_dir:
        file_name = (save_dir / "mse.pdf").expanduser()
        plt.savefig(file_name, format="pdf", bbox_inches="tight")
        log.info(f"Saved: {file_name}")

    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    g = sns.catplot(
        data=df,
        kind="boxen",
        x="model_name",
        y="dkl_norm",
        col="stage",
        row="dataset_name",
        hue="group",
        hue_order=["VAE", "SIVAE"],
        sharey="row",
        palette=palette,
        legend=True,
        gap=.1,
        width=0.9,
        showfliers=False,
    )
    g.set_axis_labels("Model", "DKL / d")
    if save_dir:
        file_name = (save_dir / "dkl.pdf").expanduser()
        plt.savefig(file_name, format="pdf", bbox_inches="tight")
        log.info(f"Saved: {file_name}")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3), constrained_layout=True)
    palette = list(sns.color_palette("colorblind", 3))
    sns.boxenplot(
        data=df[test_idx],
        x="mse",
        y="dataset_name",
        hue="model_name",
        hue_order=["VAE", "SIVAE"],
        palette=palette[1:],
        legend=False,
        gap=.1,
        ax=ax1,
        showfliers=False,
    )
    ax1.set_xscale("log")
    sns.boxenplot(
        data=df[test_idx],
        x="dkl_norm",
        y="dataset_name",
        hue="model_name",
        hue_order=["VAE", "SIVAE"],
        palette=palette[1:],
        legend=True,
        gap=.1,
        ax=ax2,
        showfliers=False,
    )
    ax1.set_ylabel("Dataset")
    ax1.set_xlabel("Frame-wise MSE")
    ax2.set_ylabel("")
    ax2.set_xlabel(r"KL Divergence")
    sns.move_legend(ax2, loc="upper left", bbox_to_anchor=(1.0, 1.0), title=None, frameon=False)
    if save_dir:
        file_name = (save_dir / "mse_dkl.pdf").expanduser()
        fig.savefig(file_name, format="pdf", bbox_inches="tight")
        log.info(f"Saved: {file_name}")

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
