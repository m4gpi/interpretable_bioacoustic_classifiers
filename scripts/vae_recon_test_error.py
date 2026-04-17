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
    'legend.fontsize': 6,
})

def main(results_dir, save_dir):
    z_score = 2.326
    if not save_dir:
        log.warn("save_dir not assigned, will not persist results")
    else:
        save_dir.mkdir(exist_ok=True, parents=True)

    con = duckdb.connect()
    df = con.execute(f"""
CREATE TABLE mae AS
SELECT
    *,
    regexp_extract(filename, 'version=(.*?)_', 1) AS version,
    regexp_extract(filename, 'scope=(.*?)\\.parquet', 1) AS dataset_name
FROM read_parquet('{results_dir}', filename=true);
    """)
    df = con.execute("SELECT * FROM mae").fetchdf()
    df["stage"] = df["dataloader_idx"].map({0: "Train", 1: "Validation", 2: "Test"})
    df["dkl_norm"] = df["dkl"] / df["latent_dim"]

    summary_stats = df.groupby(["dataset_name", "model_name", "stage", "latent_dim"])[["mae", "mse", "dkl_norm", "elbo"]].agg(["mean", "std"])
    print(summary_stats)

    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    palette = list(sns.color_palette("colorblind", 3))
    g = sns.catplot(
        data=df,
        kind="box",
        x="model_name",
        y="mae",
        col="stage",
        row="dataset_name",
        hue="latent_dim",
        sharey="row",
        palette=palette,
        legend=True,
        gap=.1,
        width=0.9,
        showfliers=False,
        whis=2.0,
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
    palette = list(sns.color_palette("colorblind", 3))
    g = sns.catplot(
        data=df,
        kind="box",
        x="model_name",
        y="mse",
        col="stage",
        row="dataset_name",
        hue="latent_dim",
        sharey="row",
        palette=palette,
        legend=True,
        gap=.1,
        width=0.9,
        showfliers=False,
        whis=2.0,
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
    palette = list(sns.color_palette("colorblind", 3))
    g = sns.catplot(
        data=df,
        kind="violin",
        x="model_name",
        y="dkl_norm",
        col="stage",
        row="dataset_name",
        hue="latent_dim",
        sharey="row",
        palette=palette,
        common_norm=True,
        density_norm="area",
        gap=.1,
        bw_adjust=0.75,
        legend=True,
        width=0.9,
    )
    g.set_axis_labels("Model", "DKL")
    if save_dir:
        file_name = (save_dir / "dkl.pdf").expanduser()
        plt.savefig(file_name, format="pdf", bbox_inches="tight")
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
