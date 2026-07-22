import argparse
import hydra
import numpy as np
import pandas as pd
import torch
import rootutils
import seaborn as sns
import logging
import warnings
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib import colors as mcolors
from pathlib import Path
from torchvision import transforms as T
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.data.soundscape_embeddings import SoundscapeEmbeddingsDataModule
from src.core.models.mil_species_detector import MILSpeciesDetector
from src.core.data.rainforest_connection import RainforestConnection
from src.core.utils.sketch import make_ax_invisible

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 10,
})

def darken_color(color, amount=0.7):
    return tuple(amount * channel for channel in mcolors.to_rgb(color))

def lighten_color(color, amount=0.7):
    return tuple(1 - (1 - channel) * amount for channel in mcolors.to_rgb(color))

@torch.no_grad()
def main(data_dir, results_dir, model_dir, save_dir: str | Path | None = None, format: str = "pdf"):
    results = []
    groups = {
        "SO_EC": {
            "vae": ["lumpy-gibson", "slow-partner", "unique-tiger"],
            "sivae": ["just-drum", "dynamic-malta", "daring-system"],
        },
        "SO_UK": {
            "vae": ["lumpy-gibson", "slow-partner", "unique-tiger"],
            "sivae": ["just-drum", "dynamic-malta", "daring-system"],
        },
        "RFCX_bird": {
            "vae": ["jumpy-engine", "quaint-pilot", "numb-chef"],
            "sivae": ["earthy-virgo", "part-armor", "secluded-montana"],
        },
        "RFCX_frog": {
            "vae": ["jumpy-engine", "quaint-pilot", "numb-chef"],
            "sivae": ["earthy-virgo", "part-armor", "secluded-montana"],
        }
    }
    results_df = pd.read_parquet(results_dir, columns=["file_i", "species_name", "label", "prob", "model", "scope"])
    for scope, scope_group in groups.items():
        for model, versions in scope_group.items():
            for version in versions:
                log.info(f"{scope} {model} {version}")
                dm = SoundscapeEmbeddingsDataModule(root=data_dir / version / scope)
                dm.setup()
                df = results_df[(results_df["model"] == version) & (results_df["scope"] == scope)].drop(["model", "scope"], axis=1).pivot(index="file_i", columns="species_name")
                clf = MILSpeciesDetector.load_from_checkpoint(model_dir / f"{version}_{scope}.ckpt", map_location="cpu")

                labels, probs  = df["label"], df["prob"]
                target_counts = clf.target_counts
                sort_idx = np.argsort(-np.array(target_counts))[:len(target_counts)]
                species_names = np.array(clf.target_names)[sort_idx]
                target_counts = target_counts[sort_idx]
                features = dm.test_data.features

                # find the average distance to apply as normalisation to account for differences in latent space diversity
                # this is not as good as mutual information between distributions, but it'll do
                z_means = torch.tensor(features.loc[:, [f"z_mean_{i}" for i in range(128)]].values, dtype=torch.float32)
                norm = torch.sum(z_means**2, dim=1, keepdims=True)
                D = torch.clamp((norm + norm.T - 2 * (z_means @ z_means.T)), min=0).sqrt()
                D = D.masked_fill(torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=0), 0)
                D = D[D != 0].flatten()
                D_avg = D.mean()

                for i, species in enumerate(species_names):
                    y_true = labels.loc[labels[species] == 1, species]
                    y_prob = probs.loc[y_true.index, species]
                    z_mean = torch.tensor(features.loc[y_true.index, [f"z_mean_{i}" for i in range(128)]].values.reshape(len(y_true), -1, 128), dtype=torch.float32)
                    z = z_mean
                    A = clf.attention_weights(z, species)
                    W = clf.species_weights(species)
                    # apply the weights of the classifier to select features, will near zero out irrelevant features
                    z = z * W
                    attn_w, idx = torch.max(A.squeeze(-1), dim=1)
                    # attn_w = attn_w.unsqueeze(-1)
                    # y_prob = torch.tensor(y_prob.to_numpy()).unsqueeze(-1)
                    # prob_diff = abs(y_prob - y_prob.T)
                    # w_diff = abs(attn_w - attn_w.T)
                    z_t = z[torch.arange(z.shape[0]), idx]
                    # pairwise euclidean distance between vectors
                    norm = torch.sum(z_t**2, dim=1, keepdims=True)
                    D = torch.clamp((norm + norm.T - 2 * (z_t @ z_t.T)), min=0).sqrt()
                    # set upper triangle to zero
                    D = D.masked_fill(torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=0), 0)
                    D = D[D != 0].flatten()
                    # normalise by the average to approximate factoring out the marginal distribution
                    D /= D_avg
                    for d in D.numpy():
                        results.append({
                            "model": model,
                            "version": version,
                            "scope": scope,
                            "distance": d,
                            "species_name": species.split("_")[-1],
                            "count": target_counts[i].item()
                        })
    df = pd.DataFrame(results)
    df["model_name"] = df["model"].str.upper()

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    palette = sns.color_palette("colorblind", 4)[1:3]

    so_uk_species = df.loc[df["scope"] == "SO_UK", "species_name"].unique()
    so_ec_species = df.loc[df["scope"] == "SO_EC", "species_name"].unique()
    rfcx_bird_species = df.loc[df["scope"] == "RFCX_bird", "species_name"].unique()
    rfcx_frog_species = df.loc[df["scope"] == "RFCX_frog", "species_name"].unique()
    num_per_ds = 10
    order = [*so_uk_species[:num_per_ds], *so_ec_species[:num_per_ds]]

    sns.boxenplot(
        df,
        x="species_name",
        y="distance",
        hue="model_name",
        order=order,
        hue_order=["VAE", "SIVAE"],
        palette=palette,
        flier_kws={"s": 3},
        ax=ax,
    )
    ax.set_ylabel("Normalised L2 Distance")
    ax.set_xlabel("Species")
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_horizontalalignment('right')
        label.set_rotation_mode('anchor')
    for line in ax.lines:
        line.set_markersize(3)
    sns.move_legend(
        ax,
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncols=2,
        title="",
    )
    if save_dir is not None:
        save_path = save_dir / f"so_z_distances.{format}"
        fig.savefig(save_path, format=format, bbox_inches="tight")
        print(f"Saved: {save_path}")

    # plot the remaining species distances
    for scope, order in zip(["SO UK", "SO EC"], [so_uk_species[num_per_ds:], so_ec_species[num_per_ds:]]):
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        sns.boxenplot(
            df,
            x="species_name",
            y="distance",
            hue="model_name",
            order=order,
            hue_order=["VAE", "SIVAE"],
            palette=palette,
            flier_kws={"s": 3},
            ax=ax,
        )
        ax.set_ylabel("Normalised L2 Distance")
        ax.set_xlabel("Species")
        for label in ax.get_xticklabels():
            label.set_rotation(60)
            label.set_horizontalalignment('right')
            label.set_rotation_mode('anchor')
        ax.set_title(scope)
        sns.move_legend(
            ax,
            loc="lower right", bbox_to_anchor=(1.0, 1.01),
            ncols=2,
            title="",
        )
        if save_dir is not None:
            save_path = save_dir / f"{scope.lower().replace(' ', '_')}_z_distances_rem.{format}"
            fig.savefig(save_path, format=format, bbox_inches="tight")
            print(f"Saved: {save_path}")

    order = [*rfcx_bird_species, *rfcx_frog_species]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.boxenplot(
        df,
        x="species_name",
        y="distance",
        hue="model_name",
        order=order,
        hue_order=["VAE", "SIVAE"],
        palette=palette,
        flier_kws={"s": 3},
        ax=ax,
    )
    ax.set_ylabel("Normalised L2 Distance")
    ax.set_xlabel("Species")
    ax.set_title("RFCX")
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_horizontalalignment('right')
        label.set_rotation_mode('anchor')
    sns.move_legend(
        ax,
        loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncols=2,
        title="",
    )
    if save_dir is not None:
        save_path = save_dir / f"rfcx_z_distances.{format}"
        fig.savefig(save_path, format=format, bbox_inches="tight")
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/data/",
    )
    parser.add_argument(
        "--results-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/results/",
    )
    parser.add_argument(
        "--model-dir",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/models/",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        required=False,
        help="file format (pdf, png, jpg, etc)",
    )
    args = parser.parse_args()
    main(**vars(args))
