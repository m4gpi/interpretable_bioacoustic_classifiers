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
from src.core.models.species_detector import SpeciesDetector
from src.core.models.base_vae import BaseVAE
from src.core.models.smooth_nifti_vae import SmoothNiftiVAE
from src.core.data.rainforest_connection import RainforestConnection
from src.core.utils.sketch import plot_mel_spectrogram, make_ax_invisible
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils import tree
from src.cli.utils.instantiators import instantiate_transforms

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.rcParams.update({
    'axes.labelsize': 12,
    'xtick.labelsize': 6,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
})

def darken_color(color, amount=0.7):
    return tuple(amount * channel for channel in mcolors.to_rgb(color))

def lighten_color(color, amount=0.7):
    return tuple(1 - (1 - channel) * amount for channel in mcolors.to_rgb(color))

@torch.no_grad()
def main(
    data_dir,
    results_dir,
    save_dir,
    format: str = "pdf",
):
    results = []
    groups = {
        "SO_EC": {
            "base_vae": [
                "v4",
                "v5",
                "v6"
            ],
            "nifti_vae": [
                "v12",
                "v17",
                "v18"
            ],
        },
        "SO_UK": {
            "base_vae": [
                "v4",
                "v5",
                "v6"
            ],
            "nifti_vae": [
                "v12",
                "v17",
                "v18"
            ],
        },
    }
    for scope, scope_group in groups.items():
        for model, versions in scope_group.items():
            for version in versions:
                log.info(f"{scope} {model} {version}")
                dm = SoundscapeEmbeddingsDataModule(
                    root=data_dir,
                    model=model,
                    version=version,
                    scope=scope,
                    transforms=None,
                )
                dm.setup()
                results_df = pd.read_parquet(results_dir, columns=["file_i", "species_name", "label", "prob", "model", "version", "scope"])
                df = (
                    results_df[
                        (results_df["model"] == model) &
                        (results_df["version"] == version) &
                        (results_df["scope"] == scope)
                    ]
                    .drop(["model", "version", "scope"], axis=1)
                    .pivot(index="file_i", columns="species_name")
                )
                clf = SpeciesDetector(
                    target_names=dm.train_data.target_names,
                    target_counts=dm.train_data.target_counts,
                    in_features=128,
                    attn_dim=10,
                    pool_method="prob_attn",
                    beta=0.0,
                    key_per_target=True,
                )
                checkpoint = torch.load(f"results/species_detectors/checkpoints/{scope.lower()}_{model}.pt:{version}.ckpt", map_location="cpu")
                clf.load_state_dict(checkpoint["state_dict"])

                labels, probs  = df["label"], df["prob"]
                target_counts = np.array(dm.train_data.target_counts)
                sort_idx = np.argsort(-np.array(dm.train_data.target_counts))[:len(dm.train_data.target_counts)]
                species_names = np.array(dm.train_data.target_names)[sort_idx]
                target_counts = target_counts[sort_idx]
                features = dm.test_data.features

                # find the average distance to apply as normalisation to account for differences in latent space diversity
                # this is an approximation to normalising by the marginal distribution
                z_all = torch.tensor(features.loc[:, [f"z_mean_{i}" for i in range(128)]].values, dtype=torch.float32)
                norm = torch.sum(z_all**2, dim=1, keepdims=True)
                D = torch.clamp((norm + norm.T - 2 * (z_all @ z_all.T)), min=0).sqrt()
                D = D.masked_fill(torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=0), 0)
                D = D[D != 0].flatten()
                D_avg = D.mean()

                for i, species in enumerate(species_names):
                    y_true = labels.loc[labels[species] > 0, species]
                    y_prob = probs.loc[y_true.index, species]
                    z_mean = torch.tensor(features.loc[y_true.index, [f"z_mean_{i}" for i in range(128)]].values.reshape(len(y_true), -1, 128), dtype=torch.float32)
                    # z_log_var = torch.tensor(features.loc[y_true.index, [f"z_log_var_{i}" for i in range(128)]].values.reshape(len(y_true), -1, 128), dtype=torch.float32)
                    z = z_mean # + torch.randn_like(z_mean) * (z_log_var * 0.5).exp()
                    A = clf.attention_weights(z, species)
                    W = clf.classifier_weights(species).unsqueeze(0)
                    # apply the weights of the classifier to select features, will near zero out irrelevant features
                    z = z * W
                    attn_w, idx = torch.max(A.squeeze(-1), dim=1)
                    # attn_w = attn_w.unsqueeze(-1)
                    # y_prob = torch.tensor(y_prob.to_numpy()).unsqueeze(-1)
                    # prob_diff = abs(y_prob - y_prob.T)
                    # w_diff = abs(attn_w - attn_w.T)
                    z_t = z[torch.arange(z.shape[0]), idx]
                    # l2 distance between vectors
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
                            "scope": scope,
                            "distance": d,
                            "species_name": species.split("_")[-1],
                            "count": target_counts[i]
                        })
    df = pd.DataFrame(results)
    # rfcx_df = df[(df["scope"] == "RFCX_bird") | (df["scope"] == "RFCX_frog")]

    fig, ax1 = plt.subplots(figsize=(8.1, 4))
    palette = ["blue", "red"] #sns.color_palette("husl", 4)[1:3]
    dark_palette = [color for color in palette]
    light_palette = [lighten_color(color, 0.6) for color in palette]
    ax2 = ax1.twinx()
    ax2_top = ax1.twiny()

    max_cats = df.groupby("scope")["species_name"].nunique().max()

    so_uk_df = df[(df["scope"] == "SO_UK")]
    so_uk_df = so_uk_df.sort_values(by="count", ascending=False)
    position, species = pd.factorize(so_uk_df["species_name"])
    so_uk_df["position"] = position
    so_uk_df["model_name"] = so_uk_df.model.map(dict(base_vae="VAE", nifti_vae="SIVAE"))
    sns.stripplot(
        so_uk_df,
        x="position",
        y="distance",
        hue="model_name",
        hue_order=["VAE", "SIVAE"],
        ax=ax1,
        palette=dark_palette,
        size=2,
        alpha=0.05,
        dodge=True,
        jitter=0.3,
        legend=True,
    )
    xticklabels = []
    for i in range(max_cats):
        cat = "" if i >= len(species) else species[i]
        xticklabels.append(cat)
    ax1.set_xticks(range(max_cats), xticklabels)
    ax1.set_ylabel("Normalised L2 Distance")
    ax1.set_xlabel("Species (UK)")
    ax1.tick_params(axis="x", rotation=90)
    ax1.set_ylim([0.0, 1.25])

    so_ec_df = df[(df["scope"] == "SO_EC")]
    so_ec_df = so_ec_df.sort_values(by="count", ascending=True)
    position, species = pd.factorize(so_ec_df["species_name"])
    so_ec_df["position"] = position.astype(str)
    so_ec_df["model_name"] = so_ec_df.model.map(dict(base_vae="VAE", nifti_vae="SIVAE"))
    sns.stripplot(
        so_ec_df,
        x="position",
        y="distance",
        hue="model_name",
        hue_order=["VAE", "SIVAE"],
        ax=ax2,
        palette=light_palette,
        size=2,
        alpha=0.25,
        dodge=True,
        jitter=0.3,
        legend=False,
    )
    xticklabels = []
    for i in range(max_cats):
        cat = "" if i <= max_cats - len(species) else species[i]
        xticklabels.append(cat)
    ax2_top.set_xlim(-0.5, len(species) - 0.5)
    ax2_top.set_xticks(range(max_cats), xticklabels)
    ax2.set_ylabel("Normalised L2 Distance")
    ax2_top.set_xlabel("Species (EC)")
    ax2_top.tick_params(axis="x", rotation=90)
    ax2.set_ylim([0.0, 1.0])

    ax2.invert_yaxis()

    handles, labels = ax1.get_legend_handles_labels()
    for h in handles:
        h.set_alpha(1.0)
        h.set_markersize(6)
    ax1.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(1.00, -0.02),
        title="Model",
        ncols=2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    save_path = save_dir / f"so_z_distances.{format}"
    fig.savefig(save_path, format=format)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--results-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
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
