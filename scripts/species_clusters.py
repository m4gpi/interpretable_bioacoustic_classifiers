import argparse
import hydra
import numpy as np
import pandas as pd
import torch
import rootutils
import seaborn as sns
import sklearn
import logging
import warnings
import yaml
import umap

warnings.filterwarnings("ignore", category=FutureWarning)

from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib import colors as mcolors
from matplotlib import cm
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
    'xtick.labelsize': 12,
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
            ],
            "nifti_vae": [
                "v12",
            ],
        },
        "SO_UK": {
            "base_vae": [
                "v4",
            ],
            "nifti_vae": [
                "v12",
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

                for i, species in enumerate(species_names):
                    y_true = labels.loc[labels[species] > 0, species]
                    y_prob = probs.loc[y_true.index, species]
                    z_mean = torch.tensor(features.loc[y_true.index, [f"z_mean_{i}" for i in range(128)]].values.reshape(len(y_true), -1, 128), dtype=torch.float32)
                    # z_log_var = torch.tensor(features.loc[y_true.index, [f"z_log_var_{i}" for i in range(128)]].values.reshape(len(y_true), -1, 128), dtype=torch.float32)
                    z = z_mean # + torch.randn_like(z_mean) * (z_log_var * 0.5).exp()
                    A = clf.attention_weights(z, species)
                    W = clf.classifier_weights(species).unsqueeze(0)
                    # apply the weights of the classifier to select features, will near zero out irrelevant features
                    z_w = z * W
                    attn_w, idx = torch.max(A.squeeze(-1), dim=1)
                    z_t = z_w[torch.arange(z_w.shape[0]), idx]
                    frame_prob = torch.sigmoid(z[torch.arange(z_w.shape[0]), idx].unsqueeze(1) @ W.transpose(-1, -2))
                    for j in range(z_t.shape[0]):
                        results.append({
                            "model": model,
                            "scope": scope,
                            **{f"z_{k}": z_t[j, k].item() for k in range(z_t.shape[1])},
                            "species_name": species.split("_")[-1],
                            "count": target_counts[i],
                            "prob": frame_prob[j].item(),
                        })
    df = pd.DataFrame(results)
    df = df[df["count"] > 50].copy()
    from src.core.utils.sketch import make_ax_invisible
    fig, axes = plt.subplots(figsize=(12, 3), ncols=5, width_ratios=[0.248, 0.248, 0.248, 0.248, 0.008])
    name_map = dict(base_vae="VAE", nifti_vae="SIVAE")
    # lims = [[-2.5, 2.5], [-2.5, 2.5], [-4.5, 4.5], [-4.5, 4.5]]
    i = 0
    for scope in df["scope"].unique():
        for model_name in df["model"].unique():
            sub_df = df[(df["model"] == model_name) & (df["scope"] == scope)].copy()
            model = sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.StandardScaler(),
                umap.UMAP(n_components=2, metric="euclidean", min_dist=0.1, n_neighbors=5),
                sklearn.preprocessing.StandardScaler(),
            )
            zs = sub_df.loc[:, [f"z_{k}" for k in range(128)]]
            x, y = np.split(model.fit_transform(zs), 2, axis=1)
            sub_df["x"] = x.squeeze()
            sub_df["y"] = y.squeeze()
            ax = axes[i]
            sns.scatterplot(
                sub_df,
                x="x",
                y="y",
                style="species_name",
                hue="prob",
                size=3,
                ax=ax,
                palette="flare",
                legend=False,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"{name_map[model_name]} ({scope.split('_')[-1]})")
            ax.set_xlim([-2.5, 2.5])
            ax.set_ylim([-2.5, 2.5])
            i +=1
    axes[0].set_xlabel("UMAP Dim 1")
    axes[1].set_xlabel("UMAP Dim 1")
    axes[2].set_xlabel("UMAP Dim 1")
    axes[3].set_xlabel("UMAP Dim 1")
    axes[0].set_ylabel("UMAP Dim 2")

    sm = cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="flare")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=axes[-1])
    cbar.set_label("p(y)", rotation=0)

    fig.tight_layout()
    save_path = save_dir / f"so_species_umap.{format}"
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
