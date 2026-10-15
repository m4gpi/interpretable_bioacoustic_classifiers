import argparse
import logging
import numpy as np
import os
import pandas as pd
import rootutils
import seaborn as sns
import sys
import torch
import umap

from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Colormap, Normalize
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import ticker
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import metrics

from torch import Tensor
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.models.vae import VAE
from src.core.utils.metrics import average_precision
from src.core.utils.sketch import make_ax_invisible
from src.core.data.sounding_out_chorus import SoundingOutChorus

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def plot_confusion_matrix(
    y_true: Tensor | NDArray,
    y_pred: Tensor | NDArray,
    classes: List | NDArray,
    cmap: Colormap = sns.color_palette("rocket", as_cmap=True),
    ax: Axes | None = None,
    figsize: Tuple[int, int] | None = (8, 6),
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    fig, ax = (plt.gcf, ax) if ax is not None else plt.subplots(figsize=figsize)
    cm = metrics.confusion_matrix(y_true, y_pred, **kwargs)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, classes, classes)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    acc = metrics.accuracy_score(y_true, y_pred)
    ax.set_title(f"Accuracy: {np.format_float_positional(acc, precision=3)}")
    return fig, ax

def pairwise_feature_importance(
    data: pd.DataFrame,
    weights: NDArray,
    group_by: str,
    feature_columns: str,
    labels: List[Tuple[str, str]],
) -> NDArray:
    if not isinstance(weights, list): weights = [weights]
    if not isinstance(labels, list): labels = [labels]
    xs = np.empty((len(labels), len(feature_columns)))
    for i, (W, class_names) in enumerate(zip(weights, labels)):
        class_name_i, class_name_j = class_names
        class_idx = (data.loc[:, group_by] == class_name_i) | (data.loc[:, group_by] == class_name_j)
        X = data.loc[class_idx, feature_columns].values
        # relative contribution of latent dimension to variation in decision
        # latent dimensions can have different scales, taking std accounts for spread in each dimension
        feature_importance = abs(X.std(axis=0) * W.squeeze())
        # normalise with some epsilon to prevent zero division
        xs[i] = feature_importance / (feature_importance.sum() + 1e-8)
    return xs

def plot_feature_importance(
    data: NDArray,
    y_tick_labels: List[str] = [],
    x_tick_labels: List[str] = [],
    cmap: Colormap = "pink_r",
    figsize: Tuple[int, int] = (15, 3),
    orientation: str = "horizontal",
    ax: Axes = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize) if not ax else (plt.gcf(), ax)
    if orientation == 'vertical':
        data = data.T
        x_tick_labels, y_tick_labels = y_tick_labels, x_tick_labels
    xticklabels = x_tick_labels if len(x_tick_labels) else list(range(data.shape[1]))
    yticklabels = y_tick_labels if len(y_tick_labels) else list(range(data.shape[0]))
    res = sns.heatmap(data, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax, **kwargs)
    if orientation == 'vertical':
        ax.yaxis.tick_right()
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)
        for label in ax.get_yticklabels():
            label.set_fontsize(6)
        for label in ax.get_xticklabels():
            label.set_fontsize(6)
    else:
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)
        for label in ax.get_xticklabels():
            label.set_fontsize(8)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1)
    return fig, ax

def cross_hyperplane(
    z: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | float,
    delta: float = 1e-3
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(W)
    d = (z @ W.T + b) / norm
    return z - (d + torch.sign(d) * delta) * (W / norm)

def main(root, save_path, seed: int = 42):
    save_path.mkdir(exist_ok=True, parents=True)
    data_dir = root / "data"
    model_name = "lumpy-gibson"

    # --------------------------------- Prepare Data --------------------------------- #

    uk_train = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_UK" / "train" / "features.parquet")
    ec_train = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_EC" / "train" / "features.parquet")
    train_features = pd.concat([uk_train, ec_train]).reset_index()

    uk_val = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_UK" / "val" / "features.parquet")
    ec_val = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_EC" / "val" / "features.parquet")
    val_features = pd.concat([uk_val, ec_val]).reset_index()

    uk_test = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_UK" / "test" / "features.parquet")
    ec_test = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_EC" / "test" / "features.parquet")
    test_features = pd.concat([uk_test, ec_test]).reset_index()

    metadata = pd.read_parquet(data_dir / "sounding_out" / "metadata.parquet").reset_index()
    data = pd.concat([train_features, val_features, test_features]).merge(metadata[["file_i", "file_name", "recorder_model", "country", "habitat"]], on="file_i", how="left")

    train_idx = data["dataloader_idx"] == 0
    test_idx = data["dataloader_idx"] == 2
    uk_idx = data["country"] == "UK"
    ec_idx = data["country"] == "EC"
    sm2_idx = data["recorder_model"] == "SM2"
    sm3_idx = data["recorder_model"] == "SM3"
    z_idx = [f"z_mean_{i}" for i in range(128)]
    z_norm_idx = [f"z_norm_{i}" for i in range(128)]

    # --------------------------------- OVO Habitat Classification --------------------------------- #

    y, habitats = data["habitat"].factorize(sort=True)
    data["y"] = y.astype(int)
    mapper = dict(zip(*habitats.factorize()))

    log.info("OVO habitat classification")
    ensemble = OneVsOneClassifier(LogisticRegression(penalty="l1", solver="saga",  max_iter=10000, random_state=seed), n_jobs=-1)
    ensemble.fit(data.loc[train_idx, z_idx], data.loc[train_idx, "y"])
    data.loc[train_idx, "y_pred"] = ensemble.predict(data.loc[train_idx, z_idx])
    data.loc[test_idx, "y_pred"] = ensemble.predict(data.loc[test_idx, z_idx])
    data["h_pred"] = data["y_pred"].map(mapper)

    log.info("Plotting confusion")
    fig = plt.figure(figsize=(11, 4), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[0.35, 0.65], height_ratios=[0.6, 0.4])
    ax = fig.add_subplot(gs[:, 0])
    grid_spec = fig.add_gridspec()
    plot_confusion_matrix(data.loc[test_idx, "y"].astype(int), data.loc[test_idx, "y_pred"].astype(int), habitats, ax=ax)

    A = { f"{a}_{b}": model for (a, b), model in list(zip(combinations([habitats[i] for i in ensemble.classes_], 2), ensemble.estimators_))  }
    B = { f"{b}_{a}": model for (a, b), model in list(zip(combinations([habitats[i] for i in ensemble.classes_], 2), ensemble.estimators_))  }
    ensemble_dict = A | B
    A = { f"{a}_{b}": (0, 1) for (a, b), model in list(zip(combinations([habitats[i] for i in ensemble.classes_], 2), ensemble.estimators_))  }
    B = { f"{b}_{a}": (1, 0) for (a, b), model in list(zip(combinations([habitats[i] for i in ensemble.classes_], 2), ensemble.estimators_))  }
    class_map = A | B

    uk_labels = [(a, b) for (a, b) in combinations(data[data.country == "UK"].habitat.unique(), 2)]
    ec_labels = [(a, b) for (a, b) in combinations(data[data.country == "EC"].habitat.unique(), 2)]
    uk_weights = [ensemble_dict[f"{a}_{b}"].coef_ for (a, b) in combinations(data[data.country == "UK"].habitat.unique(), 2)]
    ec_weights = [ensemble_dict[f"{a}_{b}"].coef_ for (a, b) in combinations(data[data.country == "EC"].habitat.unique(), 2)]

    log.info("Plotting relative feature importance")
    uk_imp = pairwise_feature_importance(data=data[train_idx & uk_idx], weights=uk_weights, labels=uk_labels, group_by="habitat", feature_columns=z_idx)
    ec_imp = pairwise_feature_importance(data=data[train_idx & ec_idx], weights=ec_weights, labels=ec_labels, group_by="habitat", feature_columns=z_idx)
    y_tick_labels = [*[fr"{i} $\Leftrightarrow$ {j}" for i, j in uk_labels], *[fr"{i} $\Leftrightarrow$ {j}" for i, j in ec_labels]]
    ax = fig.add_subplot(gs[0, 1])
    plot_feature_importance(data=np.vstack([uk_imp, ec_imp]), y_tick_labels=y_tick_labels, ax=ax, cbar_kws={'label': "Relative Importance (%)"})
    ax.xaxis.set_tick_params(labelsize=4, rotation=90)
    ax.set_ylabel("Habitat Pair")
    ax.set_xlabel("Latent Dimension")
    log.info(save_path / "habitat_prediction.pdf")
    fig.savefig(save_path / "habitat_prediction.pdf", format="pdf")
    plt.close()

    # --------------------------------- Classifier-guided Interpolation between Habitat Means --------------------------------- #

    # habitat classifier-guided latent space sampling
    ckpt_path = root / "models" / "v4" / "vae" / model_name / "step=180000.ckpt"
    log.info(f"loading model {ckpt_path}")
    model = VAE.load_from_checkpoint(ckpt_path, map_location="cuda")

    # start with habitat means and use the classifier direction to see an overall change
    # create a 3x3 grid with empty diagonal
    uk_habs = sorted(data[data["country"] == "UK"].habitat.unique())
    hab_to_idx = dict(zip(uk_habs, range(len(uk_habs))))
    rmin, rmax = np.inf, -np.inf
    residuals = torch.empty(3, 3, 192, 64)
    for h1, h2 in uk_labels:
        log.info(f"{h1} <-> {h2} interpolation")
        clf = ensemble_dict[f"{h1}_{h2}"]
        h1_idx = train_idx & (data["habitat"] == h1)
        h2_idx = train_idx & (data["habitat"] == h2)
        h1_mean = data.loc[h1_idx, z_idx].mean(axis=0).values
        h2_mean = data.loc[h2_idx, z_idx].mean(axis=0).values
        z_h1 = torch.tensor(h1_mean, device=model.device, dtype=torch.float32).unsqueeze(0)
        z_h2 = torch.tensor(h2_mean, device=model.device, dtype=torch.float32).unsqueeze(0)
        W = torch.tensor(clf.coef_, device=model.device, dtype=torch.float32)
        b = torch.tensor(clf.intercept_[0], device=model.device, dtype=torch.float32)
        # calculate direction and signed distance from hyperplane, shift to decision boundary plus some additional epsilon
        z_tilde_h1 = cross_hyperplane(z_h1, W, b)
        z_tilde_h2 = cross_hyperplane(z_h2, W, b)
        # check the class changed, h1 == h2, h2 == h1
        assert (clf.predict(z_tilde_h1.cpu().numpy()) != clf.predict(z_h1.cpu().numpy())).all()
        assert (clf.predict(z_tilde_h2.cpu().numpy()) != clf.predict(z_h2.cpu().numpy())).all()
        with torch.no_grad():
            x_hat_h1 = model.decode(z_h1.unsqueeze(0)).squeeze().cpu()
            x_hat_h2 = model.decode(z_h2.unsqueeze(0)).squeeze().cpu()
            x_tilde_h1 = model.decode(z_tilde_h1.unsqueeze(0)).squeeze().cpu()
            x_tilde_h2 = model.decode(z_tilde_h2.unsqueeze(0)).squeeze().cpu()
        # take difference and map to decibel range
        x_res_h1 = (20 / np.log(10)) * (x_tilde_h1 - x_hat_h1)
        x_res_h2 = (20 / np.log(10)) * (x_tilde_h2 - x_hat_h2)
        rmin = min(rmin, x_res_h1.min())
        rmin = min(rmin, x_res_h2.min())
        rmax = max(rmax, x_res_h1.max())
        rmax = max(rmax, x_res_h2.max())
        residuals[hab_to_idx[h1], hab_to_idx[h2]] = x_res_h1
        residuals[hab_to_idx[h2], hab_to_idx[h1]] = x_res_h2
    # plot each interpolation
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=6, width_ratios=[0.01, 0.01, 0.32, 0.32, 0.32, 0.02])
    axes = np.empty((3, 6), dtype=object)
    for i in range(3):
        for j in range(1, 5):
            axes[i, j] = fig.add_subplot(gs[i, j])
    for i in range(3):
        ax = axes[i, 1]
        make_ax_invisible(ax)
        ax.set_ylabel(uk_habs[i])
    for i in range(3):
        for j in range(2, 5):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(uk_habs[j - 2])
            if i == j - 2:
                make_ax_invisible(ax)
                continue
            residual = residuals[i][j - 2]
            bound = max(np.abs(rmin), np.abs(rmax))
            mesh = model.front_end.plot(residual.T, cmap="bwr", vmin=-bound, vmax=bound, ax=ax)
            ax.tick_params(labelbottom=False, bottom=False)
    ax = fig.add_subplot(gs[1, 0])
    make_ax_invisible(ax)
    ax.set_ylabel("Start Class")
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle("End Class")
    log.info(save_path / "so_uk_habitat_average_residuals.pdf")
    fig.savefig(save_path / "so_uk_habitat_average_residuals.pdf", format="pdf")
    plt.close()

    # start with habitat means and use the classifier direction to see an overall change
    # create a 3x3 grid with empty diagonal
    ec_habs = sorted(data[data["country"] == "EC"].habitat.unique())
    hab_to_idx = dict(zip(ec_habs, range(len(ec_habs))))
    rmin, rmax = np.inf, -np.inf
    residuals = torch.empty(3, 3, 192, 64)
    for h1, h2 in ec_labels:
        log.info(f"{h1} <-> {h2} interpolation")
        clf = ensemble_dict[f"{h1}_{h2}"]
        h1_idx = train_idx & (data["habitat"] == h1)
        h2_idx = train_idx & (data["habitat"] == h2)
        h1_mean = data.loc[h1_idx, z_idx].mean(axis=0).values
        h2_mean = data.loc[h2_idx, z_idx].mean(axis=0).values
        z_h1 = torch.tensor(h1_mean, device=model.device, dtype=torch.float32).unsqueeze(0)
        z_h2 = torch.tensor(h2_mean, device=model.device, dtype=torch.float32).unsqueeze(0)
        W = torch.tensor(clf.coef_, device=model.device, dtype=torch.float32)
        b = torch.tensor(clf.intercept_[0], device=model.device, dtype=torch.float32)
        # calculate direction and signed distance from hyperplane, shift to decision boundary plus some additional epsilon
        z_tilde_h1 = cross_hyperplane(z_h1, W, b)
        z_tilde_h2 = cross_hyperplane(z_h2, W, b)
        # check the class changed, h1 == h2, h2 == h1
        assert (clf.predict(z_tilde_h1.cpu().numpy()) != clf.predict(z_h1.cpu().numpy())).all()
        assert (clf.predict(z_tilde_h2.cpu().numpy()) != clf.predict(z_h2.cpu().numpy())).all()
        with torch.no_grad():
            x_hat_h1 = model.decode(z_h1.unsqueeze(0)).squeeze().cpu()
            x_hat_h2 = model.decode(z_h2.unsqueeze(0)).squeeze().cpu()
            x_tilde_h1 = model.decode(z_tilde_h1.unsqueeze(0)).squeeze().cpu()
            x_tilde_h2 = model.decode(z_tilde_h2.unsqueeze(0)).squeeze().cpu()
        # take difference and map to decibel range
        x_res_h1 = (20 / np.log(10)) * (x_tilde_h1 - x_hat_h1)
        x_res_h2 = (20 / np.log(10)) * (x_tilde_h2 - x_hat_h2)
        rmin = min(rmin, x_res_h1.min())
        rmin = min(rmin, x_res_h2.min())
        rmax = max(rmax, x_res_h1.max())
        rmax = max(rmax, x_res_h2.max())
        residuals[hab_to_idx[h1], hab_to_idx[h2]] = x_res_h1
        residuals[hab_to_idx[h2], hab_to_idx[h1]] = x_res_h2
    # plot each interpolation
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=6, width_ratios=[0.01, 0.01, 0.32, 0.32, 0.32, 0.02])
    axes = np.empty((3, 6), dtype=object)
    for i in range(3):
        for j in range(1, 5):
            axes[i, j] = fig.add_subplot(gs[i, j])
    for i in range(3):
        ax = axes[i, 1]
        make_ax_invisible(ax)
        ax.set_ylabel(ec_habs[i])
    for i in range(3):
        for j in range(2, 5):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(ec_habs[j - 2])
            if i == j - 2:
                make_ax_invisible(ax)
                continue
            residual = residuals[i][j - 2]
            bound = max(np.abs(rmin), np.abs(rmax))
            mesh = model.front_end.plot(residual.T, cmap="bwr", vmin=-bound, vmax=bound, ax=ax)
            ax.tick_params(labelbottom=False, bottom=False)
    ax = fig.add_subplot(gs[1, 0])
    make_ax_invisible(ax)
    ax.set_ylabel("Start Class")
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Magnitude (dB)")
    fig.suptitle("End Class")
    log.info(save_path / "so_ec_habitat_average_residuals.pdf")
    fig.savefig(save_path / "so_ec_habitat_average_residuals.pdf", format="pdf")
    plt.close()

    # --------------------------------- Classifier-guided Interpolation between Habitats by Sample --------------------------------- #

    dataset = SoundingOutChorus(data_dir / "sounding_out")
    samples = {
        "UK": {
            "UK1": [{
                "file_name": "PL-14_0_20150604_0700.wav",
                "description": "Dawn chorus & road",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }, {
                "file_name": "PL-09_0_20150605_0430.wav",
                "description": "Dawn chorus",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }],
            "UK2": [{
                "file_name": "KN-13_0_20150510_0715.wav",
                "description": "Dawn chorus & road",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }, {
                "file_name": "KN-04_0_20150510_0545.wav",
                "description": "Dawn chorus",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }],
            "UK3": [{
                "file_name": "BA-09_0_20150619_0615.wav",
                "description": "Birds & wind",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }, {
                "file_name": "BA-01_0_20150620_0432.wav",
                "description": "Birds & Insects",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }],
        },
        "EC": {
            "EC1": [{
                "file_name": "TE-05_0_20150713_0620.wav",
                "description": "Dawn chorus & wind",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }, {
                "file_name": "TE-01_0_20150713_1830.wav",
                "description": "Frogs, Birds & Insects",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }],
            "EC2": [{
                "file_name": "FS-11_0_20150808_0635.wav",
                "description": "Running water",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }, {
                "file_name": "FS-08_0_20150806_1835.wav",
                "description": "Frogs & Insects",
                "t_start": 3.072 * 10,
                "t_end": 3.072 * 10 + 3.072 * 5,
            }],
            "EC3": [{
                "file_name": "PO-14_0_20150818_0705.wav",
                "description": "Chainsaw",
                "t_start": 3.072 * 0,
                "t_end": 3.072 * 5,
            }, {
                "file_name": "PO-02_0_20150818_1845.wav",
                "description": "Frogs & Insects",
                "t_start": 3.072 * 8,
                "t_end": 3.072 * 8 + 3.072 * 5,
            }],
        }
    }
    sample_md = data[data.file_name.isin([x["file_name"] for country, group in samples.items() for site, xs in group.items() for x in xs])]
    for country, site_group in samples.items():
        for site_a, group in site_group.items():
            # fetch this sample
            for i, sample in enumerate(group):
                file_name, description, t_start, t_end = sample.values()
                log.info(f"{file_name} - {description}")
                wav = dataset.load_sample(dataset.metadata[dataset.metadata.file_name == file_name].iloc[0].file_path)
                x = model.front_end(wav.to(model.device).squeeze(0)).squeeze().cpu().T
                z = sample_md.loc[(sample_md.file_name == file_name) & (sample_md["timestep"] % 2 == 0), z_idx].values
                z = torch.tensor(z, device=model.device, dtype=torch.float32)
                # decode original
                with torch.no_grad():
                    x_hat = model.decode(z.unsqueeze(0)).squeeze().cpu()
                results = []
                # interpolate to other sites
                for site_b in samples[country].keys():
                    if site_a == site_b:
                        continue
                    clf = ensemble_dict[f"{site_a}_{site_b}"]
                    W = torch.tensor(clf.coef_, device=model.device, dtype=torch.float32)
                    b = torch.tensor(clf.intercept_[0], device=model.device, dtype=torch.float32)
                    # check z-classification, if fails, derive mask
                    t, f =  class_map[f"{site_a}_{site_b}"]
                    y_pred = clf.predict(z.cpu().numpy())
                    # calculate direction and signed distance from hyperplane
                    # shift to decision boundary plus some additional epsilon
                    delta = 1e-3
                    norm = torch.linalg.vector_norm(W)
                    d = (z @ W.T + b) / norm
                    z_tilde = z - (d + torch.sign(d) * delta) * (W / norm)
                    # use mask to overwrite latents before reconstruction so residual is null
                    z_tilde[y_pred == f] = z[y_pred == f]
                    # decode
                    with torch.no_grad():
                        x_tilde = model.decode(z_tilde.unsqueeze(0)).squeeze().cpu()
                    # calculate residual
                    x_res = (x_tilde - x_hat)
                    results.append((site_b, x_tilde, x_res))
                # plot
                t_start = int(model.front_end.sample_rate * t_start // model.front_end.fft_hop_length)
                t_end = int(model.front_end.sample_rate * t_end // model.front_end.fft_hop_length)
                fig = plt.figure(constrained_layout=True, figsize=(10, 8))
                gs = fig.add_gridspec(nrows=4, ncols=2, width_ratios=[0.98, 0.02])
                fig.suptitle(f"{site_a}: {description}")
                site_b, x_tilde_b, x_res_b = results[0]
                site_c, x_tilde_c, x_res_c = results[1]
                # original & recon
                ax = fig.add_subplot(gs[0, 0])
                x = (20 / np.log(10)) * x[t_start:t_end].T
                x_hat = (20 / np.log(10)) * x_hat[t_start:t_end].T
                vmin, vmax = min(x.min(), x_hat.min()), max(x.max(), x_hat.max())
                mesh = model.front_end.plot(x, cmap="Greys", vmin=vmin, vmax=vmax, ax=ax)
                ax.set_title("Original")
                ax = fig.add_subplot(gs[1, 0])
                mesh = model.front_end.plot(x_hat, cmap="Greys", vmin=vmin, vmax=vmax, ax=ax)
                ax.set_title("Reconstruction")
                # shared colorbar
                cax = fig.add_subplot(gs[:2,-1])
                cbar = fig.colorbar(mesh, cax=cax)
                cbar.set_label("Magnitude (dB)")
                # residuals
                x_res_b = (20 / np.log(10)) * x_res_b[t_start:t_end].T
                x_res_c = (20 / np.log(10)) * x_res_c[t_start:t_end].T
                bound = max(np.abs(x_res_b.min()), np.abs(x_res_b.max()), np.abs(x_res_c.min()), np.abs(x_res_c.max()))
                # a -> b
                ax = fig.add_subplot(gs[2, 0])
                mesh = model.front_end.plot(x_res_b, cmap="bwr", vmin=-bound, vmax=bound, ax=ax)
                ax.set_title(rf"{site_a} $\longrightarrow$ {site_b} Residual")
                # a -> c
                ax = fig.add_subplot(gs[3, 0])
                mesh = model.front_end.plot(x_res_c, cmap="bwr", vmin=-bound, vmax=bound, ax=ax)
                ax.set_title(rf"{site_a} $\longrightarrow$ {site_c} Residual")
                # shared colorbar
                cax = fig.add_subplot(gs[2:,-1])
                cbar = fig.colorbar(mesh, cax=cax)
                cbar.set_label("Magnitude (dB)")
                # save
                fig_path = save_path / f"{site_a}_{i}_interpolation.pdf"
                log.info(fig_path)
                fig.savefig(fig_path, format="pdf")
                plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/root/",
    )
    parser.add_argument(
        "--save-path",
        type=lambda p: Path(p),
        required=False,
        help="/path/to/saved/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed",
    )
    args = parser.parse_args()
    main(**vars(args))
