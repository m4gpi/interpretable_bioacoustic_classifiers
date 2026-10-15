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

from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, Normalize
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import ticker
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from torch import Tensor
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.models.vae import VAE
from src.core.utils.metrics import average_precision

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

def plot_roc_curve(
    y: NDArray,
    y_probs: NDArray,
    label_prefix: str = "",
    ax: Axes | None = None,
    figsize: Tuple[int, int] | None = (8, 6),
    **kwargs: Any,
) -> Figure:
    fig, ax = (plt.gcf(), ax) if ax is not None else plt.subplots(figsize=figsize)
    auroc = metrics.roc_auc_score(y, y_probs)
    fpr, tpr, _ = metrics.roc_curve(y, y_probs, pos_label=1)
    ax.plot(fpr, tpr, label=f"{label_prefix}AUC: {np.format_float_positional(auroc, precision=3)}", **kwargs)
    ax.plot(np.linspace(0, 1, len(fpr)), np.linspace(0, 1, len(fpr)), linestyle="dashed", color="black")
    ax.legend(loc="lower right")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    return fig

def plot_pr_curve(
    y: NDArray,
    y_probs: NDArray,
    pos_label: int = 1,
    label_prefix: str = "",
    ax: Axes | None = None,
    figsize: Tuple[int, int] | None = (8, 6),
    **kwargs: Any,
) -> Figure:
    fig, ax = (plt.gcf(), ax) if ax is not None else plt.subplots(figsize=figsize)
    mAP = average_precision(y, y_probs)
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_probs, pos_label=pos_label)
    ax.plot(recall, precision, label=f"{label_prefix}mAP: {np.format_float_positional(mAP, precision=3)}", **kwargs)
    ax.legend(loc="lower right")
    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision (PPV)")
    return fig, thresholds


def main(root, save_path, seed: int = 42):
    save_path.mkdir(exist_ok=True, parents=True)
    data_dir = root / "data"
    model_name = "lumpy-gibson"

    # --------------------------------- Prepare Data --------------------------------- #

    uk_train = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_UK" / "train" / "features.parquet")
    ec_train = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_EC" / "train" / "features.parquet")
    train_features = pd.concat([uk_train, ec_train]).reset_index()

    uk_test = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_UK" / "test" / "features.parquet")
    ec_test = pd.read_parquet(data_dir / "soundscape_vae_embeddings" / model_name / "SO_EC" / "test" / "features.parquet")
    test_features = pd.concat([uk_test, ec_test]).reset_index()

    metadata = pd.read_parquet(data_dir / "sounding_out" / "metadata.parquet").reset_index()
    data = pd.concat([train_features, test_features]).merge(metadata[["file_i", "recorder_model", "country", "habitat"]], on="file_i", how="left")

    train_idx = data["dataloader_idx"] == 0
    test_idx = data["dataloader_idx"] == 2
    uk_idx = data["country"] == "UK"
    ec_idx = data["country"] == "EC"
    sm2_idx = data["recorder_model"] == "SM2"
    sm3_idx = data["recorder_model"] == "SM3"
    z_idx = [f"z_mean_{i}" for i in range(128)]
    z_norm_idx = [f"z_norm_{i}" for i in range(128)]

    # --------------------------------- Calculate Attribute Vectors --------------------------------- #

    log.info("Calculate attribute vectors for each recorder")
    z_sm2 = data.loc[train_idx & sm2_idx, z_idx].values.mean(axis=0)
    z_sm3 = data.loc[train_idx & sm3_idx, z_idx].values.mean(axis=0)
    log.info("Calculate attribute vectors by country")
    z_sm2_uk = data.loc[train_idx & uk_idx & sm2_idx, z_idx].values.mean(axis=0)
    z_sm3_uk = data.loc[train_idx & uk_idx & sm3_idx, z_idx].values.mean(axis=0)
    z_sm2_ec = data.loc[train_idx & ec_idx & sm2_idx, z_idx].values.mean(axis=0)
    z_sm3_ec = data.loc[train_idx & ec_idx & sm3_idx, z_idx].values.mean(axis=0)

    # --------------------------------- Attribute Vector Normalisation --------------------------------- #

    log.info("Apply attribute vectors (by country) to shift samples in latent space")
    # shift SM2 to SM3
    data.loc[uk_idx & sm2_idx, z_norm_idx] = (data.loc[uk_idx & sm2_idx, z_idx] - z_sm2_uk + z_sm3_uk).values
    data.loc[ec_idx & sm2_idx, z_norm_idx] = (data.loc[ec_idx & sm2_idx, z_idx] - z_sm2_ec + z_sm3_ec).values
    # keep SM3 as is
    data.loc[uk_idx & sm3_idx, z_norm_idx] = data.loc[uk_idx & sm3_idx, z_idx].values
    data.loc[ec_idx & sm3_idx, z_norm_idx] = data.loc[ec_idx & sm3_idx, z_idx].values

    # --------------------------------- Decode Attribute Vectors --------------------------------- #

    ckpt_path = root / "models" / "v4" / "vae" / model_name / "step=180000.ckpt"
    log.info(f"loading model {ckpt_path}")
    model = VAE.load_from_checkpoint(ckpt_path)

    recorder_labels = [r"SM2", r"SM3", r"SM2 (EC)", r"SM3 (EC)", r"SM2 (UK)",  r"SM3 (UK)"]
    z = torch.stack([
        torch.tensor(z_sm2),
        torch.tensor(z_sm3),
        torch.tensor(z_sm2_ec),
        torch.tensor(z_sm3_ec),
        torch.tensor(z_sm2_uk),
        torch.tensor(z_sm3_uk),
    ]).float().to(model.device)
    z = z.view(6, 1, 128)

    log.info("Decoding attribute vectors")
    with torch.no_grad():
        x_hat = model.decode(z).exp().cpu().numpy()

    db_ref = 1.0
    x_hat_mean = x_hat.mean(axis=-2).squeeze()

    line_styles = ["-", "-", "--", "--", '-.', '-.']
    cmap = sns.color_palette("viridis", as_cmap=True)
    a = cmap(np.linspace(0.1, 0.2, 3))
    b = cmap(np.linspace(0.7, 0.8, 3))
    colors = [[a[i], b[i]] for i in range(3)]
    colors = [item for sublist in colors for item in sublist]

    log.info("Plotting decoded recorder attributes as frequency response curves")
    xs = model.front_end.mel_frequencies
    ys = 20 * np.log10(x_hat_mean / db_ref)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i in range(6):
        ax.plot(xs, ys[i], linestyle=line_styles[i], label=recorder_labels[i], color=colors[i])
    # derive log ticks
    ax.set_xscale("log")
    ax.minorticks_off()
    f_min, f_max = model.front_end.mel_min_hertz, model.front_end.mel_max_hertz
    x_tick_positions = [f_min, *[2**i for i in range(max(9, int(np.ceil(np.log2(f_min)))), int(np.floor(np.log2(f_max))) + 1)], f_max]
    x_tick_labels = [f"{int(f)}" for f in x_tick_positions]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel('Magnitude (db)')

    ax.margins(x=0)
    legend = plt.legend(ncols=len(recorder_labels)//2, loc="upper left", edgecolor=(0, 0, 0, 1), facecolor=(1, 1, 1, 0.1))
    log.info(save_path / "reconstructed_recorder_frequency_residual.pdf")
    fig.savefig(save_path / "reconstructed_recorder_frequency_residual.pdf", format="pdf")

    # --------------------------------- Predict Recorders Before/After Attribute Vector Normalisation --------------------------------- #

    train_data = data[train_idx]
    test_data = data.loc[test_idx]
    # one vs rest, where rest is singular, so binary
    y, labels = data["recorder_model"].factorize()
    shuffle_idx = torch.randperm(len(train_data)).numpy()

    X_train = train_data.loc[:, z_idx].values
    y_train = y[train_idx]
    X_test = test_data.loc[:, z_idx].values
    y_test = y[test_idx]
    X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]

    log.info("Fitting logistic regression (before norm)")
    clf = LogisticRegression(penalty="l1", solver="saga", max_iter=10000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_test)
    y_prob_clf = clf.predict_proba(X_test)

    log.info("Fitting random forest classifier (before norm)")
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)

    log.info("Plotting confusion")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11, 4), constrained_layout=True)
    plot_confusion_matrix(y_test, y_pred_clf, labels, ax=ax1)
    plot_confusion_matrix(y_test, y_pred_rf, labels, ax=ax2)
    log.info(save_path / "hardware_conf_matrix_before_norm.pdf")
    fig.savefig(save_path / "hardware_conf_matrix_before_norm.pdf", format="pdf")

    # p(SM2) = 1 - p(SM3)
    fig_c, (ax1_c, ax2_c) = plt.subplots(ncols=2, figsize=(11, 4), constrained_layout=True)
    plot_roc_curve(y_test, y_prob_clf[:, 1], label_prefix="LR (before) ", ax=ax1_c, color="#307473")
    _, thresholds = plot_pr_curve(y_test, y_prob_clf[:, 1], label_prefix="LR (before) ", ax=ax2_c, color="#307473")
    plot_roc_curve(y_test, y_prob_rf[:, 1], label_prefix="RF (before) ", ax=ax1_c, color="#3F334D")
    plot_pr_curve(y_test, y_prob_rf[:, 1], label_prefix="RF (before) ", ax=ax2_c, color="#3F334D")
    # add the baseline probability
    p_sm2 = y_train.sum() / y_train.shape[0]
    ax2_c.text(x=0.5, y=p_sm2 + 0.01, s=f"{np.format_float_positional(p_sm2, precision=3)}", fontsize=9, color="black", weight="bold")
    ax2_c.plot(np.linspace(0, 1, len(thresholds)), np.linspace(p_sm2, p_sm2, len(thresholds)), linestyle="dashed", color="black")

    X_train = train_data.loc[:, z_norm_idx].values
    X_test = test_data.loc[:, z_norm_idx].values
    X_train = X_train[shuffle_idx]

    log.info("Fitting logistic regression (after norm)")
    clf = LogisticRegression(penalty="l1", solver="saga", max_iter=10000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_test)
    y_prob_clf = clf.predict_proba(X_test)

    log.info("Fitting random forest classifier (after norm)")
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)

    log.info("Plotting confusion")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11, 4), constrained_layout=True)
    plot_confusion_matrix(y_test, y_pred_clf, labels, ax=ax1)
    plot_confusion_matrix(y_test, y_pred_rf, labels, ax=ax2)
    log.info(save_path / "hardware_prediction_after_norm.pdf")
    fig.savefig(save_path / "hardware_prediction_after_norm.pdf", format="pdf")

    plot_roc_curve(y_test, y_prob_clf[:, 1], label_prefix="LR (after) ", ax=ax1_c, color="#A6EBC9")
    plot_roc_curve(y_test, y_prob_rf[:, 1], label_prefix="RF (after) ", ax=ax1_c, color="#9A98B5")
    plot_pr_curve(y_test, y_prob_clf[:, 1], label_prefix="LR (after) ", ax=ax2_c, color="#A6EBC9")
    plot_pr_curve(y_test, y_prob_rf[:, 1], label_prefix="RF (after) ", ax=ax2_c, color="#9A98B5")
    log.info(save_path / "hardware_pred_curves.pdf")
    fig_c.savefig(save_path / "hardware_pred_curves.pdf", format="pdf")

    # --------------------------------- Per-Country Recorder UMAP by Before/After Attribute Vector Normalisation --------------------------------- #

    log.info("Fitting UK umap before")
    mapper = umap.UMAP(metric="euclidean", n_neighbors=50, min_dist=0.0, random_state=seed, n_components=2).fit(data.loc[uk_idx, z_idx])
    data.loc[uk_idx, "x_uk_i"], data.loc[uk_idx, "y_uk_i"] = np.hsplit(mapper.transform(data.loc[uk_idx, z_idx]), 2)
    log.info("Fitting EC umap before")
    mapper = umap.UMAP(metric="euclidean", n_neighbors=50, min_dist=0.0, random_state=seed, n_components=2).fit(data.loc[ec_idx, z_idx])
    data.loc[ec_idx, "x_ec_i"], data.loc[ec_idx, "y_ec_i"] = np.hsplit(mapper.transform(data.loc[ec_idx, z_idx]), 2)

    log.info("Fitting UK umap after")
    mapper = umap.UMAP(metric="euclidean", n_neighbors=50, min_dist=0.0, random_state=seed, n_components=2).fit(data.loc[uk_idx, z_norm_idx])
    data.loc[uk_idx, "x_uk_j"], data.loc[uk_idx, "y_uk_j"] = np.hsplit(mapper.transform(data.loc[uk_idx, z_norm_idx]), 2)
    log.info("Fitting EC umap after")
    mapper = umap.UMAP(metric="euclidean", n_neighbors=50, min_dist=0.0, random_state=seed, n_components=2).fit(data.loc[ec_idx, z_norm_idx])
    data.loc[ec_idx, "x_ec_j"], data.loc[ec_idx, "y_ec_j"] = np.hsplit(mapper.transform(data.loc[ec_idx, z_norm_idx]), 2)

    z_score = 2.5
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8), constrained_layout=True)

    log.info("Plotting EC umap")
    ax = axes[0, 0]
    x_mean, x_std = data.loc[ec_idx, "x_ec_i"].mean(), data.loc[ec_idx, "x_ec_i"].std()
    x_min = x_mean - z_score * x_std
    x_max = x_mean + z_score * x_std
    y_mean, y_std = data.loc[ec_idx, "y_ec_i"].mean(), data.loc[ec_idx, "y_ec_i"].std()
    y_min = y_mean - z_score * y_std
    y_max = y_mean + z_score * y_std
    sns.scatterplot(data=data[ec_idx], x="x_ec_i", y="y_ec_i", hue="recorder_model", s=25, alpha=0.5, palette=sns.color_palette("viridis", n_colors=2), legend=False, ax=ax)
    ax.set_xlim([x_min.min(), x_max.max()])
    ax.set_ylim([y_min.min(), y_max.max()])
    ax.set_xlabel("")
    ax.set_ylabel("Before norm")
    ax.set_title("EC")

    log.info("Plotting UK umap")
    ax = axes[0, 1]
    x_mean, x_std = data.loc[uk_idx, "x_uk_i"].mean(), data.loc[uk_idx, "x_uk_i"].std()
    x_min = x_mean - z_score * x_std
    x_max = x_mean + z_score * x_std
    y_mean, y_std = data.loc[uk_idx, "y_uk_i"].mean(), data.loc[uk_idx, "y_uk_i"].std()
    y_min = y_mean - z_score * y_std
    y_max = y_mean + z_score * y_std
    sns.scatterplot(data=data[uk_idx], x="x_uk_i", y="y_uk_i", hue="recorder_model", s=25, alpha=0.5, palette=sns.color_palette("viridis", n_colors=2), ax=ax)
    ax.set_xlim([x_min.min(), x_max.max()])
    ax.set_ylim([y_min.min(), y_max.max()])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("UK")

    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.04, 1), ncols=1, title="Recorder Type")

    log.info("Plotting EC umap")
    ax = axes[1, 0]
    x_mean, x_std = data.loc[ec_idx, "x_ec_j"].mean(), data.loc[ec_idx, "x_ec_j"].std()
    x_min = x_mean - z_score * x_std
    x_max = x_mean + z_score * x_std
    y_mean, y_std = data.loc[ec_idx, "y_ec_j"].mean(), data.loc[ec_idx, "y_ec_j"].std()
    y_min = y_mean - z_score * y_std
    y_max = y_mean + z_score * y_std
    sns.scatterplot(data=data[ec_idx], x="x_ec_j", y="y_ec_j", hue="recorder_model", s=25, alpha=0.5, palette=sns.color_palette("viridis", n_colors=2), legend=False, ax=ax)
    ax.set_xlim([x_min.min(), x_max.max()])
    ax.set_ylim([y_min.min(), y_max.max()])
    ax.set_xlabel("")
    ax.set_ylabel("After norm")

    log.info("Plotting UK umap")
    ax = axes[1, 1]
    x_mean, x_std = data.loc[uk_idx, "x_uk_j"].mean(), data.loc[uk_idx, "x_uk_j"].std()
    x_min = x_mean - z_score * x_std
    x_max = x_mean + z_score * x_std
    y_mean, y_std = data.loc[uk_idx, "y_uk_j"].mean(), data.loc[uk_idx, "y_uk_j"].std()
    y_min = y_mean - z_score * y_std
    y_max = y_mean + z_score * y_std
    sns.scatterplot(data=data[uk_idx], x="x_uk_j", y="y_uk_j", hue="recorder_model", s=25, alpha=0.5, palette=sns.color_palette("viridis", n_colors=2), ax=ax, legend=False)
    ax.set_xlim([x_min.min(), x_max.max()])
    ax.set_ylim([y_min.min(), y_max.max()])
    ax.set_xlabel("")
    ax.set_ylabel("")

    log.info(save_path / "recorder_umap.pdf")
    fig.savefig(save_path / "recorder_umap.pdf", format="pdf")

    # TODO: ????
    # quantify loss of reconstruction quality after applying adjustment
    # how does the recorder attribute adjstment affect the likelihood?
    # - calculate the reconstruction gaussian log probability / error for every shifted sample (all SM2 to SM3, all SM3 to SM2)

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
