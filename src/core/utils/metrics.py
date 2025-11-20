import numpy as np
import torch

from numpy.typing import NDArray
from torch.nn import functional as F

def class_balanced_binary_cross_entropy(
    y: torch.Tensor,
    y_probs: torch.Tensor,
    beta: torch.Tensor,
    samples_per_class: torch.Tensor,
    label_smoothing: float = 0.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    weights = (torch.ones_like(beta) - beta) / (torch.ones_like(samples_per_class) - torch.pow(beta, samples_per_class.clip(min=1)))
    weights = weights / weights.sum() * weights.shape[0]
    y_probs = y_probs.clamp(epsilon, 1 - epsilon)
    for values in [y, y_probs, weights]:
        assert torch.isfinite(values).all()
    y = y * (1 - label_smoothing) + (1 - y) * label_smoothing
    return (-(weights * y * y_probs.log() + (1 - y) * (1 - y_probs).log()))

def weight_regularisation(weights: torch.Tensor, lamdba: float, order: int = 2) -> torch.Tensor:
    return lamdba * torch.stack([torch.linalg.norm(layer, order) for layer in weights])

def average_precision(
    labels: NDArray,
    scores: NDArray,
    label_mask: NDArray | None = None,
    sort_descending: bool = True,
    interpolated: bool = False,
) -> NDArray:
    """
    see https://github.com/google-research/perch/blob/main/chirp/models/metrics.py
    """
    if label_mask is not None:
        # Set all masked labels to zero, and send the scores for those labels to a
        # low/high value (depending on whether we sort in descending order or not).
        # Then the masked scores+labels will not impact the average precision
        # calculation.
        labels = labels * label_mask
        extremum_score = np.min(scores) - 1.0 if sort_descending else np.max(scores) + 1.0
        scores = np.where(label_mask, scores, extremum_score)
    idx = np.argsort(scores)
    if sort_descending:
        idx = np.flip(idx, axis=-1)
    scores = np.take_along_axis(scores, idx, axis=-1)
    labels = np.take_along_axis(labels, idx, axis=-1)
    pr_curve = np.cumsum(labels, axis=-1) / (np.arange(labels.shape[-1]) + 1)
    # In case of an empty row, assign precision = 0, and avoid dividing by zero.
    mask = np.float32(np.sum(labels, axis=-1) != 0)
    raw_av_prec = np.sum(pr_curve * labels, axis=-1) / np.maximum(np.sum(labels, axis=-1), 1.0)
    return mask * raw_av_prec

def recall_at_k(
    labels: NDArray,
    probs: NDArray
) -> NDArray:
    """
    Recall at K treats a multi-label binary task as a multi-class task
    Probabilities are ranked, accuracy calculated as the proportion of correct
    classifications of all positive labels
    """
    N, M = labels.shape
    # k is the total number of positives
    actual_positive = labels.sum(axis=1)
    # for each sample, assign a rank to probabilities in descending order
    top_k_mask = np.zeros_like(probs, dtype=bool)
    for i in range(N):
        # sort probabilities in descending order
        order = np.argsort(-probs[i], kind="mergesort")
        # highest probabilites get the highest rank, highest starts at 1 and increments
        rank = np.empty_like(probs[i], dtype=int)
        rank[order] = np.arange(1, M + 1)
        # fetch the top k classes
        top_k_mask[i] = rank <= actual_positive[i]
    # how many in the positive labels are also in the predicted top k?
    predicted_positive = (top_k_mask & (labels == 1)).sum(axis=1)
    # evaluate meaningful cases where there are actually positive labels, samples with zeros are omitted
    mask = actual_positive != 0
    # recall_at_k is the ratio of predicted positive classes in the top k to those actually in the top k
    recall = predicted_positive[mask] / actual_positive[mask]
    return recall.mean(axis=0)
