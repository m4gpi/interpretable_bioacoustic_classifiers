import torch

from torch.nn import functional as F

def class_balanced_binary_cross_entropy(
    y: torch.Tensor,
    logits: torch.Tensor,
    beta: torch.Tensor,
    samples_per_class: torch.Tensor,
) -> torch.Tensor:
    weights = (torch.ones_like(beta) - beta) / (torch.ones_like(samples_per_class) - torch.pow(beta, samples_per_class.clip(min=1)))
    weights = weights / weights.sum() * weights.shape[0]
    assert torch.isfinite(weights).all()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(y).all()
    return -weights * y * probs.log() - (1 - y) * (1 - p).log()
    return F.binary_cross_entropy_with_logits(input=logits, target=y, pos_weight=weights, reduction="none")

def l1_penalty(weights: torch.Tensor, lamdba: float) -> torch.Tensor:
    return lamdba * torch.stack([torch.linalg.norm(layer, 1) for layer in weights])
