import numpy as np
import torch

from torch import Tensor
from torch.functional import F
from typing import Union, Tuple

__all__ = ["circular_boundary", "translation", "Translation"]

def circular_boundary(xx: Tensor) -> Tensor:
    x = torch.zeros_like(xx)
    mask = (xx == 0).any(dim=(-2, -1))
    x[mask] = xx[mask]
    x[~mask] = (xx[~mask] + 1) - torch.floor((xx[~mask] + 1) / 2.0) * 2.0 - 1.0
    return x

def translation(x, dx, padding_mode: str = "circular"):
    """
    Translate x by dx along the H dimension
    using a circular boundary condition along the H border. Note: translations
    where zero exists in co-ordinates are ignored to prevent division by zero

    :param Tensor x: a tensor of shape (B, C, H, W)
    :param Tensor dx: a tensor of shape (B, C, H, 1)
    :returns x_tilde: a tensor of shape (B, C, H, W)
    """
    B, C, H, W = x.size()
    mesh = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    ), dim=-1).expand(B, H, W, 2).to(x.device)
    xx, yy = mesh.chunk(2, dim=-1)
    xx = xx + dx
    if padding_mode == "circular":
        xx = circular_boundary(xx)
        padding_mode = "zeros" # doesn't matter because they're all now within bounds
    grid = torch.cat([yy, xx], dim=-1).squeeze(-2)
    x_tilde = F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    return x_tilde.view(B, C, H, W)

class Translation:
    def __init__(dx: float, padding_mode: str) -> None:
        self.dx = dx
        self.padding_mode = padding_mode

    def __call__(self, x: Tensor) -> Tensor:
        return translation(x, self.dx, self.padding_mode)
