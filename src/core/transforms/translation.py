import numpy as np
import torch

from torch.functional import F

__all__ = ["circular_boundary", "translation", "Translation"]

def circular_boundary(xx: torch.Tensor) -> torch.Tensor:
    return ((xx + 1) % 2) - 1

def translation_1d(x: torch.Tensor, delta: torch.Tensor, mode: str = "bilinear", padding_mode: str = "circular") -> torch.Tensor:
    bs, ch, fq, ts = x.shape
    x_flat = x.view(bs, ch * fq, 1, ts)
    xs = torch.linspace(-1, 1, ts, device=x.device)
    grid_x = xs.view(1, 1, ts).expand(bs, 1, ts)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    xx = grid[..., 0] + delta.view(bs, 1, 1)
    if padding_mode == "circular":
        xx = ((xx + 1) % 2) - 1
        padding_mode = "zeros"
    grid[..., 0] = xx
    x_tilde = F.grid_sample(x_flat, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return x_tilde.view(bs, ch, fq, ts)
