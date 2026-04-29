import numpy as np
import torch

from torch import Tensor
from torch.functional import F
from typing import Union, Tuple

__all__ = ["circular_boundary", "translation", "Translation"]

def translation(x, dx, mode="bilinear", padding_mode="circular"):
    B, C, H, W = x.shape
    device = x.device
    # --- flatten ---
    x_flat = x.view(B * C * H, 1, 1, W)  # treat each row as a 1D signal
    # expand dx to match flattened batch
    dx_flat = dx.view(B, 1, 1).expand(B, C, H).reshape(-1)
    # --- build 1D grid ---
    xs = torch.linspace(-1, 1, W, device=device)
    grid_x = xs.view(1, 1, W).expand(B * C * H, 1, W)
    # y is dummy (always 0 since height = 1)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (N, 1, W, 2)
    # apply shift in x
    xx = grid[..., 0] + dx_flat.view(-1, 1, 1)
    if padding_mode == "circular":
        xx = circular_boundary(xx)
        padding_mode = "zeros"
    grid[..., 0] = xx
    # --- sample ---
    out = F.grid_sample(
        x_flat,
        grid,
        mode=mode,                 # "bilinear" or "bicubic"
        padding_mode=padding_mode,
        align_corners=True
    )
    # --- reshape back ---
    out = out.view(B, C, H, W)
    return out

# def translation(x, dx, mode: str = "bilinear", padding_mode: str = "circular"):
#     B, C, H, W = x.shape
#     ys = torch.linspace(-1, 1, H, device=x.device)
#     xs = torch.linspace(-1, 1, W, device=x.device)
#     grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
#     grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
#     grid = grid.unsqueeze(0).expand(B, H, W, 2).clone()
#     dx = dx.view(B, 1, 1) # broadcast over H, W
#     # shift along W → x-coordinates
#     xx = grid[..., 0] + dx
#     if padding_mode == "circular":
#         xx = circular_boundary(xx)
#         padding_mode = "zeros" # doesn't matter because they're all now within bounds
#     grid[..., 0] = xx
#     return F.grid_sample(
#         x,
#         grid,
#         mode=mode,
#         padding_mode=padding_mode,
#         align_corners=True
#     )

def circular_boundary(xx: torch.Tensor) -> torch.Tensor:
    return ((xx + 1) % 2) - 1

# def translation(x, dx, padding_mode: str = "circular", mode: str = "bilinear"):
#     """
#     Translate x by dx along the H dimension
#     using a circular boundary condition along the H border. Note: translations
#     where zero exists in co-ordinates are ignored to prevent division by zero

#     :param Tensor x: a tensor of shape (B, C, H, W)
#     :param Tensor dx: a tensor of shape (B, C, H, 1)
#     :returns x_tilde: a tensor of shape (B, C, H, W)
#     """
#     B, C, H, W = x.size()
#     mesh = torch.stack(torch.meshgrid(
#         torch.linspace(-1, 1, H),
#         torch.linspace(-1, 1, W),
#         indexing="ij",
#     ), dim=-1).expand(B, H, W, 2).to(x.device)
#     xx, yy = mesh.chunk(2, dim=-1)
#     xx = xx + dx
#     if padding_mode == "circular":
#         xx = circular_boundary(xx)
#         padding_mode = "zeros" # doesn't matter because they're all now within bounds
#     grid = torch.cat([yy, xx], dim=-1).squeeze(-2)
#     x_tilde = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
#     return x_tilde.view(B, C, H, W)

class Translation:
    def __init__(dx: float, padding_mode: str) -> None:
        self.dx = dx
        self.padding_mode = padding_mode

    def __call__(self, x: Tensor) -> Tensor:
        return translation(x, self.dx, self.padding_mode)
