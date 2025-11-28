import lightning as L
import torch
import functools
from dataclasses import dataclass
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from torch import nn
from torch import Tensor
from typing import Any, List, Tuple

__all__ = [
    "Activation",
    "NormType",
    "ResidualConv2d",
]

class Activation(Enum):
    RELU = (nn.ReLU,)
    GELU = (nn.GELU,)
    SELU = (nn.SELU,)
    SILU = (nn.SiLU,)
    LEAK = (nn.LeakyReLU,)

    def __init__(self, init: Callable[[], nn.Module]) -> None:
        self.init = init

class NormType(Enum):
    BN1 = (nn.BatchNorm1d,)
    BN2 = (nn.BatchNorm2d,)
    LN = (functools.partial(nn.GroupNorm, 1),)
    GN = (nn.GroupNorm,)

    def __init__(self, init: Callable[[int], nn.Module]) -> None:
        self.init = init

class ResidualConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Activation,
        norm: NormType,
        dropout_prob: float = 0.5,
        padding_mode: str = 'zeros',
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
            norm.init(out_channels),
            activation.init(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm.init(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + self.shortcut(x)

def init_cnn_feature_encoder(
    block_sizes: List[int],
    block_width: int,
    block_depth: int,
    dropout_prob: float,
    padding_mode: str,
    norm_fn: NormType,
    activation_fn: Activation,
    weight_init_std: float,
) -> nn.Module:
    in_channels = block_sizes[0] * block_width
    encoder_cnn = nn.ModuleList()
    pre_process = nn.Sequential(
        nn.Conv2d(1, in_channels, kernel_size=5, padding=2, padding_mode=padding_mode),
        norm_fn.init(in_channels),
        activation_fn.init(),
        nn.Conv2d(in_channels, in_channels * 2, kernel_size=(2, 1), stride=(2, 1), padding_mode=padding_mode),
        nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
    )
    nn.init.trunc_normal_(pre_process[0].weight, std=weight_init_std)
    encoder_cnn.append(pre_process)
    in_channels = block_sizes[0] * block_width
    for block_size in block_sizes[1:]:
        block = nn.Sequential()
        out_channels = block_size * block_width
        down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        block.append(down)
        block.extend([ResidualConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=activation_fn,
            norm=norm_fn,
            dropout_prob=dropout_prob,
            padding_mode=padding_mode,
        ) for i in range(block_depth)])
        encoder_cnn.append(block)
        in_channels = out_channels
    return encoder_cnn

def init_cnn_feature_decoder(
    block_sizes: List[int],
    block_width: int,
    block_depth: int,
    dropout_prob: float,
    padding_mode: str,
    norm_fn: NormType,
    activation_fn: Activation,
) -> nn.Module:
    decoder_cnn = nn.ModuleList()
    in_channels = block_sizes[0] * block_width
    for i, block_size in enumerate(block_sizes[1:]):
        block = nn.Sequential()
        out_channels = block_size * block_width
        block.extend([ResidualConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            activation=activation_fn,
            norm=norm_fn,
            dropout_prob=dropout_prob,
            padding_mode=padding_mode,
        ) for j in range(block_depth)])
        up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        block.append(up)
        decoder_cnn.append(block)
        in_channels = out_channels
    # final expansion in time dimension
    decoder_cnn.append(nn.Sequential(
        nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
        norm_fn.init(in_channels * 2),
        activation_fn.init(),
        nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=(2, 1), stride=(2, 1)),
        norm_fn.init(in_channels),
        activation_fn.init(),
        nn.Conv2d(in_channels, 1, kernel_size=5, padding=2, padding_mode=padding_mode),
    ))
    return decoder_cnn

def init_mlp_content_encoder(
    in_channels: int,
    out_channels: int,
    feature_height: int,
    feature_width: int,
    mlp_reduction_factor: float,
    activation_fn: Activation,
    dropout_prob: float,
    out_features: int,
) -> nn.Module:
    in_features = out_channels * feature_height * feature_width
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.Flatten(),
        nn.Linear(in_features=in_features, out_features=in_features // mlp_reduction_factor),
        activation_fn.init(),
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features=in_features // mlp_reduction_factor, out_features=out_features)
    )

def init_mlp_content_decoder(
    in_features: int,
    in_channels: int,
    out_channels: int,
    feature_height: int,
    feature_width: int,
    mlp_reduction_factor: float,
    activation_fn: Activation,
    dropout_prob: float,
) -> nn.Module:
    out_features = in_channels * feature_width * feature_height
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features // mlp_reduction_factor),
        activation_fn.init(),
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features=out_features // mlp_reduction_factor, out_features=out_features),
        nn.Unflatten(-1, (in_channels, feature_height, feature_width)),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    )

def init_alignment_encoder(
    in_channels: int,
    out_channels: int,
    cnn_kernel_size: Tuple[int, int],
    flatten_start_dim: int,
    in_features: int,
    activation_fn: Activation,
    mlp_reduction_factor: int,
    out_features: int,
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=cnn_kernel_size),
        nn.Flatten(start_dim=flatten_start_dim),
        nn.Linear(in_features=in_features, out_features=in_features // mlp_reduction_factor),
        activation_fn.init(),
        nn.Linear(in_features=in_features // mlp_reduction_factor, out_features=out_features)
    )
