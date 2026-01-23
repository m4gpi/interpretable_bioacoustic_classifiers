import attrs
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

from src.core.models.components import NormType, Activation, ResidualConv2d

__all__ = ["ResidualCNNDecoder"]

@dataclass
class ResidualCNNDecoder(torch.nn.Module):
    block_sizes: List[int]
    block_width: int
    block_depth: int
    dropout_prob: float
    padding_mode: str
    norm_fn: NormType = NormType.LN
    activation_fn: Activation = Activation.RELU

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        torch.nn.Module.__init__(obj)
        return obj

    def __post_init__(self):
        self.model = torch.nn.ModuleList()
        in_channels = self.block_sizes[0] * self.block_width
        for i, block_size in enumerate(self.block_sizes[1:]):
            block = torch.nn.Sequential()
            out_channels = block_size * self.block_width
            block.extend([ResidualConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                activation=self.activation_fn,
                norm=self.norm_fn,
                dropout_prob=self.dropout_prob,
                padding_mode=self.padding_mode,
            ) for j in range(block_depth)])
            block.append(torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
            ))
            self.model.append(block)
            in_channels = out_channels
        # final expansion in time dimension
        self.model.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            self.norm_fn.init(in_channels * 2),
            self.activation_fn.init(),
            torch.nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=(2, 1), stride=(2, 1)),
            self.norm_fn.init(in_channels),
            self.activation_fn.init(),
            torch.nn.Conv2d(in_channels, 1, kernel_size=5, padding=2, padding_mode=self.padding_mode),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
