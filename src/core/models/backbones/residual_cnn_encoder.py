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

__all__ = ["ResidualCNNEncoder"]

@dataclass
class ResidualCNNEncoder(torch.nn.Module):
    block_sizes: List[int]
    block_width: int
    block_depth: int
    dropout_prob: float
    padding_mode: str
    norm_fn: NormType = NormType.LN
    activation_fn: Activation = Activation.RELU
    weight_init_std: float

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        torch.nn.Module.__init__(obj)
        return obj

    def __post_init__(self):
        in_channels = self.block_sizes[0] * self.block_width
        self.model = torch.nn.ModuleList()
        pre_process = torch.nn.Sequential(
            torch.nn.Conv2d(1, in_channels, kernel_size=5, padding=2, padding_mode=self.padding_mode),
            self.norm_fn.init(in_channels),
            self.activation_fn.init(),
            torch.nn.Conv2d(in_channels, in_channels * 2, kernel_size=(2, 1), stride=(2, 1), padding_mode=self.padding_mode),
            torch.nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
        )
        torch.nn.init.trunc_normal_(pre_process[0].weight, std=self.weight_init_std)
        self.model.append(pre_process)
        in_channels = self.block_sizes[0] * self.block_width
        for block_size in self.block_sizes[1:]:
            block = torch.nn.Sequential()
            out_channels = block_size * self.block_width
            block.append(torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
            ))
            block.extend([ResidualConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                activation=self.activation_fn,
                norm=self.norm_fn,
                dropout_prob=self.dropout_prob,
                padding_mode=self.padding_mode,
            ) for i in range(block_depth)])
            self.model.append(block)
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
