import torch
import numpy as np

from dataclasses import dataclass
from typing import Any, List, Tuple

__all__ = ["EncoderBlock", "TransformerEncoder"]

class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        mlp_ratio: int,
        num_heads: int = 1,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        self.attention = torch.nn.MultiheadAttention(
            num_heads=self.num_heads,
            embed_dim=self.num_features,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.norm_1 = torch.nn.LayerNorm(self.num_features)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_features, out_features=self.num_features * self.mlp_ratio),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=self.num_features * self.mlp_ratio, out_features=self.num_features)
        )
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.norm_2 = torch.nn.LayerNorm(self.num_features)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # compute attention using query & key and apply softmax'ed attention weight to value
        x_attn, attn_w = self.attention(x, x, x, attn_mask=attn_mask)
        # apply residual & norm
        x = self.norm_1(x_attn + x)
        # feed forward of attented representations
        x_fwd = self.dropout(self.mlp(x))
        # final residual & norm
        x = self.norm_2(x_fwd + x)
        return x, attn_w

@dataclass(unsafe_hash=True, kw_only=True)
class TransformerEncoder(torch.nn.Module):
    input_size: int
    mlp_ratio: int = 4
    depth: int = 1
    num_heads: int = 4

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        torch.nn.Module.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        self.blocks = torch.nn.ModuleList([
            EncoderBlock(
                num_features=self.input_size,
                mlp_ratio=self.mlp_ratio,
                num_heads=self.num_heads,
            )
            for _ in range(self.depth)
        ])

    def forward(self, x: torch.torch.Tensor, attn_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attentions = []
        outputs = []
        for block in self.blocks:
            x, attn_w = block(x, attn_mask=attn_mask)
            outputs.append(x)
            attentions.append(attn_w)
        return x, torch.stack(outputs), attentions
