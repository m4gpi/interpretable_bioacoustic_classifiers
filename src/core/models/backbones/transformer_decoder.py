import torch
import numpy as np

from dataclasses import dataclass
from typing import Any, List, Tuple

__all__ = ["DecoderBlock", "TransformerDecoder"]

class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        mlp_ratio: int,
        num_heads: int = 1,
        dropout_prob: float = 0.1,
    ) -> None:
        self.num_features = num_features
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        self.self_attention = torch.nn.MultiheadAttention(
            num_heads=self.num_heads,
            embed_dim=self.num_features,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.norm_1 = torch.nn.LayerNorm(self.num_features)
        self.source_target_attention = torch.nn.MultiheadAttention(
            num_heads=self.num_heads,
            embed_dim=self.num_features,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.norm_2 = torch.nn.LayerNorm(self.num_features)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_features, out_features=self.num_features * self.mlp_ratio),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=self.num_features * self.mlp_ratio, out_features=self.num_features)
        )
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.norm_3 = torch.nn.LayerNorm(self.num_features)

    def forward(self, x: torch.Tensor, z: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # apply self-attention with causal masking
        x_attn, self_attn_w = self.self_attention(x, x, x, attn_mask=attn_mask, is_causal=True)
        # apply the residual & norm
        x = self.norm_1(x_attn + x)
        # source-target attention between encoder & decoder
        # this multi-head attention mechanism in the decoder receives the query and key values from the encoder
        # and maps this to the value output at the previous attention layer in the decoder (i.e. the contextualised 
        # representation up until that point). This matches the encoder's input to the decoder's input, allowing
        # the decoder to determine which encoder input is relevant with respect to the current time-step being considered.
        x_attn, source_target_attn_w = self.source_target_attention(z, z, x)
        # apply the residual & norm
        x = self.norm_2(x_attn + x)
        # feed forward of attended representations
        x_fwd = self.dropout(self.mlp(x))
        # final residual & norm
        x = self.norm_3(x_fwd + x)
        # return output with attention weights
        return x, self_attn_w, source_target_attn_w

@dataclass(unsafe_hash=True, kw_only=True)
class TransformerDecoder(torch.nn.Module):
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
            DecoderBlock(
                num_features=self.input_size,
                mlp_ratio=self.mlp_ratio,
                num_heads=self.num_heads,
            )
            for _ in range(self.depth)
        ])

    def forward(self, x: torch.Tensor, encoder_outputs: List[torch.Tensor], attn_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self_attentions = []
        source_target_attentions = []
        for block, z in zip(self.blocks, encoder_outputs):
            x, dec_attn_w, enc_attn_w = block(x, z, attn_mask=attn_mask)
            self_attentions.append(dec_attn_w)
            source_target_attentions.append(enc_attn_w)
        return x, (torch.stack(self_attentions), torch.stack(source_target_attentions))
