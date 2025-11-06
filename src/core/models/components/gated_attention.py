import attrs
import torch

from torch.nn import functional as F

@attrs.define()
class GatedAttention(torch.nn.Module):
    in_features: int = attrs.field()
    hidden_dim: int = attrs.field()
    out_features: int = attrs.field()

    def __attrs_post_init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.attention_V = torch.nn.Linear(in_features=self.in_features, out_features=self.hidden_dim)
        self.attention_U = torch.nn.Linear(in_features=self.in_features, out_features=self.hidden_dim)
        self.attention_w = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, z: torch.Tensor, dim: int = -2) -> torch.Tensor:
        A_V = torch.tanh(self.attention_V(z)) # (N, T, D)
        A_U = torch.sigmoid(self.attention_U(z)) # (N, T, D)
        # wᵀ (tanh(V h_kᵀ) ⊙ sigmoid(U h_kᵀ))
        A = F.softmax(self.attention_w(A_V * A_U), dim=-2) # (N, T, 1)
        return A
