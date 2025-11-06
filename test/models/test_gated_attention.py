from src.core.models.components import GatedAttention

def test_gated_attention():
    model = GatedAttention(in_features=128, hidden_dim=5, out_features=1)
