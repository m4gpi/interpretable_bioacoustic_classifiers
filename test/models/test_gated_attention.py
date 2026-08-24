import pytest
import torch

from src.core.models.species_detector import GatedAttention

def test_gated_attention():
    z = torch.randn(6, 39, 128)
    model = GatedAttention(128, 10, 1, 10)
    A = model(z)
    assert A.shape == (6, 39, 10)
