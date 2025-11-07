import torch

from src.core.models import SpeciesDetector

def test_species_detector():
    species = ["a", "b"]
    model = SpeciesDetector(species=species, in_features=128, pool_method="mean")
    assert model.species == species
    assert isinstance(model.beta, torch.nn.Parameter)

    x = torch.randn(100, 34, 128)
    y = (torch.rand(100, len(species)) > 0.5).long()
    y_freq = torch.randint(high=100, low=1, size=(1, len(species))).repeat(100, 1)
    y_probs = model(x)
    model.loss(y, y_probs, y_freq)

    model = SpeciesDetector(species=species, in_features=128, pool_method="feature_attn", attn_dim=12)
    assert hasattr(model, "attention")
    assert isinstance(model.attention, torch.nn.Module)

    x = torch.randn(100, 34, 128)
    y = (torch.rand(100, len(species)) > 0.5).long()
    y_freq = torch.randint(high=100, low=1, size=(1, len(species))).repeat(100, 1)
    y_probs = model(x)
    model.loss(y, y_probs, y_freq)
