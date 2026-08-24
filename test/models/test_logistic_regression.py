import pytest
import torch

from src.core.models.species_detector import MultiLabelLogisticRegression, MultiLabelBayesianLogisticRegression

def test_logistic_regression():
    z = torch.randn(6, 39, 128)
    model = MultiLabelLogisticRegression(128, 10)
    y_prob = model(z)
    assert y_prob.shape == (6, 39, 10)

def test_logistic_regression():
    z = torch.randn(6, 39, 128)
    model = MultiLabelBayesianLogisticRegression(128, 10)
    y_prob = model(z)
    assert y_prob.shape == (6, 39, 10)
