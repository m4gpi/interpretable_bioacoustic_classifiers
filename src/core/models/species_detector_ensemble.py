import enum
import itertools
import lightning as L
import logging
import numpy as np
import pandas as pd
import hydra
import pathlib
import torch
import yaml

from dataclasses import dataclass
from omegaconf import DictConfig
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from src.core.utils import metrics
from src.core.utils import detach_values, prefix_keys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SpeciesDetectorEnsemble"]

# TODO: strategies for handling weighting across the ensemble / mixture-of-experts model?

class SpeciesDetectorEnsemble(torch.nn.Module):
    def __init__(
        self,
        target_names: List[str],
        target_counts: List[int],
        in_features: int,
        beta: float = 0.0,
        ensemble_size: int = 1,
        lamdba: float = 1e-1,
        label_smoothing: float = 0.0,
        attn_dim: int = 10,
    ) -> None:
        super().__init__()
        self.target_names = target_names
        self.target_counts = target_counts
        self.in_features = in_features
        self.ensemble_size = ensemble_size
        self.lamdba= lamdba
        self.label_smoothing = label_smoothing
        self.attn_dim = attn_dim
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.classifiers = self._init_classifier_ensemble()
        self.attention_V, self.attention_U, self.attention_w = self._init_attention_ensemble()

    def _init_classifier_ensemble(self):
        return torch.nn.ModuleDict({
            target_name: torch.nn.ModuleList([
                torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
                for i in range(self.ensemble_size)
            ])
            for target_name in self.target_names
        })

    def _init_attention_ensemble(self):
        attention_V = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
        torch.nn.init.xavier_uniform_(attention_V.weight)
        torch.nn.init.zeros_(attention_V.bias)
        attention_U = torch.nn.ModuleDict({})
        for target_name in self.target_names:
            layers = []
            for i in range(self.ensemble_size):
                layer = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                layers.append(layer)
            attention_U[target_name] = torch.nn.ModuleList(layers)
        attention_w = torch.nn.ModuleDict({})
        for target_name in self.target_names:
            layers = []
            for i in range(self.ensemble_size):
                layer = torch.nn.Linear(in_features=self.attn_dim, out_features=1, bias=False)
                torch.nn.init.xavier_uniform_(layer.weight)
                layers.append(layer)
            attention_w[target_name] = torch.nn.ModuleList(layers)
        return attention_V, attention_U, attention_w

    def pre_process(self, x: torch.Tensor, num_samples: int):
        if num_samples is None:
            return x.unsqueeze(1)
        mu, log_sigma_sq = x.chunk(2, dim=-1)
        mu = mu.unsqueeze(1).expand(-1, num_samples, -1, -1)
        log_sigma_sq = log_sigma_sq.unsqueeze(1).expand(-1, num_samples, -1, -1)
        return torch.distributions.Normal(mu, (0.5 * log_sigma_sq).exp()).rsample()

    def forward(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, num_samples: int | None = None, t: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        bs, sam, bg, ld = x.shape
        # forward pass
        y_probs = []
        attention_weights = []
        A_V = torch.tanh(self.attention_V(x))
        # iterate over species
        for target_name, clfs, attn_Us, attn_ws in zip(self.target_names, self.classifiers.values(), self.attention_U.values(), self.attention_w.values()):
            timestep_weights = []
            y_target_probs = []
            # iterate over the ensemble
            for clf, attn_U, attn_w in zip(clfs, attn_Us, attn_ws):
                A_U = torch.sigmoid(attn_U(x))
                A = F.softmax(attn_w(A_V * A_U), dim=-2)
                y_hat = (torch.sigmoid(clf(x)) * A).sum(dim=-2)
                y_target_probs.append(y_hat)
                timestep_weights.append(A)
            y_target_probs = torch.stack(y_target_probs, dim=-2)
            timestep_weights = torch.stack(timestep_weights, dim=-2)
            y_probs.append(y_target_probs)
            attention_weights.append(timestep_weights)
        y_probs = torch.cat(y_probs, dim=-1)
        attn_w = torch.cat(attention_weights, dim=-1)
        # mean over samples
        y_probs = y_probs.mean(dim=1)
        attn_w = attn_w.mean(dim=1)
        # replicate label information for each ensemble
        y = y.unsqueeze(1).expand(-1, self.ensemble_size, -1)
        samples_per_class = torch.tensor(self.target_counts, dtype=torch.int64, device=y.device).expand(self.ensemble_size, -1)
        return dict(
            x=x, s=s, y=y, y_probs=y_probs,
            attn_w=attn_w, samples_per_class=samples_per_class, target_names=self.target_names,
        )

    def predict(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.forward(*args, **kwargs)

    def loss(self, y: torch.Tensor, y_probs: torch.Tensor, samples_per_class: torch.Tensor, epsilon: float = 1e-6, **kwargs: Any) -> Dict[str, torch.Tensor]:
        # batch mean over log probabilities weighted by positive class frequency
        cel = metrics.class_balanced_binary_cross_entropy(y, y_probs, samples_per_class=samples_per_class, **self.cel_params).mean(dim=0)
        # weight sparsity penalty for classifiers
        weights = torch.cat(list(self.classifiers.parameters())[::2], dim=0).view(self.ensemble_size, -1, self.in_features)
        l1 = self.lamdba * torch.linalg.norm(weights, dim=-1, ord=1)
        # orthogonality loss for clf weights within species diversity
        w_tilde = F.normalize(weights.abs(), p=2, dim=-1).transpose(0, 1)
        I = torch.eye(w_tilde.size(1), device=w_tilde.device)
        S = (w_tilde @ w_tilde.transpose(-1, -2))
        otl = ((S - I).triu().sum(dim=[-1, -2])) / w_tilde.size(1)**2
        # sum losses
        loss = (cel + l1 + otl).sum()
        return dict(
            loss=loss,
            cel=cel.detach().sum(),
            lamdba=l1.detach().sum(),
            otl=otl.detach().mean(),
        )

    @torch.no_grad()
    def metrics(
        self,
        attn_w: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor | float]:
        seq_len = attn_w.size(1)
        max_entropy = np.log(seq_len)
        attn_w = attn_w.clamp(min=1e-8)
        attn_entropy = (-(attn_w * attn_w.log()).sum(dim=1)).cpu().numpy()
        norm_attn_entropy = attn_entropy / (max_entropy + 1e-8)
        attn_entropy_hist = np.histogram(attn_entropy.flatten(), bins=128, range=[0, max_entropy])
        return dict(
            attn_entropy_mean=attn_entropy.mean(),
            attn_entropy_std=attn_entropy.std(),
            attn_entropy_hist=attn_entropy_hist,
        )

    @torch.no_grad()
    def predict(self, y: torch.Tensor, y_probs: torch.Tensor, s: torch.Tensor, target_names: List[str], **kwargs: Any) -> pd.DataFrame:
        s = s.expand(y.size(1), y.size(2), -1).permute(2, 0, 1).flatten().cpu().numpy()
        ensemble_num = torch.arange(y_probs.size(1)).expand(y.size(0), y.size(2), -1).permute(0, 2, 1).flatten().cpu().numpy()
        target_names = list(itertools.chain(*[target_names for _ in range(y_probs.size(0)) for _ in range(y_probs.size(1))]))
        y, y_probs = y.flatten().cpu().numpy(), y_probs.flatten().cpu().numpy()
        return pd.DataFrame(
            data=list(zip(s, ensemble_num, target_names, y, y_probs)),
            columns=["file_i", "ensemble_num", "species_name", "label", "prob"]
        )

    @property
    def cel_params(self):
        return dict(beta=self.beta, label_smoothing=self.label_smoothing)

    @property
    def param_groups(self):
        return [
            self.classifiers.parameters(),
            list(self.attention_V.parameters()) + list(self.attention_U.parameters()) + list(self.attention_w.parameters()),
        ]
