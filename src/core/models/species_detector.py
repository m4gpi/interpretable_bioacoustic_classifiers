import enum
import functools
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

__all__ = ["SpeciesDetector"]

@dataclass(kw_only=True, eq=False)
class SpeciesDetector(L.LightningModule):
    target_names: List[str]
    target_counts: List[int]
    in_features: int
    beta: float | None
    ensemble_size: int = 1
    l1_penalty: float | None = None
    clf_learning_rate: float | None = None
    penalty_multiplier: int = 1.0
    label_smoothing: float = 0.0
    attn_dim: int | None = None
    attn_learning_rate: float | None = None
    attn_weight_decay: float | None = None
    train_sample_size: int | None = None
    eval_sample_size: int | None = None
    seed: int | None = None

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        self.save_hyperparameters()
        self.beta = torch.nn.Parameter(torch.tensor(self.beta, dtype=torch.float32), requires_grad=False)
        self.classifiers = torch.nn.ModuleDict({
            target_name: torch.nn.ModuleList([
                torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
                for i in range(self.ensemble_size)
            ])
            for target_name in self.target_names
        })
        if self.attn_dim is not None:
            self.attention_V = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
            torch.nn.init.xavier_uniform_(self.attention_V.weight)
            torch.nn.init.zeros_(self.attention_V.bias)
            self.attention_U = torch.nn.ModuleDict({})
            for target_name in self.target_names:
                layers = []
                for i in range(self.ensemble_size):
                    layer = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                    layers.append(layer)
                self.attention_U[target_name] = torch.nn.ModuleList(layers)
            self.attention_w = torch.nn.ModuleDict({})
            for target_name in self.target_names:
                layers = []
                for i in range(self.ensemble_size):
                    layer = torch.nn.Linear(in_features=self.attn_dim, out_features=1, bias=False)
                    torch.nn.init.xavier_uniform_(layer.weight)
                    layers.append(layer)
                self.attention_w[target_name] = torch.nn.ModuleList(layers)

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule | None, config: DictConfig, **kwargs: Any) -> None:
        device = trainer.strategy.root_device
        log.info(f"Training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(self, datamodule=data_module, ckpt_path=config.get("ckpt_path"))
        log.info(f"Testing <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.test(self, dataloaders=data_module.predict_dataloader(), ckpt_path=ckpt_path)

    def classifier_weights(self, target_name: str):
        return torch.stack([layer.weight for layer in self.classifiers[target_name]], dim=0)

    def attention_weights(self, x: torch.Tensor, target_name: str):
        attention_U = self.attention_U[target_name]
        attention_w = self.attention_w[target_name]
        A_V = torch.tanh(self.attention_V(x)) # (N, T, D)
        A_U = torch.sigmoid(attention_U(x))
        A = F.softmax(attention_w(A_V * A_U), dim=-2) # (N, T, 1)
        return A

    def species_frame_probs(self, x: torch.Tensor, target_name: str, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # if a normal distribution over features is provided, sample N times
        if num_samples is not None:
            x = self.q_samples(x, num_samples)
        clf = self.classifiers[target_name]
        attention_U = self.attention_U[target_name]
        attention_w = self.attention_w[target_name]
        A_V = torch.tanh(self.attention_V(x))
        A_U = torch.sigmoid(attention_U(z))
        A = F.softmax(attention_w(A_V * A_U), dim=-2)
        frame_probs, weighted_frame_probs = torch.sigmoid(clf(z)), torch.sigmoid(clf(x * A))
        if num_samples is not None:
            frame_probs = frame_probs.mean(dim=1)
            weighted_frame_probs = weighted_frame_probs.mean(dim=1)
            A = A.mean(dim=1)
        return frame_probs, weighted_frame_probs, A

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        y_probs = []
        attention_weights = []
        A_V = torch.tanh(self.attention_V(x)) # (bs, bag, ld)
        for target_name, clfs, attn_Us, attn_ws in zip(self.target_names, self.classifiers.values(), self.attention_U.values(), self.attention_w.values()):
            timestep_weights = []
            y_target_probs = []
            for clf, attn_U, attn_w in zip(clfs, attn_Us, attn_ws):
                A_U = torch.sigmoid(attn_U(x))
                A = F.softmax(attn_w(A_V * A_U), dim=-2) # (bs, bag, 1)
                y_hat = (torch.sigmoid(clf(x)) * A).sum(dim=-2) # (bs, 1)
                y_target_probs.append(y_hat)
                timestep_weights.append(A)
            y_target_probs = torch.stack(y_target_probs, dim=-2) # (bs, n, 1)
            timestep_weights = torch.stack(timestep_weights, dim=-2) # (bs, bag, n, 1)
            y_probs.append(y_target_probs)
            attention_weights.append(timestep_weights)
        y_probs = torch.cat(y_probs, dim=-1) # (bs, n, sp)
        attention_weights = torch.cat(attention_weights, dim=-1) # (bs, bag, n, sp)
        return y_probs, attention_weights

    def score(
        self,
        y: torch.Tensor,
        y_probs: torch.Tensor,
        s: torch.Tensor,
    ) -> pd.DataFrame:
        bs, bag, sp = y.size()
        sample_idx = s.expand(bag, -1).t().flatten().cpu()
        model_idx = torch.arange(bag).repeat(bs, 1).view(bs * bag).cpu()
        ref_column_types = dict(file_i=int, model_idx=int)
        feat_column_types = {target_name: float for target_name in self.target_names}
        column_types = (ref_column_types | feat_column_types)
        label_df = pd.DataFrame(data=dict(zip(column_types.keys(), [sample_idx, model_idx, *y.flatten(end_dim=1).cpu().t()])), columns=column_types.keys())
        label_df = label_df.astype(dtype=column_types).set_index(list(ref_column_types.keys()))
        label_df = label_df.melt(id_vars="file_i", var_name="species_name", value_name="label")
        prob_df = pd.DataFrame(data=dict(zip(column_types.keys(), [sample_idx, model_idx, *y_probs.flatten(end_dim=1).cpu().t()])), columns=column_types.keys())
        prob_df = prob_df.astype(dtype=column_types).set_index(list(ref_column_types.keys()))
        prob_df = prob_df.melt(id_vars="file_i", var_name="species_name", value_name="prob")
        return label_df.merge(probs_df, on=["file_i", "species_name"], how="inner")

    def loss(
        self,
        y: torch.Tensor,
        y_probs: torch.Tensor,
        samples_per_class: torch.Tensor,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        # batch mean over log probabilities weighted by positive class frequency
        cel = metrics.class_balanced_binary_cross_entropy(y, y_probs, samples_per_class=samples_per_class, **self.cel_params)
        # sparse penalty for clf weights
        weights = torch.cat(list(self.classifiers.parameters())[::2], dim=0).view(self.ensemble_size, -1, self.in_features)
        l1 = self.l1_penalty * torch.stack([torch.stack([torch.linalg.norm(weights[i, j], 1) for j in range(weights.size(1))]) for i in range(weights.size(0))])
        # orthogonality loss for clf weights within species diversity
        w_tilde = F.normalize(weights.transpose(0, 1).abs() + 1e-8, dim=-1)
        I = torch.eye(w_tilde.size(1), device=w_tilde.device)
        S = (w_tilde @ w_tilde.transpose(-1, -2))
        otl = ((S - I).triu().sum(dim=[-1, -2])) / w_tilde.size(1)**2
        # loss should be invariant to number of species, sum across models, average across species
        loss = (cel + l1 + otl).mean()
        return dict(
            loss=loss,
            cel=cel.detach().mean(),
            l1=l1.detach().mean(),
            otl=otl.detach().mean(),
        )

    def q_samples(self, q: torch.Tensor, num_samples: int):
        mu, log_sigma_sq = q.chunk(2, dim=-1)
        mu = mu.unsqueeze(1).expand(-1, num_samples, -1, -1)
        log_sigma_sq = log_sigma_sq.unsqueeze(1).expand(-1, num_samples, -1, -1)
        return torch.distributions.Normal(mu, (0.5 * log_sigma_sq).exp()).rsample()

    # ------------------------------ LIGHTNING FUNCS --------------------------------- #

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], num_samples: int | None = None) -> Tuple[torch.Tensor, ...]:
        x, y, s, *_ = batch
        # if a normal distribution over features is provided, sample N times
        if num_samples is not None:
            x = self.q_samples(x, num_samples)
        # forward pass
        y_probs, attn_w = self.forward(x)
        # take the mean over samples
        if num_samples is not None:
            y_probs = y_probs.mean(dim=1)
            if attn_w is not None:
                attn_w = attn_w.mean(dim=1)
        # sort our labels
        y = y.float().expand(self.ensemble_size, -1, -1).transpose(0, 1)
        samples_per_class = torch.tensor(self.target_counts, dtype=torch.int64, device=y.device).expand(self.ensemble_size, -1)
        return dict(
            y=y,
            y_probs=y_probs,
            attn_w=attn_w,
            s=s,
            samples_per_class=samples_per_class,
            target_names=self.target_names,
        )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        step_outputs = self.step(batch, num_samples=self.train_sample_size)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "train"), batch_size=batch.s.size(0), prog_bar=True, logger=False)
        return {**loss_outputs, **step_outputs}

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        step_outputs = self.step(batch, num_samples=self.eval_sample_size)
        loss_outputs = self.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=batch.s.size(0), prog_bar=True, logger=False)
        return {**loss_outputs, **step_outputs}

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> pd.DataFrame:
        step_outputs = self.step(batch, num_samples=self.eval_sample_size)
        return step_outputs

    @torch.no_grad()
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> pd.DataFrame:
        step_outputs = self.step(batch)
        return detach_values(step_outputs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = []
        params.append({'params': self.classifiers.parameters(), 'lr': self.clf_learning_rate})
        if self.attn_dim is not None:
            attn_params = list(self.attention_V.parameters()) + list(self.attention_U.parameters()) + list(self.attention_w.parameters())
            params.append({'params': attn_params, 'lr': self.attn_learning_rate, "weight_decay": self.attn_weight_decay})
        return torch.optim.Adam(params)

    @property
    def cel_params(self):
        return dict(
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
