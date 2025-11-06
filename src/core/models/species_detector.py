import attrs
import enum
import functools
import lightning as L
import numpy as np
import pandas as pd
import torch

from typing import Any, Dict, List, Tuple

from src.core.models.components import GatedAttention
from src.core.utils import metrics, weight_initialization

class POOL(enum.Enum):
    MAX = "max"
    MEAN = "mean"
    FEATURE_ATTN = "feature_attn"
    PROB_ATTN = "prop_attn"

@attrs.define(kw_only=True)
class SpeciesDetector(L.LightningModule):
    species: List[str] = attrs.field(factory=list, validator=attrs.validators.min_len(1))
    in_features: int = attrs.field(default=128, validator=attrs.validators.instance_of(int))
    attn_dim: int | None = attrs.field(default=None)
    l1_penalty: float = attrs.field(default=1e-2, validator=attrs.validators.instance_of(float))
    penalty_multiplier: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    beta: float = attrs.field(default=0.0, converter=lambda beta: torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32), requires_grad=False))
    pool_method: str = attrs.field(default="mean", converter=POOL, validator=attrs.validators.in_(POOL))
    clf_learning_rate: float = attrs.field(default=1e-1, validator=attrs.validators.instance_of(float))
    attn_learning_rate: float | None = attrs.field(default=None)

    @attn_dim.validator
    def check_positive_non_zero_integer_if_defined(self, attribute, value):
        if value is not None:
            assert isinstance(value, int) and value > 0, f"'{attribute}' must be a positive non-zero integer"

    def __attrs_post_init__(self) -> None:
        L.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.classifiers = torch.nn.ModuleDict({
            species_name: torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
            for species_name in self.species
        })
        if self.pool_method == POOL.FEATURE_ATTN or self.pool_method == POOL.PROB_ATTN:
            self.attention = torch.nn.ModuleDict({
                species_name: GatedAttention(in_features=self.in_features, hidden_dim=self.attn_dim, out_features=1)
                for species_name in self.species
            })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pool by taking a linear combination of features as a function of the data
        if self.pool_method == POOL.FEATURE_ATTN:
            return torch.cat([
                torch.sigmoid(clf((x * attn(x)).sum(dim=-2)))
                for clf, attn in zip(self.classifiers.values(), self.attention.values())
            ], dim=1)
        # pool by taking a linear function of probabilities as a function of the data
        elif self.pool_method == POOL.PROB_ATTN:
            return torch.cat([
                torch.sigmoid((clf(x) * attn(x)).sum(dim=-2))
                for clf, attn in zip(self.classifiers.values(), self.attention.values())
            ], dim=1)
        # pool by maximum probability
        elif self.pool_method == POOL.MAX:
            return torch.cat([
                torch.sigmoid(torch.max(clf(x), dim=-2).values)
                for clf in self.classifiers.values()
            ], dim=1)
        # pool by mean probability
        elif self.pool_method == POOL.MEAN:
            return torch.cat([
                torch.sigmoid(torch.mean(clf(x), dim=-2))
                for clf in self.classifiers.values()
            ], dim=1)

    def loss(self, y: torch.Tensor, y_probs: torch.Tensor, y_freq: torch.Tensor) -> Dict[str, torch.Tensor]:
        # take the mean over the batch for each logistic regression model
        weights = (torch.ones_like(self.beta) - self.beta) / (torch.ones_like(y_freq[0]) - torch.pow(self.beta, y_freq[0].clip(min=1)))
        weights = weights / weights.sum() * weights.shape[0]
        cel = (-weights * y * y_probs.log() - (1 - y) * (1 - y_probs).log()).mean(dim=0)
        # L1 regularisation with split for smooth latents (if applicable), sum the L1 penalties for each model
        l1_1 = metrics.l1_penalty([w[:, 0:64] for w in list(self.classifiers.parameters())[::2]], self.l1_penalty)
        l1_2 = metrics.l1_penalty([w[:, 64:128] for w in list(self.classifiers.parameters())[::2]], self.l1_penalty * self.penalty_multiplier)
        l1_penalty = torch.stack([l1_1, l1_2], dim=-1).sum(-1)
        # per logistic regression model, add the CEL and L1 together and then sum to get the total loss
        loss = cel + l1_penalty
        loss = loss.sum()
        return dict(
            loss=loss,
            cel=cel.detach().sum(),
            l1_penalty=l1_penalty.detach().sum(),
            l1_penalty_1=l1_1.detach().sum(),
            l1_penalty_2=l1_2.detach().sum(),
        )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y, s = batch
        logits = self.forward(x)
        loss_outputs = self.loss(y.float(), logits)
        self.log_dict({f"train/{key}": value for key, value in loss_outputs.items()}, prog_bar=True)
        return loss_outputs

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y, s = batch
        logits = self.forward(x)
        loss_outputs = self.loss(y.float(), logits)
        self.log_dict({f"val/{key}": value for key, value in loss_outputs.items()}, prog_bar=True)
        return loss_outputs

    @torch.no_grad()
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> pd.DataFrame:
        x, y, (s, samples_per_class) = batch
        logits = self.forward(x)
        # take the mean probability over samples if that dimension exists
        logits = logits.mean(dim=1) if len(logits.shape) > 3 else logits
        # return dataframe of predictions
        return (
            pd.DataFrame(
                data=y.detach().cpu(),
                columns=list(self.classifiers.keys()),
                index=s.detach().cpu().tolist()
            )
            .reset_index(names="file_i")
            .melt(id_vars="file_i", var_name="species_name", value_name="label")
            .merge(
                pd.DataFrame(
                    data=torch.sigmoid(logits).detach().cpu(),
                    columns=list(self.classifiers.keys()),
                    index=s.detach().cpu().tolist()
                )
                .reset_index(names="file_i")
                .melt(id_vars="file_i", var_name="species_name", value_name="prob"),
                on=["file_i", "species_name"],
                how="inner",
            )
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimisers = []
        optimisers.append(torch.optim.Adam(params=self.classifiers.parameters(), lr=self.clf_learning_rate))
        if self.pool_method == POOL.FEATURE_ATTN or self.pool_method == POOL.PROB_ATTN:
            optimisers.append(torch.optim.Adam(params=self.attention.parameters(), lr=self.attn_learning_rate))
        return optimisers

    def run(
        self,
        trainer: "Trainer",
        train_dataloader: torch.utils.data.DataLoader,
        device: str,
    ) -> None:
        if self.pool_method == POOL.MAX:
            self.classifiers = weight_initialization(
                in_features=self.in_features,
                dataloader=train_dataloader,
                species_names=self.species,
                loss=functools.partial(class_balanced_binary_cross_entropy_with_logits, beta=self.beta.to(device)),
                device=device,
            )
