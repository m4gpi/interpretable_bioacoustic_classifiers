import enum
import functools
import lightning as L
import logging
import numpy as np
import pandas as pd
import pathlib
import sklearn
import torch
import yaml

from dataclasses import dataclass
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from src.core.utils import metrics, weight_initialization

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = [
    "SpeciesDetector",
]

class POOL(enum.Enum):
    MAX = "max"
    MEAN = "mean"
    FEATURE_ATTN = "feature_attn"
    PROB_ATTN = "prob_attn"

@dataclass(kw_only=True, eq=False)
class SpeciesDetector(L.LightningModule):
    species_list_path: str
    in_features: int
    l1_penalty: float
    penalty_multiplier: int
    beta: float
    clf_learning_rate: float
    pool_method: str = "max"
    attn_dim: int | None = None
    attn_learning_rate: float | None = None

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        self.save_hyperparameters()
        self.pool_method = POOL(self.pool_method)
        self.species = self._load_species_list()
        self.beta = torch.nn.Parameter(torch.tensor(self.beta, dtype=torch.float32), requires_grad=False)
        self.classifiers = torch.nn.ModuleDict({
            species_name: torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
            for species_name in self.species
        })
        if self.pool_method == POOL.FEATURE_ATTN or self.pool_method == POOL.PROB_ATTN:
            self.attention_V = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
            self.attention_U = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
            self.attention_w = torch.nn.ModuleDict({
                species_name: torch.nn.Linear(in_features=self.attn_dim, out_features=1)
                for species_name in self.species
            })

    # TODO: include in pipeline somehow
    def run(
        self,
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

    def forward(self, x: torch.Tensor, species_names: List[str]) -> torch.Tensor:
        # pool by taking a linear combination of *features* as a function of the data
        if self.pool_method == POOL.FEATURE_ATTN:
            y_probs = []
            A_V = torch.tanh(self.attention_V(x)) # (N, T, D)
            A_U = torch.sigmoid(self.attention_U(x)) # (N, T, D)
            for species_name in species_names:
                clf = self.classifiers[species_name]
                attention_w = self.attention_w[species_name]
                assert clf is not None and attention_w is not None, f"'{species_name}' is not a valid species"
                A = F.softmax(attention_w(A_V * A_U), dim=-2) # (N, T, 1)
                y_species_logits = clf((x * A).sum(dim=-2))
                y_species_probs = torch.sigmoid(y_species_logits)
                y_probs.append(y_species_probs)
            return torch.cat(y_probs, dim=-1).mean(dim=1)
        # pool by taking a linear function of *probabilities* as a function of the data
        elif self.pool_method == POOL.PROB_ATTN:
            y_probs = []
            A_V = torch.tanh(self.attention_V(x)) # (N, T, D)
            A_U = torch.sigmoid(self.attention_U(x)) # (N, T, D)
            for species_name in species_names:
                clf = self.classifiers[species_name]
                attention_w = self.attention_w[species_name]
                assert clf is not None and attention_w is not None, f"'{species_name}' is not a valid species"
                A = F.softmax(attention_w(A_V * A_U), dim=-2) # (N, T, 1)
                y_species_logits = clf(x)
                y_species_probs = torch.sigmoid(y_species_logits)
                # y_species_probs = torch.logsumexp((A + 1e-6).log() + F.logsigmoid(y_species_logits), dim=-2).exp()
                y_species_probs = (y_species_probs * A).sum(dim=-2)
                y_probs.append(y_species_probs)
            return torch.cat(y_probs, dim=-1).mean(dim=1)
        # pool by maximum probability
        elif self.pool_method == POOL.MAX:
            y_probs = []
            for species_name in species_names:
                clf = self.classifiers[species_name]
                assert clf is not None, f"'{species_name}' is not a valid species"
                y_species_logits = torch.max(clf(x), dim=-2).values
                y_species_probs = torch.sigmoid(y_species_logits)
                y_probs.append(y_species_probs)
            return torch.cat(y_probs, dim=-1).mean(dim=1)
        # pool by mean probability
        elif self.pool_method == POOL.MEAN:
            y_probs = []
            for species_name in species_names:
                clf = self.classifiers[species_name]
                assert clf is not None, f"'{species_name}' is not a valid species"
                y_species_logits = torch.mean(clf(x), dim=-2)
                y_species_probs = torch.sigmoid(y_species_logits)
                y_probs.append(y_species_probs)
            return torch.cat(y_probs, dim=-1).mean(dim=1)

    def loss(self, y: torch.Tensor, y_probs: torch.Tensor, y_freq: List[float], epsilon: float = 1e-6) -> Dict[str, torch.Tensor]:
        # batch mean over log probabilities weighted by positive class frequency
        cel = metrics.class_balanced_binary_cross_entropy(y, y_probs, self.beta, torch.tensor(y_freq, dtype=torch.int64).to(y.device))
        # L1 regularisation with split for smooth latents (if applicable), sum the L1 penalties for each model
        l1_1 = metrics.l1_penalty([weights[:, 0:64] for weights in list(self.classifiers.parameters())[::2]], self.l1_penalty)
        l1_2 = metrics.l1_penalty([weights[:, 64:128] for weights in list(self.classifiers.parameters())[::2]], self.l1_penalty * self.penalty_multiplier)
        l1_penalty = torch.stack([l1_1, l1_2], dim=-1).sum(-1)
        # per logistic regression model, add the CEL and L1 together and then sum to get the total loss
        loss = (cel + l1_penalty).sum()
        return dict(
            loss=loss,
            cel=cel.detach().sum(),
            l1_penalty=l1_penalty.detach().sum(),
            l1_penalty_1=l1_1.detach().sum(),
            l1_penalty_2=l1_2.detach().sum(),
        )

    def evaluate(self, batch_predictions: List[pd.DataFrame]) -> None:
        predictions = pd.concat(batch_predictions)
        scores = self.score(predictions)
        import code; code.interact(local=locals())
        print(scores.to_markdown())

    def score(self, results: pd.DataFrame) -> pd.DataFrame:
        scores = []
        for species_name in results.species_name.unique():
            y = results.loc[results.species_name == species_name, "label"].values
            y_prob = results.loc[results.species_name == species_name, "prob"].values
            if np.isnan(y_prob).any():
                prop_nans = np.isnan(y_prob).sum() / len(y_prob)
                log.warning(f"NaNs found in predicted probabilities for {species_name} with a proportional count of {prop_nans}")
                y_prob = np.nan_to_num(y_prob, nan=0.0)
            assert not np.isnan(y).any(), f"NaNs found in true labels for {species_name}"
            scores.append(dict(
                species_name=species_name,
                mAP=metrics.average_precision(y, y_prob),
                auROC=sklearn.metrics.roc_auc_score(y, y_prob),
            ))
        return pd.DataFrame(data=scores).set_index("species_name")

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y, _, y_freq = batch
        y_probs = self.forward(x, list(y_freq.keys()))
        loss_outputs = self.loss(y.float(), y_probs, list(y_freq.values()))
        self.log_dict({f"train/{key}": value for key, value in loss_outputs.items()}, prog_bar=True)
        return loss_outputs

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y, _, y_freq = batch
        y_probs = self.forward(x, list(y_freq.keys()))
        loss_outputs = self.loss(y.float(), y_probs, list(y_freq.values()))
        self.log_dict({f"val/{key}": value for key, value in loss_outputs.items()}, prog_bar=True)
        return loss_outputs

    @torch.no_grad()
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], batch_idx: int) -> pd.DataFrame:
        x, y, s, y_freq = batch
        y_probs = self.forward(x, list(y_freq.keys()))
        return (
            pd.DataFrame(
                data=y.detach().cpu(),
                columns=list(y_freq.keys()),
                index=s.detach().cpu().tolist()
            )
            .reset_index(names="file_i")
            .melt(id_vars="file_i", var_name="species_name", value_name="label")
            .merge(
                pd.DataFrame(
                    data=y_probs.detach().cpu(),
                    columns=list(y_freq.keys()),
                    index=s.detach().cpu().tolist()
                )
                .reset_index(names="file_i")
                .melt(id_vars="file_i", var_name="species_name", value_name="prob"),
                on=["file_i", "species_name"],
                how="inner",
            )
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = []
        params.append({'params': self.classifiers.parameters(), 'lr': self.clf_learning_rate})
        if self.pool_method == POOL.FEATURE_ATTN or self.pool_method == POOL.PROB_ATTN:
            attn_params = list(self.attention_V.parameters()) + list(self.attention_U.parameters()) + list(self.attention_w.parameters())
            params.append({'params': attn_params, 'lr': self.attn_learning_rate})
        return torch.optim.Adam(params)

    def _load_species_list(self):
        with open(self.species_list_path, "r") as f:
            return yaml.safe_load(f.read())
