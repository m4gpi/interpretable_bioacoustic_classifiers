import enum
import functools
import lightning as L
import logging
import numpy as np
import pandas as pd
import pathlib
import torch
import yaml

from dataclasses import dataclass
from omegaconf import DictConfig
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from src.core.utils import metrics

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SpeciesDetector"]

class POOL(enum.Enum):
    MAX = "max"
    MEAN = "mean"
    FEATURE_ATTN = "feature_attn"
    PROB_ATTN = "prob_attn"

@dataclass(kw_only=True, eq=False)
class SpeciesDetector(L.LightningModule):
    target_names: List[str]
    target_counts: List[int]
    in_features: int
    l1_penalty: float
    penalty_multiplier: int
    beta: float
    clf_learning_rate: float
    label_smoothing: float = 0.0
    pool_method: str = "max"
    attn_dim: int | None = None
    attn_learning_rate: float | None = None
    train_sample_size: int = 1
    eval_sample_size: int = 1

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        self.save_hyperparameters()
        self.pool_method = POOL(self.pool_method)
        self.beta = torch.nn.Parameter(torch.tensor(self.beta, dtype=torch.float32), requires_grad=False)
        self.classifiers = torch.nn.ModuleDict({
            target_name: torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
            for target_name in self.target_names
        })
        if self.pool_method == POOL.FEATURE_ATTN or self.pool_method == POOL.PROB_ATTN:
            self.attention_V = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
            self.attention_U = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
            self.attention_w = torch.nn.ModuleDict({
                target_name: torch.nn.Linear(in_features=self.attn_dim, out_features=1)
                for target_name in self.target_names
            })

    def run(
        self,
        trainer: L.Trainer,
        config: DictConfig,
        data_module: L.LightningDataModule | None = None,
        train_dataloader: torch.utils.data.DataLoader | None = None,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        test_dataloader: torch.utils.data.DataLoader | None = None,
        device: int | None = None,
    ) -> None:
        if not device:
            device = trainer.strategy.root_device
            log.info(f"Device: {device}")

        if config.get("train"):
            if self.pool_method == POOL.MAX:
                if data_module is not None:
                    data_module.setup(stage="fit")
                    train_dataloader = data_module.train_dataloader()
                assert train_dataloader is not None, f"No training dataloader provided, either provide a data module or dataloaders"
                log.info(f"Applying weight initialisation procedure for {self.pool_method}")
                self._max_pool_weight_initialization(train_dataloader, device)
                log.info(f"Weight initialisation complete")

            log.info("Starting training!")
            if data_module is not None:
                trainer.fit(model=self, datamodule=data_module, ckpt_path=config.get("ckpt_path"))
            else:
                trainer.fit(
                    model=self,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path=config.get("ckpt_path", None)
                )

        predictions = None
        if config.get("test"):
            log.info("Starting prediction!")
            ckpt_path = trainer.checkpoint_callback.best_model_path

            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for prediction...")
                ckpt_path = None

            if data_module is not None:
                predictions = trainer.predict(model=self, datamodule=data_module, ckpt_path=ckpt_path, return_predictions=True)
            else:
                predictions = trainer.predict(model=self, dataloaders=test_dataloader, ckpt_path=ckpt_path, return_predictions=True)

        return predictions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # pool by taking a linear combination of *features* as a function of the data
        if self.pool_method == POOL.FEATURE_ATTN:
            y_probs = []
            attn_w = []
            A_V = torch.tanh(self.attention_V(x)) # (N, T, D)
            # TODO: try sigmoid layer different for target?
            A_U = torch.sigmoid(self.attention_U(x)) # (N, T, D)
            for target_name in self.target_names:
                clf = self.classifiers[target_name]
                attention_w = self.attention_w[target_name]
                assert clf is not None and attention_w is not None, f"'{target_name}' is not a valid target"
                A = F.softmax(attention_w(A_V * A_U), dim=-2) # (N, T, 1)
                y_target_logits = clf((x * A).sum(dim=-2))
                y_target_probs = torch.sigmoid(y_target_logits)
                y_probs.append(y_target_probs)
                attn_w.append(A)
            return torch.cat(y_probs, dim=-1), torch.cat(attn_w, dim=-1)
        # pool by taking a linear function of *probabilities* as a function of the data
        elif self.pool_method == POOL.PROB_ATTN:
            y_probs = []
            attn_w = []
            A_V = torch.tanh(self.attention_V(x)) # (N, T, D)
            A_U = torch.sigmoid(self.attention_U(x)) # (N, T, D)
            for target_name in self.target_names:
                clf = self.classifiers[target_name]
                attention_w = self.attention_w[target_name]
                assert clf is not None and attention_w is not None, f"'{target_name}' is not a valid target"
                A = F.softmax(attention_w(A_V * A_U), dim=-2) # (N, T, 1)
                y_target_logits = clf(x)
                y_target_probs = torch.sigmoid(y_target_logits)
                # y_target_probs = torch.logsumexp((A + 1e-6).log() + F.logsigmoid(y_target_logits), dim=-2).exp()
                y_target_probs = (y_target_probs * A).sum(dim=-2)
                y_probs.append(y_target_probs)
                attn_w.append(A)
            return torch.cat(y_probs, dim=-1), torch.cat(attn_w, dim=-1)
        # pool by maximum probability
        elif self.pool_method == POOL.MAX:
            y_probs = []
            for target_name in self.target_names:
                clf = self.classifiers[target_name]
                assert clf is not None, f"'{target_name}' is not a valid target"
                y_target_logits = torch.max(clf(x), dim=-2).values
                y_target_probs = torch.sigmoid(y_target_logits)
                y_probs.append(y_target_probs)
            return torch.cat(y_probs, dim=-1), None
        # pool by mean probability
        elif self.pool_method == POOL.MEAN:
            y_probs = []
            for target_name in self.target_names:
                clf = self.classifiers[target_name]
                assert clf is not None, f"'{target_name}' is not a valid target"
                y_target_logits = torch.mean(clf(x), dim=-2)
                y_target_probs = torch.sigmoid(y_target_logits)
                y_probs.append(y_target_probs)
            return torch.cat(y_probs, dim=-1), None

    def loss(self, y: torch.Tensor, y_probs: torch.Tensor, epsilon: float = 1e-6) -> Dict[str, torch.Tensor]:
        # batch mean over log probabilities weighted by positive class frequency
        cel = metrics.class_balanced_binary_cross_entropy(
            y, y_probs,
            beta=self.beta,
            samples_per_class=torch.tensor(self.target_counts, dtype=torch.int64).to(y.device),
            label_smoothing=self.label_smoothing,
        ).mean(dim=0)
        # L1 regularisation with split for smooth latents (if applicable), sum the L1 penalties for each model
        # TODO: parameterise these indices
        l1_1 = metrics.l1_penalty([weights[:, 0:64] for weights in list(self.classifiers.parameters())[::2]], self.l1_penalty)
        l1_2 = metrics.l1_penalty([weights[:, 64:128] for weights in list(self.classifiers.parameters())[::2]], self.l1_penalty * self.penalty_multiplier)
        l1_penalty = torch.stack([l1_1, l1_2], dim=-1).sum(dim=-1)
        # per logistic regression model, add the CEL and L1 together and then sum to get the total loss
        loss = (cel + l1_penalty).sum()
        return dict(
            loss=loss,
            cel=cel.detach().sum(),
            l1_penalty=l1_penalty.detach().sum(),
            l1_penalty_1=l1_1.detach().sum(),
            l1_penalty_2=l1_2.detach().sum(),
        )

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], num_samples: int = 1) -> Tuple[torch.Tensor, ...]:
        q_z, y, s = batch
        mean, log_var = q_z.chunk(2, dim=-1)
        mean = mean.unsqueeze(1).expand(-1, num_samples, -1, -1)
        log_var = log_var.unsqueeze(1).expand(-1, num_samples, -1, -1)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        y_probs, attn_w = self.forward(z)
        y_probs = y_probs.mean(dim=1)
        if attn_w is not None:
            attn_w = attn_w.mean(dim=1)
        return y, y_probs, attn_w, s

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        y, y_probs, _, s = self.model_step(batch, num_samples=self.train_sample_size)
        loss_outputs = self.loss(y.float(), y_probs)
        self.log_dict({f"train/{key}": value for key, value in loss_outputs.items()}, prog_bar=True, batch_size=s.size(0))
        return {**loss_outputs, "y_probs": y_probs, "y": y, "s": s, "target_names": self.target_names}

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        y, y_probs, _, s = self.model_step(batch, num_samples=self.eval_sample_size)
        loss_outputs = self.loss(y.float(), y_probs)
        self.log_dict({f"val/{key}": value for key, value in loss_outputs.items()}, prog_bar=True, batch_size=s.size(0))
        return {**loss_outputs, "y_probs": y_probs, "y": y, "s": s, "target_names": self.target_names}

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> pd.DataFrame:
        y, y_probs, _, s = self.model_step(batch, num_samples=self.eval_sample_size)
        return {"y_probs": y_probs, "y": y, "s": s, "target_names": self.target_names}

    @torch.no_grad()
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> pd.DataFrame:
        y, y_probs, _, s, y_freq = self.model_step(batch)
        y_probs = y_probs.mean(dim=1)
        return {"y_probs": y_probs, "y": y, "s": s, "target_names": self.target_counts}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = []
        params.append({'params': self.classifiers.parameters(), 'lr': self.clf_learning_rate})
        if self.pool_method == POOL.FEATURE_ATTN or self.pool_method == POOL.PROB_ATTN:
            attn_params = list(self.attention_V.parameters()) + list(self.attention_U.parameters()) + list(self.attention_w.parameters())
            params.append({'params': attn_params, 'lr': self.attn_learning_rate})
        return torch.optim.Adam(params)

    def _load_target_list(self):
        with open(self.target_list_path, "r") as f:
            return yaml.safe_load(f.read())

    def _max_pool_weight_initialization(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str,
        num_initialisations: int = 100,
    ) -> None:
        """
        Create a set of num_initialisations linear classifiers for each target and evaluate
        performance on the training set. Pick the best linear classifier for each target to use as an initialisation point.
        """
        self.to(device)
        classifiers = torch.nn.ModuleDict({})
        # J initialisations for each target
        for target_name in self.classifiers.keys():
            classifiers[target_name] = torch.nn.ModuleList([
                torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
                for i in range(num_initialisations)
            ])
        classifiers.to(device)
        # evaluate k layers on entire training set, select the best performing model on the training set
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                q_z, y, _ = batch
                q_z, y = q_z.to(device), y.to(device)
                mean, log_var = q_z.chunk(2, dim=-1)
                z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
                for j in range(num_initialisations):
                    y_probs = torch.cat([torch.sigmoid(torch.max(layers[j](z), dim=-2).values) for layers in classifiers.values()], dim=-1)
                    loss = metrics.class_balanced_binary_cross_entropy(y, y_probs, self.beta, torch.tensor(self.target_counts, dtype=torch.int64).to(y.device))
                    losses.append(loss)
        # sample-wise sum
        losses = torch.stack(losses, dim=1).sum(dim=0) # (N, S)
        # identify the weights with the lowest losses for each target
        indices = losses.argmin(dim=0)
        # for each target select the layer within its subset of layers that yields the lowest loss
        for idx, (target_name, layers) in zip(indices, classifiers.items()):
            self.classifiers[target_name].load_state_dict(layers[idx].state_dict())
        # delete unused layers from memory
        del classifiers

