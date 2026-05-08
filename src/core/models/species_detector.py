import enum
import itertools
import lightning as L
import logging
import numpy as np
import pandas as pd
import hydra
import omegaconf
import pathlib
import torch
import yaml

from dataclasses import dataclass
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from src.core.utils import metrics
from src.core.utils import Batch, TensorDict, detach_values, prefix_keys, histogram_to_wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SpeciesDetector"]

class SpeciesDetector(L.LightningModule):
    def __init__(
        self,
        target_names: List[str],
        target_counts: List[int],
        in_features: int,
        beta: float = 0.0,
        lamdba : float = 1e-1,
        label_smoothing: float = 0.0,
        pool_method: str = "max",
        attn_dim: int = 10,
        clf_learning_rate: float = 1e-2,
        attn_learning_rate: float = 1e-3,
        attn_weight_decay: float = 1e-3,
        train_sample_size: int | None = 1,
        eval_sample_size: int | None = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.target_names = target_names
        self.target_counts = target_counts
        self.in_features = in_features
        self.lamdba = lamdba
        self.label_smoothing = label_smoothing
        self.attn_dim = attn_dim

        self.clf_learning_rate = clf_learning_rate
        self.attn_learning_rate = attn_learning_rate
        self.attn_weight_decay = attn_weight_decay
        self.train_sample_size = train_sample_size
        self.eval_sample_size = eval_sample_size

        self.beta = torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32), requires_grad=False)
        self.classifiers = self._init_classifiers()
        self.attention_V, self.attention_U, self.attention_w = self._init_attention()

    def _init_classifiers(self):
        return torch.nn.ModuleDict({
            target_name: torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
            for target_name in self.target_names
        })

    def _init_attention(self):
        # gated attention mechanism
        attention_V = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
        # all layers initialized according to Glorot & Bengio (2010) and biases set to zero
        torch.nn.init.xavier_uniform_(attention_V.weight)
        torch.nn.init.zeros_(attention_V.bias)
        attention_U = torch.nn.ModuleDict({})
        for target_name in self.target_names:
            layer = torch.nn.Linear(in_features=self.in_features, out_features=self.attn_dim)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            attention_U[target_name] = layer
        # no biases needed for attention weight layer
        attention_w = torch.nn.ModuleDict({})
        for target_name in self.target_names:
            layer = torch.nn.Linear(in_features=self.attn_dim, out_features=1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight)
            attention_w[target_name] = layer
        return attention_V, attention_U, attention_w

    def pre_process(self, x: torch.Tensor):
        num_samples = self.train_sample_size if self.training else self.eval_sample_size
        if num_samples is None:
            return x.unsqueeze(1)
        mean, log_var = x.chunk(2, dim=-1)
        mean = mean.unsqueeze(1).expand(-1, num_samples, -1, -1)
        log_var = log_var.unsqueeze(1).expand(-1, num_samples, -1, -1)
        return mean + torch.randn_like(mean) * (0.5 * log_var).exp()

    def forward(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
        # forward pass
        y_probs = []
        attn_w = []
        A_V = torch.tanh(self.attention_V(x)) # (N, T, D)
        for target_name in self.target_names:
            clf = self.classifiers[target_name]
            attention_w = self.attention_w[target_name]
            attention_U = self.attention_U[target_name]
            A_U = torch.sigmoid(attention_U(x))
            A = F.softmax(attention_w(A_V * A_U), dim=-2) # (N, T, 1)
            y_target_probs = (torch.sigmoid(clf(x)) * A).sum(dim=-2)
            y_probs.append(y_target_probs)
            attn_w.append(A)
        y_probs, attn_w = torch.cat(y_probs, dim=-1), torch.cat(attn_w, dim=-1)
        y_probs = y_probs.mean(dim=1)
        attn_w = attn_w.mean(dim=1)
        samples_per_class = torch.tensor(self.target_counts, dtype=torch.int64, device=y.device)
        return dict(
            y=y, y_probs=y_probs,
            attn_w=attn_w, samples_per_class=samples_per_class,
            s=s, target_names=self.target_names,
        )

    def loss(self, y: torch.Tensor, y_probs: torch.Tensor, samples_per_class: torch.Tensor, epsilon: float = 1e-6, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = TensorDict()
        # batch mean over log probabilities weighted by positive class frequency
        cel = metrics.class_balanced_binary_cross_entropy(y, y_probs, samples_per_class=samples_per_class, **self.cel_params).mean(dim=0)
        outputs |= dict(cel=cel.detach().sum())
        # sparse penalty for clf weights
        weights = torch.cat(list(self.classifiers.parameters())[::2], dim=0)
        l1 = self.lamdba * torch.linalg.norm(weights, dim=-1, ord=1)
        outputs |= dict(l1=l1.detach().sum())
        # per logistic regression model, add the CEL and L1 together and then sum to get the total loss
        loss = (cel + l1).sum()
        outputs |= dict(loss=loss)
        return outputs

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
        s = s.expand(y.size(-1), -1).permute(1, 0).flatten().cpu().numpy()
        target_names = list(itertools.chain(*[target_names for _ in range(y_probs.size(0))]))
        y, y_probs = y.flatten().cpu().numpy(), y_probs.flatten().cpu().numpy()
        return pd.DataFrame(
            data=list(zip(s, target_names, y, y_probs)),
            columns=["file_i", "species_name", "label", "prob"]
        )

    @property
    def param_groups(self):
        return [
            self.classifiers.parameters(),
            list(self.attention_V.parameters()) + list(self.attention_U.parameters()) + list(self.attention_w.parameters()),
        ]

    @property
    def cel_params(self):
        return dict(beta=self.beta, label_smoothing=self.label_smoothing)

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        # run training
        log.info(f"Training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        checkpoint_path, resume = config.get("ckpt_path"), config.get("resume")
        if checkpoint_path is not None and resume:
            log.info(f"Resuming from {checkpoint_path}")
            trainer.fit(self, datamodule=data_module, ckpt_path=checkpoint_path)
        elif config.get("ckpt_path"):
            log.info(f"Loading state dict from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint["state_dict"], strict=False)
            trainer.fit(self, datamodule=data_module)
        else:
            trainer.fit(self, datamodule=data_module)
        # persist the model configuration
        checkpoint_dir = pathlib.Path(trainer.checkpoint_callback.dirpath)
        if checkpoint_dir.exists():
            config_path = checkpoint_dir / "config.yaml"
            log.info(f"Saving model configuration to {config_path}")
            omegaconf.OmegaConf.save(config, config_path)
        # running test
        log.info(f"Testing <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, datamodule=data_module)

    def step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self.forward(**batch, t=self.trainer.global_step, **kwargs)
        loss_outputs = self.loss(**step_outputs, t=self.trainer.global_step)
        step_outputs = detach_values(step_outputs)
        return loss_outputs, step_outputs

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(loss_outputs, "train"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        metrics = histogram_to_wandb(self.metrics(**step_outputs))
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs | metrics, "train")))
        return loss_outputs | step_outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        metrics = histogram_to_wandb(self.metrics(**step_outputs))
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs | metrics, "val")))
        return loss_outputs | step_outputs

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predict(**batch, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.predict(**batch, **kwargs)

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        num_samples = self.train_sample_size if dataloader_idx == 0 else self.eval_sample_size
        x = self.pre_process(batch.x, num_samples)
        return Batch(x=x, **{k: batch[k] for k in batch.keys() if k != "x"})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        groups = zip([self.clf_learning_rate, self.attn_learning_rate], [0.0, self.attn_weight_decay], self.param_groups)
        return torch.optim.Adam([
            dict(lr=learning_rate, params=params, weight_decay=weight_decay)
            for learning_rate, weight_decay, params in groups
        ])
