import math
import enum
import collections
import itertools
import functools
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
from typing import Any, Callable, Dict, List, Tuple

from src.core.utils import metrics
from src.core.utils import Batch, TensorDict, detach_values, prefix_keys, histogram_to_wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["MILSpeciesDetector"]

class L1:
    def __call__(self, weights: torch.Tensor, dim: int, **kwargs: Any) -> torch.Tensor:
        return metrics.l1(weights, dim=dim)

class L2:
    def __call__(self, weights: torch.Tensor, dim: int, **kwargs: Any) -> torch.Tensor:
        return metrics.l2(weights, dim=dim)

class Elastic:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def __call__(self, weights: torch.Tensor, dim: int, **kwargs: Any) -> torch.Tensor:
        return metrics.elastic(weights, dim=dim, alpha=self.alpha)

class MultiLabelLogisticRegression(torch.nn.Module):
    def __init__(self, num_features: int, num_targets: int) -> None:
        super().__init__()
        weight = torch.empty(num_targets, num_features, 1)
        bias = torch.empty(num_targets, 1)
        self.reset_parameters(weight, bias)
        self.weight = torch.nn.Parameter(weight.squeeze())
        self.bias = torch.nn.Parameter(bias.squeeze())

    def reset_parameters(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        y_prob = torch.sigmoid(x @ self.weight.t() + self.bias)
        return y_prob,

    def get_weights(self):
        return self.weight

class MultiLabelBayesianLogisticRegression(torch.nn.Module):
    def __init__(self, num_features: int, num_targets: int) -> None:
        super().__init__()
        # NB: isotropic multivariate gaussian
        weight_mu = torch.empty(num_targets, num_features, 1)
        bias_mu = torch.empty(num_targets, 1)
        self.reset_parameters(weight_mu, bias_mu)
        self.weight_mu = torch.nn.Parameter(weight_mu.squeeze())
        self.bias_mu = torch.nn.Parameter(bias_mu.squeeze())
        weight_log_var = torch.empty(num_targets, num_features, 1)
        bias_log_var = torch.empty(num_targets, 1)
        self.reset_parameters(weight_log_var, bias_log_var)
        self.weight_log_var = torch.nn.Parameter(weight_log_var.squeeze())
        self.bias_log_var = torch.nn.Parameter(bias_log_var.squeeze())

    def reset_parameters(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # predict the mean and variance of activations
        mu_a = x @ self.weight_mu.t() + self.bias_mu
        sigma_sq_a = x.pow(2) @ self.weight_log_var.exp().t() + self.bias_log_var.exp()
        # approximating the expectation of a sigmoid under a Gaussian distribution using the mackay approximation
        y_prob = torch.sigmoid(mu_a / torch.sqrt(1.0 + torch.pi * sigma_sq_a / 8.0))
        return y_prob, mu_a, sigma_sq_a

    def get_weights(self):
        return self.weight_mu

class GatedAttention(torch.nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, num_targets: int) -> None:
        super().__init__()
        self.A_V_weight = torch.nn.Parameter(torch.empty(in_features, hidden_dim))
        self.A_V_bias = torch.nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters(self.A_V_weight, self.A_V_bias)
        self.A_U_weight = torch.nn.Parameter(torch.empty(num_targets, in_features, hidden_dim))
        self.A_U_bias = torch.nn.Parameter(torch.empty(num_targets, 1, hidden_dim))
        self.reset_parameters(self.A_U_weight, self.A_U_bias)
        self.A_w = torch.nn.Parameter(torch.empty(num_targets, hidden_dim, out_features))
        self.reset_parameters(self.A_w)

    def reset_parameters(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        torch.nn.init.xavier_uniform_(weight)
        if bias is not None:
            torch.nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor, target_i: int | None = None) -> torch.Tensor:
        if target_i is not None:
            A_V = torch.tanh(x @ self.A_V_weight + self.A_V_bias) # (N, T, D)
            A_U = torch.sigmoid(x @ self.A_U_weight[target_i] + self.A_U_bias[target_i]) # (N, T, D)
            A = F.softmax((A_V * A_U) @ self.A_w[target_i], dim=-2) # (N, T, 1)
            A = A.squeeze(-1)
        else:
            x = x.unsqueeze(1) # (N, 1, T, D)
            A_V = torch.tanh(x @ self.A_V_weight + self.A_V_bias) # (N, 1, T, D)
            A_U = torch.sigmoid(x @ self.A_U_weight + self.A_U_bias) # (N, C, T, D)
            A = F.softmax((A_V * A_U) @ self.A_w, dim=-2) # (N, C, T, 1)
            A = A.squeeze(-1).transpose(-1, -2) # (N, T, C)
        return A

    def get_weights(self):
        return self.A_V_weight, self.A_U_weight, self.A_w

class MILSpeciesDetector(L.LightningModule):
    @staticmethod
    def to_buffer_matrix(str_array: List[str]) -> torch.Tensor:
        ords = list(map(lambda s: list(map(ord, s)), str_array))
        buf = torch.zeros((len(ords), len(sorted(ords, key=len, reverse=True)[0])), dtype=torch.int32)
        for i, o in enumerate(ords):
            buf[i, :len(o)] = torch.tensor(o)
        return buf

    @staticmethod
    def from_buffer_matrix(buf: torch.Tensor) -> List[str]:
        return [
            "".join((map(chr, buf[i,torch.nonzero(buf[i])].squeeze())))
            for i in range(buf.size(0))
        ]

    @property
    def target_index(self):
        return torch.arange(self.target_names_enc.size(0))

    @functools.cached_property
    def target_names(self):
        return self.from_buffer_matrix(self.target_names_enc)

    def species_weights(self, species_name: str) -> torch.Tensor:
        weights = self.classifiers.get_weights()
        return weights[self.target_names.index(species_name)]

    def attention_weights(self, x: torch.Tensor, species_name: str) -> torch.Tensor:
        return self.attention(x, self.target_names.index(species_name))

    def __init__(
        self,
        target_names: List[str],
        target_counts: List[str],
        attention: omegaconf.DictConfig,
        classifiers: omegaconf.DictConfig,
        clf_regulariser: omegaconf.DictConfig,
        attn_regulariser: omegaconf.DictConfig,
        beta: float = 0.0,
        gamma_clf: float = 0.0,
        gamma_attn: float = 0.0,
        label_smoothing: float = 0.0,
        clf_learning_rate: float = 1e-2,
        attn_learning_rate: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.gamma_clf = gamma_clf
        self.gamma_attn = gamma_attn
        self.label_smoothing = label_smoothing
        self.clf_learning_rate = clf_learning_rate
        self.attn_learning_rate = attn_learning_rate

        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("target_names_enc", self.to_buffer_matrix(target_names))
        self.register_buffer("target_counts", torch.tensor(target_counts, dtype=torch.int64))

        self.attention = hydra.utils.instantiate(attention, num_targets=self.target_names_enc.size(0))
        self.classifiers = hydra.utils.instantiate(classifiers, num_targets=self.target_names_enc.size(0))
        self.clf_regulariser = hydra.utils.instantiate(clf_regulariser)
        self.attn_regulariser = hydra.utils.instantiate(attn_regulariser)

    def forward(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        attn_w = self.attention(x)
        y_t_probs, mu_a, sigma_sq_a = self.classifiers(x)
        y_probs = (y_t_probs * attn_w).sum(dim=-2)
        return dict(
            y=y, y_probs=y_probs, y_t_probs=y_t_probs, attn_w=attn_w, s=s,
            mu_a=mu_a, sigma_sq_a=sigma_sq_a,
            samples_per_class=self.target_counts,
        )

    def loss(self, y: torch.Tensor, y_probs: torch.Tensor, samples_per_class: torch.Tensor, epsilon: float = 1e-6, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = TensorDict()
        # first take the mean over VAE posterior samples
        y_probs = y_probs.mean(dim=1)
        # batch mean over log probabilities weighted by positive class frequency
        cel = metrics.class_balanced_binary_cross_entropy(y, y_probs, samples_per_class=samples_per_class, **self.cel_params).mean(dim=0)
        outputs |= dict(cel=cel.detach().sum())
        # sparse penalty for clf weights
        clf_reg = self.gamma_clf * self.clf_regulariser(self.classifiers.get_weights(), dim=-1)
        outputs |= {f"clf_{self.clf_regulariser.__class__.__name__.lower()}": clf_reg.detach().sum()}
        # l2 regularisation on attn model weights
        attn_reg = self.gamma_attn * sum([self.attn_regulariser(weights, dim=[-1, -2]) for weights in self.attention.get_weights()])
        outputs |= {f"attn_l2": attn_reg.detach().sum()}
        # sum the cel, clf_reg and attn_reg for logistic regression
        # total loss sums across models
        loss = (cel + clf_reg + attn_reg).sum()
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
        metrics = dict(collections.ChainMap(*[{
            f"attn_entropy_{target_name}_mean": a_e.mean(),
            f"attn_entropy_{target_name}_std": a_e.std(),
            f"a_e_{target_name}_hist": np.histogram(a_e.flatten(), bins=128, range=[0, max_entropy])
        } for target_name, a_e in zip(self.target_names, attn_entropy.T)]))
        return metrics

    @torch.no_grad()
    def predict(
        self,
        y: torch.Tensor,
        y_probs: torch.Tensor,
        s: torch.Tensor,
        **kwargs: Any
    ) -> pd.DataFrame:
        s = s.unsqueeze(1).expand(-1, self.target_names_enc.size(0))
        species_i = torch.arange(self.target_names_enc.size(0), device=y_probs.device).unsqueeze(0).expand(y_probs.size(0), -1)
        # target_names = np.array(list(itertools.chain(*[self.target_names for _ in range(y_probs.size(0))])))
        data = torch.stack([s.ravel(), species_i.ravel(), y.ravel(), y_probs.ravel()], dim=1)
        return data

    @property
    def cel_params(self):
        return dict(beta=self.beta, label_smoothing=self.label_smoothing)

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = False):
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
        checkpoint_name = trainer.checkpoint_callback.filename
        config_path = checkpoint_dir / "config.yaml"
        # wierd bug the checkpoint isnt saving, so do manually now
        trainer.save_checkpoint(checkpoint_dir / f"{checkpoint_name}.ckpt")
        log.info(f"Saving model configuration to {config_path}")
        omegaconf.OmegaConf.save(config, config_path)
        # run validation
        log.info(f"Final validation run <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.limit_val_batches = 1.0
        trainer.validate(self, datamodule=data_module)
        # running test
        if config.get("test"):
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
        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs | metrics, "train")))
        return loss_outputs | step_outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        metrics = histogram_to_wandb(self.metrics(**step_outputs))
        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs | metrics, "val")))
        return loss_outputs | step_outputs

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx)
        return loss_outputs | step_outputs

    @torch.no_grad()
    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        loss_outputs, step_outputs = self.step(batch, batch_idx)
        return loss_outputs | step_outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim_groups = zip([self.clf_learning_rate, self.attn_learning_rate], [self.classifiers.parameters(), self.attention.parameters()])
        return torch.optim.Adam([dict(lr=learning_rate, params=params) for learning_rate, params in optim_groups])

