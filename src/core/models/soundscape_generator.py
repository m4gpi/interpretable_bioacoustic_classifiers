import hydra
import lightning as L
import pathlib
import logging
import numpy as np
import torch
import wandb

from dataclasses import dataclass
from omegaconf import DictConfig
from torch.nn import functional as F
from typing import Any, Dict, Callable, List, Tuple

from src.core.models.backbones.transformer_encoder import TransformerEncoder
from src.core.utils import detach_values, prefix_keys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

__all__ = ["SoundscapeGenerator"]

@dataclass(unsafe_hash=True, kw_only=True)
class SoundscapeGenerator(L.LightningModule):
    num_classes: 512
    embedding_dim: int = 64
    attn_num_enc_heads: int = 2
    attn_num_dec_heads: int = 2
    attn_enc_depth: int = 2
    attn_dec_depth: int = 2
    attn_mlp_ratio: int = 4

    learning_rate: float = 4e-5
    optimiser_cls: str = "torch.optim.AdamW"
    optimiser_config: DictConfig | None = None
    scheduler_cls: str | None = None
    scheduler_config: DictConfig | None = None
    scheduler_interval: str = "step"
    scheduler_frequency: int = 1

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        log.info(f"Beginning training <{config.model.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.fit(
            self,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )

    def __new__(cls, *args: Any, **kwargs: Any):
        obj = object.__new__(cls)
        L.LightningModule.__init__(obj)
        return obj

    def __post_init__(self) -> None:
        # map vqvae codebook indices to embeddings
        self.embedding = torch.nn.Embedding(self.num_classes, self.embedding_dim)
        # learn a token that initializes the auto-regressive process
        self.init_token = torch.nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        # use encoded attentions to decode sequence autoregressively
        self.sequence_decoder = TransformerEncoder(
            input_size=self.embedding_dim,
            mlp_ratio=self.attn_mlp_ratio,
            depth=self.attn_dec_depth,
            num_heads=self.attn_num_dec_heads,
        )
        self.projection = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.num_classes)
        self._reset_cache()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        x = x.flatten(start_dim=1)
        x_e = self.embedding(x)
        x_e = x_e + self.positional_encoding(x_e.size(1), x_e.size(2)).to(x_e.device)
        attn_mask = self.causal_mask(x_e.size(1)).to(x_e.device)
        x_in = torch.cat([self.init_token.expand(x_e.size(0), 1, -1), x_e[:, :-1]], dim=1)
        x_out, _, attn_w = self.sequence_decoder(x_in, attn_mask=attn_mask)
        logits = self.projection(x_out)
        return dict(x=x, logits=logits, attn_w=attn_w)

    @staticmethod
    def positional_encoding(sequence_len: int, embedding_dim: int) -> torch.Tensor:
        positional_embedding = torch.zeros(sequence_len, embedding_dim)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        return positional_embedding

    @staticmethod
    def causal_mask(sequence_len: int) -> torch.Tensor:
        neg_inf = torch.from_numpy(np.ones((sequence_len, sequence_len)) * -np.inf)
        return torch.triu(neg_inf, diagonal=1).float()

    def loss(self, x: torch.Tensor, logits: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        cel = F.cross_entropy(logits.flatten(end_dim=1), x.flatten(end_dim=1), reduction="none").unflatten(0, (x.size(0), x.size(1)))
        # loss scale???
        cel = cel.sum(dim=1).mean()
        return dict(loss=cel, cel=cel.detach())

    def training_step(self, batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self.forward(batch.x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        self.training_step_outputs.append(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "train"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs, "train")))
        return loss_outputs | step_outputs

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch, batch_idx: int) -> None:
        self.training_step_outputs.clear()

    @torch.no_grad()
    def validation_step(self, batch: Tuple, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self.forward(batch.x, **kwargs)
        loss_outputs = self.loss(**step_outputs)
        self.validation_step_outputs.append(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, "val"), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(loss_outputs, "val")))
        return loss_outputs | step_outputs

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor], batch, batch_idx: int) -> None:
        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimiser_config = DictConfig(dict(_target_=self.optimiser_cls, **(self.optimiser_config or {})))
        optimiser = hydra.utils.instantiate(optimiser_config, params=self.parameters(), lr=self.learning_rate)
        if self.scheduler_cls is not None:
            scheduler_config = DictConfig(dict(_target_=self.scheduler_cls, **(self.scheduler_config or {})))
            scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimiser)
            return [optimiser], [dict(
                scheduler=scheduler,
                interval=self.scheduler_interval,
                frequency=self.scheduler_frequency
            )]
        return optimiser

    def _reset_cache(self):
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []
