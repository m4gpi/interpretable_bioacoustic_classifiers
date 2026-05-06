import hydra
import lightning as L
import torch
import logging
import omegaconf
import pathlib
import wandb

from torch.optim import Optimizer
from typing import Any, Dict, Tuple, List

from src.core.utils import Batch, detach_values, prefix_keys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Algorithm(L.LightningModule):
    def __init__(self, model: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.strict_loading = False
        self._reset_cache()

    def run(self, trainer: L.Trainer, data_module: L.LightningDataModule, config: Dict[str, Any], test: bool = True):
        # run training
        log.info(f"Training <{config.algorithm.get('_target_')}> on <{config.data.get('_target_')}>")
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
        log.info(f"Testing <{config.algorithm.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, datamodule=data_module)

    def forward(self, batch: Batch, batch_idx: int, stage: str, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self.model(**batch, t=self.trainer.global_step, **kwargs)
        loss_outputs = self.model.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        self.log_dict(prefix_keys(loss_outputs, stage), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        return {**loss_outputs, **step_outputs}

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch, batch_idx, "train")
        return outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch, batch_idx, "val")
        return outputs

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if self.logger is not None and getattr(self.logger, "experiment") is not None and self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            metrics = self._process_metrics(self.model.metrics(**outputs))
            self.logger.experiment.log(dict(step=self.trainer.global_step, **prefix_keys(metrics, "train")))

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        metrics = self._process_metrics(self.model.metrics(**outputs))
        self.logger.experiment.log(dict(step=self.trainer.global_step, **prefix_keys(metrics, "val")))

    @torch.no_grad()
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.model.predict(**batch, **kwargs)

    @torch.no_grad()
    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.model.predict(**batch, **kwargs)

    @staticmethod
    def _process_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for k, v in metrics.items():
            if k.endswith("hist"):
                v = wandb.Histogram(np_histogram=v)
            results[k] = v
        return results

    def _reset_cache(self):
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []
