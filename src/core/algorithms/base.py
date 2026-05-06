import hydra
import lightning as L
import torch
import logging
import omegaconf
import wandb

from matplotlib import pyplot as plt
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
        plt.switch_backend('agg')
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
        config_path = pathlib.Path(trainer.checkpoint_callback.dirpath) / "config.yaml"
        log.info(f"Saving model configuration to {config_path}")
        omegaconf.OmegaConf.save(config, config_path)
        # running test
        log.info(f"Testing <{config.algorithm.get('_target_')}> on <{config.data.get('_target_')}>")
        trainer.test(self, dataloaders=data_module.predict_dataloader())

    def forward(self, batch: Batch, batch_idx: int, stage: str, **kwargs: Any) -> Dict[str, torch.Tensor]:
        step_outputs = self.model(**batch, t=self.trainer.global_step, **kwargs)
        loss_outputs = self.model.loss(**step_outputs)
        step_outputs = detach_values(step_outputs)
        metrics = self._process_metrics(self.model.metrics(**step_outputs))
        self.log_dict(prefix_keys(loss_outputs, stage), batch_size=batch.x.size(0), prog_bar=True, logger=False)
        if self.logger is not None and getattr(self.logger, "experiment") is not None:
            self.logger.experiment.log(dict(global_step=self.trainer.global_step, **prefix_keys(metrics | loss_outputs, stage)))
        return {**loss_outputs, **step_outputs}

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch, batch_idx, "train")
        self.training_step_outputs.append(outputs)
        return outputs

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch, batch_idx, "val")
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        self.training_step_outputs.clear()

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Batch, batch_idx: int, **kwargs: Any) -> None:
        if batch_idx < 4 and len(self.validation_step_outputs):
            step_outputs = self.validation_step_outputs[0]
            self.on_batch_end(step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_batch_end(self, step_outputs, stage: str, min_num_samples: int = 6):
        figures = self.model.tracking_figures(**step_outputs)
        for fig_name, fig in figures:
            if self.logger is not None and getattr(self.logger, "experiment") is not None:
                self.logger.experiment.log({f"{stage}/{fig_name}": wandb.Image(fig)})
            plt.close(fig)

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
