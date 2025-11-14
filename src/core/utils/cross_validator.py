import abc
import attrs
import hydra
import lightning as L
import logging
import torch

from omegaconf import DictConfig
from typing import Any, Dict, List, Tuple

from src.core.utils.hyperparams import Hyperparams
from src.cli.utils.instantiators import instantiate_loggers, instantiate_callbacks

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@attrs.define()
class CrossValidator(abc.ABC):
    hyperparams: Hyperparams = attrs.field()
    num_folds: int = attrs.field(default=7, validator=attrs.validators.instance_of(int))

    datasets: List = attrs.field(init=False)

    def setup(self, data_module: L.LightningDataModule) -> List[Any]:
        log.info(f"Setting up data module for cross-validation")
        return data_module.cross_validation_setup(num_folds=self.num_folds)

    def run(
        self,
        fold_id: int,
        train_dataloader_params: Dict[str, Any],
        val_dataloader_params: Dict[str, Any],
        test_dataloader_params: Dict[str, Any],
        model_params: Dict[str, Any],
        model_config: DictConfig,
        trainer_config: DictConfig,
        callback_config: DictConfig,
        logger_config: DictConfig,
        hyperparams: Dict[str, Any],
        seed: int,
    ) -> Any:
        L.seed_everything(seed, workers=True)

        log.info(f"Instantiating dataloaders")
        train_dataloader = torch.utils.data.DataLoader(**train_dataloader_params)
        val_dataloader = torch.utils.data.DataLoader(**val_dataloader_params)
        test_dataloader = torch.utils.data.DataLoader(**test_dataloader_params)

        log.info(f"Instantiating model <{model_config._target_}> with {model_params} and overrides {hyperparams}")
        model = hydra.utils.instantiate(model_config, **model_params, **hyperparams)

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(callback_config)

        log.info("Instantiating loggers...")
        loggers: List[Logger] = instantiate_loggers(logger_config)

        log.info(f"Instantiating trainer <{trainer_config._target_}>")
        trainer: L.Trainer = hydra.utils.instantiate(trainer_config, callbacks=callbacks, logger=loggers)

        if loggers:
            for logger in loggers:
                logger.log_hyperparams({
                    # TODO: figure out how to do this nicely
                    # "data": dict(cfg.data),
                    "model": dict(**model_config, **model_params, **hyperparams),
                    "logger": dict(logger_config),
                    "trainer": dict(trainer_config),
                })

        results = model.run(
            trainer,
            config=dict(train=True, test=True),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
        )
        return results

