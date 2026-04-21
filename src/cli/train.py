import functools
import hydra
import json
import lightning as L
import logging
import os
import pathlib
import rootutils
import torch
import wandb

from omegaconf import DictConfig, OmegaConf
from typing import Any, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers, instantiate_transforms
from src.cli.utils import filter_kwargs_for_callable

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    OmegaConf.update(config, "run_id", os.urandom(16).hex(), force_add=True)

    if config.get("seed"):
        L.seed_everything(config.seed, workers=True)

    log.info(f"Instantiating datamodule <{config.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(config.data)
    data_module.setup()

    log.info(f"Instantiating model <{config.algorithm._target_}>")
    algorithm: L.LightningModule = hydra.utils.instantiate(config.algorithm)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = instantiate_callbacks(config.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(config.get("logger"))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        for logger in loggers:
            logger.log_hyperparams({
                "algorithm": dict(config.algorithm),
                "data": dict(config.data),
                "logger": dict(config.logger),
                "trainer": dict(config.trainer),
            })

    try:
        algorithm.run(trainer=trainer, data_module=data_module, config=config)
    except Exception as e:
        log.error(e)
    finally:
        if wandb.run is not None:
            wandb.finish()

@hydra.main(
    version_base="1.3",
    config_path=str(rootutils.find_root() / "config"),
    config_name="train.yaml"
)
def main(config: DictConfig):
    train(config)

if __name__ == "__main__":
    main()
