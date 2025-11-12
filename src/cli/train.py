import hydra
import lightning as L
import logging
import rootutils
import torch

from omegaconf import DictConfig
from typing import Any, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "data_module": data_module,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        logger.log_hyperparams(object_dict)

    model.run(trainer, data_module, cfg)

@hydra.main(version_base="1.3", config_path="../../config", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
