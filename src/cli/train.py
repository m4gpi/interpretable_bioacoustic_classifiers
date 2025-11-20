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

from src.cli.utils import run_id
from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    OmegaConf.update(cfg, "run_id", os.urandom(6).hex(), force_add=True)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    results_dir = pathlib.Path(cfg.get("results_dir")).expanduser()
    (results_dir / "config").mkdir(parents=True, exist_ok=True)
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(raw_config, results_dir / "config" / f"{cfg.get('run_id')}.yaml")
    log.info(json.dumps(raw_config, indent=1))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.setup(stage="fit")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, **data_module.data.model_params)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        for logger in loggers:
            logger.log_hyperparams({
                "name": cfg.get("name"),
                "data": dict(cfg.data),
                "model": dict(cfg.model),
                "logger": dict(cfg.logger),
                "trainer": dict(cfg.trainer),
            })

    try:
        model.run(trainer, cfg, data_module=data_module)
    except Exception as e:
        log.error(e)
    finally:
        if wandb.run is not None:
            wandb.finish()

@hydra.main(version_base="1.3", config_path="../../config", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
