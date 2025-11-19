import os
import hydra
import lightning as L
import logging
import pathlib
import rootutils
import torch
import json
import wandb

from omegaconf import DictConfig, OmegaConf
from typing import Any, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers

OmegaConf.register_new_resolver("gen_run_id", lambda: os.urandom(6).hex())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    run_id = cfg.get("run_id") or os.urandom(6).hex()
    results_dir = pathlib.Path(cfg.get("results_dir")).expanduser() / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = str(results_dir)
    log.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=1))
    OmegaConf.save(cfg, results_dir / "config.yaml")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.setup(stage="fit")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, **data_module.data.model_params)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"), id=run_id)

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

    model.run(trainer, cfg, data_module=data_module)

    if wandb.run is not None:
        wandb.finish()

@hydra.main(version_base="1.3", config_path="../../config", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
