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
from src.cli.utils import filter_kwargs_for_callable, mnemonic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("mnemonic", lambda: mnemonic(os.urandom(8).hex()))

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    OmegaConf.update(cfg, "run_id", os.urandom(16).hex(), force_add=True)
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    log.info(json.dumps(raw_config, indent=1))
    # results_dir = pathlib.Path(cfg.get("paths").get("results_dir")).expanduser()
    # (results_dir / "config").mkdir(parents=True, exist_ok=True)
    # OmegaConf.save(raw_config, results_dir / "config" / f"{cfg.get('run_id')}.yaml")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating transforms...")
    transforms: List[L.Callback] = instantiate_transforms(cfg.get("transforms"))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data, transforms=transforms)
    data_module.setup(stage="fit")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_cls = hydra.utils.get_class(cfg.model._target_)
    filtered_params = filter_kwargs_for_callable(model_cls.__init__, data_module.data.model_params)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, **filtered_params)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        for logger in loggers:
            logger.log_hyperparams(raw_config)

    try:
        model.run(
            trainer=trainer,
            config=cfg,
            data_module=data_module
        )
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
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
