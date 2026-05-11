import hydra
import json
import lightning as L
import logging
import os
import pathlib
import rootutils
import torch
import wandb
import yaml

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from typing import Any, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.cli.utils import filter_kwargs_for_callable, mnemonic, load_yaml

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("mnemonic", lambda: mnemonic(os.urandom(8).hex()))
OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("sub", lambda x, y: int(x) - int(y))
OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
OmegaConf.register_new_resolver("div", lambda x, y: int(x) // int(y))
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("pow", lambda x, y: int(x) ** int(y))
OmegaConf.register_new_resolver("yaml_load", load_yaml)

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("run_id") is None:
        OmegaConf.update(cfg, "run_id", mnemonic(os.urandom(16).hex()), force_add=True)
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    log.info(json.dumps(raw_config, indent=1))

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.setup(stage="fit")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_cls = hydra.utils.get_class(cfg.model._target_)
    filtered_params = filter_kwargs_for_callable(model_cls.__init__, data_module.data.model_params)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False, **filtered_params)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    if cfg.trainer.devices == 1 and cfg.num_gpus > 1:
        gpu_num = HydraConfig.get().job.num % cfg.num_gpus
        available_devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        log.info(f"Using GPU NUM: {gpu_num}, GPU ID: {available_devices[gpu_num]} from available devices {available_devices}")
        cfg.trainer.devices = [gpu_num]
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
