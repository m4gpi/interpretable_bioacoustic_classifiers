import attrs
import hydra
import json
import logging
import os
import pathlib
import rootutils
import torch
import warnings
import wandb

from omegaconf import OmegaConf, DictConfig
from typing import Any, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.cli.utils import filter_kwargs_for_callable, mnemonic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("mnemonic", lambda: mnemonic(os.urandom(8).hex()))

def evaluate(cfg):
    if cfg.get("run_id") is None:
        OmegaConf.update(cfg, "run_id", mnemonic(os.urandom(16).hex()), force_add=True)
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    log.info(json.dumps(raw_config, indent=1))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_cls = hydra.utils.get_class(cfg.model._target_)
    if (ckpt_path := cfg.get("ckpt_path")):
        model = model_cls.load_from_checkpoint(ckpt_path, map_location=torch.device("cuda"))
    else:
        filtered_params = filter_kwargs_for_callable(model_cls.__init__, data_module.data.model_params)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False, **filtered_params)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    if cfg.get("trainer"):
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    else:
        trainer = None

    if loggers:
        for logger in loggers:
            logger.log_hyperparams({
                "data": dict(cfg.data),
                "model": dict(cfg.model),
                "logger": dict(cfg.logger),
                "trainer": dict(cfg.trainer),
            })

    try:
        evaluator = hydra.utils.instantiate(cfg.evaluator)
        evaluator(
            trainer=trainer,
            model=model,
            data_module=data_module,
            config=cfg
        )
    except Exception as e:
        log.error(e)
    finally:
        if wandb.run is not None:
            wandb.finish()

@hydra.main(
    version_base="1.3",
    config_path=str(rootutils.find_root() / "config"),
    config_name="eval.yaml"
)
def main(cfg: DictConfig):
    evaluate(cfg)

if __name__ == "__main__":
    main()
