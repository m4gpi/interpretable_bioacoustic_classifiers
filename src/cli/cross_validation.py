import abc
import attrs
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

def cross_validation(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log.info(f"Instantiating hyperparams <{cfg.hyperparams._target_}>")
    hyperparams = hydra.utils.instantiate(cfg.hyperparams)

    log.info(f"Instantiating cross validator <{cfg.cross_validator._target_}>")
    cross_validator = hydra.utils.instantiate(cfg.cross_validator, hyperparams=hyperparams)

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data, seed=cfg.get("seed"))

    folds = cross_validator.setup(data_module)

    results = []
    runs = [(fold, combination) for combination in hyperparams.combinations() for fold in folds]
    log.info(f"Starting cross-validation with {len(runs)} independent runs")
    for fold, combination in runs:
        result = cross_validator.run(
            model_config=cfg.get("model"),
            trainer_config=cfg.get("trainer"),
            callback_config=cfg.get("callbacks"),
            logger_config=cfg.get("logger"),
            **fold,
            hyperparams=combination,
            seed=cfg.get("seed"),
        )
        results.append(result)

    log.info(f"Cross-validation complete")

    if cfg.get("evaluator"):
        log.info(f"Instantiating evaluator <{cfg.evaluator._target_}>")
        evaluator = hydra.utils.instantiate(cfg.evaluator)

        log.info(f"Evaluating results")
        evaluator.run(results)

@hydra.main(version_base="1.3", config_path="../../config", config_name="cross_validation.yaml")
def main(cfg: DictConfig):
    cross_validation(cfg)

if __name__ == "__main__":
    main()

