import abc
import attrs
import copy
import hydra
import itertools
import lightning as L
import logging
import multiprocessing
import rootutils
import torch

from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from typing import Any, Callable, Iterable, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)
multiprocessing.set_start_method("spawn", force=True)

def process_concurrent(func: Callable, pool_args: List[Any], num_workers: int) -> Iterable:
    if num_workers > 0:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap(func, pool_args):
                yield result
    else:
        for result in map(func, pool_args):
            yield result

def worker_run(kwargs: Dict[str, Any]):
    return run(**kwargs)

def run(
    fold_id: int,
    model_params: Dict[str, Any],
    dataset_config: Dict[str, Any],
    train_dataloader_params: Dict[str, Any],
    val_dataloader_params: Dict[str, Any],
    test_dataloader_params: Dict[str, Any],
    config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    device: int,
) -> Any:
    seed = config.get("seed")
    L.seed_everything(seed, workers=True)

    log.info(f"Instantiating dataloaders")
    dataset_config = config.get("data")
    dataset_config.update(dataset_config)
    train_dataloader = torch.utils.data.DataLoader(**train_dataloader_params)
    val_dataloader = torch.utils.data.DataLoader(**val_dataloader_params)
    test_dataloader = torch.utils.data.DataLoader(**test_dataloader_params)

    model_config = config.get("model")
    model_config.update(model_params)
    model_config.update(hyperparams)
    log.info(f"Instantiating model <{model_config['_target_']}> with {model_params} and overrides {hyperparams}")
    model = hydra.utils.instantiate(model_config)

    callback_config = config.get("callbacks")
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(callback_config)

    logger_config = config.get("logger")
    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(logger_config)

    trainer_config = config.get("trainer")
    log.info(f"Instantiating trainer <{trainer_config['_target_']}>")
    trainer: L.Trainer = hydra.utils.instantiate(trainer_config, callbacks=callbacks, logger=loggers, devices=[device])

    run_dict = {
        "data": dataset_config,
        "model": model_config,
        "logger": logger_config,
        "trainer": trainer_config,
    }

    if loggers:
        log.info("Logging hyperparameters")
        for logger in loggers:
            logger.log_hyperparams(run_dict)

    results = model.run(
        trainer,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
    )
    return results


def cross_validation(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    num_workers = cfg.get("num_workers", 0)
    num_folds = cfg.get("num_folds")
    devices = cfg.get("devices", [])
    assert num_folds > 3, "'num_folds' must be 3 or more"
    assert len(devices), "GPU 'devices' are required in the cross_validation configuration"

    log.info(f"Instantiating hyperparams <{cfg.hyperparams._target_}>")
    hyperparams = hydra.utils.instantiate(cfg.hyperparams)

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data, seed=cfg.get("seed"))

    log.info(f"Setting up data module for cross-validation")
    folds = data_module.cross_validation_setup(num_folds=num_folds)

    combinations = hyperparams.combinations()
    assert len(combinations), "No hyperparameters declared to search"

    config = OmegaConf.to_container(cfg, resolve=True)
    runs = []
    for i, (fold, combination) in enumerate([(fold, combination) for combination in combinations for fold in folds]):
        runs.append(dict(**fold, hyperparams=combination, config=copy.deepcopy(config), device=devices[i % len(devices)]))

    results = []
    log.info(f"Starting {num_folds}-folds cross-validation with {len(runs)} independent runs using {num_workers} workers across GPUs: {devices}")
    for result in process_concurrent(worker_run, runs, num_workers=num_workers):
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

