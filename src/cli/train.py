import hydra
import lightning as L
import logging
import rootutils
import torch

from omegaconf import DictConfig
from typing import Any, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating checkpointer...")
    checkpointer = hydra.utils.instantiate(cfg.get("checkpoint"))

    log.info("Instantiating logger...")
    logger: List[Logger] = hydra.utils.instantiate(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[checkpointer], logger=logger)

    object_dict = {
        "cfg": cfg,
        "data_module": data_module,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("predict"):
        log.info("Starting prediction!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for prediction...")
            ckpt_path = None
        predictions = trainer.predict(model=model, datamodule=data_module, ckpt_path=ckpt_path, return_predictions=True)
        log.info(f"Prediction ckpt path: {ckpt_path}")
        if hasattr(model, "evaluate") and callable(model.evaluate):
            model.evaluate(predictions)

@hydra.main(version_base="1.3", config_path="../../config", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
