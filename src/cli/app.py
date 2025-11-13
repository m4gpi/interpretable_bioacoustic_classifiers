import hydra
import logging
import rootutils
import torch

from omegaconf import DictConfig
from typing import Any, List, Dict, Tuple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../../config", config_name="app.yaml")
def main(cfg: DictConfig):
    if not len(cfg.app):
        log.info(f"No app specified, select one")
        return

    log.info(f"Instantiating app <{cfg.app._target_}>")
    app = hydra.utils.instantiate(cfg.app)
    log.info(f"Starting app <{cfg.app._target_}>")
    app.setup(cfg).run(debug=True)

if __name__ == '__main__':
    main()
