import os
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf

class RunIDCallback(Callback):
    def on_run_start(self, config, **kwargs):
        run_id = os.urandom(6).hex()
        OmegaConf.update(config, "run_id", run_id, force_add=True)

    def on_job_start(self, config, **kwargs):
        run_id = os.urandom(6).hex()
        OmegaConf.update(config, "run_id", run_id, force_add=True)

    def on_multirun_job_start(self, config, **kwargs):
        run_id = os.urandom(6).hex()
        OmegaConf.update(config, "run_id", run_id, force_add=True)
