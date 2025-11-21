import os
from typing import List
from joblib import Parallel, delayed

from hydra.core.launcher import Launcher
from hydra.core.utils import JobRuntime, JobReturn
from hydra.core.singleton import Singleton
from hydra.plugins.launcher import Launcher as HydraLauncher
from hydra.types import HydraContext, TaskFunction


class JoblibGPULauncher(HydraLauncher):
    def __init__(self, gpus: List[int]):
        """
        gpus: list of available GPU IDs, e.g. [0,1,2,3]
        """
        self.gpus = gpus

    def setup(self, hydra_context: HydraContext, task_function: TaskFunction) -> None:
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(self, job_overrides: List[List[str]], initial_job_idx: int) -> List[JobReturn]:
        """
        Launches each Hydra job using Joblib in parallel.
        """

        num_jobs = len(job_overrides)

        def run_single_job(job_idx: int):
            # --- Choose GPU for this job ---
            gpu_id = self.gpus[job_idx % len(self.gpus)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            print(f"[Launcher] Job {job_idx} assigned to GPU {gpu_id}")

            # --- Run the Hydra job ---
            ret = self.hydra_context.job(
                overrides=job_overrides[job_idx],
                job_num=job_idx + initial_job_idx,
            )

            return ret

        # Run jobs with Joblib
        results = Parallel(n_jobs=len(self.gpus))(
            delayed(run_single_job)(i) for i in range(num_jobs)
        )

        return results
