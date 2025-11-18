#!/bin/bash

set -a
source .env
set +a
mkdir -p ./logs

sbatch --cpus-per-task=$CORES --mem-per-cpu="${MEM_PER_CPU}G" ./slurm/jobs/pipeline.job
