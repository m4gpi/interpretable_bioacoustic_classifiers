#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline uv run main.py train +experiment=eai_mil_species_detector_cv_sweep paths=vili hydra.launcher.n_jobs=16
