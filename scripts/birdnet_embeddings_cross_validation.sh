#!/bin/bash

wandb offline

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run main.py train +experiment=birdnet_mil_species_detector_cv_sweep paths=vili hydra.launcher.n_jobs=16
