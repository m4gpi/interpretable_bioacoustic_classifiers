#!/bin/bash

# last run on 22.05.2026
# commit ref: c1d3a34bb724ce9610420923469a064fec0bfd8f

RESULTS_DIR=/its/home/kag25/experiments/v1/species_detectors_l1/
CKPT_DIR=/its/home/kag25/models/v1/species_detectors_l1/
STORAGE=/its/home/kag25/data
GPU_IDS=1

# |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# | 14237 | vae     | SO_UK     |     399 |                0.1  |     0.01 |               0.0005 |               0.001 | 0.742926 | 0.259842  | 1.00394  |
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=silly-byte scope=SO_UK seed=8 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=meek-zebra scope=SO_UK seed=16 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=rude-money scope=SO_UK seed=24 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001 model.regularisation_mode=L1

# |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# | 13567 | vae     | SO_EC     |    1749 |                0.1  |     0.01 |               0.001  |               0.001 | 0.809463 | 0.197051  | 1.01039  |
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=silly-byte scope=SO_EC seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=meek-zebra scope=SO_EC seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=rude-money scope=SO_EC seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001 model.regularisation_mode=L1

# # |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# # |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# # |  9403 | vae     | RFCX_bird |    1549 |                0.01 |     0.01 |               0.001  |               0.001 | 0.710596 | 0.0750152 | 0.785612 |
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=tusked-chief scope=RFCX_bird seed=8 trainer.max_epochs=1550 model.clf_learning_rate=0.01 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=ultimate-story scope=RFCX_bird seed=16 trainer.max_epochs=1550 model.clf_learning_rate=0.01 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=misty-lecture scope=RFCX_bird seed=24 trainer.max_epochs=1550 model.clf_learning_rate=0.01 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001

# # |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# # |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# # | 10004 | vae     | RFCX_frog |      99 |                0.05 |     0.01 |               0.001  |               0.01  | 0.86909  | 0.19846   | 1.06755  |
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=tusked-chief scope=RFCX_frog seed=8 trainer.max_epochs=100 model.clf_learning_rate=0.05 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.01
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=ultimate-story scope=RFCX_frog seed=16 trainer.max_epochs=100 model.clf_learning_rate=0.05 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.01
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=misty-lecture scope=RFCX_frog seed=24 trainer.max_epochs=100 model.clf_learning_rate=0.05 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.01

# # |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# # |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# # |  6726 | sivae   | SO_UK     |     799 |                0.1  |     0.01 |               0.0005 |               0.01  | 0.765789 | 0.277352  | 1.04436  |
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=tan-ohio scope=SO_UK seed=8 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=small-peru scope=SO_UK seed=16 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=brave-vincent scope=SO_UK seed=24 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001 model.regularisation_mode=L1

# # |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# # |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# # |  5175 | sivae   | SO_EC     |    1249 |                0.1  |     0.01 |               0.001  |               0.001 | 0.832921 | 0.271675  | 1.10853  |
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=tan-ohio scope=SO_EC seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=small-peru scope=SO_EC seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001 model.regularisation_mode=L1
CUDA_VISIBLE_DEVICES=$GPU_IDS WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=brave-vincent scope=SO_EC seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.lamdba=0.001 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001 model.regularisation_mode=L1

# # |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# # |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# # |  1668 | sivae   | RFCX_bird |    1699 |                0.1  |     0.01 |               0.0005 |               0.001 | 0.781016 | 0.152203  | 0.933219 |
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=uncanny-burma scope=RFCX_bird seed=8 trainer.max_epochs=1700 model.clf_learning_rate=0.1 model.lamdba=0.01 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=detailed-ticket scope=RFCX_bird seed=16 trainer.max_epochs=1700 model.clf_learning_rate=0.1 model.lamdba=0.01 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=mossy-andrea scope=RFCX_bird seed=24 trainer.max_epochs=1700 model.clf_learning_rate=0.1 model.lamdba=0.01 model.attn_learning_rate=0.0005 model.attn_weight_decay=0.001

# # |       | model   | scope     |   epoch |   clf_learning_rate |   lamdba |   attn_learning_rate |   attn_weight_decay |    auROC |        AP |    score |
# # |------:|:--------|:----------|--------:|--------------------:|---------:|---------------------:|--------------------:|---------:|----------:|---------:|
# # |  2498 | sivae   | RFCX_frog |     549 |                0.01 |     0.01 |               0.001  |               0.001 | 0.871875 | 0.246894  | 1.11877  |
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=uncanny-burma scope=RFCX_frog seed=8 trainer.max_epochs=550 model.clf_learning_rate=0.01 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=detailed-ticket scope=RFCX_frog seed=16 trainer.max_epochs=550 model.clf_learning_rate=0.01 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001
# WANDB_MODE=offline uv run main.py train +experiment=species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=mossy-andrea scope=RFCX_frog seed=24 trainer.max_epochs=550 model.clf_learning_rate=0.01 model.lamdba=0.01 model.attn_learning_rate=0.001 model.attn_weight_decay=0.001
