#!/bin/bash

GPU_IDS=0
VERSION=v4
RESULTS_DIR=$ROOT/experiments/$VERSION/species_detectors/
CKPT_DIR=$ROOT/models/$VERSION/species_detectors/

wandb offline

# ----------------------------------------------------------------------------------------- #
# ----------------------------------- VAE representations --------------------------------- #
# ----------------------------------------------------------------------------------------- #

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=lumpy-gibson scope=SO_UK seed=8 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=slow-partner scope=SO_UK seed=16 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=unique-tiger scope=SO_UK seed=24 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=lumpy-gibson scope=SO_EC seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=slow-partner scope=SO_EC seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=unique-tiger scope=SO_EC seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=jumpy-engine scope=RFCX_bird seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.01 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=quaint-pilot scope=RFCX_bird seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.01 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=numb-chef scope=RFCX_bird seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.01 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=jumpy-engine scope=RFCX_frog seed=8 trainer.max_epochs=600 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=quaint-pilot scope=RFCX_frog seed=16 trainer.max_epochs=600 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=numb-chef scope=RFCX_frog seed=24 trainer.max_epochs=600 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

# ------------------------------------------------------------------------------------------- #
# ----------------------------------- SIVAE representations --------------------------------- #
# ------------------------------------------------------------------------------------------- #

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=just-drum scope=SO_UK seed=8 trainer.max_epochs=350 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=dynamic-malta scope=SO_UK seed=16 trainer.max_epochs=350  model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=daring-system scope=SO_UK seed=24 trainer.max_epochs=350 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=just-drum scope=SO_EC seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=dynamic-malta scope=SO_EC seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=daring-system scope=SO_EC seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=earthy-virgo scope=RFCX_bird seed=8 trainer.max_epochs=2000 model.clf_learning_rate=0.05 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=part-armor scope=RFCX_bird seed=16 trainer.max_epochs=2000 model.clf_learning_rate=0.05 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=secluded-montana scope=RFCX_bird seed=24 trainer.max_epochs=2000 model.clf_learning_rate=0.05 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=earthy-virgo scope=RFCX_frog seed=8 trainer.max_epochs=1900 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=part-armor scope=RFCX_frog seed=16 trainer.max_epochs=1900 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=secluded-montana scope=RFCX_frog seed=24 trainer.max_epochs=1900 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

# --------------------------------------------------------------------------------------------- #
# ----------------------------------- BirdNET representations --------------------------------- #
# --------------------------------------------------------------------------------------------- #

RESULTS_DIR=$ROOT/experiments/$VERSION/birdnet_species_detectors/
CKPT_DIR=$ROOT/models/$VERSION/birdnet_species_detectors/

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_8 scope=SO_UK seed=8 trainer.max_epochs=300 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_16 scope=SO_UK seed=16 trainer.max_epochs=300 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_24 scope=SO_UK seed=24 trainer.max_epochs=300 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_8 scope=SO_EC seed=8 trainer.max_epochs=1000 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_16 scope=SO_EC seed=16 trainer.max_epochs=1000 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_24 scope=SO_EC seed=24 trainer.max_epochs=1000 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_8 scope=RFCX_bird seed=8 trainer.max_epochs=600 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_16 scope=RFCX_bird seed=16 trainer.max_epochs=600 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_24 scope=RFCX_bird seed=24 trainer.max_epochs=600 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.001 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_8 scope=RFCX_frog seed=8 trainer.max_epochs=950 model.clf_learning_rate=0.01 model.gamma_clf=0.0001 model.attn_learning_rate=0.001 model.gamma_attn=0.01
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_16 scope=RFCX_frog seed=16 trainer.max_epochs=950 model.clf_learning_rate=0.01 model.gamma_clf=0.0001 model.attn_learning_rate=0.001 model.gamma_attn=0.01
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=birdnet_mil_species_detector paths=$DEVICE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=birdnet_24 scope=RFCX_frog seed=24 trainer.max_epochs=950 model.clf_learning_rate=0.01 model.gamma_clf=0.0001 model.attn_learning_rate=0.001 model.gamma_attn=0.01
