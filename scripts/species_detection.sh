#!/bin/bash

# last run on 22.05.2026
# commit ref: c1d3a34bb724ce9610420923469a064fec0bfd8f

GPU_IDS=0
RESULTS_DIR=/mnt/data0/kag25/experiments/v4/species_detectors/
CKPT_DIR=/mnt/data0/kag25/models/v4/species_detectors/
STORAGE=/mnt/data0/kag25/data

wandb offline

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=lumpy-gibson scope=SO_UK seed=8 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=slow-partner scope=SO_UK seed=16 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=unique-tiger scope=SO_UK seed=24 trainer.max_epochs=750 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=lumpy-gibson scope=SO_EC seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=slow-partner scope=SO_EC seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=unique-tiger scope=SO_EC seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=jumpy-engine scope=RFCX_bird seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.01 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=quaint-pilot scope=RFCX_bird seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.01 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=numb-chef scope=RFCX_bird seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.01 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=jumpy-engine scope=RFCX_frog seed=8 trainer.max_epochs=600 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=quaint-pilot scope=RFCX_frog seed=16 trainer.max_epochs=600 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=numb-chef scope=RFCX_frog seed=24 trainer.max_epochs=600 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=just-drum scope=SO_UK seed=8 trainer.max_epochs=350 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=dynamic-malta scope=SO_UK seed=16 trainer.max_epochs=350  model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=daring-system scope=SO_UK seed=24 trainer.max_epochs=350 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.001 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=just-drum scope=SO_EC seed=8 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=dynamic-malta scope=SO_EC seed=16 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=daring-system scope=SO_EC seed=24 trainer.max_epochs=1950 model.clf_learning_rate=0.03 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=earthy-virgo scope=RFCX_bird seed=8 trainer.max_epochs=2000 model.clf_learning_rate=0.05 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=part-armor scope=RFCX_bird seed=16 trainer.max_epochs=2000 model.clf_learning_rate=0.05 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=secluded-montana scope=RFCX_bird seed=24 trainer.max_epochs=2000 model.clf_learning_rate=0.05 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=earthy-virgo scope=RFCX_frog seed=8 trainer.max_epochs=1900 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=part-armor scope=RFCX_frog seed=16 trainer.max_epochs=1900 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py train +experiment=mil_species_detector paths.data_dir=$STORAGE paths.results_dir=$RESULTS_DIR paths.checkpoint_dir=$CKPT_DIR model_name=secluded-montana scope=RFCX_frog seed=24 trainer.max_epochs=1900 model.clf_learning_rate=0.1 model.gamma_clf=0.001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
