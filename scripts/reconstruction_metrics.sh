#!/bin/bash

GPU_IDS=0
MODEL_DIR=/mnt/data0/kag25/models/v4
RESULTS_DIR=/mnt/data0/kag25/experients/v4/vae_evaluation
CKPT_STEP="step\=180000.ckpt"

wandb offline

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae run_id=lumpy-gibson seed=8 paths=vili data=sounding_out_chorus ckpt_path=$MODEL_DIR/vae/lumpy-gibson/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae run_id=slow-partner seed=16 paths=vili data=sounding_out_chorus ckpt_path=$MODEL_DIR/vae/slow-partner/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae run_id=unique-tiger seed=24 paths=vili data=sounding_out_chorus ckpt_path=$MODEL_DIR/vae/unique-tiger/$CKPT_STEP paths.results_dir=$RESULTS_DIR

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae run_id=jumpy-engine seed=8 paths=vili data=rainforest_connection ckpt_path=$MODEL_DIR/vae/jumpy-engine/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae run_id=quaint-pilot seed=16 paths=vili data=rainforest_connection ckpt_path=$MODEL_DIR/vae/quaint-pilot/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae run_id=numb-chef seed=24 paths=vili data=rainforest_connection ckpt_path=$MODEL_DIR/vae/numb-chef/$CKPT_STEP paths.results_dir=$RESULTS_DIR

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae run_id=just-drum seed=8 paths=vili data=sounding_out_chorus ckpt_path=$MODEL_DIR/sivae/just-drum/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae run_id=daring-system seed=16 paths=vili data=sounding_out_chorus ckpt_path=$MODEL_DIR/sivae/daring-system/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae run_id=dymamic-malta seed=24 paths=vili data=sounding_out_chorus ckpt_path=$MODEL_DIR/sivae/dymamic-malta/$CKPT_STEP paths.results_dir=$RESULTS_DIR

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae run_id=earthy-virgo seed=8 paths=vili data=rainforest_connection ckpt_path=$MODEL_DIR/sivae/earthy-virgo/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae run_id=part-armor seed=16 paths=vili data=rainforest_connection ckpt_path=$MODEL_DIR/sivae/part-armor/$CKPT_STEP paths.results_dir=$RESULTS_DIR
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae run_id=secluded-montana seed=24 paths=vili data=rainforest_connection ckpt_path=$MODEL_DIR/sivae/secluded-montana/$CKPT_STEP paths.results_dir=$RESULTS_DIR
