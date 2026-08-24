#!/bin/bash

GPU_IDS=0
MODEL_DIR=/mnt/data0/kag25/models/v4
DATA_PATH=/mnt/data0/kag25/data/soundscape_vae_embeddings
CKPT_STEP="step\=180000.ckpt"

# VAE
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=8 run_id=lumpy-gibson paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/lumpy-gibson/$CKPT_STEP data.scope=UK paths.results_dir=$DATA_PATH/lumpy-gibson/SO_UK
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=16 run_id=slow-partner paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/slow-partner/$CKPT_STEP data.scope=UK paths.results_dir=$DATA_PATH/slow-partner/SO_UK
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=24 run_id=unique-tiger paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/unique-tiger/$CKPT_STEP data.scope=UK paths.results_dir=$DATA_PATH/unique-tiger/SO_UK
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=8 run_id=lumpy-gibson paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/lumpy-gibson/$CKPT_STEP data.scope=EC paths.results_dir=$DATA_PATH/lumpy-gibson/SO_EC
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=16 run_id=slow-partner paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/slow-partner/$CKPT_STEP data.scope=EC paths.results_dir=$DATA_PATH/slow-partner/SO_EC
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=24 run_id=unique-tiger paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/unique-tiger/$CKPT_STEP data.scope=EC paths.results_dir=$DATA_PATH/unique-tiger/SO_EC

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=8 run_id=jumpy-engine paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/jumpy-engine/$CKPT_STEP data.scope=bird paths.results_dir=$DATA_PATH/jumpy-engine/RFCX_bird
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=16 run_id=quaint-pilot paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/quaint-pilot/$CKPT_STEP data.scope=bird paths.results_dir=$DATA_PATH/quaint-pilot/RFCX_bird
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=24 run_id=numb-chef paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/numb-chef/$CKPT_STEP data.scope=bird paths.results_dir=$DATA_PATH/numb-chef/RFCX_bird
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=8 run_id=jumpy-engine paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/jumpy-engine/$CKPT_STEP data.scope=frog paths.results_dir=$DATA_PATH/jumpy-engine/RFCX_frog
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=16 run_id=quaint-pilot paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/quaint-pilot/$CKPT_STEP data.scope=frog paths.results_dir=$DATA_PATH/quaint-pilot/RFCX_frog
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=vae seed=24 run_id=numb-chef paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/vae/numb-chef/$CKPT_STEP data.scope=frog paths.results_dir=$DATA_PATH/numb-chef/RFCX_frog

# SIVAE
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=8 run_id=just-drum paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/just-drum/$CKPT_STEP data.scope=UK paths.results_dir=$DATA_PATH/just-drum/SO_UK
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=16 run_id=daring-system paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/daring-system/$CKPT_STEP data.scope=UK paths.results_dir=$DATA_PATH/daring-system/SO_UK
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=24 run_id=dynamic-malta paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/dynamic-malta/$CKPT_STEP data.scope=UK paths.results_dir=$DATA_PATH/dynamic-malta/SO_UK
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=8 run_id=just-drum paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/just-drum/$CKPT_STEP data.scope=EC paths.results_dir=$DATA_PATH/just-drum/SO_EC
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=16 run_id=daring-system paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/daring-system/$CKPT_STEP data.scope=EC paths.results_dir=$DATA_PATH/daring-system/SO_EC
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=24 run_id=dynamic-malta paths=vili data=sounding_out_chorus evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/dynamic-malta/$CKPT_STEP data.scope=EC paths.results_dir=$DATA_PATH/dynamic-malta/SO_EC

CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=8 run_id=earthy-virgo paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/earthy-virgo/$CKPT_STEP data.scope=bird paths.results_dir=$DATA_PATH/earthy-virgo/RFCX_bird
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=16 run_id=part-armor paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/part-armor/$CKPT_STEP data.scope=bird paths.results_dir=$DATA_PATH/part-armor/RFCX_bird
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=24 run_id=secluded-montana paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/secluded-montana/$CKPT_STEP data.scope=bird paths.results_dir=$DATA_PATH/secluded-montana/RFCX_bird
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=8 run_id=earthy-virgo paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/earthy-virgo/$CKPT_STEP data.scope=frog paths.results_dir=$DATA_PATH/earthy-virgo/RFCX_frog
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=16 run_id=part-armor paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/part-armor/$CKPT_STEP data.scope=frog paths.results_dir=$DATA_PATH/part-armor/RFCX_frog
CUDA_VISIBLE_DEVICES=$GPU_IDS uv run main.py eval +experiment=sivae seed=24 run_id=secluded-montana paths=vili data=rainforest_connection evaluator=lightning_predict ckpt_path=$MODEL_DIR/sivae/secluded-montana/$CKPT_STEP data.scope=frog paths.results_dir=$DATA_PATH/secluded-montana/RFCX_frog
