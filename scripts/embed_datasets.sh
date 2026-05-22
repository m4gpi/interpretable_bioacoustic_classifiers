#!/bin/bash

wandb offline
CKPT_PATH=/its/home/kag25/models
DATA_PATH=/its/home/kag25/data/soundscape_vae_embeddings

# VAE
uv run main.py eval +experiment=vae seed=8 run_id=silly-byte data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/silly-byte/step\=180000.ckpt" data.scope=UK paths.results_dir=$DATA_PATH/silly-byte/SO_UK
uv run main.py eval +experiment=vae seed=16 run_id=meek-zebra data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/meek-zebra/step\=180000.ckpt" data.scope=UK paths.results_dir=$DATA_PATH/meek-zebra/SO_UK
uv run main.py eval +experiment=vae seed=24 run_id=rude-money data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/rude-money/step\=180000.ckpt" data.scope=UK paths.results_dir=$DATA_PATH/rude-money/SO_UK
uv run main.py eval +experiment=vae seed=8 run_id=silly-byte data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/silly-byte/step\=180000.ckpt" data.scope=EC paths.results_dir=$DATA_PATH/silly-byte/SO_EC
uv run main.py eval +experiment=vae seed=16 run_id=meek-zebra data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/meek-zebra/step\=180000.ckpt" data.scope=EC paths.results_dir=$DATA_PATH/meek-zebra/SO_EC
uv run main.py eval +experiment=vae seed=24 run_id=rude-money data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/rude-money/step\=180000.ckpt" data.scope=EC paths.results_dir=$DATA_PATH/rude-money/SO_EC
uv run main.py eval +experiment=vae seed=8 run_id=tusked-chief data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/tusked-chief/step\=180000.ckpt" data.scope=bird paths.results_dir=$DATA_PATH/tusked-chief/RFCX_bird
uv run main.py eval +experiment=vae seed=16 run_id=ultimate-story data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/ultimate-story/step\=180000.ckpt" data.scope=bird paths.results_dir=$DATA_PATH/ultimate-story/RFCX_bird
uv run main.py eval +experiment=vae seed=24 run_id=misty-lecture data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/misty-lecture/step\=180000.ckpt" data.scope=bird paths.results_dir=$DATA_PATH/misty-lecture/RFCX_bird
uv run main.py eval +experiment=vae seed=8 run_id=tusked-chief data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/tusked-chief/step\=180000.ckpt" data.scope=frog paths.results_dir=$DATA_PATH/tusked-chief/RFCX_frog
uv run main.py eval +experiment=vae seed=16 run_id=ultimate-story data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/ultimate-story/step\=180000.ckpt" data.scope=frog paths.results_dir=$DATA_PATH/ultimate-story/RFCX_frog
uv run main.py eval +experiment=vae seed=24 run_id=misty-lecture data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/vae/misty-lecture/step\=180000.ckpt" data.scope=frog paths.results_dir=$DATA_PATH/misty-lecture/RFCX_frog

# SIVAE
uv run main.py eval +experiment=sivae seed=8 run_id=tan-ohio data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/tan-ohio/step\=180000.ckpt" data.scope=UK paths.results_dir=$DATA_PATH/tan-ohio/SO_UK
uv run main.py eval +experiment=sivae seed=16 run_id=brave-vincent data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/brave-vincent/step\=180000.ckpt" data.scope=UK paths.results_dir=$DATA_PATH/brave-vincent/SO_UK
uv run main.py eval +experiment=sivae seed=24 run_id=small-peru data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/small-peru/step\=180000.ckpt" data.scope=UK paths.results_dir=$DATA_PATH/small-peru/SO_UK
uv run main.py eval +experiment=sivae seed=8 run_id=tan-ohio data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/tan-ohio/step\=180000.ckpt" data.scope=EC paths.results_dir=$DATA_PATH/tan-ohio/SO_EC
uv run main.py eval +experiment=sivae seed=16 run_id=brave-vincent data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/brave-vincent/step\=180000.ckpt" data.scope=EC paths.results_dir=$DATA_PATH/brave-vincent/SO_EC
uv run main.py eval +experiment=sivae seed=24 run_id=small-peru data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/small-peru/step\=180000.ckpt" data.scope=EC paths.results_dir=$DATA_PATH/small-peru/SO_EC
uv run main.py eval +experiment=sivae seed=8 run_id=uncanny-burma data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/uncanny-burma/step\=180000.ckpt" data.scope=bird paths.results_dir=$DATA_PATH/uncanny-burma/RFCX_bird
uv run main.py eval +experiment=sivae seed=16 run_id=detailed-ticket data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/detailed-ticket/step\=180000.ckpt" data.scope=bird paths.results_dir=$DATA_PATH/detailed-ticket/RFCX_bird
uv run main.py eval +experiment=sivae seed=24 run_id=mossy-andrea data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/mossy-andrea/step\=180000.ckpt" data.scope=bird paths.results_dir=$DATA_PATH/mossy-andrea/RFCX_bird
uv run main.py eval +experiment=sivae seed=8 run_id=uncanny-burma data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/uncanny-burma/step\=180000.ckpt" data.scope=frog paths.results_dir=$DATA_PATH/uncanny-burma/RFCX_frog
uv run main.py eval +experiment=sivae seed=16 run_id=detailed-ticket data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/detailed-ticket/step\=180000.ckpt" data.scope=frog paths.results_dir=$DATA_PATH/detailed-ticket/RFCX_frog
uv run main.py eval +experiment=sivae seed=24 run_id=mossy-andrea data=rainforest_connection evaluator=lightning_predict "ckpt_path=$CKPT_PATH/v1/sivae/mossy-andrea/step\=180000.ckpt" data.scope=frog paths.results_dir=$DATA_PATH/mossy-andrea/RFCX_frog
