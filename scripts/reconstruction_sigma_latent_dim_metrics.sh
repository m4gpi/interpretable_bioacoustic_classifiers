#!/bin/bash

RESULTS_DIR=./results/reconstruction_variance/metrics.parquet
CKPT_PATH=./checkpoints/reconstruction_variance_expt

wandb offline

mkdir -p $RESULTS_DIR

uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v0/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v0_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v1/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v1_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v2/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v2_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v3/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v3_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v4/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v4_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v5/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v5_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v6/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v6_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v7/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v7_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v8/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v8_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v9/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v9_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v10/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v10_scope\=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v11/model_60000.ckpt" data.scope=UK callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v11_scope\=SO_UK.parquet"

uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v0/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v0_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v1/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v1_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v2/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v2_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v3/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v3_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v4/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v4_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v5/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v5_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v6/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v6_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v7/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v7_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v8/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v8_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v9/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v9_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v10/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v10_scope\=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=$CKPT_PATH/vae.pt:v11/model_60000.ckpt" data.scope=EC callbacks.vae_metrics.save_path="$RESULTS_DIR/version\=v11_scope\=SO_EC.parquet"
