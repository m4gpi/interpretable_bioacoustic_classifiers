#!/bin/bash

RESULTS_DIR=./mae.parquet/
mkdir -p $RESULTS_DIR

WANDB_MODE=offline uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v4/model.ckpt" data.scope=UK evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v4_scope=SO_UK.parquet"
WANDB_MODE=offline uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v5/model.ckpt" data.scope=UK evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v5_scope=SO_UK.parquet"
WANDB_MODE=offline uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v6/model.ckpt" data.scope=UK evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v6_scope=SO_UK.parquet"

WANDB_MODE=offline uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v4/model.ckpt" data.scope=EC evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v4_scope=SO_EC.parquet"
WANDB_MODE=offline uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v5/model.ckpt" data.scope=EC evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v5_scope=SO_EC.parquet"
WANDB_MODE=offline uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v6/model.ckpt" data.scope=EC evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v6_scope=SO_EC.parquet"

WANDB_MODE=offline uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/base_vae.pt:v7/model.ckpt" evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v7_scope=RFCX.parquet"
WANDB_MODE=offline uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/base_vae.pt:v8/model.ckpt" evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v8_scope=RFCX.parquet"
WANDB_MODE=offline uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/base_vae.pt:v9/model.ckpt" evaluator=test paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v9_scope=RFCX.parquet"

mv $RESULTS_DIR/config /tmp/results_configs
