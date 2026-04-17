#!/bin/bash

RESULTS_DIR=./results/vae_sivae_reconstruction_mae/mae.parquet
mkdir -p $RESULTS_DIR

wandb offline

uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v4/model.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v4_scope=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v5/model.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v5_scope=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v6/model.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v6_scope=SO_UK.parquet"

uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v4/model.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v4_scope=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v5/model.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v5_scope=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v6/model.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v6_scope=SO_EC.parquet"

uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/base_vae.pt:v7/model.ckpt" paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v7_scope=RFCX.parquet"
uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/base_vae.pt:v8/model.ckpt" paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v8_scope=RFCX.parquet"
uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/base_vae.pt:v9/model.ckpt" paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_vae_version=v9_scope=RFCX.parquet"

uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v12/model.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v12_scope=SO_UK.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v17/model.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v17_scope=SO_UK.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v18/model.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v18_scope=SO_UK.parquet"

uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v12/model.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v12_scope=SO_EC.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v17/model.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v17_scope=SO_EC.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v18/model.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v18_scope=SO_EC.parquet"

uv run main.py eval +experiment=sivae data=rainforest_connection "ckpt_path=./models/nifti_vae.pt:v14/model.ckpt" paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v14_scope=RFCX.parquet"
uv run main.py eval +experiment=sivae data=rainforest_connection "ckpt_path=./models/nifti_vae.pt:v15/model.ckpt" paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v15_scope=RFCX.parquet"
uv run main.py eval +experiment=sivae data=rainforest_connection "ckpt_path=./models/nifti_vae.pt:v16/model.ckpt" paths.results_dir=$RESULTS_DIR
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_vae_version=v16_scope=RFCX.parquet"

rm -rf $RESULTS_DIR/config
