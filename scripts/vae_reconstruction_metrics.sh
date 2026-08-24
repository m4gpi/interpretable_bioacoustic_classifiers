#!/bin/bash

RESULTS_DIR=./results/vae_dimensionality_reconstruction/mae.parquet
mkdir -p $RESULTS_DIR

wandb offline

uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v4/model.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR model.latent_dim=128
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_version=v4_latent_dim=128_scope=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/vae.pt:v1/step\=180000.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR model.latent_dim=256
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_version=v1_latent_dim=256_scope=SO_UK.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/vae.pt:v2/step\=180000.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR model.latent_dim=512
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_version=v2_latent_dim=512_scope=SO_UK.parquet"

uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/base_vae.pt:v4/model.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR model.latent_dim=128
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_version=v4_latent_dim=128_scope=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/vae.pt:v1/step\=180000.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR model.latent_dim=256
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_version=v1_latent_dim=256_scope=SO_EC.parquet"
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/vae.pt:v2/step\=180000.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR model.latent_dim=512
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=base_version=v2_latent_dim=512_scope=SO_EC.parquet"

uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v12/model_v1.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR model.latent_dim=128
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_version=v12_latent_dim=128_scope=SO_UK.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/sivae.pt:v1/step\=180000.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR model.latent_dim=256
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_version=v1_latent_dim=256_scope=SO_UK.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/sivae.pt:v2/step\=180000.ckpt" data.scope=UK paths.results_dir=$RESULTS_DIR model.latent_dim=512
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_version=v2_latent_dim=512_scope=SO_UK.parquet"

uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/nifti_vae.pt:v12/model_v1.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR model.latent_dim=128
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_version=v12_latent_dim=128_scope=SO_EC.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/sivae.pt:v1/step\=180000.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR model.latent_dim=256
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_version=v1_latent_dim=256_scope=SO_EC.parquet"
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/sivae.pt:v2/step\=180000.ckpt" data.scope=EC paths.results_dir=$RESULTS_DIR model.latent_dim=512
mv $RESULTS_DIR/metrics.parquet "$RESULTS_DIR/model_name=nifti_version=v2_latent_dim=512_scope=SO_EC.parquet"

rm -rf $RESULTS_DIR/config
