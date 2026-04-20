#!/bin/bash

RESULTS_DIR=./data/soundscape_vae_embeddings
CKPT_PATH=./checkpoints/reconstruction_variance_expt

wandb offline

mkdir -p $RESULTS_DIR

uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v0/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v1/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v1/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v2/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v2/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v3/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v3/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v4/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v4/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v5/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v5/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v6/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v6/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v7/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v7/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v8/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v8/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v9/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v9/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v10/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v10/SO_UK"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v11/model_60000.ckpt" data.scope=UK callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v11/SO_UK"

uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v0/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v1/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v2/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v3/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v4/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v5/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v6/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v7/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v8/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v9/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v10/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
uv run main.py eval +experiment=vae data=sounding_out_chorus evaluator=lightning_predict "ckpt_path=$CKPT_PATH/vae.pt:v11/model_60000.ckpt" data.scope=EC callbacks.vae_embeddings.save_path="$RESULTS_DIR/vae.pt:v0/SO_EC"
