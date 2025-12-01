#!/bin/bash

RESULTS_DIR=./results/test

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=base_vae \
   data.version=v4 \
   data.scope=SO_UK \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=so_uk_base_vae.pt:v4 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=nifti_vae \
   data.version=v12 \
   data.scope=SO_UK \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=so_uk_nifti_vae.pt:v12 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=smooth_nifti_vae \
   data.version=v0 \
   data.scope=SO_UK \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   model.penalty_multiplier=2 \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=so_uk_smooth_nifti_vae.pt:v0 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=base_vae \
   data.version=v4 \
   data.scope=SO_EC \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=so_ec_base_vae.pt:v4 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=nifti_vae \
   data.version=v12 \
   data.scope=SO_EC \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=so_ec_nifti_vae.pt:v12 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=smooth_nifti_vae \
   data.version=v0 \
   data.scope=SO_EC \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   model.penalty_multiplier=2 \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=so_ec_smooth_nifti_vae.pt:v0 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=base_vae \
   data.version=v8 \
   data.scope=RFCX_bird \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_base_vae.pt:v8 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=nifti_vae \
   data.version=v14 \
   data.scope=RFCX_bird \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_nifti_vae.pt:v14 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=smooth_nifti_vae \
   data.version=v3 \
   data.scope=RFCX_bird \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   model.penalty_multiplier=2 \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_smooth_nifti_vae.pt:v3 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=base_vae \
   data.version=v8 \
   data.scope=RFCX_frog \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_base_vae.pt:v8 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=nifti_vae \
   data.version=v14 \
   data.scope=RFCX_frog \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_nifti_vae.pt:v14 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/train.py \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   data.model=smooth_nifti_vae \
   data.version=v3 \
   data.scope=RFCX_frog \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   model.penalty_multiplier=2 \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_smooth_nifti_vae.pt:v3 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/eval.py \
   +experiment=birdnet \
   data=sounding_out_chorus \
   data.scope=SO_UK \
   data.root=/its/home/kag25/data/sounding_out \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/eval.py \
   +experiment=birdnet \
   data=sounding_out_chorus \
   data.scope=SO_EC \
   data.root=/its/home/kag25/data/sounding_out \
   paths.results_dir=$RESULTS_DIR

uv run src/cli/eval.py \
   +experiment=birdnet \
   data=rainforest_connection \
   data.scope=RFCX_bird \
   data.root=/its/home/kag25/data/rainforest_connection \
   paths.results_dir=$RESULTS_DIR
