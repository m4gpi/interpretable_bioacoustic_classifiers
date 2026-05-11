#!/bin/bash

DATA_DIR = "/mnt/data0/kag25"

uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  seed=8 \
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  'data.root=${paths.data_dir}/soundscape_vae_embeddings/silly-byte/SO_UK' \
  data.num_folds=5 \
  data.fold_id=0,1,2,3,4 \
  model.eval_sample_size=100 \
  model.lamdba=1.e-1,5.e-2,1.e-2 \
  model.clf_learning_rate=1.e-1,5.e-2,3.e-2,1.e-2 \
  model.attn_weight_decay=1.e-2,1.e-3 \
  model.attn_learning_rate=1.e-3,5.e-4 \
  paths.results_dir=./results/v1/species_cross_validation

uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  seed=16 \
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.root=$DATA_DIR/soundscape_vae_embeddings/meek-zebra/SO_UK \
  data.num_folds=5 \
  data.fold_id=0,1,2,3,4 \
  model.eval_sample_size=100 \
  model.lamdba=1.e-1,5.e-2,1.e-2 \
  model.clf_learning_rate=1.e-1,5.e-2,3.e-2,1.e-2 \
  model.attn_weight_decay=1.e-2,1.e-3 \
  model.attn_learning_rate=1.e-3,5.e-4 \
  paths.results_dir=./results/v1/species_cross_validation

uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  seed=24 \
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.root=$DATA_DIR/soundscape_vae_embeddings/rude-money/SO_UK \
  data.num_folds=5 \
  data.fold_id=0,1,2,3,4 \
  model.eval_sample_size=100 \
  model.lamdba=1.e-1,5.e-2,1.e-2 \
  model.clf_learning_rate=1.e-1,5.e-2,3.e-2,1.e-2 \
  model.attn_weight_decay=1.e-2,1.e-3 \
  model.attn_learning_rate=1.e-3,5.e-4 \
  paths.results_dir=./results/v1/species_cross_validation

uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  seed=8 \
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.root=$DATA_DIR/soundscape_vae_embeddings/tan-ohio/SO_UK \
  data.num_folds=5 \
  data.fold_id=0,1,2,3,4 \
  model.eval_sample_size=100 \
  model.lamdba=1.e-1,5.e-2,1.e-2 \
  model.clf_learning_rate=1.e-1,5.e-2,3.e-2,1.e-2 \
  model.attn_weight_decay=1.e-2,1.e-3 \
  model.attn_learning_rate=1.e-3,5.e-4 \
  paths.results_dir=./results/v1/species_cross_validation

uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  seed=16 \
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.root=$DATA_DIR/soundscape_vae_embeddings/small-peru/SO_UK \
  data.num_folds=5 \
  data.fold_id=0,1,2,3,4 \
  model.eval_sample_size=100 \
  model.lamdba=1.e-1,5.e-2,1.e-2 \
  model.clf_learning_rate=1.e-1,5.e-2,3.e-2,1.e-2 \
  model.attn_weight_decay=1.e-2,1.e-3 \
  model.attn_learning_rate=1.e-3,5.e-4 \
  paths.results_dir=./results/v1/species_cross_validation

uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  seed=24 \
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.root=$DATA_DIR/soundscape_vae_embeddings/brave-vincent/SO_UK \
  data.num_folds=5 \
  data.fold_id=0,1,2,3,4 \
  model.eval_sample_size=100 \
  model.lamdba=1.e-1,5.e-2,1.e-2 \
  model.clf_learning_rate=1.e-1,5.e-2,3.e-2,1.e-2 \
  model.attn_weight_decay=1.e-2,1.e-3 \
  model.attn_learning_rate=1.e-3,5.e-4 \
  paths.results_dir=./results/v1/species_cross_validation
