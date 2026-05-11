#!/bin/bash

uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  seed=8 \
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.root=silly-byte/SO_EC \
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
  data.root=meek-zebra/SO_EC \
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
  data.root=rude-money/SO_EC \
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
  'trainer.devices=[0]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.root=tan-ohio,small-peru,brave-vincent
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
  data.root=tan-ohio/SO_EC \
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
  data.root=small-peru/SO_EC \
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
  data.root=brave-vincent/SO_EC \
  data.num_folds=5 \
  data.fold_id=0,1,2,3,4 \
  model.eval_sample_size=100 \
  model.lamdba=1.e-1,5.e-2,1.e-2 \
  model.clf_learning_rate=1.e-1,5.e-2,3.e-2,1.e-2 \
  model.attn_weight_decay=1.e-2,1.e-3 \
  model.attn_learning_rate=1.e-3,5.e-4 \
  paths.results_dir=./results/v1/species_cross_validation
