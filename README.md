# Interpretable Bioacoustic Classifiers

## Single Training Run:
```
uv run src/cli/train.py \
  +experiment=species_detector \
  name="runs/$(openssl rand -hex 6)" \
  trainer.max_epochs=1000 \
  trainer.check_val_every_n_epoch=50 \
  data.model=smooth_nifti_vae \
  data.version=v0 \
  data.scope=SO_UK \
  model.eval_sample_size=50 \
```

## Hyperparameter Sweep
```sh
uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=7 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  name="cross_validation/$(openssl rand -hex 6)" \
  'trainer.devices=[0]' \
  trainer.max_epochs=1000 \
  trainer.check_val_every_n_epoch=50 \
  data.num_folds=7 \
  data.fold_id=0,1,2,3,4,5,6 \
  data.model=smooth_nifti_vae \
  data.version=v0,v1,v2 \
  data.scope=SO_UK \
  model.pool_method=max,mean,feature_attn,prob_attn \
  model.attn_learning_rate=1e-2,5e-3 \
  model.clf_learning_rate=1e-1,5e-2 \
  model.label_smoothing=0.0 \
  model.eval_sample_size=50 \
  paths.results_dir=/mnt/data0/kag25/interpretable_bioacoustic_classifiers
```

```sh
uv run src/cli/train.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=7 \
  hydra.launcher.batch_size=1 \
  +experiment=species_detector \
  name="attention_cross_validation/$(openssl rand -hex 6)" \
  'trainer.devices=[3]' \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  data.num_folds=7 \
  data.fold_id=0,1,2,3,4,5,6 \
  data.model=smooth_nifti_vae \
  data.version=v0,v1,v2 \
  data.scope=SO_UK \
  model.pool_method=feature_attn,prob_attn \
  model.attn_learning_rate=1e-2,5e-3,1e-3,5e-4,1e-4 \
  model.clf_learning_rate=1e-1,5e-2,1e-2,5e-3,1e-3 \
  model.label_smoothing=0.0 \
  model.eval_sample_size=50 \
  paths.results_dir=/mnt/data0/kag25/interpretable_bioacoustic_classifiers
```
