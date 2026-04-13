#!/bin/bash

RESULTS_DIR=./results/birdnet_cross_validation/

WANDB_MODE=offline uv run src/cli/train.py --multirun \
   hydra/launcher=joblib \
   hydra.launcher.n_jobs=4 \
   hydra.launcher.batch_size=1 \
   +experiment=species_detector \
   trainer.max_epochs=1000 \
   trainer.check_val_every_n_epoch=50 \
   data.root=./data/soundscape_birdnet_embeddings \
   data.model=birdnet \
   data.scope=SO_UK,SO_EC,RFCX_bird,RFCX_frog \
   data.version=v0 \
   data.num_folds=5 \
   data.fold_id=0,1,2,3,4 \
   model.in_features=1024 \
   model.train_sample_size=null \
   model.eval_sample_size=null \
   model.pool_method=prob_attn \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2,1.e-3 \
   model.attn_learning_rate=1.e-3,5.e-4 \
   model.l1_penalty=5.e-2,1.e-2 \
   model.clf_learning_rate=5.e-2,3.e-2,1.e-2 \
   model.key_per_target=true \
   paths.results_dir=$RESULTS_DIR
