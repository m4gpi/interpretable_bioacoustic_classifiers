#!/bin/bash

RESULTS_DIR=./results/species_detectors/

uv run main.py eval \
   +experiment=birdnet \
   data=rainforest_connection \
   data.scope=RFCX_bird \
   data.root=/its/home/kag25/data/rainforest_connection \
   paths.results_dir=$RESULTS_DIR

# model                  base_vae
# scope                 RFCX_bird
# pool_method           prob_attn
# clf_learning_rate          0.01
# l1_penalty                 0.01
# attn_learning_rate        0.001
# attn_weight_decay         0.001
# epoch                      1949
# auROC                  0.859588
# AP                     0.261723
# score                  1.121311

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v7 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_base_vae.pt:v7 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v8 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
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

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v9 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-2 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_base_vae.pt:v9 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

# model                 nifti_vae
# scope                 RFCX_bird
# pool_method           prob_attn
# clf_learning_rate          0.05
# l1_penalty                 0.01
# attn_learning_rate       0.0005
# attn_weight_decay         0.001
# epoch                      1999
# auROC                  0.914302
# AP                     0.353441
# score                  1.267742

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=2000 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v14 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=5.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_nifti_vae.pt:v14 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=2000 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v15 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=5.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_nifti_vae.pt:v15 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=2000 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v16 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=5.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_nifti_vae.pt:v16 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

# model                 smooth_nifti_vae
# scope                        RFCX_bird
# pool_method                  prob_attn
# clf_learning_rate                  0.1
# l1_penalty                        0.01
# attn_learning_rate               0.001
# attn_weight_decay                0.001
# epoch                              999
# auROC                         0.900189
# AP                            0.315831
# score                          1.21602

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=1000 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v3 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=1.e-3 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-1 \
   model.key_per_target=true \
   model.penalty_multiplier=2 \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_smooth_nifti_vae.pt:v3 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=1000 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v4 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=1.e-3 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-1 \
   model.key_per_target=true \
   model.penalty_multiplier=2 \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_smooth_nifti_vae.pt:v4 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[2]' \
   trainer.max_epochs=1000 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v5 \
   data.scope=RFCX_bird \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=1.e-3 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-1 \
   model.key_per_target=true \
   model.penalty_multiplier=2 \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_bird_smooth_nifti_vae.pt:v5 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR
