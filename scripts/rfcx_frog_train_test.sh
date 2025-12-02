#!/bin/bash

RESULTS_DIR=/mnt/data0/kag25/species_detectors/

# model                  base_vae
# scope                 RFCX_frog
# pool_method           prob_attn
# clf_learning_rate          0.03
# l1_penalty                 0.01
# attn_learning_rate       0.0005
# attn_weight_decay         0.001
# epoch                       599
# auROC                   0.89515
# AP                     0.297165
# score                  1.192315

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=600 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v7 \
   data.scope=RFCX_frog \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=3.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_base_vae.pt:v7 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=600 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v8 \
   data.scope=RFCX_frog \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=3.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_base_vae.pt:v8 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=600 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v9 \
   data.scope=RFCX_frog \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=3.e-2 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_base_vae.pt:v9 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

# model                 nifti_vae
# scope                 RFCX_frog
# pool_method           prob_attn
# clf_learning_rate           0.1
# l1_penalty                 0.01
# attn_learning_rate       0.0005
# attn_weight_decay         0.001
# epoch                      1899
# auROC                  0.926267
# AP                     0.368286
# score                  1.294552

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=1900 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v14 \
   data.scope=RFCX_frog \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-1 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_nifti_vae.pt:v14 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=1900 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v15 \
   data.scope=RFCX_frog \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-1 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_nifti_vae.pt:v15 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=1900 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v16 \
   data.scope=RFCX_frog \
   data.val_prop=0.0 \
   model.pool_method=prob_attn \
   model.eval_sample_size=100 \
   model.attn_dim=10 \
   model.attn_weight_decay=1.e-3 \
   model.attn_learning_rate=5.e-4 \
   model.l1_penalty=1.e-2 \
   model.clf_learning_rate=1.e-1 \
   model.key_per_target=true \
   callbacks.model_checkpoint.dirpath=$RESULTS_DIR/checkpoints \
   callbacks.model_checkpoint.filename=rfcx_frog_nifti_vae.pt:v16 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

# model                 smooth_nifti_vae
# scope                        RFCX_frog
# pool_method                  prob_attn
# clf_learning_rate                  0.1
# l1_penalty                        0.01
# attn_learning_rate               0.001
# attn_weight_decay                0.001
# epoch                              499
# auROC                         0.901212
# AP                            0.291005
# score                         1.192217

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=500 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v3 \
   data.scope=RFCX_frog \
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
   callbacks.model_checkpoint.filename=rfcx_frog_smooth_nifti_vae.pt:v3 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=500 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v4 \
   data.scope=RFCX_frog \
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
   callbacks.model_checkpoint.filename=rfcx_frog_smooth_nifti_vae.pt:v4 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[3]' \
   trainer.max_epochs=500 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v5 \
   data.scope=RFCX_frog \
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
   callbacks.model_checkpoint.filename=rfcx_frog_smooth_nifti_vae.pt:v5 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR
