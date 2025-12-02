#!/bin/bash

RESULTS_DIR=/mnt/data0/kag25/species_detectors/

# uv run main.py eval \
#    +experiment=birdnet \
#    data=sounding_out_chorus \
#    data.scope=SO_EC \
#    data.root=/its/home/kag25/data/sounding_out \
#    paths.results_dir=$RESULTS_DIR

# model                  base_vae
# scope                     SO_EC
# pool_method           prob_attn
# clf_learning_rate          0.03
# l1_penalty                 0.01
# attn_learning_rate       0.0005
# attn_weight_decay         0.001
# epoch                      1949
# auROC                  0.873897
# AP                      0.33703
# score                  1.210927

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v4 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_base_vae.pt:v4 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v5 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_base_vae.pt:v5 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=base_vae \
   data.version=v6 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_base_vae.pt:v6 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

# model                 nifti_vae
# scope                     SO_EC
# pool_method           prob_attn
# clf_learning_rate          0.03
# l1_penalty                 0.01
# attn_learning_rate       0.0005
# attn_weight_decay         0.001
# epoch                      1949
# auROC                  0.880108
# AP                     0.362223
# score                  1.242331

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v12 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_nifti_vae.pt:v12 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v17 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_nifti_vae.pt:v17 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1950 \
   +trainer.limit_val_batches=0.0 \
   data.model=nifti_vae \
   data.version=v18 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_nifti_vae.pt:v18 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

# model                 smooth_nifti_vae
# scope                            SO_EC
# pool_method                  prob_attn
# clf_learning_rate                  0.1
# l1_penalty                        0.01
# attn_learning_rate               0.001
# attn_weight_decay                0.001
# epoch                             1949
# auROC                         0.871199
# AP                            0.324339
# score                         1.195538

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v0 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_smooth_nifti_vae.pt:v0 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v1 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_smooth_nifti_vae.pt:v1 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR

uv run main.py train \
   +experiment=species_detector \
   project=species_detector \
   'trainer.devices=[0]' \
   trainer.max_epochs=1 \
   +trainer.limit_val_batches=0.0 \
   data.model=smooth_nifti_vae \
   data.version=v2 \
   data.scope=SO_EC \
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
   callbacks.model_checkpoint.filename=so_ec_smooth_nifti_vae.pt:v2 \
   callbacks.model_checkpoint.monitor=null \
   callbacks.model_checkpoint.save_last=null \
   paths.results_dir=$RESULTS_DIR
