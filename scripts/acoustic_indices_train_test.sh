#!/bin/bash

# Cross Val Results
#                  model      scope  clf_learning_rate  attn_learning_rate  gamma_clf  gamma_attn  epoch     auROC        AP     score
# 543   acoustic_indices  RFCX_bird               0.05              0.0005     0.0001       0.010    899  0.719535  0.048480  0.768015
# 1910  acoustic_indices  RFCX_frog               0.10              0.0010     0.0001       0.001    999  0.852172  0.148170  1.000342
# 2792  acoustic_indices      SO_EC               0.10              0.0005     0.0001       0.001    999  0.793947  0.130623  0.924570
# 3926  acoustic_indices      SO_UK               0.10              0.0010     0.0001       0.001    999  0.737458  0.235823  0.973282

wandb offline

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=eai_mil_species_detector paths=$DEVICE hydra.launcher.n_jobs=3 'model_name=acoustic_indices_${seed}' seed=8,16,24 scope=SO_UK trainer.max_epochs=1000 model.clf_learning_rate=0.1 model.gamma_clf=0.0001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=eai_mil_species_detector paths=$DEVICE hydra.launcher.n_jobs=3 'model_name=acoustic_indices_${seed}' seed=8,16,24 scope=SO_EC trainer.max_epochs=1000 model.clf_learning_rate=0.1 model.gamma_clf=0.0001 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=eai_mil_species_detector paths=$DEVICE hydra.launcher.n_jobs=3 'model_name=acoustic_indices_${seed}' seed=8,16,24 scope=RFCX_bird trainer.max_epochs=900 model.clf_learning_rate=0.05 model.gamma_clf=0.0001 model.attn_learning_rate=0.0005 model.gamma_attn=0.01
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=eai_mil_species_detector paths=$DEVICE hydra.launcher.n_jobs=3 'model_name=acoustic_indices_${seed}' seed=8,16,24 scope=RFCX_frog trainer.max_epochs=1000 model.clf_learning_rate=0.1 model.gamma_clf=0.0001 model.attn_learning_rate=0.001 model.gamma_attn=0.001
