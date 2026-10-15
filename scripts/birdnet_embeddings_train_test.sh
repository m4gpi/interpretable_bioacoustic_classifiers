#!/bin/bash

# cross val results
#         model      scope  clf_learning_rate  l1_penalty  attn_learning_rate  attn_weight_decay  epoch     auROC        AP     score
# 138   birdnet  RFCX_bird               0.01        0.05              0.0010              0.001    599  0.962361  0.550907  1.513268
# 586   birdnet  RFCX_frog               0.01        0.01              0.0010              0.010    949  0.926537  0.394483  1.321020
# 1112  birdnet      SO_EC               0.01        0.05              0.0005              0.001    999  0.917581  0.518909  1.436491
# 1602  birdnet      SO_UK               0.01        0.05              0.0005              0.001    299  0.885744  0.563532  1.449276

wandb offline

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=birdnet_mil_species_detector hydra.launcher.n_jobs=3 paths=$DEVICE 'model_name=birdnet_${seed}' scope=SO_UK seed=8,16,24 trainer.max_epochs=300 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=birdnet_mil_species_detector hydra.launcher.n_jobs=3 paths=$DEVICE 'model_name=birdnet_${seed}' scope=SO_EC seed=8,16,24 trainer.max_epochs=1000 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.0005 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=birdnet_mil_species_detector hydra.launcher.n_jobs=3 paths=$DEVICE 'model_name=birdnet_${seed}' scope=RFCX_bird seed=8,16,24 trainer.max_epochs=600 model.clf_learning_rate=0.01 model.gamma_clf=0.0005 model.attn_learning_rate=0.001 model.gamma_attn=0.001
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run src/cli/train.py --multirun +experiment=birdnet_mil_species_detector hydra.launcher.n_jobs=3 paths=$DEVICE 'model_name=birdnet_${seed}' scope=RFCX_frog seed=8,16,24 trainer.max_epochs=950 model.clf_learning_rate=0.01 model.gamma_clf=0.0001 model.attn_learning_rate=0.001 model.gamma_attn=0.01
