#!/bin/bash

uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/v1/vae/silly-byte/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=silly-byte seed=8
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/v1/vae/meek-zebra/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=meek-zebra seed=16
uv run main.py eval +experiment=vae data=sounding_out_chorus "ckpt_path=./models/v1/vae/rude-money/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=rude-money seed=24

uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/v1/vae/tusked-chief/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=tusked-chief seed=8
uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/v1/vae/ultimate-story/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=ultimate-story seed=16
uv run main.py eval +experiment=vae data=rainforest_connection "ckpt_path=./models/v1/vae/misty-lecture/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=misty-lecture seed=24

uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/v1/sivae/tan-ohio/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=tan-ohio seed=8
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/v1/sivae/brave-vincent/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=brave-vincent seed=16
uv run main.py eval +experiment=sivae data=sounding_out_chorus "ckpt_path=./models/v1/sivae/small-peru/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=small-peru seed=24

uv run main.py eval +experiment=sivae data=rainforest_connection "ckpt_path=./models/v1/sivae/uncanny-burma/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=uncanny-burma seed=8
uv run main.py eval +experiment=sivae data=rainforest_connection "ckpt_path=./models/v1/sivae/detailed-ticket/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=detailed-ticket seed=16
uv run main.py eval +experiment=sivae data=rainforest_connection "ckpt_path=./models/v1/sivae/mossy-andrea/step\=180000.ckpt" paths.results_dir=./results/v1/vae_evaluation run_id=mossy-andrea seed=24

