#!/bin/bash

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run src/cli/eval.py --multirun \
  +experiment=tcvae \
  hydra.launcher.n_jobs=1 \
  run_id=antsy-denmark,elite-virus,fancy-cotton,oval-permit,pale-spirit,scenic-popcorn,tame-viva,varied-madrid,naive-zipper,caring-goblin \
  data=sounding_out_chorus \
  evaluator=likelihood_degradation \
  'ckpt_path=/its/home/kag25/checkpoints/v4/tcvae_ema_alpha_sweep/${run_id}/step\=30000.ckpt' \
  'paths.results_dir=/its/home/kag25/experiments/v4/tcvae_likelihood_degradation/${run_id}.parquet/' \
  data.eval_batch_size=1
