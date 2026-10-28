#!/bin/bash

CUDA_VISIBLE_DEVICES=3,4,5,6,7 WANDB_MODE=offline uv run src/cli/eval.py --multirun \
  +experiment=tcvae \
  hydra.launcher.n_jobs=5 \
  paths=vili \
  run_id=antsy-denmark,elite-virus,fancy-cotton,oval-permit,pale-spirit,scenic-popcorn,tame-viva,varied-madrid,naive-zipper,caring-goblin \
  data=sounding_out_chorus \
  evaluator=likelihood_degradation \
  'evaluator.save_path=${paths.results_dir}/tcvae_likelihood_degradation/${run_id}.parquet/' \
  'ckpt_path=${paths.checkpoint_dir}/v4/tcvae_ema_alpha_sweep/${run_id}/step\=30000.ckpt' \
  data.eval_batch_size=1
