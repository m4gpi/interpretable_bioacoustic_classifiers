import argparse
import numpy as np
import pathlib
import pandas as pd
import torch
import logging
import rootutils
import yaml

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

model_map = {
    "tan-ohio": "sivae",
    "brave-vincent": "sivae",
    "small-peru": "sivae",
    "uncanny-burma": "sivae",
    "detailed-ticket": "sivae",
    "mossy-andrea": "sivae",
    "silly-byte": "vae",
    "meek-zebra": "vae",
    "rude-money": "vae",
    "misty-lecture": "vae",
    "ultimate-story": "vae",
    "tusked-chief": "vae",
}

def main(results_dir):
    log.info(f"Loading {results_dir}... this may take a moment...")
    df = pd.read_parquet(results_dir)
    df["version"] = df["model"]
    df["model"] = df["version"].map(model_map)
    hyper_params = ["clf_learning_rate", "lamdba", "attn_learning_rate", "attn_weight_decay"]
    df = df.drop_duplicates(subset=["model", "species_name", "version", "scope", "epoch", "fold_id", *hyper_params])
    df["score"] = df["auROC"] + df["AP"]
    metrics = ["auROC", "AP", "score"]
    # for each run take the mean score across species
    species_agg_df = df.groupby(["model", "version", "scope", "epoch", "fold_id", *hyper_params], dropna=False)[metrics].mean().reset_index()
    # for each model and dataset, take the mean score across folds
    folds_df = species_agg_df.groupby(["model", "version", "scope", "epoch", *hyper_params], dropna=False)[metrics].mean().reset_index()
    # take the mean across model seeds, for each data subset and epoch
    model_df = folds_df.groupby(["model", "scope", "epoch", *hyper_params], dropna=False)[metrics].mean().reset_index()
    # find the best epoch for each run
    best_epoch_idx = model_df.groupby(["model", "scope", *hyper_params], dropna=False)["score"].idxmax()
    results_df = model_df.iloc[best_epoch_idx].copy()
    # find the best hyper-parameters across each model and dataset
    summary_df = results_df.loc[results_df.groupby(["model", "scope"])["score"].idxmax()]
    print(summary_df.to_markdown())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=lambda p: pathlib.Path(p),
        required=True,
        help="/path/to/results.parquet/",
    )
    args = parser.parse_args()
    main(**vars(args))

