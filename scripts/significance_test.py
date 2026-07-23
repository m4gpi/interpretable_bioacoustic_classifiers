import argparse
import pathlib
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

model_map = {
    "birdnet_native": "birdnet_oob",
    "birdnet_8": "birdnet_ft",
    "birdnet_16": "birdnet_ft",
    "birdnet_24": "birdnet_ft",
    "just-drum": "sivae",
    "dynamic-malta": "sivae",
    "daring-system": "sivae",
    "earthy-virgo": "sivae",
    "part-armor": "sivae",
    "secluded-montana": "sivae",
    "lumpy-gibson": "vae",
    "slow-partner": "vae",
    "unique-tiger": "vae",
    "jumpy-engine": "vae",
    "quaint-pilot": "vae",
    "numb-chef": "vae",
}

def main(results_dir, left: str, right: str):
    scores = pd.read_parquet(results_dir / "test_scores.parquet").reset_index()

    scores["run"] = scores["model"]
    scores["model"] = scores["run"].map(model_map)

    print(f"{left} vs {right}\n")
    for scope in ["SO_UK", "SO_EC", "RFCX_bird"]:
        left_scores = scores.loc[(scores.model == left) & (scores.scope == scope)].groupby("species_name")[["AP", "auROC", ]].mean()
        right_scores = scores.loc[(scores.model == right) & (scores.scope == scope)].groupby("species_name")[["AP", "auROC"]].mean()
        species = list(set(left_scores.index).intersection(set(right_scores.index)))
        left_scores = left_scores.loc[species].values
        right_scores = right_scores.loc[species].values
        # Perform Wilcoxon signed-rank test between the two arrays (paired by species)
        stat, p_value = wilcoxon(left_scores[:, 0], right_scores[:, 0])  # mAP
        print(scope)
        print(f"Wilcoxon test for mAP: statistic={stat}, p-value={p_value}")
        stat, p_value = wilcoxon(left_scores[:, 1], right_scores[:, 1])  # auROC
        print(f"Wilcoxon test for auROC: statistic={stat}, p-value={p_value}")

        score_diff_mean = (left_scores - right_scores).mean(axis=0)
        print(f"AP: {score_diff_mean[0]}")
        print(f"auROC diff: {score_diff_mean[1]}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=lambda p: pathlib.Path(p),
        required=True,
        help="/path/to/data/",
    )
    parser.add_argument(
        "--left",
        type=str,
        required=True,
        help="model name left side of wilcoxon test",
    )
    parser.add_argument(
        "--right",
        type=str,
        required=True,
        help="model name right side of wilcoxon test",
    )
    args = parser.parse_args()
    main(**vars(args))
