import attrs
import birdnet
import hydra
import functools
import os
import pandas as pd
import pathlib
import requests
import rootutils
import warnings

from omegaconf import OmegaConf, DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.utils import metrics

BIRDNET_LABEL_TXT_FILE = (
    "https://raw.githubusercontent.com/kahst/BirdNET-Analyzer"
    "/refs/tags/v1.5.0/birdnet_analyzer/checkpoints/V2.4/"
    "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
)

def encode(cfg):
    run_id = os.urandom(6).hex()
    OmegaConf.update(cfg, "run_id", run_id, force_add=True)

    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup()
    data = data_module.test_data
    birdnet_targets = [label for label in requests.get(BIRDNET_LABEL_TXT_FILE).text.split("\n")]
    target_names = list(set(birdnet_targets).intersection(set(data.target_names)))
    birdnet_params = dict(min_confidence=0.0, species_filter=set(target_names))

    labels = data.labels.reset_index()
    results = []
    for file_path, predictions in birdnet.predict_species_within_audio_files_mp(data.data_dir / labels.file_name, **birdnet_params):
        record = labels[labels["file_name"] == file_path.name].iloc[0]
        y = record[target_names]
        y_prob = pd.Series({target: 0.0 for target in target_names})
        for window, prediction in predictions.items():
            for target, prob in prediction.items():
                y_prob[target] = max(prob, y_prob[target])
        for target in target_names:
            results.append({
                "file_i": record.file_i,
                "species_name": target,
                "prob": y_prob[target],
                "label": y[target],
            })
    results = pd.DataFrame(results)
    results["model"] = "BirdNET"
    results["version"] = "version"
    results["scope"] = data_module.scope
    scores = metrics.score(results)
    scores["run_id"] = run_id
    scores["model"] = "BirdNET"
    scores["version"] = "version"
    scores["scope"] = data_module.scope
    print(scores.to_markdown())
    summary_stats = scores.groupby("run_id").agg(
        auROC_mean=("auROC", "mean"),
        auROC_std=("auROC", "std"),
        AP_mean=("AP", "mean"),
        AP_std=("AP", "std"),
    ).reset_index()
    results_pivot = results.pivot(columns="species_name", index="file_i")
    summary_stats["recall_at_k"] = metrics.recall_at_k(
        results_pivot["label"].to_numpy(),
        results_pivot["prob"].to_numpy(),
    )
    print(summary_stats.to_markdown())
    results_dir = pathlib.Path(cfg.get("results_dir")) / "test_results.parquet"
    results_dir.mkdir(exist_ok=True, parents=True)
    scores_dir = pathlib.Path(cfg.get("results_dir")) / "test_scores.parquet"
    scores_dir.mkdir(exist_ok=True, parents=True)
    results.to_parquet(results_dir / f"{run_id}.parquet")
    scores.to_parquet(scores_dir / f"{run_id}.parquet")

@hydra.main(version_base="1.3", config_path="../../config", config_name="eval.yaml")
def main(cfg: DictConfig):
    encode(cfg)

if __name__ == "__main__":
    main()
