import argparse
import pandas as pd
import pathlib
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.core.utils import metrics

def main(
    results_dir: pathlib.Path,
    save_dir: pathlib.Path,
) -> None:
    results = pd.read_parquet(results_dir / "test_results.parquet", columns=["file_i", "species_name", "label", "prob", "model", "version", "scope"])
    scores = pd.read_parquet(results_dir / "test_scores.parquet", columns=["species_name", "AP", "auROC", "model", "scope", "version"])

    name_map = {
        "birdnet": "BirdNET V2.4",
        "base_vae": "VAE",
        "nifti_vae": "SIVAE",
        "smooth_nifti_vae": "TSSIVAE",
    }

    results["model_class"] = results["model"].map(name_map)
    scores["model_class"] = scores["model"].map(name_map)

    score_summary = scores.groupby(["model", "model_class", "version", "scope"]).agg(
        auROC=("auROC", "mean"),
        mAP=("AP", "mean"),
    ).reset_index()

    recall_results = []
    for scope in results.scope.unique():
        # pivot results for only this dataset
        results_pivot = (
            results[results.scope == scope]
            .drop("scope", axis=1)
            .pivot(columns="species_name", index=["file_i", "model", "model_class", "version"])
            .dropna(axis=1, how="any") # drop nulls for species we can't compare
        )
        recall_df = (
            results_pivot
            .reset_index()
            .groupby(["model", "model_class", "version"])
            .apply(lambda df: metrics.recall_at_k(df["label"].to_numpy(), df["prob"].to_numpy()))
            .reset_index()
            .rename(columns={0: "recall_at_k"})
        )
        recall_df["scope"] = scope
        recall_results.append(recall_df)
    recall_df = pd.concat(recall_results)
    score_summary = (
        score_summary
        .merge(recall_df, how="inner", on=["model", "scope", "version", "model_class"])
        # calculate the mean/std across model class
        .groupby(["model_class", "scope"]).agg(
            auROC_mean=("auROC", "mean"),
            auROC_std=("auROC", "std"),
            mAP_mean=("mAP", "mean"),
            mAP_std=("mAP", "std"),
            top1_mean=("recall_at_k", "mean"),
            top1_std=("recall_at_k", "std"),
        )
        .reset_index()
    )
    print(score_summary.to_markdown())

    # format results into latex tables
    score_name_map = {
        "auROC": "auROC",
        "mAP": "mAP",
        "top1": "Top-1", # ??? not sure why this is called top-1, since thats not what the score function actually is
    }
    for mean_col, std_col in [("auROC_mean", "auROC_std"), ("mAP_mean", "mAP_std"), ("top1_mean", "top1_std")]:
        score_name = score_name_map[mean_col.split("_")[0]]
        score_summary[score_name] = score_summary[[mean_col, std_col]].apply(lambda row: f"{row[mean_col]:.2f} ({(0.0 if pd.isna(row[std_col]) else row[std_col]):.3f})", axis=1)

    scores = score_summary[["model_class", "scope", *score_name_map.values()]]
    df = pd.concat([
        scores,
        scores["scope"].str.split("_", expand=True).rename(columns={0: "dataset", 1: "subset"})
    ], axis=1)
    df["scope"] = df["scope"].str.replace("_", " ")

    save_dir.mkdir(exist_ok=True, parents=True)

    for dataset in df.dataset.unique():
        dataset_results = []
        for subset in df.loc[df["dataset"] == dataset, "subset"].unique():
            df1 = (
                df[(df["subset"] == subset) & (df["dataset"] == dataset)]
                .drop(["dataset", "subset"], axis=1)
                .melt(id_vars=["model_class", "scope"])
                .pivot(index="model_class", columns=["scope", "variable"], values="value")
            )
            df1.columns.names = ["Dataset", "Metric"]
            dataset_results.append(df1)
        results_table = pd.concat(dataset_results, axis=1)
        results_table.index.name = "Model"

        with open(save_dir / f"P{dataset}_scores_table.latex", "w+") as f:
            with pd.option_context("max_colwidth", 1000):
                latex = results_table.style.to_latex()
                f.write(latex)
                print(latex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=lambda p: pathlib.Path(p).expanduser(),
        required=True,
        help="/path/to/results/dir",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: pathlib.Path(p).expanduser(),
        required=False,
        help="/path/to/saved/",
    )
    args = parser.parse_args()
    main(**vars(args))
