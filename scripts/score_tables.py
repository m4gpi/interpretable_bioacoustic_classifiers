import argparse
import pandas as pd
from pathlib import Path

def strip_stat(col):
    dataset, metric = col
    base_metric = metric.replace(' mean', '').replace(' std', '')
    return (dataset, base_metric)

def main(
    birdnet_results_path: Path,
    birdnet_scores_path: Path,
    vae_clf_results_path: Path,
    vae_clf_scores_path: Path,
    label_frequency_path: Path,
    save_dir: Path,
    frequency_threshold: int = 25,
) -> None:
    s = pd.concat([
        pd.read_parquet(birdnet_scores_path),
        pd.read_parquet(vae_clf_scores_path).reset_index(),
    ])
    R = pd.concat([
        pd.read_parquet(birdnet_results_path),
        pd.read_parquet(vae_clf_results_path),
    ])
    s["dataset_name"] = s["dataset"] + " " + s["scope"].str.split("==", expand=True)[1].str.replace("'", "")
    R["dataset_name"] = R["dataset"] + " " + R["scope"].str.split("==", expand=True)[1].str.replace("'", "")
    df3 = pd.read_parquet(label_frequency_path).reset_index()
    #innee remove species less than label frequency threshold
    s = s.merge(df3, on=["species_name", "dataset", "scope"], how="left")
    s = s[s.frequency >= frequency_threshold]
    R = R.merge(df3, on=["species_name", "dataset", "scope"], how="left")
    R = R[R.frequency >= frequency_threshold]

    save_dir.mkdir(exist_ok=True, parents=True)
    dataset_names = s.dataset_name.unique()
    name_map = {
        "Base": "Classic VAE",
        "NIFTI": "Shift Invariant\nVAE",
        "Smooth NIFTI": "Shift Invariant \&\nTemporally Smooth\nVAE",
        "BirdNET V2.4": "BirdNET V2.4",
    }
    s["model_class"] = s["model_class"].map(name_map)
    R["model_class"] = R["model_class"].map(name_map)
    # scores are computed mean and standard deviations across random seeds, mean across species, deviation across models
    # birdnet gets null std dev due to only one confidence threshold / model to evaluate

    # Top-1 Scores, i.e. how well do the binary classifiers perform at a multiclass categorical task
    R["k"] = R.groupby(["dataset_name", "file_i", "model_class", "model_id"])["label"].transform("sum")
    R["rank"] = R.groupby(["dataset_name", "file_i", "model_class", "model_id"])["prob"].rank(method="first", ascending=False)
    top_k = R[R["rank"] <= R["k"]].copy()
    predicted_positive = top_k.groupby(["dataset_name", "file_i", "model_class", "model_id"])["label"].sum()
    actual_positive = R[R["label"] == 1].groupby(["dataset_name", "file_i", "model_class", "model_id"])["label"].count()
    top_1_scores = (predicted_positive / actual_positive).reset_index()
    top_1_results = []
    for model_key in name_map.values():
        model_results = []
        for dataset_name in dataset_names:
            model_dataset_mean = top_1_scores[top_1_scores.model_class.isin([model_key]) & (top_1_scores["dataset_name"] == dataset_name)].groupby("model_id")[["label"]].mean()
            model_mean = model_dataset_mean.mean().rename(dict(label="Top-1 mean"))
            model_std = model_dataset_mean.std().rename(dict(label="Top-1 std"))
            df = pd.concat([model_mean, model_std])
            df["dataset_name"] = dataset_name
            df["model_class"] = model_key
            model_results.append(df)
        model_results = pd.concat(model_results, axis=1).T
        top_1_results.append(model_results)
    top_1_results = pd.concat(top_1_results).melt(id_vars=["dataset_name", "model_class"]).set_index("dataset_name")

    score_results = []
    for model_key in name_map.values():
        model_results = []
        for dataset_name in dataset_names:
            # mean and std of scores for each model class
            auROC_mean = s[s.model_class.isin([model_key]) & (s["dataset_name"] == dataset_name)].groupby("model_id")[["auROC"]].mean().mean().rename(dict(auROC="auROC mean"))
            auROC_std = s[s.model_class.isin([model_key]) & (s["dataset_name"] == dataset_name)].groupby("model_id")[["auROC"]].mean().std().rename(dict(auROC="auROC std"))
            mAP_mean = s[s.model_class.isin([model_key]) & (s["dataset_name"] == dataset_name)].groupby("model_id")[["mAP"]].mean().mean().rename(dict(mAP="mAP mean"))
            mAP_std = s[s.model_class.isin([model_key]) & (s["dataset_name"] == dataset_name)].groupby("model_id")[["mAP"]].mean().std().rename(dict(mAP="mAP std"))
            df = pd.concat([auROC_mean, auROC_std, mAP_mean, mAP_std])
            df["dataset_name"] = dataset_name
            df["model_class"] = model_key
            model_results.append(df)
        model_results = pd.concat(model_results, axis=1).T
        score_results.append(model_results)
    score_results = pd.concat(score_results).melt(id_vars=["dataset_name", "model_class"]).set_index("dataset_name")

    df = pd.concat([score_results, top_1_results]).reset_index()
    df["dataset"] = df["dataset_name"].str.split(" ", expand=True)[0]

    # Evaluate RFCX
    pivot_df = df[df.dataset == "RFCX"].pivot(index='model_class', columns=['dataset_name', 'variable'], values='value')
    means = pivot_df.loc[:, pivot_df.columns.get_level_values(1).str.contains('mean')]
    stds = pivot_df.loc[:, pivot_df.columns.get_level_values(1).str.contains('std')]

    means.columns = [strip_stat(col) for col in means.columns]
    stds.columns = [strip_stat(col) for col in stds.columns]

    combined = means.copy()
    for col in means.columns:
        mean_vals = means[col]
        std_vals = stds[col]
        r = []
        for m, s in zip(mean_vals, std_vals):
            if pd.notna(m) and pd.notna(s):
                value = f"{m:.2f} ({s:.3f})"
            elif pd.notna(m) and not pd.notna(s):
                value = f"{m:.2f}"
            else:
                value = ""
            r.append(value)
        combined[col] = r
    combined.columns = pd.MultiIndex.from_tuples(combined.columns, names=['Dataset', 'Metric'])
    combined = combined.sort_index(axis=1, level=[0, 1])
    combined[combined.isna()] = "-"
    combined = combined.iloc[[-1, -2, -3, -4]]

    with open(save_dir / "RFCX_scores_table.latex", "w+") as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(combined.style.to_latex())
            print(combined.to_markdown())

    # Evaluate SO
    pivot_df = df[df.dataset == "SO"].pivot(index='model_class', columns=['dataset_name', 'variable'], values='value')
    means = pivot_df.loc[:, pivot_df.columns.get_level_values(1).str.contains('mean')]
    stds = pivot_df.loc[:, pivot_df.columns.get_level_values(1).str.contains('std')]


    means.columns = [strip_stat(col) for col in means.columns]
    stds.columns = [strip_stat(col) for col in stds.columns]

    combined = means.copy()
    for col in means.columns:
        mean_vals = means[col]
        std_vals = stds[col]
        r = []
        for m, s in zip(mean_vals, std_vals):
            if pd.notna(m) and pd.notna(s):
                value = f"{m:.2f} ({s:.3f})"
            elif pd.notna(m) and not pd.notna(s):
                value = f"{m:.2f}"
            else:
                value = ""
            r.append(value)
        combined[col] = r
    combined.columns = pd.MultiIndex.from_tuples(combined.columns, names=['Dataset', 'Metric'])
    combined = combined.sort_index(axis=1, level=[0, 1])
    combined[combined.isna()] = "-"
    combined = combined.iloc[[-1, -2, -3, -4]]

    with open(save_dir / "SO_scores_table.latex", "w+") as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(combined.style.to_latex())
            print(combined.to_markdown())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--birdnet-results-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/birdnet/results.parquet",
    )
    parser.add_argument(
        "--birdnet-scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/birdnet/scores.parquet",
    )
    parser.add_argument(
        "--vae-clf-results-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/vae_clf/results.parquet",
    )
    parser.add_argument(
        "--vae-clf-scores-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/vae_clf/scores.parquet",
    )
    parser.add_argument(
        "--label-frequency-path",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/train_label_frequency.parquet",
    )
    parser.add_argument(
        "--save-dir",
        type=lambda p: Path(p),
        required=True,
        help="/path/to/saved/",
    )
    args = parser.parse_args()
    main(**vars(args))
