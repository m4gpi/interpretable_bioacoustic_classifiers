import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

print("stuff")
path = 'results/species_detectors/'

scores_file = path + 'test_scores.parquet'
results_file = path + 'test_results.parquet'

scores_df = pd.read_parquet(scores_file)
results_df = pd.read_parquet(results_file)

print("Scores (top 5 rows):")
print(scores_df.head())

print("\nResults (top 5 rows):")
print(results_df.head())

species_list = scores_df[scores_df["model"] == "birdnet"].species_name.tolist()

def get_model_performance_vector(model, subset):
    results = {}
    for species in species_list:
        filtered = scores_df[
            (scores_df['species_name'] == species) &
            (scores_df['model'] == model) &
            (scores_df['scope'] == subset)
        ]
        if not filtered.empty:
            arr = filtered[['AP', 'auROC']].to_numpy().mean(axis=0)
            results[species] = arr
    results_array = np.stack([results[species] for species in results.keys()], axis=0)
    return results, results_array

for scope in ["SO_UK", "SO_EC", "RFCX_bird"]:
    performance, perf_array = get_model_performance_vector("nifti_vae", scope)
    performance, perf_array2 = get_model_performance_vector("base_vae", scope)

    # Perform Wilcoxon signed-rank test between the two arrays (paired by species)
    stat, p_value = wilcoxon(perf_array[:, 0], perf_array2[:, 0])  # mAP
    print(scope)
    print(f"Wilcoxon test for mAP: statistic={stat}, p-value={p_value}")
    stat, p_value = wilcoxon(perf_array[:, 1], perf_array2[:, 1])  # auROC
    print(f"Wilcoxon test for auROC: statistic={stat}, p-value={p_value}")

    score_diff_mean = (perf_array - perf_array2).mean(axis=0)
    print(f"AP: {score_diff_mean[0]}")
    print(f"auROC diff: {score_diff_mean[1]}")
    print()

exit()
