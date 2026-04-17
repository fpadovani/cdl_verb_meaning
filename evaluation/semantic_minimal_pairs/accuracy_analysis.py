import pandas as pd
from pathlib import Path
import re
from itertools import combinations
from scipy.stats import ttest_rel


DATASETS = ["bnc", "candor", "wiki", "cds"]
CONDITIONS = ["o", "sh1", "np", "w"]

CHECKPOINTS = {
    "candor": {"o": "check-5233", "w": "check-5244", "sh1": "check-5233", "np": "check-5233"},
    "cds":    {"o": "check-6088", "w": "check-6100", "sh1": "check-6088", "np": "check-6088"},
    "wiki":   {"o": "check-7960", "w": "check-4973", "sh1": "check-4961", "np": "check-4961"},
    "bnc":    {"o": "check-7800", "w": "check-4879", "sh1": "check-4879", "np": "check-4879"},
}

BASE_FOLDER = "./evaluation/semantic_minimal_pairs/evaluation_scores/new_clean"


def extract_seed(filename):
    match = re.search(r"_(\d+)\.csv$", filename)
    return int(match.group(1)) if match else None


def compute_scores_per_seed(base_folder):
    base_folder = Path(base_folder)
    rows = []

    for dataset in DATASETS:
        dataset_path = base_folder / dataset
        if not dataset_path.exists():
            print(f"Skipping missing folder: {dataset}")
            continue

        for cond in CONDITIONS:
            for f in dataset_path.glob(f"{dataset}_{cond}_*.csv"):
                seed = extract_seed(f.name)
                if seed is None:
                    print(f"Could not extract seed from {f.name}")
                    continue

                df = pd.read_csv(f)
                checkpoint = CHECKPOINTS[dataset][cond]
                df_ckpt = df[df["checkpoint"] == checkpoint]

                if df_ckpt.empty:
                    print(f"No checkpoint {checkpoint} in {f.name}")
                    continue

                bin_cols = [c for c in df_ckpt.columns if c.startswith("bin_")]
                accuracy = df_ckpt[bin_cols].mean(axis=1).iloc[0]

                rows.append({
                    "dataset": dataset,
                    "condition": cond,
                    "seed": seed,
                    "accuracy": accuracy,
                })

    return pd.DataFrame(rows)


def print_summary(df):
    print("\n--- Mean accuracy per dataset × condition (averaged over seeds) ---")
    summary = df.groupby(["dataset", "condition"])["accuracy"].mean()
    for (dataset, cond), val in summary.items():
        print(f"  {dataset:8s} | {cond:4s}: {val:.4f}")


def compute_deltas_and_ttests(df):
    df_pivot = df.pivot(index=["dataset", "seed"], columns="condition", values="accuracy")

    df_delta = (
        df_pivot
        .dropna(subset=["o", "sh1"])
        .assign(delta=lambda x: x["o"] - x["sh1"])
        .reset_index()[["dataset", "seed", "delta"]]
    )
    df_delta.to_csv("delta_o_minus_sh1_per_seed.csv", index=False)
    print(f"\nSaved: delta_o_minus_sh1_per_seed.csv")

    pivot = df_delta.pivot(index="seed", columns="dataset", values="delta")

    results = []
    for d1, d2 in combinations(pivot.columns.tolist(), 2):
        pair = pivot[[d1, d2]].dropna()
        if len(pair) < 2:
            continue
        t_stat, p_val = ttest_rel(pair[d1], pair[d2])
        results.append({"dataset_1": d1, "dataset_2": d2, "n_seeds": len(pair),
                         "t_stat": round(t_stat, 4), "p_value": round(p_val, 4)})

    results_df = pd.DataFrame(results)
    print("\n--- Paired t-tests on delta(o - sh1) across datasets ---")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    df = compute_scores_per_seed(BASE_FOLDER)

    print("\n--- Per-seed scores ---")
    print(df)
    df.to_csv("per_seed_scores.csv", index=False)
    print("Saved: per_seed_scores.csv")

    print_summary(df)
    compute_deltas_and_ttests(df)
