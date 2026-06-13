import os
import glob
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("root", type=str)
parser.add_argument("alg", type=str)

args = parser.parse_args()

root = args.root
alg = args.alg.lower()

results = []

for dataset in sorted(os.listdir(root)):

    dataset_path = os.path.join(root, dataset)

    if not os.path.isdir(dataset_path):
        continue

    if dataset == "plots":
        continue

    # --------------------------------------------------
    # Select files belonging to requested algorithm
    # --------------------------------------------------
    if alg == "fedssp":
        accuracy_pattern = "*accuracy_fedSSP*.csv"
        metrics_pattern = "*metrics_fedSSP*.csv"

    elif alg == "fedstar":
        accuracy_pattern = "*accuracy_fedstar*.csv"
        metrics_pattern = "*metrics_fedstar*.csv"

    elif alg == "selftrain":
        accuracy_pattern = "*accuracy_selftrain*.csv"
        metrics_pattern = None

    else:
        raise ValueError(
            f"Unknown algorithm: {alg}"
        )

    accuracy_files = sorted(
        glob.glob(
            os.path.join(
                dataset_path,
                accuracy_pattern
            )
        )
    )

    if metrics_pattern is not None:
        metrics_files = sorted(
            glob.glob(
                os.path.join(
                    dataset_path,
                    metrics_pattern
                )
            )
        )
    else:
        metrics_files = []

    if len(accuracy_files) == 0:
        print(f"Skipping {dataset}")
        continue

    final_accs = []
    final_times = []

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------
    for f in accuracy_files:

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        if "test_acc" not in df.columns:
            continue

        final_accs.append(
            df["test_acc"].mean()
        )

    # --------------------------------------------------
    # Time
    # --------------------------------------------------
    for f in metrics_files:

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        if (
                "total_time" in df.columns
                and len(df) > 0
        ):
            final_times.append(
                df["total_time"].iloc[-1]
            )

    if len(final_accs) == 0:
        continue

    result = {
        "config": dataset,
        "mean_acc": np.mean(final_accs),
        "std_acc": np.std(final_accs),
        "num_runs": len(final_accs)
    }

    if len(final_times) > 0:
        result["mean_time"] = np.mean(final_times)
        result["std_time"] = np.std(final_times)
    else:
        result["mean_time"] = np.nan
        result["std_time"] = np.nan

    results.append(result)

summary_df = pd.DataFrame(results)

summary_df = summary_df.sort_values(
    "mean_acc",
    ascending=False
)

out_path = os.path.join(
    root,
    f"{alg}_summary.csv"
)

summary_df.to_csv(
    out_path,
    index=False
)

print("\n==============================")
print(f"Algorithm: {alg}")
print("==============================")
print(summary_df)

print(
    f"\nSaved summary to: {out_path}"
)