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
alg = args.alg

results = []

for dataset in sorted(os.listdir(root)):

    dataset_path = os.path.join(root, dataset)

    if not os.path.isdir(dataset_path):
        continue

    if dataset == "plots":
        continue

    csv_files = glob.glob(
        os.path.join(dataset_path, "*.csv")
    )

    if len(csv_files) == 0:
        print(f"Skipping {dataset}")
        continue

    final_accs = []
    final_times = []

    for f in csv_files:

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        # ----------------------------------
        # SelfTrain
        # ----------------------------------
        if "test_acc" in df.columns:

            run_acc = df["test_acc"].mean()

            final_accs.append(run_acc)

        # ----------------------------------
        # FedSSP / FedStar
        # ----------------------------------
        elif "mean_acc" in df.columns:

            final_accs.append(
                df["mean_acc"].iloc[-1]
            )

            if "total_time" in df.columns:
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

print(summary_df)
print(f"\nSaved summary to: {out_path}")