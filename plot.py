import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("root", type=str)
parser.add_argument("alg", type=str)

args = parser.parse_args()

root = args.root
alg = args.alg


def process_folder(folder_path, plot_dir):

    csv_files = sorted(
        glob.glob(
            os.path.join(
                folder_path,
                f"*metrics_{alg}*.csv"
            )
        )
    )

    if len(csv_files) == 0:
        print(f"No CSV files in {folder_path}")
        return

    dfs = []

    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass

    if len(dfs) == 0:
        return

    rounds = dfs[0]["round"].values

    accs = np.stack([
        df["mean_acc"].values
        for df in dfs
    ])

    times = np.stack([
        df["total_time"].values
        for df in dfs
    ])

    mean_acc = accs.mean(axis=0)
    std_acc = accs.std(axis=0)

    mean_time = times.mean(axis=0)

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # Accuracy vs Round
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(
        rounds,
        mean_acc,
        label="Mean Accuracy"
    )

    plt.fill_between(
        rounds,
        mean_acc - std_acc,
        mean_acc + std_acc,
        alpha=0.3,
        label="±1 std"
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title(
        f"{os.path.basename(folder_path)} ({alg})"
    )
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            plot_dir,
            "accuracy_vs_round.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Accuracy vs Time
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(
        mean_time,
        mean_acc,
        label="Mean Accuracy"
    )

    plt.fill_between(
        mean_time,
        mean_acc - std_acc,
        mean_acc + std_acc,
        alpha=0.3,
        label="±1 std"
    )

    plt.xlabel("Wall-clock Time (s)")
    plt.ylabel("Accuracy")
    plt.title(
        f"{os.path.basename(folder_path)} ({alg})"
    )
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            plot_dir,
            "accuracy_vs_time.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved plots to: {plot_dir}"
    )


for dataset in sorted(os.listdir(root)):

    dataset_path = os.path.join(
        root,
        dataset
    )

    if not os.path.isdir(dataset_path):
        continue

    if dataset == "plots":
        continue

    plot_dir = os.path.join(
        root,
        "plots",
        dataset
    )

    process_folder(
        dataset_path,
        plot_dir
    )