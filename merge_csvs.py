import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATASET = ["chem", "biochem", "chemcv", "biochemsn", "biosncv", "chemsncv"]
SPEPCTRAL_MODES = ["identity"]
NUM_RUNS = 5
NUM_ROUNDS = 200


def _as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _read_metrics_file(path):
    last_accuracies = []
    all_accuracies = []
    preprocessing_durations = []

    try:
        with open(path, mode="r") as file:
            csv_reader = csv.reader(file)
            seen_rounds = set()
            next(csv_reader, None)

            for row in csv_reader:
                if not row:
                    continue

                round_id = row[0]
                if round_id in seen_rounds:
                    continue

                seen_rounds.add(round_id)
                all_accuracies.append(float(row[1]))

                if int(round_id) == NUM_ROUNDS:
                    last_accuracies.append(float(row[1]))
                    preprocessing_durations.append(float(row[5]))
    except FileNotFoundError as e:
        print(f"Skipping {e}")

    return last_accuracies, all_accuracies, preprocessing_durations


def _mean_and_std(values):
    return float(np.mean(values)), float(np.std(values))


def main():
    datasets = _as_list(DATASET)

    round_rows = []
    summary_rows = []
    round_accs_per_dataset = {}
    round_stds_per_dataset = {}

    for dataset in datasets:
        dataset_round_accs = {}
        dataset_round_stds = {}
        dataset_last_accuracies = {}

        for spectral_mode in SPEPCTRAL_MODES:
            last_accuracies = []
            all_accuracies = []
            preprocessing_durations = []

            for run_idx in range(1, NUM_RUNS + 1):
                path = f"outputs/raw/{dataset}/{spectral_mode}/{run_idx}_metrics_fedSSP_rw_dg.csv"
                run_last, run_all, run_pre = _read_metrics_file(path)
                last_accuracies.extend(run_last)
                all_accuracies.extend(run_all)
                preprocessing_durations.extend(run_pre)

            if len(all_accuracies) == 0:
                print(f"No data for {dataset} / {spectral_mode}")
                continue

            count_runs = len(all_accuracies) // NUM_ROUNDS
            if count_runs == 0:
                print(f"Not enough data for {dataset} / {spectral_mode}")
                continue

            round_accs = []
            round_stds = []
            for round_idx in range(NUM_ROUNDS):
                current_round_accs = [
                    all_accuracies[run_idx * NUM_ROUNDS + round_idx]
                    for run_idx in range(count_runs)
                ]
                round_accs.append(float(np.mean(current_round_accs)))
                round_stds.append(float(np.std(current_round_accs)))

            dataset_round_accs[spectral_mode] = round_accs
            dataset_round_stds[spectral_mode] = round_stds
            dataset_last_accuracies[spectral_mode] = last_accuracies

            round_rows.extend([
                [dataset, spectral_mode, round_idx + 1, round_accs[round_idx], round_stds[round_idx]]
                for round_idx in range(NUM_ROUNDS)
            ])

            last_mean, last_std = _mean_and_std(last_accuracies)
            all_mean, all_std = _mean_and_std(all_accuracies)
            pre_mean, pre_std = _mean_and_std(preprocessing_durations)
            summary_rows.append([
                dataset,
                spectral_mode,
                last_mean,
                last_std,
                all_mean,
                all_std,
                pre_mean,
                pre_std,
            ])

        kruskal_values = [np.atleast_1d(dataset_last_accuracies.get(mode)) for mode in SPEPCTRAL_MODES if dataset_last_accuracies.get(mode) is not None]
        if len(kruskal_values) == len(SPEPCTRAL_MODES) and all(len(values) > 0 for values in kruskal_values):
            try:
                kruskal_test = stats.kruskal(*kruskal_values)
                print(dataset, kruskal_test)
            except Exception as e:
                print(f"Skipping Kruskal for {dataset}: {e}")

        if dataset_round_accs:
            round_accs_per_dataset[dataset] = dataset_round_accs
            round_stds_per_dataset[dataset] = dataset_round_stds

    with open("outputs/raw/combined_rounds.csv", mode="w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["dataset", "spectral_mode", "round", "acc-mean", "acc-std"])
        csv_writer.writerows(round_rows)

    with open("outputs/raw/combined_summary.csv", mode="w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["dataset", "spectral_mode", "last_mean", "last_std", "all_mean", "all_std", "pre_mean", "pre_std"])
        csv_writer.writerows(summary_rows)

    plottable_datasets = list(round_accs_per_dataset.keys())
    if len(plottable_datasets) == 0:
        return

    ncols = 1 if len(plottable_datasets) == 1 else min(2, len(plottable_datasets))
    nrows = math.ceil(len(plottable_datasets) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, dataset in enumerate(plottable_datasets):
        ax = axes[idx // ncols][idx % ncols]
        for spectral_mode in SPEPCTRAL_MODES:
            if spectral_mode not in round_accs_per_dataset[dataset]:
                continue

            rounds = range(1, NUM_ROUNDS + 1)
            mean_values = round_accs_per_dataset[dataset][spectral_mode]
            std_values = round_stds_per_dataset[dataset][spectral_mode]
            ax.plot(rounds, mean_values, label=spectral_mode)
            ax.fill_between(rounds, np.subtract(mean_values, std_values), np.add(mean_values, std_values), alpha=0.2)

        ax.set_title(dataset)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(10, NUM_ROUNDS)
        ax.set_ylim(0.65, 0.8)
        ax.legend()

    for idx in range(len(plottable_datasets), nrows * ncols):
        fig.delaxes(axes[idx // ncols][idx % ncols])

    fig.tight_layout()
    fig.savefig("results_full.png")
    fig.savefig("results_cropped.png")


if __name__ == "__main__":
    main()
