import csv
import numpy as np

DATASET="biosncv"
SPEPCTRAL_MODES=["full","identity","topk","chebyshev"]
NUM_RUNS=8

def main():
    last_accuracies_dict = {}
    all_accuracies_dict = {}
    preprocessing_durations_dict = {}
    for spectral_mode in SPEPCTRAL_MODES:
        last_accuracies = []
        all_accuracies = []
        preprocessing_durations = []

        for i in range(1, NUM_RUNS+1):
            path = f"outputs/raw/{DATASET}/{spectral_mode}/{i}_metrics_fedSSP_rw_dg.csv"

            try:
                with open(path, mode="r") as file:
                    csv_reader = csv.reader(file)
                    _ = next(csv_reader)

                    for row in csv_reader:
                        if int(row[0]) == 200:
                            #print("Appending")
                            last_accuracies.append(float(row[1]))
                            preprocessing_durations.append(float(row[5]))
                        all_accuracies.append(float(row[1]))

                    print(last_accuracies)
                    print(preprocessing_durations)
                    #print(all_accuracies)
            except FileNotFoundError as e:
                print(f"Skipping {e}")

        last_accuracies_dict[f"{spectral_mode}_mean"] = np.mean(last_accuracies)
        last_accuracies_dict[f"{spectral_mode}_std"] = np.std(last_accuracies)
        all_accuracies_dict[f"{spectral_mode}_mean"] = np.mean(all_accuracies)
        all_accuracies_dict[f"{spectral_mode}_std"] = np.std(all_accuracies)
        preprocessing_durations_dict[f"{spectral_mode}_mean"] = np.mean(preprocessing_durations)
        preprocessing_durations_dict[f"{spectral_mode}_std"] = np.std(preprocessing_durations)
    print(last_accuracies_dict)

    path=f"outputs/raw/combined.csv"
    with open(path, mode="w", newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["spectral_mode","last_mean","last_std","all_mean","all_std","pre_mean","pre_std"])

        for spectral_mode in SPEPCTRAL_MODES:
            rows = [spectral_mode, last_accuracies_dict[f"{spectral_mode}_mean"], last_accuracies_dict[f"{spectral_mode}_std"],
                    all_accuracies_dict[f"{spectral_mode}_mean"],all_accuracies_dict[f"{spectral_mode}_std"],
                    preprocessing_durations_dict[f"{spectral_mode}_mean"],preprocessing_durations_dict[f"{spectral_mode}_std"]]
            csv_writer.writerow(rows)

if __name__ == "__main__":
    main()