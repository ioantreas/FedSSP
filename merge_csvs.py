import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET="biosncv"
SPEPCTRAL_MODES=["full","identity","topk","chebyshev"]
NUM_RUNS=8

def main():
    last_accuracies_dict = {}
    all_accuracies_dict = {}
    preprocessing_durations_dict = {}
    round_accs_per_mode = {}
    round_stds_per_mode = {}
    for spectral_mode in SPEPCTRAL_MODES:
        last_accuracies = []
        all_accuracies = []
        preprocessing_durations = []

        for i in range(1, NUM_RUNS+1):
            path = f"outputs/raw/{DATASET}/{spectral_mode}/{i}_metrics_fedSSP_rw_dg.csv"

            try:
                with open(path, mode="r") as file:
                    csv_reader = csv.reader(file)
                    seen_rounds = set()
                    _ = next(csv_reader)
                    #count = 0
                    for row in csv_reader:
                        
                        if row[0] in seen_rounds:
                            continue
                        #count += 1
                        seen_rounds.add(row[0])
                        if int(row[0]) == 200:
                            #print("Appending")
                            last_accuracies.append(float(row[1]))
                            preprocessing_durations.append(float(row[5]))
                        all_accuracies.append(float(row[1]))
                    #print(count)
                    #print(last_accuracies)
                    #print(preprocessing_durations)
                    #print(all_accuracies)
            except FileNotFoundError as e:
                print(f"Skipping {e}")
        print(len(all_accuracies))
        count_runs = len(all_accuracies) // 200
        print(count_runs)
        round_accs = []
        round_stds = []
        for round in range(200):
            current_round_accs = []
            for i in range(count_runs):
                current_round_accs.append(all_accuracies[i*200 + round])
            round_accs.append(np.mean(current_round_accs))
            round_stds.append(np.std(current_round_accs))
        


        path=f"outputs/raw/combined_{spectral_mode}.csv"
        with open(path, mode="w", newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["spectral_mode","round","acc-mean","acc-std"])

            # rows = [spectral_mode, last_accuracies_dict[f"{spectral_mode}_mean"], last_accuracies_dict[f"{spectral_mode}_std"],
            #         all_accuracies_dict[f"{spectral_mode}_mean"],all_accuracies_dict[f"{spectral_mode}_std"],
            #         preprocessing_durations_dict[f"{spectral_mode}_mean"],preprocessing_durations_dict[f"{spectral_mode}_std"]]
            # csv_writer.writerow(rows)
            for i in range(200):
                csv_writer.writerow([spectral_mode, i+1, round_accs[i], round_stds[i]])
            #csv_writer.writerows([[spectral_mode]*200, [range(1, 201)], round_accs, round_stds])

        round_accs_per_mode[spectral_mode] = round_accs
        round_stds_per_mode[spectral_mode] = round_stds

        last_accuracies_dict[f"{spectral_mode}_mean"] = np.mean(last_accuracies)
        last_accuracies_dict[f"{spectral_mode}_std"] = np.std(last_accuracies)
        all_accuracies_dict[f"{spectral_mode}_mean"] = np.mean(all_accuracies)
        all_accuracies_dict[f"{spectral_mode}_std"] = np.std(all_accuracies)
        preprocessing_durations_dict[f"{spectral_mode}_mean"] = np.mean(preprocessing_durations)
        preprocessing_durations_dict[f"{spectral_mode}_std"] = np.std(preprocessing_durations)
    #print(last_accuracies_dict)

    path=f"outputs/raw/combined.csv"
    with open(path, mode="w", newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["spectral_mode","last_mean","last_std","all_mean","all_std","pre_mean","pre_std"])

        for spectral_mode in SPEPCTRAL_MODES:
            rows = [spectral_mode, last_accuracies_dict[f"{spectral_mode}_mean"], last_accuracies_dict[f"{spectral_mode}_std"],
                    all_accuracies_dict[f"{spectral_mode}_mean"],all_accuracies_dict[f"{spectral_mode}_std"],
                    preprocessing_durations_dict[f"{spectral_mode}_mean"],preprocessing_durations_dict[f"{spectral_mode}_std"]]
            csv_writer.writerow(rows)
    
    for spectral_mode in SPEPCTRAL_MODES:
        rounds = range(1, 201)
        mean_values = round_accs_per_mode[spectral_mode]
        std_values = round_stds_per_mode[spectral_mode]
        plt.plot(rounds, mean_values, label=spectral_mode)
        plt.fill_between(rounds, np.subtract(mean_values, std_values), np.add(mean_values, std_values), alpha=0.2)
        plt.legend()
    #plt.show()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.xlim(10, 200)
    plt.savefig("results")

if __name__ == "__main__":
    main()