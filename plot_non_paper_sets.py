import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Load all runs for one setting
# --------------------------------------------------

def load_runs(folder, alg):

    files = sorted(
        glob.glob(
            os.path.join(
                folder,
                f"*metrics_{alg}*.csv"
            )
        )
    )

    if len(files) == 0:
        return None

    runs = []

    for f in files:
        try:
            runs.append(pd.read_csv(f))
        except Exception:
            pass

    if len(runs) == 0:
        return None

    return runs


# --------------------------------------------------
# Aggregate over seeds
# --------------------------------------------------

def aggregate(runs):

    rounds = runs[0]["round"].values

    accs = np.stack([
        r["mean_acc"].values
        for r in runs
    ])

    times = np.stack([
        r["total_time"].values
        for r in runs
    ])

    return {
        "rounds": rounds,
        "mean_acc": accs.mean(axis=0),
        "std_acc": accs.std(axis=0),
        "mean_time": times.mean(axis=0)
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "fedssp_root",
        type=str
    )

    parser.add_argument(
        "fedstar_root",
        type=str
    )

    parser.add_argument(
        "--outfile",
        default="fedssp_vs_fedstar.png"
    )

    args = parser.parse_args()

    # -----------------------------
    # Only the 3 settings used
    # -----------------------------

    selected = {
        "biochem": "BIO-SM",
        "biochemsn": "BIO-SM-SN",
        "biosncv": "BIO-SN-CV",
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 7)
    )

    top_titles = [
        "(a) BIO-SM",
        "(b) BIO-SM-SN",
        "(c) BIO-SN-CV"
    ]

    bottom_titles = [
        "(d) BIO-SM",
        "(e) BIO-SM-SN",
        "(f) BIO-SN-CV"
    ]

    for col, (setting, name) in enumerate(selected.items()):

        # ----------------------------------
        # Load data
        # ----------------------------------

        fedssp_runs = load_runs(
            os.path.join(args.fedssp_root, setting),
            "fedSSP"
        )

        fedstar_runs = load_runs(
            os.path.join(args.fedstar_root, setting),
            "fedstar"
        )

        if fedssp_runs is None:
            print(f"Missing FedSSP data for {setting}")
            continue

        if fedstar_runs is None:
            print(f"Missing FedStar data for {setting}")
            continue

        fedssp = aggregate(fedssp_runs)
        fedstar = aggregate(fedstar_runs)

        # ==================================
        # TOP ROW
        # Accuracy vs Round
        # ==================================

        ax = axes[0, col]

        ax.plot(
            fedssp["rounds"],
            fedssp["mean_acc"],
            linewidth=2,
            label="FedSSP"
        )

        ax.fill_between(
            fedssp["rounds"],
            fedssp["mean_acc"] - fedssp["std_acc"],
            fedssp["mean_acc"] + fedssp["std_acc"],
            alpha=0.25
        )

        ax.plot(
            fedstar["rounds"],
            fedstar["mean_acc"],
            linewidth=2,
            label="FedStar"
        )

        ax.fill_between(
            fedstar["rounds"],
            fedstar["mean_acc"] - fedstar["std_acc"],
            fedstar["mean_acc"] + fedstar["std_acc"],
            alpha=0.25
        )

        ax.set_title(
            top_titles[col],
            fontsize=11
        )

        ax.set_xlabel(
            "Communication Round"
        )

        if col == 0:
            ax.set_ylabel(
                "Accuracy"
            )

        ax.grid(True)

        # ==================================
        # BOTTOM ROW
        # Accuracy vs Time
        # ==================================

        ax = axes[1, col]

        ax.plot(
            fedssp["mean_time"],
            fedssp["mean_acc"],
            linewidth=2,
            label="FedSSP"
        )

        ax.fill_between(
            fedssp["mean_time"],
            fedssp["mean_acc"] - fedssp["std_acc"],
            fedssp["mean_acc"] + fedssp["std_acc"],
            alpha=0.25
        )

        ax.plot(
            fedstar["mean_time"],
            fedstar["mean_acc"],
            linewidth=2,
            label="FedStar"
        )

        ax.fill_between(
            fedstar["mean_time"],
            fedstar["mean_acc"] - fedstar["std_acc"],
            fedstar["mean_acc"] + fedstar["std_acc"],
            alpha=0.25
        )

        ax.set_title(
            bottom_titles[col],
            fontsize=11
        )

        ax.set_xlabel(
            "Time (s)"
        )

        if col == 0:
            ax.set_ylabel(
                "Accuracy"
            )

        ax.grid(True)

    # ----------------------------------
    # Shared legend
    # ----------------------------------

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.savefig(
        args.outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved figure to: {args.outfile}"
    )


if __name__ == "__main__":
    main()