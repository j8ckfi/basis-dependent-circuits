"""Generate extended-control figures from completed JSON results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FIGURES_DIR, RESULTS_DIR, ensure_dirs


def load_json(name: str):
    with (RESULTS_DIR / name).open() as f:
        return json.load(f)


def plot_shared_trajectory() -> None:
    data = load_json("shared_trajectory_results.json")
    groups: dict[int, list[dict]] = {}
    for row in data["results"]:
        groups.setdefault(int(row["fork_point"]), []).append(row)

    xs = sorted(groups)
    baseline = [np.mean([r["branch_b_baseline_accuracy"] for r in groups[x]]) for x in xs]
    unaligned = [np.mean([r["unaligned_transplant_accuracy"] for r in groups[x]]) for x in xs]
    procrustes = [np.mean([r["procrustes_transplant_accuracy"] for r in groups[x]]) for x in xs]

    plt.figure(figsize=(7, 4))
    plt.plot(xs, baseline, marker="o", label="Recipient baseline")
    plt.plot(xs, unaligned, marker="o", label="Unaligned transplant")
    plt.plot(xs, procrustes, marker="o", label="Procrustes transplant")
    plt.xlabel("Shared training steps before fork")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shared_trajectory.png", dpi=200)
    plt.close()


def plot_partial_retrain() -> None:
    data = load_json("partial_retrain_rescue_results.json")
    plt.figure(figsize=(7, 4))
    for row in data["results"]:
        xs = [p["k"] for p in row["trajectory"]]
        ys = [p["accuracy"] for p in row["trajectory"]]
        plt.plot(xs, ys, marker="o", label=f"{row['donor_seed']}->{row['recipient_seed']}")
    plt.xscale("symlog", linthresh=100)
    plt.xlabel("Retraining steps K")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "partial_retrain_rescue.png", dpi=200)
    plt.close()


def plot_stitching_adapters() -> None:
    data = load_json("stitching_adapters_results.json")
    labels = [row["condition"] for row in data["conditions"]]
    values = [row["adapter_accuracy"] for row in data["conditions"]]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.axhline(data["baseline_accuracy"], color="black", linestyle="--", linewidth=1, label="Recipient baseline")
    plt.axhline(data["transplant_accuracy"], color="gray", linestyle=":", linewidth=1, label="Bare transplant")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.xticks(rotation=20, ha="right")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stitching_adapters.png", dpi=200)
    plt.close()


def plot_cross_task() -> None:
    data = load_json("cross_task_transplant_results.json")
    labels = ["Recipient baseline", "Cross-task transplant"]
    values = [data["baseline_accuracy"], data["cross_task_transplant_accuracy"]]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_task_transplant.png", dpi=200)
    plt.close()


def plot_shared_basis() -> None:
    data = load_json("shared_basis_training_results.json")
    rows = sorted(data["results"], key=lambda row: row["lambda"])
    xs = [row["lambda"] for row in rows]
    cka_l0 = [row["cka"]["post_l0"] for row in rows]
    cka_l1 = [row["cka"]["post_l1"] for row in rows]
    unaligned = [row["unaligned_transplant_accuracy"] for row in rows]
    procrustes = [row["procrustes_transplant_accuracy"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, cka_l0, marker="o", label="CKA post-L0")
    ax1.plot(xs, cka_l1, marker="o", label="CKA post-L1")
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_xlabel("Alignment loss weight")
    ax1.set_ylabel("CKA")
    ax1.set_ylim(0, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(xs, unaligned, marker="s", linestyle="--", color="tab:red", label="Unaligned")
    ax2.plot(xs, procrustes, marker="s", linestyle=":", color="tab:purple", label="Procrustes")
    ax2.set_ylabel("Transplant accuracy")
    ax2.set_ylim(0, 1.02)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shared_basis_training.png", dpi=200)
    plt.close(fig)


def plot_pythia_scale() -> None:
    data = load_json("pythia_scale_portability_results.json")
    labels = [row["recipient"].replace("EleutherAI/pythia-", "") for row in data["pairs"]]
    baseline = [row["recipient_baseline_accuracy"] for row in data["pairs"]]
    zero = [row["zero_head_accuracy"] for row in data["pairs"]]
    transplant = [row["unaligned_transplant_accuracy"] for row in data["pairs"]]
    x = np.arange(len(labels))

    plt.figure(figsize=(9, 4))
    plt.scatter(x, baseline, label="Recipient baseline", s=24)
    plt.scatter(x, zero, label="Zero selected head", s=24)
    plt.scatter(x, transplant, label="Unaligned transplant", s=24)
    plt.ylabel("Accuracy")
    plt.ylim(0, max(0.2, max(baseline) + 0.04))
    plt.xticks(x, labels, rotation=70, ha="right", fontsize=7)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pythia_scale_portability.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    plot_shared_trajectory()
    plot_partial_retrain()
    plot_stitching_adapters()
    plot_cross_task()
    plot_shared_basis()
    plot_pythia_scale()
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
