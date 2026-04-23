"""
Experiment A: Per-example agreement across 20 induction networks.

Question: Do 20 networks that hit ~96% accuracy compute the same function,
or just agree in aggregate?

Protocol:
1. Build fixed eval set (1024 sequences, seed=9999)
2. For each of 20 final checkpoints, run forward pass and record argmax predictions
   at induction-masked positions
3. Compute all 190 pairwise agreements (fraction where pred_i == pred_j)
4. Compare to chance-agreement baseline
5. Cross-reference with shared critical-head identity

Output:
- analysis/per_example_agreement_results.json
- analysis/plots/per_example_agreement.png

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/per_example_agreement.py
"""

import sys
import json
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from src.model import GPT, GPTConfig
from analysis.transplant_unified_eval import (
    build_fixed_eval_set,
    load_model,
    MODEL_CONFIG,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
CKPT_DIR = BASE_DIR / "checkpoints" / "induction_content_match"
N_SEEDS = 20
RESULTS_PATH = Path(__file__).parent / "per_example_agreement_results.json"
PLOT_PATH = Path(__file__).parent / "plots" / "per_example_agreement.png"
SEEDS_JSON = Path(__file__).parent / "content_matching_20_seeds_results_v2.json"


def checkpoint_path(seed: int) -> Path:
    """Return path to final checkpoint for a given seed."""
    if seed == 2:
        step = "step_030000"
    else:
        step = "step_020000"
    return CKPT_DIR / f"induction_seed{seed}" / step / "model.safetensors"


# ---------------------------------------------------------------------------
# Per-position prediction extraction
# ---------------------------------------------------------------------------

def collect_predictions(model: GPT, fixed_eval_set) -> tuple:
    """Run forward pass on all batches, collect predictions and mask info.

    Returns:
        preds_flat: np.ndarray shape (N_total_masked_positions,) — argmax token IDs
        targets_flat: np.ndarray shape (N_total_masked_positions,) — ground truth
        A parallel pair so positions align across seeds (same eval set).

    Actually we return per-batch flat arrays for position-aligned comparison.
    We need all seeds to share the same position indexing, so we flatten all
    induction-masked positions across the whole eval set in order.
    """
    all_preds = []
    all_targets = []

    for inputs_np, targets_np, mask_np in fixed_eval_set:
        inputs_mx = mx.array(inputs_np)
        logits = model(inputs_mx)
        mx.eval(logits)

        logits_np = np.array(logits)
        preds = logits_np.argmax(axis=-1)   # (B, T)
        at_mask = mask_np > 0.5             # (B, T)

        # Flatten masked positions in order (row-major)
        all_preds.append(preds[at_mask])
        all_targets.append(targets_np[at_mask])

    preds_flat = np.concatenate(all_preds)
    targets_flat = np.concatenate(all_targets)
    return preds_flat, targets_flat


def count_masked_positions(fixed_eval_set) -> int:
    """Count total number of induction-masked positions."""
    total = 0
    for _, _, mask_np in fixed_eval_set:
        total += (mask_np > 0.5).sum()
    return int(total)


# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------

def pairwise_agreement(preds_i: np.ndarray, preds_j: np.ndarray,
                       targets: np.ndarray) -> dict:
    """Compute agreement statistics between two seeds at masked positions."""
    agree = preds_i == preds_j
    correct_i = preds_i == targets
    correct_j = preds_j == targets

    # Fraction where both predict same token (target-agnostic)
    frac_agree = float(agree.mean())

    # Fraction where both correct
    both_correct = correct_i & correct_j
    frac_both_correct = float(both_correct.mean())

    # Fraction where i correct, j wrong
    frac_i_only = float((correct_i & ~correct_j).mean())

    # Fraction where j correct, i wrong
    frac_j_only = float((~correct_i & correct_j).mean())

    # Fraction where both wrong
    both_wrong = ~correct_i & ~correct_j
    frac_both_wrong = float(both_wrong.mean())

    # Among both-wrong, fraction that agree (same wrong token)
    n_both_wrong = int(both_wrong.sum())
    if n_both_wrong > 0:
        frac_wrong_agree = float((agree & both_wrong).sum() / n_both_wrong)
    else:
        frac_wrong_agree = float("nan")

    return {
        "frac_agree": frac_agree,
        "frac_both_correct": frac_both_correct,
        "frac_i_only_correct": frac_i_only,
        "frac_j_only_correct": frac_j_only,
        "frac_both_wrong": frac_both_wrong,
        "frac_wrong_and_agree": frac_wrong_agree,
        "n_both_wrong": n_both_wrong,
    }


# ---------------------------------------------------------------------------
# Chance-agreement baseline
# ---------------------------------------------------------------------------

def chance_agreement_baseline(acc_i: float, acc_j: float, vocab_size: int = 512) -> float:
    """Naive chance agreement assuming errors distributed uniformly over vocab.

    If both correct with probs p_i, p_j — they agree on the target token.
    If both wrong (1-p_i)*(1-p_j) — they agree on a random wrong token
    with prob 1/(vocab_size - 1) ≈ 1/vocab_size.
    """
    both_correct = acc_i * acc_j
    both_wrong = (1 - acc_i) * (1 - acc_j)
    wrong_agree = both_wrong / (vocab_size - 1)
    return both_correct + wrong_agree


# ---------------------------------------------------------------------------
# Critical-head cross-reference
# ---------------------------------------------------------------------------

def load_critical_head_data() -> dict:
    """Load per-seed critical head info from the 20-seed results JSON."""
    with open(SEEDS_JSON) as f:
        data = json.load(f)
    # Build seed -> critical head (layer, index) mapping
    head_map = {}
    for entry in data["per_seed"]:
        seed = entry["seed"]
        head_map[seed] = (entry["critical_head_layer"], entry["critical_head_index"])
    return head_map


def shared_critical_head(head_map: dict, seed_i: int, seed_j: int) -> bool:
    """Return True if both seeds use the same (layer, head) as critical head."""
    return head_map.get(seed_i) == head_map.get(seed_j)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT A: Per-example agreement across 20 seeds")
    print("=" * 70)

    # Step 1: Build fixed eval set
    print("\n[Step 1] Building fixed eval set...")
    fixed_eval_set = build_fixed_eval_set()
    n_masked = count_masked_positions(fixed_eval_set)
    print(f"  Total induction-masked positions: {n_masked}")

    # Step 2: Load per-seed critical head data
    print("\n[Step 2] Loading per-seed critical head data...")
    head_map = load_critical_head_data()
    print(f"  Critical heads: { {s: f'L{l}H{h}' for s, (l, h) in head_map.items()} }")

    # Step 3: Run forward passes for all 20 seeds
    print("\n[Step 3] Collecting predictions for all 20 seeds...")
    seed_preds = {}      # seed -> preds_flat np.ndarray
    seed_targets = None  # shared across all seeds (same eval set)
    seed_accuracies = {} # seed -> scalar accuracy

    for seed in range(N_SEEDS):
        ckpt = checkpoint_path(seed)
        if not ckpt.exists():
            print(f"  [WARNING] Seed {seed}: checkpoint not found at {ckpt}, skipping")
            continue

        print(f"  Loading seed {seed} ({ckpt.name} in {ckpt.parent.name})...", end=" ", flush=True)
        model = load_model(str(ckpt), MODEL_CONFIG)
        preds, targets = collect_predictions(model, fixed_eval_set)

        if seed_targets is None:
            seed_targets = targets
        else:
            # Sanity check: targets must be identical for all seeds
            assert np.array_equal(targets, seed_targets), \
                f"Target mismatch for seed {seed} — eval set is not deterministic!"

        acc = float((preds == targets).mean())
        seed_accuracies[seed] = acc
        seed_preds[seed] = preds

        print(f"acc={acc:.4f}, n_positions={len(preds)}")

    available_seeds = sorted(seed_preds.keys())
    print(f"\n  Available seeds: {available_seeds} ({len(available_seeds)} total)")

    # Step 4: Pairwise agreement for all 190 pairs
    print("\n[Step 4] Computing pairwise agreement for all pairs...")
    pairs = list(itertools.combinations(available_seeds, 2))
    print(f"  Total pairs: {len(pairs)}")

    pairwise_results = []
    frac_agree_vals = []
    chance_baseline_vals = []
    shared_head_vals = []

    for idx, (i, j) in enumerate(pairs):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Pair {idx+1}/{len(pairs)}: seed {i} vs seed {j}")

        stats = pairwise_agreement(seed_preds[i], seed_preds[j], seed_targets)
        chance = chance_agreement_baseline(
            seed_accuracies[i], seed_accuracies[j], vocab_size=512
        )
        same_head = shared_critical_head(head_map, i, j)

        entry = {
            "seed_i": i,
            "seed_j": j,
            "acc_i": seed_accuracies[i],
            "acc_j": seed_accuracies[j],
            "shared_critical_head": same_head,
            "critical_head_i": f"L{head_map[i][0]}H{head_map[i][1]}" if i in head_map else None,
            "critical_head_j": f"L{head_map[j][0]}H{head_map[j][1]}" if j in head_map else None,
            "chance_agreement_baseline": chance,
            **stats,
        }
        pairwise_results.append(entry)
        frac_agree_vals.append(stats["frac_agree"])
        chance_baseline_vals.append(chance)
        shared_head_vals.append(int(same_head))

    frac_agree_arr = np.array(frac_agree_vals)
    chance_arr = np.array(chance_baseline_vals)
    shared_arr = np.array(shared_head_vals, dtype=bool)

    # Step 5: Summary statistics
    print("\n[Step 5] Computing summary statistics...")

    per_seed_baselines = {str(s): seed_accuracies[s] for s in available_seeds}

    mean_agree = float(frac_agree_arr.mean())
    std_agree = float(frac_agree_arr.std())
    min_agree = float(frac_agree_arr.min())
    max_agree = float(frac_agree_arr.max())

    mean_chance = float(chance_arr.mean())

    # Agreement split by shared vs different critical head
    if shared_arr.sum() > 0:
        mean_agree_shared = float(frac_agree_arr[shared_arr].mean())
        std_agree_shared = float(frac_agree_arr[shared_arr].std())
        n_shared = int(shared_arr.sum())
    else:
        mean_agree_shared = float("nan")
        std_agree_shared = float("nan")
        n_shared = 0

    if (~shared_arr).sum() > 0:
        mean_agree_diff = float(frac_agree_arr[~shared_arr].mean())
        std_agree_diff = float(frac_agree_arr[~shared_arr].std())
        n_diff = int((~shared_arr).sum())
    else:
        mean_agree_diff = float("nan")
        std_agree_diff = float("nan")
        n_diff = 0

    # Correlation: agreement vs min accuracy of pair
    min_acc_arr = np.array([min(r["acc_i"], r["acc_j"]) for r in pairwise_results])
    corr_acc = float(np.corrcoef(min_acc_arr, frac_agree_arr)[0, 1])

    # Correlation: agreement vs shared_critical_head (point-biserial)
    if shared_arr.sum() > 0 and (~shared_arr).sum() > 0:
        corr_head = float(np.corrcoef(shared_arr.astype(float), frac_agree_arr)[0, 1])
    else:
        corr_head = float("nan")

    summary = {
        "n_seeds": len(available_seeds),
        "n_pairs": len(pairs),
        "n_masked_positions": n_masked,
        "mean_pairwise_agreement": mean_agree,
        "std_pairwise_agreement": std_agree,
        "min_pairwise_agreement": min_agree,
        "max_pairwise_agreement": max_agree,
        "mean_chance_baseline": mean_chance,
        "mean_above_chance_pp": (mean_agree - mean_chance) * 100,
        "n_pairs_shared_critical_head": n_shared,
        "n_pairs_diff_critical_head": n_diff,
        "mean_agreement_shared_head_pairs": mean_agree_shared,
        "std_agreement_shared_head_pairs": std_agree_shared,
        "mean_agreement_diff_head_pairs": mean_agree_diff,
        "std_agreement_diff_head_pairs": std_agree_diff,
        "correlation_agreement_vs_min_accuracy": corr_acc,
        "correlation_agreement_vs_shared_head": corr_head,
    }

    print(f"\n  Mean pairwise agreement : {mean_agree*100:.2f}%")
    print(f"  SD                      : {std_agree*100:.2f}%")
    print(f"  Min / Max               : {min_agree*100:.2f}% / {max_agree*100:.2f}%")
    print(f"  Mean chance baseline    : {mean_chance*100:.2f}%")
    print(f"  Difference (above chance): {(mean_agree - mean_chance)*100:.2f} pp")
    print(f"\n  Shared critical head pairs ({n_shared}): {mean_agree_shared*100:.2f}% +/- {std_agree_shared*100:.2f}%")
    print(f"  Diff  critical head pairs ({n_diff}): {mean_agree_diff*100:.2f}% +/- {std_agree_diff*100:.2f}%")
    print(f"\n  Corr(agreement, min_acc)  : {corr_acc:.4f}")
    print(f"  Corr(agreement, same_head): {corr_head:.4f}")

    # Step 6: Save JSON
    print("\n[Step 6] Saving results JSON...")
    results = {
        "experiment": "per_example_agreement",
        "question": "Do 20 networks at ~96% accuracy compute the same function, or just agree in aggregate?",
        "eval_protocol": {
            "n_sequences": 1024,
            "data_seed": 9999,
            "vocab_size": 512,
            "seq_len": 64,
            "n_bigrams": 6,
            "n_masked_positions": n_masked,
        },
        "per_seed_baselines": per_seed_baselines,
        "pairwise_agreements": pairwise_results,
        "summary": summary,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {RESULTS_PATH}")

    # Step 7: Plot
    print("\n[Step 7] Generating plots...")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Plot 1: Histogram of pairwise agreements ---
    ax = axes[0]
    ax.hist(frac_agree_arr * 100, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean_agree * 100, color="firebrick", linewidth=2, linestyle="--",
               label=f"Mean: {mean_agree*100:.1f}%")
    ax.axvline(mean_chance * 100, color="darkorange", linewidth=2, linestyle=":",
               label=f"Chance: {mean_chance*100:.1f}%")
    ax.set_xlabel("Pairwise Agreement (%)")
    ax.set_ylabel("Number of Pairs")
    ax.set_title(f"Pairwise Agreement Distribution\n(n={len(pairs)} pairs, {n_masked} positions)")
    ax.legend(fontsize=9)

    # --- Plot 2: Agreement vs min accuracy of pair ---
    ax = axes[1]
    colors_head = np.where(shared_arr, "steelblue", "darkorange")
    ax.scatter(min_acc_arr * 100, frac_agree_arr * 100,
               c=colors_head, alpha=0.5, s=20, edgecolors="none")

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label=f"Same critical head (n={n_shared})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange",
               markersize=8, label=f"Diff critical head (n={n_diff})"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("min(acc_i, acc_j) (%)")
    ax.set_ylabel("Pairwise Agreement (%)")
    ax.set_title(f"Agreement vs Min Accuracy\n(r={corr_acc:.3f})")

    # --- Plot 3: Agreement by shared critical head (box plot) ---
    ax = axes[2]
    data_shared = frac_agree_arr[shared_arr] * 100 if n_shared > 0 else np.array([])
    data_diff = frac_agree_arr[~shared_arr] * 100 if n_diff > 0 else np.array([])

    bp = ax.boxplot(
        [d for d in [data_diff, data_shared] if len(d) > 0],
        tick_labels=[lbl for d, lbl in [(data_diff, "Diff head"), (data_shared, "Same head")]
                     if len(d) > 0],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    colors_box = ["darkorange", "steelblue"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Pairwise Agreement (%)")
    ax.set_title(f"Agreement by Critical Head Identity\n(r={corr_head:.3f})")

    # Add mean labels
    if n_diff > 0:
        ax.text(1, data_diff.mean() + 0.2, f"{data_diff.mean():.1f}%",
                ha="center", fontsize=9, color="darkorange", fontweight="bold")
    if n_shared > 0:
        offset = 1 if n_diff == 0 else 2
        ax.text(offset, data_shared.mean() + 0.2, f"{data_shared.mean():.1f}%",
                ha="center", fontsize=9, color="steelblue", fontweight="bold")

    plt.suptitle(
        f"Per-Example Agreement: 20 Induction Networks\n"
        f"Mean agreement={mean_agree*100:.1f}%  Chance={mean_chance*100:.1f}%  "
        f"Above-chance={((mean_agree - mean_chance)*100):.1f} pp",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {PLOT_PATH}")

    # Final summary line
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mean pairwise agreement = {mean_agree*100:.1f}%")
    print(f"  Chance baseline         = {mean_chance*100:.1f}%")
    print(f"  Difference              = {(mean_agree - mean_chance)*100:.1f} pp")
    print(f"  Shared-head pairs       = {mean_agree_shared*100:.1f}% (n={n_shared})")
    print(f"  Diff-head pairs         = {mean_agree_diff*100:.1f}% (n={n_diff})")
    print("=" * 70)


if __name__ == "__main__":
    main()
