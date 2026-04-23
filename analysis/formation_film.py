"""
Circuit Formation Film: Developmental trajectory across 20 seeds.

QUESTION: Do all seeds follow the same DEVELOPMENTAL TRAJECTORY
(same order of circuit component formation) even if they end up
with different head assignments?

MODEL: SHORT-training induction model (2000 steps, positional shortcut exists)
CONFIG: GPTConfig(n_layers=4, n_heads=4, d_model=128, d_ff=512, vocab_size=50, ctx_len=64)
DATA:   InductionDataset(vocab_size=50, seq_len=64)
"""

import sys
import json
import time
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import InductionDataset
from src.circuit_mapping import profile_all_heads

# ============================================================
# Config
# ============================================================

MODEL_CONFIG = GPTConfig(
    n_layers=4,
    n_heads=4,
    d_model=128,
    d_ff=512,
    vocab_size=50,
    ctx_len=64,
)

N_SEEDS = 20
N_SAMPLES = 32          # per-checkpoint profiling batch size (fast)
N_PERM_SAMPLES = 32     # for permutation sensitivity
N_PERMUTATIONS = 4      # fewer permutations for speed

CKPT_BASE = Path(__file__).parent.parent / "checkpoints" / "seed_divergence"
PLOTS_DIR = Path(__file__).parent / "plots"
RESULTS_PATH = Path(__file__).parent / "formation_film_results.json"

# Crystallization thresholds — calibrated to what this positional-shortcut model achieves.
# Peak values across seeds: copy_score up to ~1.1, prev_token ~0.13, induction ~0.05,
# perm_sensitivity ~0.28, task_accuracy ~0.08. Thresholds set at ~50th percentile of peak.
THRESHOLDS = {
    "copy_score":       0.40,   # 50th pct of peak; rises early from OV diagonal formation
    "prev_token_score": 0.08,   # modest but real prev-token specialization
    "induction_score":  0.035,  # weak induction signal (positional shortcut dominates)
    "perm_sensitivity": 0.18,   # content-sensitivity threshold
    "task_accuracy":    0.05,   # 5% accuracy on induction positions (above chance ~2%)
}


# ============================================================
# Per-checkpoint metrics
# ============================================================

def compute_induction_score(model: GPT, dataset: InductionDataset, n_samples: int) -> np.ndarray:
    """Induction score per head: attention to the position where the current token previously appeared.

    For the repeated-bigram task, the second occurrence of token A should attend back
    to the first occurrence of A. We proxy this as:
      mean attention at position (pos2) to position (pos1) where A first appeared.

    Since this requires knowing which positions are induction positions, we use a
    tractable approximation: for each sequence, find bigram repetitions and check
    if the relevant attention head looks back at the matching token.

    Returns: (n_layers, n_heads) array of induction scores.
    """
    n_layers = len(model.blocks)
    n_heads = model.config.n_heads
    scores = np.zeros((n_layers, n_heads))
    count = 0

    rng = np.random.default_rng(7)

    batch = dataset.generate_batch(n_samples)
    inputs_np = np.array(batch[0])   # (B, T)
    mask_np   = np.array(batch[2])   # (B, T) — 1 at induction positions

    _ = model(mx.array(inputs_np))
    patterns = model.get_attention_patterns()  # list of (B, n_heads, T, T)

    for b in range(inputs_np.shape[0]):
        seq = inputs_np[b]   # (T,)
        ind_positions = np.where(mask_np[b] > 0)[0]

        for pos2 in ind_positions:
            # Find the most recent prior occurrence of seq[pos2] before pos2
            token = seq[pos2]
            prior = np.where(seq[:pos2] == token)[0]
            if len(prior) == 0:
                continue
            pos1 = prior[-1]  # most recent prior occurrence

            for layer_idx in range(n_layers):
                attn = np.array(patterns[layer_idx])  # (B, n_heads, T, T)
                # attn[b, h, pos2, pos1] = attention weight from pos2 to pos1
                for h in range(n_heads):
                    scores[layer_idx, h] += attn[b, h, pos2, pos1]
            count += 1

    if count > 0:
        scores /= count
    return scores


def compute_permutation_sensitivity_fast(
    model: GPT,
    dataset: InductionDataset,
    n_samples: int,
    n_permutations: int,
    seed: int = 0,
) -> np.ndarray:
    """Fast permutation sensitivity: (n_layers, n_heads)."""
    rng = np.random.default_rng(seed)
    vocab_size = model.config.vocab_size
    n_layers = len(model.blocks)
    n_heads = model.config.n_heads

    batch = dataset.generate_batch(n_samples)
    inputs_orig = np.array(batch[0])

    _ = model(mx.array(inputs_orig))
    orig_patterns = [np.array(p) for p in model.get_attention_patterns()]

    total_l2 = np.zeros((n_layers, n_heads), dtype=np.float64)

    for _ in range(n_permutations):
        pi = rng.permutation(vocab_size).astype(np.int32)
        inputs_perm = pi[inputs_orig]

        _ = model(mx.array(inputs_perm))
        perm_patterns = [np.array(p) for p in model.get_attention_patterns()]

        for layer_idx in range(n_layers):
            orig = orig_patterns[layer_idx]
            perm = perm_patterns[layer_idx]
            diff = orig - perm
            l2 = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=(0, 2))
            total_l2[layer_idx] += l2

    return total_l2 / n_permutations


def compute_task_accuracy(model: GPT, dataset: InductionDataset, n_samples: int) -> float:
    """Accuracy at induction positions (second occurrence of bigram)."""
    batch = dataset.generate_batch(n_samples)
    inputs, targets, mask = batch

    logits = model(inputs)  # (B, T, V)
    preds = np.array(mx.argmax(logits, axis=-1))  # (B, T)
    targets_np = np.array(targets)
    mask_np = np.array(mask)

    correct = ((preds == targets_np) * mask_np).sum()
    total = mask_np.sum()
    if total == 0:
        return 0.0
    return float(correct / total)


def extract_metrics_at_checkpoint(
    ckpt_path: Path,
    dataset: InductionDataset,
    n_samples: int = N_SAMPLES,
    n_perm_samples: int = N_PERM_SAMPLES,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 0,
) -> dict:
    """Load checkpoint and compute all per-head metrics.

    Returns dict with keys: induction_score, prev_token_score, copy_score,
    perm_sensitivity (all as (n_layers, n_heads) arrays), plus task_accuracy.
    """
    model = GPT(MODEL_CONFIG)
    model.load_weights(str(ckpt_path / "model.safetensors"))
    mx.eval(model.parameters())

    # Profile heads (gets prev_token_score, copy_score, etc.)
    profiles = profile_all_heads(model, dataset, n_samples=n_samples)

    n_layers = MODEL_CONFIG.n_layers
    n_heads  = MODEL_CONFIG.n_heads

    prev_token = np.zeros((n_layers, n_heads))
    copy_sc    = np.zeros((n_layers, n_heads))

    for p in profiles:
        prev_token[p.layer, p.head] = p.prev_token_score
        copy_sc[p.layer, p.head]    = p.copy_score

    # Induction score (attention back to prior matching token)
    ind_score = compute_induction_score(model, dataset, n_samples=n_samples)

    # Permutation sensitivity
    perm_sens = compute_permutation_sensitivity_fast(
        model, dataset,
        n_samples=n_perm_samples,
        n_permutations=n_permutations,
        seed=seed,
    )

    # Task accuracy
    acc = compute_task_accuracy(model, dataset, n_samples=n_samples)

    # Read training loss from checkpoint metrics.json if available
    train_loss = None
    metrics_file = ckpt_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        train_loss = m.get("loss")

    return {
        "induction_score":  ind_score.tolist(),
        "prev_token_score": prev_token.tolist(),
        "copy_score":       copy_sc.tolist(),
        "perm_sensitivity": perm_sens.tolist(),
        "task_accuracy":    acc,
        "train_loss":       train_loss,
    }


# ============================================================
# Crystallization step detection
# ============================================================

def find_crystallization_step(
    steps: list[int],
    values: list[float],
    threshold: float,
) -> Optional[int]:
    """Return the first step where value crosses threshold (and stays above for 2 consecutive steps).

    Returns None if threshold is never crossed.
    """
    for i, (step, val) in enumerate(zip(steps, values)):
        if val >= threshold:
            # Confirm with next step if available
            if i + 1 < len(values) and values[i + 1] >= threshold:
                return step
            elif i == len(steps) - 1:
                return step  # Last checkpoint, accept
    return None


# ============================================================
# Main analysis
# ============================================================

def run_formation_film():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = InductionDataset(vocab_size=50, seq_len=64, n_bigrams=8, seed=42)

    all_seed_data = {}
    all_crystallization = {}

    total_seeds = N_SEEDS
    t_start = time.time()

    for seed_idx in range(total_seeds):
        seed_dir = CKPT_BASE / f"induction_seed{seed_idx}"
        if not seed_dir.exists():
            print(f"  [seed {seed_idx}] directory not found, skipping")
            continue

        ckpt_dirs = sorted(seed_dir.glob("step_*"))
        if not ckpt_dirs:
            print(f"  [seed {seed_idx}] no checkpoints found, skipping")
            continue

        print(f"\n[seed {seed_idx}/{total_seeds-1}] Processing {len(ckpt_dirs)} checkpoints...")

        seed_steps = []
        seed_metrics = []

        for ckpt_i, ckpt_path in enumerate(ckpt_dirs):
            step = int(ckpt_path.name.split("_")[1])
            if ckpt_i % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  step {step:5d}  [{ckpt_i+1}/{len(ckpt_dirs)}]  elapsed={elapsed:.0f}s")

            metrics = extract_metrics_at_checkpoint(
                ckpt_path,
                dataset,
                n_samples=N_SAMPLES,
                n_perm_samples=N_PERM_SAMPLES,
                n_permutations=N_PERMUTATIONS,
                seed=seed_idx,
            )
            seed_steps.append(step)
            seed_metrics.append(metrics)

        all_seed_data[str(seed_idx)] = {
            "steps": seed_steps,
            "metrics": seed_metrics,
        }

        # Compute crystallization steps for this seed
        # Use max over all heads as the "seed-level" metric at each step
        cryst = {}
        for metric_name, threshold in THRESHOLDS.items():
            if metric_name == "task_accuracy":
                values = [m["task_accuracy"] for m in seed_metrics]
            else:
                # max over all (layer, head) at each step
                values = [float(np.max(m[metric_name])) for m in seed_metrics]

            step_c = find_crystallization_step(seed_steps, values, threshold)
            cryst[metric_name] = step_c

        all_crystallization[str(seed_idx)] = cryst
        print(f"  Crystallization steps: {cryst}")

    print(f"\nAll seeds processed in {time.time()-t_start:.0f}s")

    # ============================================================
    # Compute aggregate developmental trajectories
    # ============================================================
    steps_union = sorted({s for d in all_seed_data.values() for s in d["steps"]})

    # Build (n_seeds, n_steps, n_layers, n_heads) arrays per metric
    # For plotting: take max over heads per seed per step
    metric_names_ts = ["copy_score", "prev_token_score", "induction_score", "perm_sensitivity"]

    # Align all seeds to the same steps list
    seed_ids = sorted(all_seed_data.keys(), key=int)

    # For each metric: array shape (n_seeds, n_steps)
    trajectory_arrays = {}
    for mn in metric_names_ts:
        arr = []
        for sid in seed_ids:
            d = all_seed_data[sid]
            step2metric = {s: float(np.max(m[mn])) for s, m in zip(d["steps"], d["metrics"])}
            row = [step2metric.get(s, np.nan) for s in steps_union]
            arr.append(row)
        trajectory_arrays[mn] = np.array(arr)  # (n_seeds, n_steps)

    # Task accuracy
    acc_arr = []
    for sid in seed_ids:
        d = all_seed_data[sid]
        step2acc = {s: m["task_accuracy"] for s, m in zip(d["steps"], d["metrics"])}
        row = [step2acc.get(s, np.nan) for s in steps_union]
        acc_arr.append(row)
    trajectory_arrays["task_accuracy"] = np.array(acc_arr)

    # ============================================================
    # Compute ordering statistics
    # ============================================================
    ordering_analysis = analyze_ordering(all_crystallization, seed_ids)
    print("\nOrdering analysis:")
    print(json.dumps(ordering_analysis, indent=2))

    # ============================================================
    # Save results JSON
    # ============================================================
    # Convert numpy to python for JSON serialization
    def to_json_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        return obj

    results = {
        "config": {
            "n_seeds": N_SEEDS,
            "n_samples": N_SAMPLES,
            "thresholds": THRESHOLDS,
            "model": "GPTConfig(n_layers=4, n_heads=4, d_model=128, d_ff=512, vocab_size=50, ctx_len=64)",
        },
        "per_seed_data": to_json_serializable(all_seed_data),
        "crystallization_steps": to_json_serializable(all_crystallization),
        "ordering_analysis": to_json_serializable(ordering_analysis),
        "steps_union": steps_union,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ============================================================
    # Plotting
    # ============================================================
    plot_formation_film(steps_union, trajectory_arrays, seed_ids)
    plot_formation_ordering(all_crystallization, seed_ids)

    # ============================================================
    # Report
    # ============================================================
    report_trajectory_universality(ordering_analysis, all_crystallization, seed_ids)

    return results


# ============================================================
# Ordering analysis
# ============================================================

def analyze_ordering(
    all_crystallization: dict,
    seed_ids: list[str],
) -> dict:
    """Compute aggregate ordering statistics from crystallization steps.

    Checks: for each pair of metrics (A, B), how often does A crystallize before B?
    """
    metrics_ordered = ["copy_score", "prev_token_score", "induction_score", "task_accuracy"]

    # Collect crystallization steps per metric
    cryst_by_metric = {m: [] for m in metrics_ordered}
    for sid in seed_ids:
        cryst = all_crystallization[sid]
        for m in metrics_ordered:
            val = cryst.get(m)
            cryst_by_metric[m].append(val)

    # Pairwise ordering: how often does metric_A precede metric_B?
    pairwise = {}
    for i, mA in enumerate(metrics_ordered):
        for j, mB in enumerate(metrics_ordered):
            if i >= j:
                continue
            key = f"{mA}_before_{mB}"
            count_ab = 0
            count_ba = 0
            count_tie = 0
            count_neither = 0
            for sid in seed_ids:
                cA = all_crystallization[sid].get(mA)
                cB = all_crystallization[sid].get(mB)
                if cA is None and cB is None:
                    count_neither += 1
                elif cA is None:
                    count_ba += 1  # B formed, A never did
                elif cB is None:
                    count_ab += 1  # A formed, B never did
                elif cA < cB:
                    count_ab += 1
                elif cB < cA:
                    count_ba += 1
                else:
                    count_tie += 1

            n_valid = count_ab + count_ba + count_tie
            pairwise[key] = {
                f"{mA}_first": count_ab,
                f"{mB}_first": count_ba,
                "tie": count_tie,
                "neither_formed": count_neither,
                "n_seeds_with_data": n_valid,
                "fraction_A_before_B": count_ab / max(n_valid, 1),
                "dominant_order": mA if count_ab > count_ba else (mB if count_ba > count_ab else "tie"),
            }

    # Compute median crystallization step per metric
    median_steps = {}
    for m, vals in cryst_by_metric.items():
        valid = [v for v in vals if v is not None]
        median_steps[m] = {
            "median": float(np.median(valid)) if valid else None,
            "mean":   float(np.mean(valid))   if valid else None,
            "std":    float(np.std(valid))     if valid else None,
            "n_crystallized": len(valid),
            "n_total": len(vals),
        }

    # Canonical order: sort by median crystallization step
    canonical_order = sorted(
        [m for m in metrics_ordered if median_steps[m]["median"] is not None],
        key=lambda m: median_steps[m]["median"],
    )

    # Check consistency: fraction of seeds that match canonical order
    n_matching_canonical = 0
    for sid in seed_ids:
        cryst = all_crystallization[sid]
        # Get sorted order for this seed (only metrics that crystallized)
        seed_order = sorted(
            [m for m in metrics_ordered if cryst.get(m) is not None],
            key=lambda m: cryst[m],
        )
        # Check if seed_order is a subsequence of canonical_order in the same relative order
        if is_consistent_ordering(seed_order, canonical_order):
            n_matching_canonical += 1

    return {
        "pairwise_ordering": pairwise,
        "median_crystallization_steps": median_steps,
        "canonical_order": canonical_order,
        "n_seeds_matching_canonical": n_matching_canonical,
        "n_seeds_total": len(seed_ids),
        "fraction_matching_canonical": n_matching_canonical / max(len(seed_ids), 1),
    }


def is_consistent_ordering(seed_order: list, canonical_order: list) -> bool:
    """Check if seed_order's relative ordering is consistent with canonical_order."""
    # For each pair (A, B) in seed_order, check same relative order in canonical
    canon_rank = {m: i for i, m in enumerate(canonical_order)}
    for i, mA in enumerate(seed_order):
        for j, mB in enumerate(seed_order):
            if i >= j:
                continue
            # In seed: mA before mB
            # In canon: mA should also be before mB
            if mA in canon_rank and mB in canon_rank:
                if canon_rank[mA] >= canon_rank[mB]:
                    return False  # Reversed in canonical
    return True


# ============================================================
# Plot 1: Formation film (time series overlay)
# ============================================================

def plot_formation_film(
    steps: list[int],
    trajectory_arrays: dict,
    seed_ids: list[str],
):
    """One row per metric: overlay all 20 seeds, highlight mean + 90% CI."""
    metric_display = [
        ("copy_score",       "Copy Score\n(OV diagonal dominance)",   "#2196F3"),
        ("prev_token_score", "Prev-Token Score\n(attention to i-1)",  "#4CAF50"),
        ("induction_score",  "Induction Score\n(attention to prior match)", "#FF5722"),
        ("task_accuracy",    "Task Accuracy\n(induction position acc.)", "#9C27B0"),
    ]

    steps_arr = np.array(steps)
    n_rows = len(metric_display)

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        "Circuit Formation Film: 20 Seeds × 80 Checkpoints\n"
        "Do all seeds follow the same developmental trajectory?",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for ax, (metric_name, label, color) in zip(axes, metric_display):
        arr = trajectory_arrays[metric_name]  # (n_seeds, n_steps)

        # Plot individual seeds (light, thin)
        for i in range(arr.shape[0]):
            ax.plot(steps_arr, arr[i], color=color, alpha=0.18, linewidth=0.8)

        # Mean and CI
        mean = np.nanmean(arr, axis=0)
        p5   = np.nanpercentile(arr, 5,  axis=0)
        p95  = np.nanpercentile(arr, 95, axis=0)
        sem  = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])

        ax.fill_between(steps_arr, p5, p95, alpha=0.25, color=color, label="90% CI")
        ax.plot(steps_arr, mean, color=color, linewidth=2.5, label="Mean", zorder=5)

        # Threshold line
        threshold = THRESHOLDS.get(metric_name)
        if threshold is not None:
            ax.axhline(threshold, color="gray", linewidth=1.2, linestyle="--",
                       alpha=0.7, label=f"threshold={threshold}")

        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
        ax.set_xlim(steps_arr[0], steps_arr[-1])
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Training Step", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = PLOTS_DIR / "formation_film.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# Plot 2: Formation ordering (crystallization step distributions)
# ============================================================

def plot_formation_ordering(
    all_crystallization: dict,
    seed_ids: list[str],
):
    """Histogram of crystallization steps per metric across seeds."""
    metrics_ordered = [
        ("copy_score",       "Copy Score",       "#2196F3"),
        ("prev_token_score", "Prev-Token Score",  "#4CAF50"),
        ("induction_score",  "Induction Score",   "#FF5722"),
        ("task_accuracy",    "Task Accuracy",     "#9C27B0"),
        ("perm_sensitivity", "Perm. Sensitivity", "#FF9800"),
    ]

    fig, axes = plt.subplots(1, len(metrics_ordered), figsize=(16, 4), sharey=False)

    fig.suptitle(
        "Crystallization Step Distributions Across 20 Seeds\n"
        "Earlier = faster formation; overlapping distributions = synchronized development",
        fontsize=12, fontweight="bold",
    )

    max_step = 2000
    bins = np.arange(0, max_step + 100, 100)

    for ax, (metric_name, label, color) in zip(axes, metrics_ordered):
        vals = []
        n_none = 0
        for sid in seed_ids:
            v = all_crystallization[sid].get(metric_name)
            if v is not None:
                vals.append(v)
            else:
                n_none += 1

        if vals:
            ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
            median_val = np.median(vals)
            ax.axvline(median_val, color="black", linewidth=2, linestyle="-",
                       label=f"median={int(median_val)}")
            ax.legend(fontsize=8)

        ax.set_xlabel("Training Step", fontsize=9)
        ax.set_ylabel("# Seeds", fontsize=9)
        ax.set_title(f"{label}\n(n={len(vals)}/{len(seed_ids)} crystallized)", fontsize=9)
        ax.set_xlim(0, max_step + 50)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if n_none > 0:
            ax.text(0.97, 0.97, f"{n_none} never\ncrystallized",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="red", alpha=0.8)

    plt.tight_layout()
    out_path = PLOTS_DIR / "formation_ordering.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# Report
# ============================================================

def report_trajectory_universality(
    ordering_analysis: dict,
    all_crystallization: dict,
    seed_ids: list[str],
):
    print("\n" + "=" * 70)
    print("DEVELOPMENTAL TRAJECTORY REPORT")
    print("=" * 70)

    median_steps = ordering_analysis["median_crystallization_steps"]
    canonical = ordering_analysis["canonical_order"]
    frac_match = ordering_analysis["fraction_matching_canonical"]
    n_match = ordering_analysis["n_seeds_matching_canonical"]
    n_total = ordering_analysis["n_seeds_total"]

    print("\nMedian crystallization steps (lower = forms earlier):")
    for m in canonical:
        info = median_steps[m]
        med = info["median"]
        std = info["std"]
        n_c = info["n_crystallized"]
        print(f"  {m:25s}: median={med:6.0f}  std={std:5.0f}  [{n_c}/{n_total} seeds crystallized]")

    print(f"\nCanonical developmental order: {' → '.join(canonical)}")
    print(f"Seeds matching canonical order: {n_match}/{n_total} ({frac_match*100:.0f}%)")

    print("\nPairwise ordering (how often A forms before B):")
    for key, info in ordering_analysis["pairwise_ordering"].items():
        dom = info["dominant_order"]
        frac = info["fraction_A_before_B"]
        n = info["n_seeds_with_data"]
        parts = key.split("_before_")
        mA, mB = parts[0], parts[1]
        print(f"  {mA:25s} vs {mB:25s}: {dom} first  ({frac*100:.0f}% A-first, n={n})")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if frac_match >= 0.80:
        print(f"\n  UNIVERSAL TRAJECTORY: {frac_match*100:.0f}% of seeds follow the same")
        print(f"  developmental order: {' → '.join(canonical)}")
        print(f"\n  This STRONGLY SUPPORTS the 'function is fated' hypothesis:")
        print(f"  circuit components crystallize in a stereotyped sequence independent")
        print(f"  of initialization, even when final head assignments differ.")
    elif frac_match >= 0.60:
        print(f"\n  MOSTLY CONSISTENT TRAJECTORY: {frac_match*100:.0f}% of seeds match")
        print(f"  canonical order {' → '.join(canonical)}")
        print(f"\n  Moderate support for 'function is fated'. Some seeds deviate,")
        print(f"  which may reflect alternative developmental paths or noise.")
    else:
        print(f"\n  VARIABLE TRAJECTORY: only {frac_match*100:.0f}% of seeds match")
        print(f"  canonical order {' → '.join(canonical)}")
        print(f"\n  Trajectory variability is itself interesting: seeds find different")
        print(f"  routes to the same final functional circuit. This CHALLENGES the")
        print(f"  strong 'fated' claim but leaves open whether endpoints converge.")


# ============================================================
# Reanalysis from existing results (avoids re-running 1600 checkpoints)
# ============================================================

def reanalyze_from_results(results_path: Path = RESULTS_PATH):
    """Re-run ordering analysis and regenerate plots from saved results JSON.

    Use this when thresholds have changed but checkpoint data is already collected.
    """
    print(f"Loading saved results from {results_path}...")
    with open(results_path) as f:
        saved = json.load(f)

    all_seed_data = saved["per_seed_data"]
    steps_union = saved["steps_union"]
    seed_ids = sorted(all_seed_data.keys(), key=int)

    print(f"Loaded {len(seed_ids)} seeds, {len(steps_union)} steps")
    print(f"Using updated thresholds: {THRESHOLDS}")

    # Recompute crystallization steps with new thresholds
    all_crystallization = {}
    for sid in seed_ids:
        d = all_seed_data[sid]
        seed_steps = d["steps"]
        seed_metrics = d["metrics"]

        cryst = {}
        for metric_name, threshold in THRESHOLDS.items():
            if metric_name == "task_accuracy":
                values = [m["task_accuracy"] for m in seed_metrics]
            else:
                values = [float(np.max(m[metric_name])) for m in seed_metrics]
            cryst[metric_name] = find_crystallization_step(seed_steps, values, threshold)

        all_crystallization[sid] = cryst

    # Build trajectory arrays for plotting
    metric_names_ts = ["copy_score", "prev_token_score", "induction_score", "perm_sensitivity"]
    trajectory_arrays = {}
    for mn in metric_names_ts:
        arr = []
        for sid in seed_ids:
            d = all_seed_data[sid]
            step2metric = {s: float(np.max(m[mn])) for s, m in zip(d["steps"], d["metrics"])}
            row = [step2metric.get(s, np.nan) for s in steps_union]
            arr.append(row)
        trajectory_arrays[mn] = np.array(arr)

    acc_arr = []
    for sid in seed_ids:
        d = all_seed_data[sid]
        step2acc = {s: m["task_accuracy"] for s, m in zip(d["steps"], d["metrics"])}
        row = [step2acc.get(s, np.nan) for s in steps_union]
        acc_arr.append(row)
    trajectory_arrays["task_accuracy"] = np.array(acc_arr)

    # Ordering analysis
    ordering_analysis = analyze_ordering(all_crystallization, seed_ids)

    # Update saved results with new analysis
    def to_json_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        return obj

    saved["crystallization_steps"] = to_json_serializable(all_crystallization)
    saved["ordering_analysis"] = to_json_serializable(ordering_analysis)
    saved["config"]["thresholds"] = THRESHOLDS

    with open(results_path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"Updated results saved to {results_path}")

    # Regenerate plots
    plot_formation_film(steps_union, trajectory_arrays, seed_ids)
    plot_formation_ordering(all_crystallization, seed_ids)

    # Report
    report_trajectory_universality(ordering_analysis, all_crystallization, seed_ids)

    return saved


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import sys as _sys
    print("=" * 70)
    print("CIRCUIT FORMATION FILM")
    print(f"Seeds: 0–{N_SEEDS-1}  |  Checkpoints: 0–2000 (step 25)")
    print(f"Profiling batch size: {N_SAMPLES}  |  Permutations: {N_PERMUTATIONS}")
    print("=" * 70)

    # If --reanalyze flag passed or results already exist, skip checkpoint loading
    if "--reanalyze" in _sys.argv or (RESULTS_PATH.exists() and "--force" not in _sys.argv):
        if RESULTS_PATH.exists():
            print(f"Results file found at {RESULTS_PATH}")
            print("Reanalyzing with current thresholds (pass --force to re-extract from checkpoints)")
            results = reanalyze_from_results()
        else:
            results = run_formation_film()
    else:
        results = run_formation_film()

    print("\nDone.")
