"""
Experiment E: Minimum Functional Unit Sweep

Three sweeps to find the minimum rescuing subset of weights when transplanting
from donor (seed0) into recipient (seed1).

Donor:    checkpoints/induction_content_match/induction_seed0/step_020000/model.safetensors
Recipient: checkpoints/induction_content_match/induction_seed1/step_020000/model.safetensors

Sweeps:
  1. Structured subsets (~30-40 named combos)
  2. Random subsets (~70 evals, target fractions 5%..90%)
  3. Greedy peel-back (one-pass from all-15 toward minimum)

Output:
  analysis/minimum_unit_sweep_results.json
  analysis/plots/minimum_unit_sweep.png

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/minimum_unit_sweep.py
"""

import sys
import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import InductionDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CONFIG = GPTConfig(
    n_layers=2,
    n_heads=4,
    d_model=128,
    d_ff=512,
    vocab_size=512,
    ctx_len=64,
    dropout=0.0,
)

BASE_DIR = Path(__file__).parent.parent

DONOR_CKPT = (
    "checkpoints/induction_content_match/induction_seed0/step_020000/model.safetensors"
)
RECIPIENT_CKPT = (
    "checkpoints/induction_content_match/induction_seed1/step_020000/model.safetensors"
)

RESULTS_PATH = Path(__file__).parent / "minimum_unit_sweep_results.json"
PLOT_PATH = Path(__file__).parent / "plots" / "minimum_unit_sweep.png"

# All 15 tensors for a 2-layer GPT (no bias terms)
ALL_KEYS = [
    "wte.weight",
    "wpe.weight",
    "blocks.0.ln1.weight",
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.ln2.weight",
    "blocks.0.mlp.up_proj.weight",
    "blocks.0.mlp.down_proj.weight",
    "blocks.1.ln1.weight",
    "blocks.1.attn.qkv_proj.weight",
    "blocks.1.attn.out_proj.weight",
    "blocks.1.ln2.weight",
    "blocks.1.mlp.up_proj.weight",
    "blocks.1.mlp.down_proj.weight",
    "ln_f.weight",
]

RESCUE_THRESHOLD = 0.50

# Eval protocol (matches transplant_unified_eval.py)
EVAL_N_SEQUENCES = 1024
EVAL_BATCH_SIZE = 32
EVAL_DATA_SEED = 9999
EVAL_VOCAB_SIZE = 512
EVAL_SEQ_LEN = 64
EVAL_N_BIGRAMS = 6

# Reference values from prior experiments
BASELINE_DONOR_ACC = 0.964    # seed0 baseline
BASELINE_ABLATED_ACC = 0.118  # recipient with L0H1 ablated

FixedEvalSet = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Eval utilities (mirrors transplant_unified_eval.py)
# ---------------------------------------------------------------------------

def build_fixed_eval_set() -> FixedEvalSet:
    dataset = InductionDataset(
        vocab_size=EVAL_VOCAB_SIZE,
        seq_len=EVAL_SEQ_LEN,
        n_bigrams=EVAL_N_BIGRAMS,
        seed=EVAL_DATA_SEED,
    )
    n_batches = EVAL_N_SEQUENCES // EVAL_BATCH_SIZE
    fixed_batches: FixedEvalSet = []
    for _ in range(n_batches):
        inputs_mx, targets_mx, mask_mx = dataset.generate_batch(EVAL_BATCH_SIZE)
        mx.eval(inputs_mx); mx.eval(targets_mx); mx.eval(mask_mx)
        fixed_batches.append((
            np.array(inputs_mx),
            np.array(targets_mx),
            np.array(mask_mx),
        ))
    total = sum(b[0].shape[0] for b in fixed_batches)
    print(f"  [eval set] {len(fixed_batches)} batches x {EVAL_BATCH_SIZE} = {total} sequences "
          f"(seed={EVAL_DATA_SEED})")
    return fixed_batches


def evaluate_on_fixed_set(model_fn, fixed_eval_set: FixedEvalSet) -> float:
    per_seq_fracs = []
    for inputs_np, targets_np, mask_np in fixed_eval_set:
        inputs_mx = mx.array(inputs_np)
        logits = model_fn(inputs_mx)
        mx.eval(logits)
        logits_np = np.array(logits)
        preds = logits_np.argmax(axis=-1)
        at_mask = mask_np > 0.5
        for b in range(inputs_np.shape[0]):
            n_mask = at_mask[b].sum()
            if n_mask > 0:
                per_seq_fracs.append(float((preds[b] == targets_np[b])[at_mask[b]].sum()) / n_mask)
            else:
                per_seq_fracs.append(0.0)
    return float(np.mean(per_seq_fracs))


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------

def get_flat_weights(model: GPT) -> Dict[str, np.ndarray]:
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    return {k: np.array(v) for k, v in flat.items()}


def splice_tensors(recipient_weights: dict, donor_weights: dict, transplant_keys: list) -> dict:
    """Return new weight dict: recipient overwritten by donor for keys in transplant_keys."""
    w = dict(recipient_weights)
    for k in transplant_keys:
        w[k] = donor_weights[k].copy()
    return w


def build_model_from_weights(weights_dict: dict, model_config: GPTConfig) -> GPT:
    model = GPT(model_config)
    model.load_weights([(k, mx.array(v)) for k, v in weights_dict.items()])
    mx.eval(model.parameters())
    return model


def load_model(weights_path: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


def param_count(weights_dict: dict, keys: list) -> int:
    return sum(weights_dict[k].size for k in keys)


def total_params(weights_dict: dict) -> int:
    return sum(v.size for v in weights_dict.values())


def eval_splice(recipient: dict, donor: dict, keys: list,
                fixed_eval_set: FixedEvalSet, config: GPTConfig) -> float:
    spliced = splice_tensors(recipient, donor, keys)
    model = build_model_from_weights(spliced, config)
    return evaluate_on_fixed_set(model, fixed_eval_set)


# ---------------------------------------------------------------------------
# Named subset definitions for Sweep 1
# ---------------------------------------------------------------------------

def get_structured_subsets() -> List[Tuple[str, List[str]]]:
    """Return (name, keys) pairs for all named structured subsets."""
    subsets = []

    # --- Individual tensors (15) ---
    for k in ALL_KEYS:
        subsets.append((f"individual: {k}", [k]))

    # --- By role per layer ---
    subsets.append(("L0 attn (qkv+out)", [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
    ]))
    subsets.append(("L0 MLP (up+down)", [
        "blocks.0.mlp.up_proj.weight",
        "blocks.0.mlp.down_proj.weight",
    ]))
    subsets.append(("L0 LN (ln1+ln2)", [
        "blocks.0.ln1.weight",
        "blocks.0.ln2.weight",
    ]))
    subsets.append(("L1 attn (qkv+out)", [
        "blocks.1.attn.qkv_proj.weight",
        "blocks.1.attn.out_proj.weight",
    ]))
    subsets.append(("L1 MLP (up+down)", [
        "blocks.1.mlp.up_proj.weight",
        "blocks.1.mlp.down_proj.weight",
    ]))
    subsets.append(("L1 LN (ln1+ln2)", [
        "blocks.1.ln1.weight",
        "blocks.1.ln2.weight",
    ]))

    # --- By role across layers ---
    subsets.append(("all qkv_proj (L0+L1)", [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.1.attn.qkv_proj.weight",
    ]))
    subsets.append(("all out_proj (L0+L1)", [
        "blocks.0.attn.out_proj.weight",
        "blocks.1.attn.out_proj.weight",
    ]))
    subsets.append(("all MLP up (L0+L1)", [
        "blocks.0.mlp.up_proj.weight",
        "blocks.1.mlp.up_proj.weight",
    ]))
    subsets.append(("all MLP down (L0+L1)", [
        "blocks.0.mlp.down_proj.weight",
        "blocks.1.mlp.down_proj.weight",
    ]))
    subsets.append(("all LayerNorms (L0+L1+ln_f)", [
        "blocks.0.ln1.weight",
        "blocks.0.ln2.weight",
        "blocks.1.ln1.weight",
        "blocks.1.ln2.weight",
        "ln_f.weight",
    ]))

    # --- Embeddings variants ---
    subsets.append(("wte only", ["wte.weight"]))
    subsets.append(("wpe only", ["wpe.weight"]))
    subsets.append(("wte+wpe", ["wte.weight", "wpe.weight"]))

    # --- Complete blocks ---
    l0_block = [
        "blocks.0.ln1.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.ln2.weight",
        "blocks.0.mlp.up_proj.weight",
        "blocks.0.mlp.down_proj.weight",
    ]
    l1_block = [
        "blocks.1.ln1.weight",
        "blocks.1.attn.qkv_proj.weight",
        "blocks.1.attn.out_proj.weight",
        "blocks.1.ln2.weight",
        "blocks.1.mlp.up_proj.weight",
        "blocks.1.mlp.down_proj.weight",
    ]
    subsets.append(("L0 full block", l0_block))
    subsets.append(("L1 full block", l1_block))
    subsets.append(("both full blocks (L0+L1)", l0_block + l1_block))

    # --- Full minus one block ---
    all_except_l0 = [k for k in ALL_KEYS if k not in l0_block]
    all_except_l1 = [k for k in ALL_KEYS if k not in l1_block]
    subsets.append(("all except L0 block", all_except_l0))
    subsets.append(("all except L1 block", all_except_l1))

    # --- Combinations ---
    subsets.append(("L0 attn + embeddings", [
        "wte.weight", "wpe.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
    ]))
    subsets.append(("L0 full + embeddings + ln_f", [
        "wte.weight", "wpe.weight",
    ] + l0_block + ["ln_f.weight"]))
    subsets.append(("L0+L1 full (everything except embeddings)",
                    l0_block + l1_block))

    return subsets


# ---------------------------------------------------------------------------
# Sweep 1: Structured subsets
# ---------------------------------------------------------------------------

def run_sweep1(recipient: dict, donor: dict,
               fixed_eval_set: FixedEvalSet,
               total_p: int) -> List[dict]:
    print("\n" + "=" * 70)
    print("SWEEP 1: Structured named subsets")
    print("=" * 70)

    subsets = get_structured_subsets()
    results = []
    n = len(subsets)

    for i, (name, keys) in enumerate(subsets):
        acc = eval_splice(recipient, donor, keys, fixed_eval_set, MODEL_CONFIG)
        p_frac = param_count(donor, keys) / total_p
        results.append({
            "name": name,
            "keys": keys,
            "param_fraction": round(p_frac, 6),
            "accuracy": round(acc, 6),
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  Sweep 1 iteration {i+1}/{n}: '{name}' -> acc={acc:.4f}  "
                  f"({len(keys)} tensors, {p_frac*100:.1f}% params)")

    return results


# ---------------------------------------------------------------------------
# Sweep 2: Random subsets
# ---------------------------------------------------------------------------

def run_sweep2(recipient: dict, donor: dict,
               fixed_eval_set: FixedEvalSet,
               total_p: int,
               rng_seed: int = 2024) -> List[dict]:
    print("\n" + "=" * 70)
    print("SWEEP 2: Random tensor-subsets by parameter fraction")
    print("=" * 70)

    target_pcts = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    n_per_target = 10
    rng = np.random.default_rng(rng_seed)

    # Pre-compute per-key param counts
    key_params = {k: donor[k].size for k in ALL_KEYS}
    n_keys = len(ALL_KEYS)

    results = []
    sample_id = 0
    total_expected = len(target_pcts) * n_per_target

    for target_pct in target_pcts:
        target_params = int(total_p * target_pct)
        tolerance = total_p * 0.05  # ±5% of total

        collected = 0
        max_attempts = 5000
        attempts = 0

        while collected < n_per_target and attempts < max_attempts:
            attempts += 1
            # Random subset of keys (at least 1, at most all)
            k_count = rng.integers(1, n_keys + 1)
            chosen_idx = rng.choice(n_keys, size=k_count, replace=False)
            chosen_keys = [ALL_KEYS[i] for i in sorted(chosen_idx)]
            actual_params = sum(key_params[k] for k in chosen_keys)

            if abs(actual_params - target_params) <= tolerance:
                acc = eval_splice(recipient, donor, chosen_keys,
                                  fixed_eval_set, MODEL_CONFIG)
                actual_pct = actual_params / total_p
                results.append({
                    "sample_id": sample_id,
                    "target_pct": target_pct,
                    "actual_pct": round(actual_pct, 6),
                    "n_keys": len(chosen_keys),
                    "accuracy": round(acc, 6),
                    "keys": chosen_keys,
                })
                sample_id += 1
                collected += 1

                if sample_id % 10 == 0 or sample_id == total_expected:
                    print(f"  Sweep 2 sample {sample_id}/{total_expected}: "
                          f"target={target_pct*100:.0f}% actual={actual_pct*100:.1f}% "
                          f"-> acc={acc:.4f}")

        if collected < n_per_target:
            print(f"  WARNING: target_pct={target_pct*100:.0f}%: only found "
                  f"{collected}/{n_per_target} valid subsets in {attempts} attempts")

    print(f"  Sweep 2 complete: {len(results)} samples")
    return results


# ---------------------------------------------------------------------------
# Sweep 3: Greedy peel-back
# ---------------------------------------------------------------------------

def run_sweep3(recipient: dict, donor: dict,
               fixed_eval_set: FixedEvalSet,
               total_p: int) -> dict:
    print("\n" + "=" * 70)
    print("SWEEP 3: Greedy peel-back (one-pass)")
    print("=" * 70)

    # Start with all 15 tensors transplanted (should give ~0.86)
    current_subset = list(ALL_KEYS)
    acc = eval_splice(recipient, donor, current_subset, fixed_eval_set, MODEL_CONFIG)
    p_frac = param_count(donor, current_subset) / total_p
    print(f"  Baseline (all 15 tensors): acc={acc:.4f}, {p_frac*100:.1f}% params")

    trace = []

    for tensor in ALL_KEYS:
        if tensor not in current_subset:
            continue

        candidate = [k for k in current_subset if k != tensor]
        if len(candidate) == 0:
            # Never remove the last tensor without testing
            test_acc = eval_splice(recipient, donor, candidate,
                                   fixed_eval_set, MODEL_CONFIG)
        else:
            test_acc = eval_splice(recipient, donor, candidate,
                                   fixed_eval_set, MODEL_CONFIG)

        remaining_params = param_count(donor, candidate) if candidate else 0
        p_frac_after = remaining_params / total_p

        if test_acc >= RESCUE_THRESHOLD:
            # Commit removal
            current_subset = candidate
            action = "REMOVED"
        else:
            action = "KEPT"

        trace.append({
            "tensor_tested_for_removal": tensor,
            "action": action,
            "acc_after": round(test_acc, 6),
            "param_fraction_after": round(p_frac_after, 6),
            "subset_size_after": len(candidate),
        })

        print(f"  [{action}] {tensor}: acc={test_acc:.4f}, "
              f"{p_frac_after*100:.1f}% params, {len(candidate)} tensors remaining")

    final_acc = eval_splice(recipient, donor, current_subset, fixed_eval_set, MODEL_CONFIG)
    final_p_frac = param_count(donor, current_subset) / total_p if current_subset else 0.0

    print(f"\n  Final subset ({len(current_subset)} tensors, {final_p_frac*100:.1f}% params): "
          f"acc={final_acc:.4f}")
    print(f"  Tensors: {current_subset}")

    return {
        "trace": trace,
        "final_subset": current_subset,
        "final_acc": round(final_acc, 6),
        "final_param_fraction": round(final_p_frac, 6),
        "final_n_tensors": len(current_subset),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(sweep1: List[dict], sweep2: List[dict], sweep3: dict,
              all_transplant_acc: float, zero_transplant_acc: float):
    Path(PLOT_PATH.parent).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sweep 2: random (gray background)
    s2_x = [r["actual_pct"] for r in sweep2]
    s2_y = [r["accuracy"] for r in sweep2]
    ax.scatter(s2_x, s2_y, c="gray", alpha=0.5, s=25, label="Sweep 2 (random)", zorder=2)

    # Sweep 1: structured (blue)
    s1_x = [r["param_fraction"] for r in sweep1]
    s1_y = [r["accuracy"] for r in sweep1]
    ax.scatter(s1_x, s1_y, c="steelblue", alpha=0.8, s=40, label="Sweep 1 (structured)", zorder=3)

    # Label a few key structured points
    key_labels = {
        "L0 attn (qkv+out)": "L0 attn",
        "L0 full block": "L0 block",
        "both full blocks (L0+L1)": "L0+L1 blocks",
        "L0 full + embeddings + ln_f": "L0+emb+ln_f",
        "all except L1 block": "all\\ -L1",
        "L0 attn + embeddings": "L0attn+emb",
    }
    for r in sweep1:
        short = key_labels.get(r["name"])
        if short:
            ax.annotate(short, (r["param_fraction"], r["accuracy"]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color="steelblue")

    # Sweep 3: greedy trace (red markers, connected)
    trace = sweep3["trace"]
    if trace:
        # Build the path: start from all-transplant, then each step
        greedy_x = [1.0]  # all-15 baseline point
        greedy_y = [all_transplant_acc]
        cumulative_keys = list(ALL_KEYS)
        for step in trace:
            greedy_x.append(step["param_fraction_after"])
            greedy_y.append(step["acc_after"])
        ax.plot(greedy_x, greedy_y, c="red", alpha=0.6, lw=1.5, zorder=4)
        ax.scatter(greedy_x, greedy_y, c="red", s=30, label="Sweep 3 (greedy)", zorder=5)

    # Mark final greedy point prominently
    ax.scatter([sweep3["final_param_fraction"]], [sweep3["final_acc"]],
               c="red", s=120, marker="*", zorder=6,
               label=f"Greedy final ({sweep3['final_n_tensors']} tensors)")
    ax.annotate(f"greedy min\n({sweep3['final_n_tensors']}T, "
                f"{sweep3['final_param_fraction']*100:.1f}%)",
                (sweep3["final_param_fraction"], sweep3["final_acc"]),
                textcoords="offset points", xytext=(6, -14),
                fontsize=8, color="red")

    # Reference lines
    ax.axhline(RESCUE_THRESHOLD, color="orange", lw=1.5, ls="--",
               label=f"Rescue threshold ({RESCUE_THRESHOLD:.2f})")
    ax.axhline(BASELINE_DONOR_ACC, color="green", lw=1.2, ls=":",
               label=f"Donor baseline ({BASELINE_DONOR_ACC:.3f})")
    ax.axhline(BASELINE_ABLATED_ACC, color="purple", lw=1.2, ls=":",
               label=f"Recipient ablated ({BASELINE_ABLATED_ACC:.3f})")
    ax.axhline(zero_transplant_acc, color="black", lw=1.2, ls=":",
               label=f"Recipient baseline ({zero_transplant_acc:.3f})")

    ax.set_xlabel("Transplanted parameter fraction", fontsize=12)
    ax.set_ylabel("Induction accuracy", fontsize=12)
    ax.set_title("Minimum Functional Unit Sweep\n"
                 "(donor=seed0, recipient=seed1, transplant from donor)", fontsize=12)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PLOT_PATH), dpi=150)
    plt.close()
    print(f"  Plot saved to: {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("EXPERIMENT E: MINIMUM FUNCTIONAL UNIT SWEEP")
    print(f"Donor:     {DONOR_CKPT}")
    print(f"Recipient: {RECIPIENT_CKPT}")
    print(f"Eval: {EVAL_N_SEQUENCES} seqs, seed={EVAL_DATA_SEED}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Build fixed eval set
    # -----------------------------------------------------------------------
    print("\n[Step 1] Building fixed eval set...")
    fixed_eval_set = build_fixed_eval_set()

    # -----------------------------------------------------------------------
    # 2. Load models and get flat weights
    # -----------------------------------------------------------------------
    print("\n[Step 2] Loading models...")
    donor_model = load_model(str(BASE_DIR / DONOR_CKPT), MODEL_CONFIG)
    recipient_model = load_model(str(BASE_DIR / RECIPIENT_CKPT), MODEL_CONFIG)
    donor_w = get_flat_weights(donor_model)
    recipient_w = get_flat_weights(recipient_model)
    print(f"  Keys: {sorted(donor_w.keys())}")

    total_p = total_params(donor_w)
    print(f"  Total parameters: {total_p:,}")

    # Verify we have exactly the expected keys
    assert set(donor_w.keys()) == set(ALL_KEYS), \
        f"Key mismatch! Got: {sorted(donor_w.keys())}"

    # -----------------------------------------------------------------------
    # 3. Baselines
    # -----------------------------------------------------------------------
    print("\n[Step 3] Baselines...")
    all_transplant_acc = eval_splice(recipient_w, donor_w, ALL_KEYS,
                                     fixed_eval_set, MODEL_CONFIG)
    zero_transplant_acc = evaluate_on_fixed_set(recipient_model, fixed_eval_set)
    print(f"  All-transplant (donor weights in recipient): {all_transplant_acc:.4f}")
    print(f"  Zero-transplant (recipient unchanged):       {zero_transplant_acc:.4f}")
    print(f"  Expected donor baseline: ~{BASELINE_DONOR_ACC}")

    # -----------------------------------------------------------------------
    # 4. Sweep 1 — Structured subsets
    # -----------------------------------------------------------------------
    sweep1_results = run_sweep1(recipient_w, donor_w, fixed_eval_set, total_p)

    # -----------------------------------------------------------------------
    # 5. Sweep 2 — Random subsets
    # -----------------------------------------------------------------------
    sweep2_results = run_sweep2(recipient_w, donor_w, fixed_eval_set, total_p)

    # -----------------------------------------------------------------------
    # 6. Sweep 3 — Greedy peel-back
    # -----------------------------------------------------------------------
    sweep3_results = run_sweep3(recipient_w, donor_w, fixed_eval_set, total_p)

    # -----------------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------------
    rescuing_s1 = [r for r in sweep1_results if r["accuracy"] >= RESCUE_THRESHOLD]
    if rescuing_s1:
        min_rescuing = min(rescuing_s1, key=lambda r: r["param_fraction"])
    else:
        min_rescuing = None

    summary = {
        "all_transplant_acc": round(all_transplant_acc, 6),
        "zero_transplant_acc": round(zero_transplant_acc, 6),
        "rescue_threshold": RESCUE_THRESHOLD,
        "minimum_rescuing_unit_sweep1": (
            {
                "name": min_rescuing["name"],
                "n_tensors": len(min_rescuing["keys"]),
                "param_fraction": min_rescuing["param_fraction"],
                "accuracy": min_rescuing["accuracy"],
            }
            if min_rescuing else None
        ),
        "greedy_minimum": {
            "n_tensors": sweep3_results["final_n_tensors"],
            "param_fraction": sweep3_results["final_param_fraction"],
            "accuracy": sweep3_results["final_acc"],
            "subset": sweep3_results["final_subset"],
        },
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  All-transplant accuracy:    {summary['all_transplant_acc']:.4f}")
    print(f"  Zero-transplant accuracy:   {summary['zero_transplant_acc']:.4f}")
    if min_rescuing:
        print(f"  Sweep1 smallest rescuing:   '{min_rescuing['name']}' "
              f"({len(min_rescuing['keys'])} tensors, "
              f"{min_rescuing['param_fraction']*100:.1f}% params, "
              f"acc={min_rescuing['accuracy']:.4f})")
    else:
        print("  Sweep1: no structured subset rescued (>= 0.50)")
    print(f"  Greedy minimum:             {sweep3_results['final_n_tensors']} tensors, "
          f"{sweep3_results['final_param_fraction']*100:.1f}% params, "
          f"acc={sweep3_results['final_acc']:.4f}")

    # -----------------------------------------------------------------------
    # 8. Save results JSON
    # -----------------------------------------------------------------------
    output = {
        "sweep1": sweep1_results,
        "sweep2": sweep2_results,
        "sweep3": sweep3_results,
        "summary": summary,
        "eval_protocol": {
            "n_sequences": EVAL_N_SEQUENCES,
            "data_seed": EVAL_DATA_SEED,
            "vocab_size": EVAL_VOCAB_SIZE,
            "seq_len": EVAL_SEQ_LEN,
            "n_bigrams": EVAL_N_BIGRAMS,
            "batch_size": EVAL_BATCH_SIZE,
        },
        "config": {
            "donor_ckpt": DONOR_CKPT,
            "recipient_ckpt": RECIPIENT_CKPT,
            "total_params": total_p,
            "rescue_threshold": RESCUE_THRESHOLD,
            "random_sweep_seed": 2024,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # 9. Plot
    # -----------------------------------------------------------------------
    print("\n[Step 9] Generating plot...")
    make_plot(sweep1_results, sweep2_results, sweep3_results,
              all_transplant_acc, zero_transplant_acc)

    elapsed = time.time() - t0
    print(f"\n[Done] Total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
