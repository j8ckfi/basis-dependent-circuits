"""
Experiment B: Training-time portability curve.

Question: When during training does non-portability emerge?

Protocol:
1. Build fixed eval set once (reuses build_fixed_eval_set from transplant_unified_eval).
2. Load seed 1 final checkpoint as the RECIPIENT (fixed throughout).
3. Collect recipient's pre-L0-LN and post-L0 activations on alignment batch (once).
4. Enumerate seed 0's intermediate checkpoints (step_000000 to step_020000).
5. For each donor checkpoint:
   a. Load donor model
   b. Evaluate donor accuracy (donor_acc)
   c. Collect donor's pre-L0-LN and post-L0 activations
   d. Compute R_in, R_out via Procrustes (donor -> recipient)
   e. Compute R2 alignment quality metrics
   f. Extract donor's L0H3 weights
   g. Transplant UNALIGNED: donor L0H3 -> recipient L0H1 -> eval -> unaligned_acc
   h. Transplant ALIGNED: rotate via two-interface -> eval -> aligned_acc
   i. Record (step, donor_acc, unaligned_acc, aligned_acc, r2_pre_l0, r2_post_l0)
6. Cross-reference grokking_analysis.json to annotate transition region.

Output:
  analysis/training_time_portability_results.json
  analysis/plots/training_time_portability.png

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/training_time_portability.py
"""

import sys
import json
import copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import InductionDataset

# Reuse utilities from transplant_unified_eval
from analysis.transplant_unified_eval import (
    MODEL_CONFIG,
    EVAL_VOCAB_SIZE,
    EVAL_SEQ_LEN,
    EVAL_N_BIGRAMS,
    build_fixed_eval_set,
    evaluate_on_fixed_set,
    load_model,
    get_head_weights,
    transplant_head,
    collect_pre_l0_ln,
    collect_post_l0,
    procrustes_orthogonal,
    rotate_head_weights_two_interface,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent

RECIPIENT_CKPT = (
    "checkpoints/induction_content_match/induction_seed1/step_020000/model.safetensors"
)

DONOR_SEED0_DIR = (
    BASE_DIR / "checkpoints/induction_content_match/induction_seed0"
)

GROKKING_JSON = (
    BASE_DIR / "checkpoints/induction_content_match/grokking_analysis.json"
)

RESULTS_PATH = Path(__file__).parent / "training_time_portability_results.json"
PLOT_PATH = Path(__file__).parent / "plots" / "training_time_portability.png"

# Donor head: seed0 L0H3 (critical head)
DONOR_LAYER = 0
DONOR_HEAD = 3

# Recipient slot: seed1 L0H1 (critical slot)
RECIPIENT_LAYER = 0
RECIPIENT_HEAD = 1

# Alignment batch seed (consistent with existing analysis)
ALIGN_SEED = 1234
ALIGN_BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Alignment R2 helper
# ---------------------------------------------------------------------------

def compute_alignment_r2(X_source: np.ndarray, X_target: np.ndarray, R: np.ndarray) -> float:
    """R^2 of X_source @ R vs X_target (matches basis_aligned_transplant_v2.py)."""
    X_rotated = X_source @ R
    residuals = X_target - X_rotated
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((X_target - X_target.mean(axis=0)) ** 2))
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Enumerate donor checkpoints
# ---------------------------------------------------------------------------

def get_donor_checkpoints(seed0_dir: Path) -> list:
    """Return sorted list of (step_int, ckpt_path) for all step_XXXXXX dirs."""
    ckpts = []
    for d in seed0_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            step_str = d.name[len("step_"):]
            try:
                step_int = int(step_str)
            except ValueError:
                continue
            ckpt_file = d / "model.safetensors"
            if ckpt_file.exists():
                ckpts.append((step_int, ckpt_file))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


# ---------------------------------------------------------------------------
# Load grokking transition info
# ---------------------------------------------------------------------------

def load_grokking_info() -> dict:
    """Load grokking_analysis.json; return transition region bounds."""
    if not GROKKING_JSON.exists():
        print("  [grokking] grokking_analysis.json not found — no annotation")
        return {}

    with open(GROKKING_JSON) as f:
        data = json.load(f)

    # data is a list of {step, accuracy, ...} sorted by step
    # Find the transition: first step where accuracy exceeds 0.2 and last step below 0.7
    transition_start = None
    transition_end = None
    for entry in data:
        step = entry["step"]
        acc = entry.get("accuracy", 0.0)
        if transition_start is None and acc > 0.15:
            transition_start = step
        if acc < 0.70:
            transition_end = step

    info = {
        "raw": data,
        "transition_start": transition_start,
        "transition_end": transition_end,
    }
    print(f"  [grokking] transition_start={transition_start}, transition_end={transition_end}")
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("EXPERIMENT B: Training-Time Portability Curve")
    print("Question: When during training does non-portability emerge?")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Build fixed eval set (once)
    # ------------------------------------------------------------------
    print("\n[Step 1] Building fixed eval set...")
    fixed_eval_set = build_fixed_eval_set()

    # ------------------------------------------------------------------
    # Step 2: Load recipient (seed 1 final) — fixed throughout
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Loading recipient model (seed1 final)...")
    print(f"  {RECIPIENT_CKPT}")
    recipient = load_model(str(BASE_DIR / RECIPIENT_CKPT), MODEL_CONFIG)
    recipient_acc, _ = evaluate_on_fixed_set(recipient, fixed_eval_set)
    print(f"  Recipient baseline accuracy: {recipient_acc:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Collect recipient activations for alignment (once, reused)
    # ------------------------------------------------------------------
    print("\n[Step 3] Collecting recipient activations (align batch, seed=1234)...")
    align_dataset = InductionDataset(
        vocab_size=EVAL_VOCAB_SIZE,
        seq_len=EVAL_SEQ_LEN,
        n_bigrams=EVAL_N_BIGRAMS,
        seed=ALIGN_SEED,
    )
    align_inputs_mx, _, _ = align_dataset.generate_batch(ALIGN_BATCH_SIZE)
    mx.eval(align_inputs_mx)

    recip_pre_l0_ln = collect_pre_l0_ln(recipient, align_inputs_mx)   # (B*T, d_model)
    recip_post_l0   = collect_post_l0(recipient, align_inputs_mx)     # (B*T, d_model)
    print(f"  Recipient pre-L0-LN shape: {recip_pre_l0_ln.shape}")
    print(f"  Recipient post-L0 shape:   {recip_post_l0.shape}")

    # ------------------------------------------------------------------
    # Step 4: Enumerate donor checkpoints
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Enumerating donor checkpoints in {DONOR_SEED0_DIR}...")
    donor_ckpts = get_donor_checkpoints(DONOR_SEED0_DIR)
    print(f"  Found {len(donor_ckpts)} checkpoints: "
          f"step {donor_ckpts[0][0]} to step {donor_ckpts[-1][0]}")

    # ------------------------------------------------------------------
    # Step 5: Load grokking transition info for annotation
    # ------------------------------------------------------------------
    print("\n[Step 5] Loading grokking transition info...")
    grokking_info = load_grokking_info()

    # ------------------------------------------------------------------
    # Step 6: Loop over donor checkpoints
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Running portability curve ({len(donor_ckpts)} checkpoints)...")
    print("-" * 72)
    print(f"{'Step':>8} | {'DonorAcc':>9} | {'Unaligned':>9} | {'Aligned':>9} | {'R2_pre':>7} | {'R2_post':>7}")
    print("-" * 72)

    curve_entries = []

    for idx, (step, ckpt_path) in enumerate(donor_ckpts):
        # (a) Load donor model
        try:
            donor = load_model(str(ckpt_path), MODEL_CONFIG)
        except Exception as e:
            print(f"  [step {step:6d}] LOAD ERROR: {e} — skipping")
            continue

        # (b) Evaluate donor accuracy
        try:
            donor_acc, _ = evaluate_on_fixed_set(donor, fixed_eval_set)
        except Exception as e:
            print(f"  [step {step:6d}] EVAL ERROR: {e} — skipping")
            continue

        # (c) Collect donor activations on same alignment batch
        try:
            donor_pre_l0_ln = collect_pre_l0_ln(donor, align_inputs_mx)
            donor_post_l0   = collect_post_l0(donor, align_inputs_mx)
        except Exception as e:
            print(f"  [step {step:6d}] ACTIVATION ERROR: {e} — skipping")
            continue

        # (d) Compute R_in, R_out via Procrustes (donor -> recipient)
        try:
            R_in  = procrustes_orthogonal(donor_pre_l0_ln, recip_pre_l0_ln)
            R_out = procrustes_orthogonal(donor_post_l0,   recip_post_l0)
        except Exception as e:
            print(f"  [step {step:6d}] PROCRUSTES ERROR: {e} — skipping")
            continue

        # (e) Compute R2 alignment quality
        r2_pre  = compute_alignment_r2(donor_pre_l0_ln, recip_pre_l0_ln, R_in)
        r2_post = compute_alignment_r2(donor_post_l0,   recip_post_l0,   R_out)

        # (f) Extract donor's L0H3 weights
        try:
            donor_weights = get_head_weights(donor, DONOR_LAYER, DONOR_HEAD)
        except Exception as e:
            print(f"  [step {step:6d}] WEIGHT ERROR: {e} — skipping")
            continue

        # (g) Transplant UNALIGNED: donor L0H3 -> recipient L0H1
        try:
            m_unaligned = transplant_head(recipient, donor_weights, RECIPIENT_LAYER, RECIPIENT_HEAD)
            unaligned_acc, _ = evaluate_on_fixed_set(m_unaligned, fixed_eval_set)
        except Exception as e:
            print(f"  [step {step:6d}] UNALIGNED TRANSPLANT ERROR: {e} — skipping")
            continue

        # (h) Transplant ALIGNED: rotate then insert
        try:
            aligned_weights = rotate_head_weights_two_interface(donor_weights, R_in, R_out)
            m_aligned = transplant_head(recipient, aligned_weights, RECIPIENT_LAYER, RECIPIENT_HEAD)
            aligned_acc, _ = evaluate_on_fixed_set(m_aligned, fixed_eval_set)
        except Exception as e:
            print(f"  [step {step:6d}] ALIGNED TRANSPLANT ERROR: {e} — skipping")
            continue

        # (i) Record
        entry = {
            "step": step,
            "donor_acc": float(donor_acc),
            "unaligned_acc": float(unaligned_acc),
            "aligned_acc": float(aligned_acc),
            "r2_pre_l0": float(r2_pre),
            "r2_post_l0": float(r2_post),
        }
        curve_entries.append(entry)

        # Print progress every checkpoint (they're meaningful)
        print(f"  {step:8d} | {donor_acc:9.4f} | {unaligned_acc:9.4f} | {aligned_acc:9.4f} | {r2_pre:7.4f} | {r2_post:7.4f}")

        # Flush stdout for real-time progress
        sys.stdout.flush()

        # Free donor model (MLX keeps arrays alive; explicit del helps)
        del donor, m_unaligned, m_aligned

    print("-" * 72)
    print(f"  Completed {len(curve_entries)} / {len(donor_ckpts)} checkpoints")

    # ------------------------------------------------------------------
    # Step 7: Append recipient baseline entry
    # ------------------------------------------------------------------
    recipient_entry = {
        "step": "recipient_baseline",
        "donor_acc": None,
        "unaligned_acc": None,
        "aligned_acc": None,
        "recipient_acc": float(recipient_acc),
        "r2_pre_l0": None,
        "r2_post_l0": None,
        "note": "Seed1 final checkpoint — fixed recipient throughout",
    }

    # ------------------------------------------------------------------
    # Step 8: Save JSON results
    # ------------------------------------------------------------------
    results = {
        "experiment": "B — Training-time portability curve",
        "question": "When during training does non-portability emerge?",
        "protocol": {
            "recipient": RECIPIENT_CKPT,
            "donor_seed": 0,
            "donor_head": f"L{DONOR_LAYER}H{DONOR_HEAD}",
            "recipient_slot": f"L{RECIPIENT_LAYER}H{RECIPIENT_HEAD}",
            "alignment": "two-interface Procrustes (R_in for Q/K/V, R_out for out_proj)",
            "align_seed": ALIGN_SEED,
            "align_batch_size": ALIGN_BATCH_SIZE,
        },
        "recipient_baseline": recipient_acc,
        "grokking_transition": {
            "transition_start_step": grokking_info.get("transition_start"),
            "transition_end_step": grokking_info.get("transition_end"),
        },
        "curve": curve_entries,
        "recipient_entry": recipient_entry,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Results -> {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Step 9: Plot
    # ------------------------------------------------------------------
    print("\n[Step 9] Generating plot...")
    _make_plot(curve_entries, recipient_acc, grokking_info, PLOT_PATH)
    print(f"[Saved] Plot -> {PLOT_PATH}")

    print("\n[Done]")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plot(curve_entries: list, recipient_acc: float, grokking_info: dict, plot_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not curve_entries:
        print("  [plot] No data to plot.")
        return

    steps          = np.array([e["step"] for e in curve_entries])
    donor_accs     = np.array([e["donor_acc"] for e in curve_entries])
    unaligned_accs = np.array([e["unaligned_acc"] for e in curve_entries])
    aligned_accs   = np.array([e["aligned_acc"] for e in curve_entries])
    r2_pre         = np.array([e["r2_pre_l0"] for e in curve_entries])
    r2_post        = np.array([e["r2_post_l0"] for e in curve_entries])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Experiment B: Training-Time Portability Curve\n"
                 "Donor = Seed0 L0H3, Recipient = Seed1 L0H1 (fixed final ckpt)",
                 fontsize=12, fontweight="bold")

    # Grokking transition shading
    t_start = grokking_info.get("transition_start")
    t_end   = grokking_info.get("transition_end")
    shade_kwargs = dict(alpha=0.15, color="orange", label="Grokking transition region")

    # --- Top subplot: accuracy curves ---
    ax1.axhline(recipient_acc, color="gray", linestyle="--", linewidth=1.2,
                label=f"Recipient baseline ({recipient_acc:.3f})")
    ax1.plot(steps, donor_accs,     color="#1f77b4", linewidth=2.0, marker="o",
             markersize=3, label="Donor acc (seed0 L0H3 model)")
    ax1.plot(steps, unaligned_accs, color="#d62728", linewidth=1.8, marker="s",
             markersize=3, label="Unaligned transplant acc")
    ax1.plot(steps, aligned_accs,   color="#2ca02c", linewidth=1.8, marker="^",
             markersize=3, label="Aligned transplant acc (Procrustes)")

    if t_start is not None and t_end is not None:
        ax1.axvspan(t_start, t_end, **shade_kwargs)

    ax1.set_ylabel("Induction Accuracy", fontsize=11)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Top: Accuracy vs. Training Step", fontsize=10)

    # --- Bottom subplot: R2 alignment quality ---
    ax2.plot(steps, r2_pre,  color="#9467bd", linewidth=1.8, marker="o",
             markersize=3, label="R² pre-L0-LN (Q/K/V interface)")
    ax2.plot(steps, r2_post, color="#8c564b", linewidth=1.8, marker="s",
             markersize=3, label="R² post-L0 (out_proj interface)")

    if t_start is not None and t_end is not None:
        ax2.axvspan(t_start, t_end, alpha=0.15, color="orange")

    ax2.set_xlabel("Training Step (Donor / Seed 0)", fontsize=11)
    ax2.set_ylabel("Alignment R²", fontsize=11)
    ax2.set_ylim(-0.1, 1.05)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Bottom: Alignment Quality vs. Training Step", fontsize=10)

    # Annotation: grokking region label
    if t_start is not None and t_end is not None:
        mid = (t_start + t_end) / 2
        ax1.text(mid, 0.50, "Grokking\ntransition",
                 ha="center", va="center", fontsize=7, color="darkorange",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkorange", alpha=0.7))

    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
