"""
Circuit transplant experiment: test whether induction heads are functionally
transplantable across seeds.

Background:
- Seed 0: L0H3 is critical head (0.81 accuracy drop when ablated)
- Seed 1: L0H1 and L0H2 are critical heads (0.85 and 0.77 drops)

Question: Can we take seed 0's L0H3 and transplant it into seed 1's slot?
If functional universality is real, the transplanted head should preserve
function even though the surrounding weights are different.

Experiments:
1. Baseline: measure accuracy for both models
2. Transplant: copy seed0 L0H3 into seed1 L0H3 (non-critical slot) — control
3. Transplant: copy seed0 L0H3 into seed1 L0H1 (critical slot) — key experiment
4. Control A: copy a random/non-critical seed0 L0 head into seed1 L0H1 — should hurt
5. Control B: copy seed0 L0H3 into seed1 L0H1 with shuffled weights — null control
6. Selective ablation + replacement: zero seed1 L0H1, then replace with seed0 L0H3

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/circuit_transplant.py
"""

import sys
import copy
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Optional

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

SEED0_CKPT = (
    "checkpoints/induction_content_match/induction_seed0/step_020000/model.safetensors"
)
SEED1_CKPT = (
    "checkpoints/induction_content_match/induction_seed1/step_020000/model.safetensors"
)

BASE_DIR = Path(__file__).parent.parent
RESULTS_PATH = Path(__file__).parent / "circuit_transplant_results.json"

# Key heads identified from ablation analysis
SEED0_CRITICAL_HEAD = (0, 3)   # L0H3: drops 0.81 when ablated
SEED1_CRITICAL_HEAD = (0, 1)   # L0H1: drops 0.85 when ablated
SEED1_SECONDARY_HEAD = (0, 2)  # L0H2: drops 0.77 when ablated

# A non-critical seed0 L0 head to use as control (not H3)
SEED0_NONCRITICAL_HEAD = (0, 0)  # presumed non-critical


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Weight extraction / transplantation
# ---------------------------------------------------------------------------

def get_head_weights(model: GPT, layer: int, head: int) -> dict:
    """Extract QKV rows and out_proj columns for a single attention head.

    QKV projection shape: (3*d_model, d_model)
      Q rows: [H*d_head : (H+1)*d_head]
      K rows: [d_model + H*d_head : d_model + (H+1)*d_head]
      V rows: [2*d_model + H*d_head : 2*d_model + (H+1)*d_head]

    out_proj shape: (d_model, d_model)
      columns for head H: [:, H*d_head : (H+1)*d_head]
    """
    d_model = model.config.d_model
    d_head = model.config.d_head

    q_start = head * d_head
    q_end = (head + 1) * d_head

    qkv_w = model.blocks[layer].attn.qkv_proj.weight  # (3*d_model, d_model)
    out_w = model.blocks[layer].attn.out_proj.weight   # (d_model, d_model)

    q_rows = np.array(qkv_w[q_start:q_end, :])
    k_rows = np.array(qkv_w[d_model + q_start : d_model + q_end, :])
    v_rows = np.array(qkv_w[2 * d_model + q_start : 2 * d_model + q_end, :])
    out_cols = np.array(out_w[:, q_start:q_end])

    return {
        "q": q_rows,    # (d_head, d_model)
        "k": k_rows,    # (d_head, d_model)
        "v": v_rows,    # (d_head, d_model)
        "out": out_cols,  # (d_model, d_head)
    }


def transplant_head(
    recipient_model: GPT,
    donor_weights: dict,
    target_layer: int,
    target_head: int,
) -> GPT:
    """Return a new GPT model with donor head weights inserted at target slot.

    MLX arrays are immutable so we work in numpy, rebuild the weight arrays,
    and reload via load_weights on a fresh model instance.
    """
    config = recipient_model.config
    d_model = config.d_model
    d_head = config.d_head

    h = target_head
    q_start = h * d_head
    q_end = (h + 1) * d_head

    # Pull all current weights into numpy
    qkv_w = np.array(recipient_model.blocks[target_layer].attn.qkv_proj.weight)
    out_w = np.array(recipient_model.blocks[target_layer].attn.out_proj.weight)

    # Splice in donor weights
    qkv_w[q_start:q_end, :] = donor_weights["q"]
    qkv_w[d_model + q_start : d_model + q_end, :] = donor_weights["k"]
    qkv_w[2 * d_model + q_start : 2 * d_model + q_end, :] = donor_weights["v"]
    out_w[:, q_start:q_end] = donor_weights["out"]

    # Collect all weights from recipient as flat dict using tree_flatten
    flat = dict(nn.utils.tree_flatten(recipient_model.parameters()))
    weights = {k: np.array(v) for k, v in flat.items()}

    # Override the modified layer
    qkv_key = f"blocks.{target_layer}.attn.qkv_proj.weight"
    out_key = f"blocks.{target_layer}.attn.out_proj.weight"
    weights[qkv_key] = qkv_w
    weights[out_key] = out_w

    # Build a fresh model and load the modified weights
    new_model = GPT(config)
    # load_weights accepts list of (str, mx.array) pairs
    mlx_weights = [(k, mx.array(v)) for k, v in weights.items()]
    new_model.load_weights(mlx_weights)
    mx.eval(new_model.parameters())
    return new_model


def zero_head(
    model: GPT,
    target_layer: int,
    target_head: int,
) -> GPT:
    """Return a new model with the specified head's out_proj columns zeroed.

    We zero only the out_proj columns (the write path) which prevents the
    head from contributing anything to the residual stream — equivalent to
    ablation during forward pass but baked into the weights.
    """
    config = model.config
    d_head = config.d_head
    h = target_head
    q_start = h * d_head
    q_end = (h + 1) * d_head

    donor_zeros: dict = {
        "q": np.zeros((d_head, config.d_model), dtype=np.float32),
        "k": np.zeros((d_head, config.d_model), dtype=np.float32),
        "v": np.zeros((d_head, config.d_model), dtype=np.float32),
        "out": np.zeros((config.d_model, d_head), dtype=np.float32),
    }
    return transplant_head(model, donor_zeros, target_layer, target_head)


# ---------------------------------------------------------------------------
# Accuracy measurement (mirrors ablation.py)
# ---------------------------------------------------------------------------

def compute_induction_accuracy(
    model: GPT,
    dataset: InductionDataset,
    n_batches: int = 20,
    batch_size: int = 32,
) -> float:
    """Top-1 accuracy at induction-masked positions."""
    correct = 0
    total = 0

    for _ in range(n_batches):
        inputs, targets, mask = dataset.generate_batch(batch_size)
        logits = model(inputs)
        mx.eval(logits)

        logits_np = np.array(logits)
        targets_np = np.array(targets)
        mask_np = np.array(mask)

        preds = logits_np.argmax(axis=-1)
        at_mask = mask_np > 0.5

        correct += int((preds == targets_np)[at_mask].sum())
        total += int(at_mask.sum())

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Verify transplant actually changed weights (sanity check)
# ---------------------------------------------------------------------------

def verify_transplant_changed(
    original: GPT,
    transplanted: GPT,
    layer: int,
    head: int,
) -> bool:
    """Return True if at least one weight differs at the transplanted head."""
    d_model = original.config.d_model
    d_head = original.config.d_head
    h = head
    q_start = h * d_head
    q_end = (h + 1) * d_head

    orig_qkv = np.array(original.blocks[layer].attn.qkv_proj.weight)
    new_qkv = np.array(transplanted.blocks[layer].attn.qkv_proj.weight)

    orig_slice = orig_qkv[q_start:q_end, :]
    new_slice = new_qkv[q_start:q_end, :]
    changed = not np.allclose(orig_slice, new_slice)
    return changed


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CIRCUIT TRANSPLANT EXPERIMENT")
    print("Testing functional universality of induction heads across seeds")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print(f"\nLoading seed 0 model from: {SEED0_CKPT}")
    model0 = load_model(str(BASE_DIR / SEED0_CKPT), MODEL_CONFIG)

    print(f"Loading seed 1 model from: {SEED1_CKPT}")
    model1 = load_model(str(BASE_DIR / SEED1_CKPT), MODEL_CONFIG)

    # Fixed-seed dataset for reproducible evaluation
    dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Baselines
    # -----------------------------------------------------------------------
    print("\n--- Baselines ---")
    acc0 = compute_induction_accuracy(model0, dataset)
    acc1 = compute_induction_accuracy(model1, dataset)
    print(f"  Seed 0 baseline accuracy: {acc0:.4f}")
    print(f"  Seed 1 baseline accuracy: {acc1:.4f}")

    results["baselines"] = {
        "seed0": acc0,
        "seed1": acc1,
    }

    # -----------------------------------------------------------------------
    # 2. Extract seed 0's critical head (L0H3)
    # -----------------------------------------------------------------------
    donor_L = SEED0_CRITICAL_HEAD[0]
    donor_H = SEED0_CRITICAL_HEAD[1]
    donor_weights = get_head_weights(model0, donor_L, donor_H)
    print(f"\nExtracted seed0 L{donor_L}H{donor_H} weights "
          f"(Q shape {donor_weights['q'].shape})")

    # Extract seed 0's non-critical head for control
    ctrl_L = SEED0_NONCRITICAL_HEAD[0]
    ctrl_H = SEED0_NONCRITICAL_HEAD[1]
    control_weights = get_head_weights(model0, ctrl_L, ctrl_H)
    print(f"Extracted seed0 L{ctrl_L}H{ctrl_H} weights (non-critical control)")

    # -----------------------------------------------------------------------
    # 3. Transplant seed0 L0H3 -> seed1 L0H3  (non-critical slot)
    #    Expect: minimal effect since seed1 doesn't rely on L0H3
    # -----------------------------------------------------------------------
    print("\n--- Experiment 1: transplant seed0 L0H3 -> seed1 L0H3 (non-critical slot) ---")
    model1_t1 = transplant_head(model1, donor_weights, target_layer=0, target_head=3)
    changed = verify_transplant_changed(model1, model1_t1, layer=0, head=3)
    print(f"  Weight change verified: {changed}")
    acc_t1 = compute_induction_accuracy(model1_t1, dataset)
    print(f"  Accuracy after transplant (non-critical slot): {acc_t1:.4f}  "
          f"(delta vs seed1 baseline: {acc_t1 - acc1:+.4f})")

    results["transplant_seed0_L0H3_into_seed1_L0H3"] = {
        "description": "seed0 critical head inserted into seed1 non-critical slot",
        "hypothesis": "minimal effect — seed1 doesn't use L0H3",
        "accuracy": acc_t1,
        "delta_vs_seed1_baseline": acc_t1 - acc1,
        "weight_change_verified": changed,
    }

    # -----------------------------------------------------------------------
    # 4. Transplant seed0 L0H3 -> seed1 L0H1  (seed1's critical slot)
    #    Key experiment: does functional universality hold?
    # -----------------------------------------------------------------------
    print("\n--- Experiment 2: transplant seed0 L0H3 -> seed1 L0H1 (critical slot) ---")
    model1_t2 = transplant_head(model1, donor_weights, target_layer=0, target_head=1)
    changed2 = verify_transplant_changed(model1, model1_t2, layer=0, head=1)
    print(f"  Weight change verified: {changed2}")
    acc_t2 = compute_induction_accuracy(model1_t2, dataset)
    print(f"  Accuracy after transplant (critical slot): {acc_t2:.4f}  "
          f"(delta vs seed1 baseline: {acc_t2 - acc1:+.4f})")

    results["transplant_seed0_L0H3_into_seed1_L0H1"] = {
        "description": "seed0 critical head inserted into seed1 critical slot",
        "hypothesis": "if universal, accuracy should be preserved (near seed1 baseline)",
        "accuracy": acc_t2,
        "delta_vs_seed1_baseline": acc_t2 - acc1,
        "weight_change_verified": changed2,
    }

    # -----------------------------------------------------------------------
    # 5. Control A: copy seed0 non-critical head (L0H0) -> seed1 L0H1
    #    Expect: significant drop (non-critical head can't replace critical one)
    # -----------------------------------------------------------------------
    print("\n--- Control A: transplant seed0 L0H0 (non-critical) -> seed1 L0H1 ---")
    model1_ctrl_a = transplant_head(model1, control_weights, target_layer=0, target_head=1)
    changed_ca = verify_transplant_changed(model1, model1_ctrl_a, layer=0, head=1)
    print(f"  Weight change verified: {changed_ca}")
    acc_ctrl_a = compute_induction_accuracy(model1_ctrl_a, dataset)
    print(f"  Accuracy (non-critical donor): {acc_ctrl_a:.4f}  "
          f"(delta: {acc_ctrl_a - acc1:+.4f})")

    results["control_a_noncritical_donor_into_seed1_L0H1"] = {
        "description": "seed0 non-critical head into seed1 critical slot",
        "hypothesis": "should hurt — non-critical head lacks induction function",
        "accuracy": acc_ctrl_a,
        "delta_vs_seed1_baseline": acc_ctrl_a - acc1,
        "weight_change_verified": changed_ca,
    }

    # -----------------------------------------------------------------------
    # 6. Control B: seed0 L0H3 weights shuffled -> seed1 L0H1
    #    Destroys functionality while preserving weight statistics.
    #    Expect: significant drop (any weights don't work, only functional ones do)
    # -----------------------------------------------------------------------
    print("\n--- Control B: seed0 L0H3 (shuffled weights) -> seed1 L0H1 ---")
    rng = np.random.default_rng(seed=0)
    shuffled_weights = {
        "q": rng.permutation(donor_weights["q"].flatten()).reshape(donor_weights["q"].shape).astype(np.float32),
        "k": rng.permutation(donor_weights["k"].flatten()).reshape(donor_weights["k"].shape).astype(np.float32),
        "v": rng.permutation(donor_weights["v"].flatten()).reshape(donor_weights["v"].shape).astype(np.float32),
        "out": rng.permutation(donor_weights["out"].flatten()).reshape(donor_weights["out"].shape).astype(np.float32),
    }
    model1_ctrl_b = transplant_head(model1, shuffled_weights, target_layer=0, target_head=1)
    changed_cb = verify_transplant_changed(model1, model1_ctrl_b, layer=0, head=1)
    print(f"  Weight change verified: {changed_cb}")
    acc_ctrl_b = compute_induction_accuracy(model1_ctrl_b, dataset)
    print(f"  Accuracy (shuffled donor): {acc_ctrl_b:.4f}  "
          f"(delta: {acc_ctrl_b - acc1:+.4f})")

    results["control_b_shuffled_donor_into_seed1_L0H1"] = {
        "description": "seed0 L0H3 with shuffled weights into seed1 critical slot",
        "hypothesis": "null control — shuffled weights should destroy function",
        "accuracy": acc_ctrl_b,
        "delta_vs_seed1_baseline": acc_ctrl_b - acc1,
        "weight_change_verified": changed_cb,
    }

    # -----------------------------------------------------------------------
    # 7. Selective ablation + replacement
    #    Step A: zero seed1 L0H1 → confirm accuracy drop
    #    Step B: replace zeroed L0H1 with seed0 L0H3 → does accuracy recover?
    # -----------------------------------------------------------------------
    print("\n--- Selective ablation + replacement ---")

    # Step A: ablate seed1 L0H1 by zeroing weights
    model1_zeroed = zero_head(model1, target_layer=0, target_head=1)
    acc_zeroed = compute_induction_accuracy(model1_zeroed, dataset)
    print(f"  Seed1 after zeroing L0H1: {acc_zeroed:.4f}  "
          f"(drop: {acc1 - acc_zeroed:+.4f})")

    results["selective_ablation_seed1_L0H1_zeroed"] = {
        "description": "seed1 with L0H1 completely zeroed (weight-level ablation)",
        "accuracy": acc_zeroed,
        "drop_vs_baseline": acc1 - acc_zeroed,
    }

    # Step B: take the zeroed model and transplant seed0 L0H3 into that L0H1 slot
    model1_recovered = transplant_head(model1_zeroed, donor_weights, target_layer=0, target_head=1)
    changed_rec = verify_transplant_changed(model1_zeroed, model1_recovered, layer=0, head=1)
    print(f"  Weight change verified: {changed_rec}")
    acc_recovered = compute_induction_accuracy(model1_recovered, dataset)
    recovery_delta = acc_recovered - acc_zeroed
    print(f"  Seed1 after replace zeroed L0H1 with seed0 L0H3: {acc_recovered:.4f}  "
          f"(recovery delta vs zeroed: {recovery_delta:+.4f}, "
          f"vs original baseline: {acc_recovered - acc1:+.4f})")

    results["selective_ablation_then_replace_seed0_L0H3"] = {
        "description": "zeroed seed1 L0H1 then inserted seed0 L0H3",
        "accuracy_after_zero": acc_zeroed,
        "accuracy_after_replace": acc_recovered,
        "recovery_delta": recovery_delta,
        "delta_vs_seed1_baseline": acc_recovered - acc1,
        "weight_change_verified": changed_rec,
    }

    # -----------------------------------------------------------------------
    # 8. Interpret results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    seed1_baseline = acc1
    key_transplant_acc = acc_t2  # transplant into critical slot

    # Threshold: within 10% of baseline is "preserved function"
    preservation_threshold = 0.10

    transplant_preserves = abs(key_transplant_acc - seed1_baseline) < preservation_threshold * seed1_baseline
    control_hurts = (seed1_baseline - acc_ctrl_a) > preservation_threshold * seed1_baseline
    shuffled_hurts = (seed1_baseline - acc_ctrl_b) > preservation_threshold * seed1_baseline
    recovery_happened = recovery_delta > preservation_threshold * seed1_baseline

    print(f"\n  Seed 1 baseline accuracy:        {seed1_baseline:.4f}")
    print(f"  After transplant (critical slot): {key_transplant_acc:.4f}  "
          f"({'PRESERVED' if transplant_preserves else 'DEGRADED'})")
    print(f"  Control A (non-critical donor):   {acc_ctrl_a:.4f}  "
          f"({'hurts as expected' if control_hurts else 'did NOT hurt as expected'})")
    print(f"  Control B (shuffled weights):     {acc_ctrl_b:.4f}  "
          f"({'hurts as expected' if shuffled_hurts else 'did NOT hurt as expected'})")
    print(f"  Recovery after ablation+replace:  {recovery_delta:+.4f}  "
          f"({'RECOVERY OBSERVED' if recovery_happened else 'no clear recovery'})")

    if transplant_preserves and control_hurts and shuffled_hurts:
        interpretation = (
            "STRONG EVIDENCE FOR FUNCTIONAL UNIVERSALITY: "
            "The induction function appears to be stored in the head weights themselves "
            "(transplantable), not in the interactions with surrounding weights. "
            "The transplanted head from seed 0 preserves function in seed 1's context, "
            "while controls (non-critical donor, shuffled weights) both degrade performance. "
            "This supports 'function is fated, wiring is contingent': the same functional "
            "computation emerges in different weight configurations, and the head-level "
            "computation is portable across seeds."
        )
    elif transplant_preserves and not (control_hurts and shuffled_hurts):
        interpretation = (
            "AMBIGUOUS: transplant preserved function but controls did not hurt as expected. "
            "Multiple interpretations possible — any L0 head weights may work in this context."
        )
    elif not transplant_preserves and shuffled_hurts:
        interpretation = (
            "EVIDENCE AGAINST SIMPLE FUNCTIONAL UNIVERSALITY: "
            "Transplanting the functional head did NOT preserve seed 1's performance, "
            "suggesting the induction computation is NOT simply stored in head weights alone. "
            "The head function depends on INTERACTIONS with surrounding weights — "
            "the wiring context (Q/K/V projections operating in seed 1's residual stream basis) "
            "matters. This argues against naive weight-level transplantability, even if the "
            "computational motif is universal at a higher level of abstraction."
        )
    else:
        interpretation = (
            "INCONCLUSIVE: results do not clearly separate transplant from controls. "
            "More experiments needed."
        )

    if recovery_happened:
        interpretation += (
            " The recovery experiment (ablate then replace) provides additional evidence: "
            "inserting seed 0's head into the zeroed slot partially restores function."
        )

    print(f"\n  CONCLUSION: {interpretation}")

    results["interpretation"] = {
        "transplant_preserves_function": transplant_preserves,
        "control_a_hurts_as_expected": control_hurts,
        "control_b_hurts_as_expected": shuffled_hurts,
        "recovery_observed": recovery_happened,
        "conclusion": interpretation,
    }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
