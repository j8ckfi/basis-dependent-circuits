"""
Basis-aligned circuit transplant experiment.

Tests whether transplant failure (experiment F in circuit_transplant.py) is due to
basis mismatch correctable via Procrustes rotation, or whether circuits are truly
non-portable across seeds.

Background:
- Unaligned transplant of seed0 L0H3 -> seed1 L0H1: accuracy 0.090 (vs baseline 0.964)
- Hypothesis: both networks may compute the SAME function but in DIFFERENT bases of the
  residual stream. If we align the bases first (Procrustes), the transplant might succeed.

Approach:
1. Collect residual stream activations after L0 from both models on the same batch
2. Compute orthogonal Procrustes alignment: R = argmin ||X0 @ R - X1||_F, R^T R = I
3. Rotate seed0 L0H3 weights into seed1's basis before transplanting
4. Compare: unaligned vs aligned vs random-rotation vs non-critical-aligned transplant

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/basis_aligned_transplant.py
"""

import sys
import json
import copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import InductionDataset


# ---------------------------------------------------------------------------
# Constants (match circuit_transplant.py)
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
RESULTS_PATH = Path(__file__).parent / "basis_aligned_transplant_results.json"

SEED0_CRITICAL_HEAD = (0, 3)   # L0H3: drops 0.81 when ablated
SEED1_CRITICAL_HEAD = (0, 1)   # L0H1: drops 0.85 when ablated
SEED0_NONCRITICAL_HEAD = (0, 0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Residual stream collection
# ---------------------------------------------------------------------------

def collect_residual_stream_after_layer0(
    model: GPT,
    inputs: mx.array,
) -> np.ndarray:
    """Return the residual stream after layer 0 as a 2D numpy array.

    Uses model.get_residual_stream() which returns [after_embed, after_block0, after_block1].
    Index 1 = after block 0 = what layer 1 reads from.

    Returns:
        activations: shape (B*T, d_model)
    """
    residuals = model.get_residual_stream(inputs)
    # residuals[1] is after block 0, shape (B, T, d_model)
    after_l0 = residuals[1]
    mx.eval(after_l0)
    arr = np.array(after_l0)  # (B, T, d_model)
    B, T, D = arr.shape
    return arr.reshape(B * T, D)


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def compute_procrustes_rotation(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    """Compute orthogonal R minimizing ||X0 @ R - X1||_F.

    Solution: SVD of X1.T @ X0 = U S V.T  =>  R = U @ V.T

    Args:
        X0: (N, d_model) activations from seed 0
        X1: (N, d_model) activations from seed 1

    Returns:
        R: (d_model, d_model) orthogonal rotation matrix such that X0 @ R ≈ X1
    """
    # M = X1.T @ X0, shape (d_model, d_model)
    M = X1.T @ X0
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    R = U @ Vt  # (d_model, d_model), orthogonal
    return R.astype(np.float32)


def compute_alignment_r2(X0: np.ndarray, X1: np.ndarray, R: np.ndarray) -> float:
    """Compute R^2 of X0 @ R vs X1 (how well rotation explains X1 from X0).

    R^2 = 1 - SS_res / SS_tot
    """
    X0_rotated = X0 @ R
    residuals = X1 - X0_rotated
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((X1 - X1.mean(axis=0)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_frobenius_alignment(X0: np.ndarray, X1: np.ndarray, R: np.ndarray) -> float:
    """Fraction of Frobenius norm explained: 1 - ||X0@R - X1||_F / ||X1||_F."""
    norm_residual = np.linalg.norm(X0 @ R - X1, "fro")
    norm_x1 = np.linalg.norm(X1, "fro")
    if norm_x1 == 0:
        return 1.0
    return float(1.0 - norm_residual / norm_x1)


# ---------------------------------------------------------------------------
# Weight extraction (mirrors circuit_transplant.py)
# ---------------------------------------------------------------------------

def get_head_weights(model: GPT, layer: int, head: int) -> dict:
    """Extract Q/K/V rows and out_proj columns for a single attention head.

    QKV weight shape: (3*d_model, d_model)
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
        "q": q_rows,      # (d_head, d_model)
        "k": k_rows,      # (d_head, d_model)
        "v": v_rows,      # (d_head, d_model)
        "out": out_cols,  # (d_model, d_head)
    }


# ---------------------------------------------------------------------------
# Rotation of head weights
# ---------------------------------------------------------------------------

def rotate_head_weights(weights: dict, R: np.ndarray) -> dict:
    """Rotate head weights to align with target model's residual stream basis.

    Convention: R maps seed0 basis -> seed1 basis, i.e., X0 @ R ≈ X1.

    For projections that READ from the residual stream (Q, K, V):
        qkv_proj computes: h = x @ W_qkv.T  (x has shape (T, d_model))
        In seed0 basis: h = x0 @ W0.T
        In seed1 basis: x1 = x0 @ R, so x0 = x1 @ R.T
        For same output: x1 @ R.T @ W0.T = x1 @ (W0 @ R).T
        => new_W = W0 @ R  (right-multiply each row by R)

    For the output projection that WRITES to the residual stream (out):
        out_proj computes: y = v_out @ W_out.T  where v_out has shape (T, d_model)
        But out_proj takes concatenated head outputs (d_model dim), not the head slice directly.
        Actually: the head value output is written via out_cols (d_model, d_head slice).
        The contribution to residual is: v_head @ out_cols.T  where v_head is (T, d_head)
        After rotation, seed1's residual is in a different basis.
        For the output to land in seed1's basis: we need R.T @ out_cols (left-multiply)
        i.e., new_out = R.T @ out_cols

    Note on LayerNorm: Q/K/V actually read from LayerNorm(x), not x directly.
    LayerNorm is not rotation-invariant, so perfect alignment is impossible.
    We still apply the rotation as the best linear approximation.
    """
    q = weights["q"]    # (d_head, d_model)
    k = weights["k"]    # (d_head, d_model)
    v = weights["v"]    # (d_head, d_model)
    out = weights["out"]  # (d_model, d_head)

    # Q, K, V read from the residual stream: right-multiply by R
    new_q = q @ R       # (d_head, d_model)
    new_k = k @ R       # (d_head, d_model)
    new_v = v @ R       # (d_head, d_model)

    # out writes to residual stream: left-multiply by R.T
    new_out = R.T @ out  # (d_model, d_head)

    return {
        "q": new_q.astype(np.float32),
        "k": new_k.astype(np.float32),
        "v": new_v.astype(np.float32),
        "out": new_out.astype(np.float32),
    }


def apply_random_rotation(weights: dict, rng: np.random.Generator, d_model: int) -> dict:
    """Apply a uniformly random orthogonal rotation to the head weights.

    Used as a control: random rotation should be no better than unaligned transplant.
    """
    # Generate random orthogonal matrix via QR decomposition of random Gaussian
    A = rng.standard_normal((d_model, d_model)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return rotate_head_weights(weights, Q)


# ---------------------------------------------------------------------------
# Transplantation (mirrors circuit_transplant.py exactly)
# ---------------------------------------------------------------------------

def transplant_head(
    recipient_model: GPT,
    donor_weights: dict,
    target_layer: int,
    target_head: int,
) -> GPT:
    """Return a new GPT model with donor head weights inserted at target slot."""
    config = recipient_model.config
    d_model = config.d_model
    d_head = config.d_head

    h = target_head
    q_start = h * d_head
    q_end = (h + 1) * d_head

    qkv_w = np.array(recipient_model.blocks[target_layer].attn.qkv_proj.weight)
    out_w = np.array(recipient_model.blocks[target_layer].attn.out_proj.weight)

    qkv_w[q_start:q_end, :] = donor_weights["q"]
    qkv_w[d_model + q_start : d_model + q_end, :] = donor_weights["k"]
    qkv_w[2 * d_model + q_start : 2 * d_model + q_end, :] = donor_weights["v"]
    out_w[:, q_start:q_end] = donor_weights["out"]

    flat = dict(nn.utils.tree_flatten(recipient_model.parameters()))
    weights = {k: np.array(v) for k, v in flat.items()}

    qkv_key = f"blocks.{target_layer}.attn.qkv_proj.weight"
    out_key = f"blocks.{target_layer}.attn.out_proj.weight"
    weights[qkv_key] = qkv_w
    weights[out_key] = out_w

    new_model = GPT(config)
    mlx_weights = [(k, mx.array(v)) for k, v in weights.items()]
    new_model.load_weights(mlx_weights)
    mx.eval(new_model.parameters())
    return new_model


def verify_transplant_changed(
    original: GPT,
    transplanted: GPT,
    layer: int,
    head: int,
) -> bool:
    """Return True if at least one weight differs at the transplanted head."""
    d_model = original.config.d_model
    d_head = original.config.d_head
    q_start = head * d_head
    q_end = (head + 1) * d_head

    orig_qkv = np.array(original.blocks[layer].attn.qkv_proj.weight)
    new_qkv = np.array(transplanted.blocks[layer].attn.qkv_proj.weight)
    return not np.allclose(orig_qkv[q_start:q_end, :], new_qkv[q_start:q_end, :])


def compute_weight_checksum(model: GPT, layer: int, head: int) -> float:
    """L2 norm of Q rows as a checksum to verify weight changes."""
    d_head = model.config.d_head
    q_start = head * d_head
    q_end = (head + 1) * d_head
    qkv_w = np.array(model.blocks[layer].attn.qkv_proj.weight)
    return float(np.linalg.norm(qkv_w[q_start:q_end, :]))


# ---------------------------------------------------------------------------
# Accuracy measurement (mirrors circuit_transplant.py)
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
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BASIS-ALIGNED CIRCUIT TRANSPLANT EXPERIMENT")
    print("Testing whether transplant failure is due to correctable basis mismatch")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print(f"\nLoading seed 0 model from: {SEED0_CKPT}")
    model0 = load_model(str(BASE_DIR / SEED0_CKPT), MODEL_CONFIG)

    print(f"Loading seed 1 model from: {SEED1_CKPT}")
    model1 = load_model(str(BASE_DIR / SEED1_CKPT), MODEL_CONFIG)

    # Fixed-seed dataset for reproducible evaluation (same as circuit_transplant.py)
    eval_dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)

    results = {}

    # -----------------------------------------------------------------------
    # Baselines (re-confirm to ensure consistency)
    # -----------------------------------------------------------------------
    print("\n--- Baselines ---")
    acc0 = compute_induction_accuracy(model0, eval_dataset)
    acc1 = compute_induction_accuracy(model1, eval_dataset)
    print(f"  Seed 0 baseline accuracy: {acc0:.4f}")
    print(f"  Seed 1 baseline accuracy: {acc1:.4f}")

    results["baselines"] = {"seed0": acc0, "seed1": acc1}

    # -----------------------------------------------------------------------
    # Collect residual stream activations after layer 0 on a fixed batch
    # Use a separate fixed dataset for alignment (256 sequences)
    # -----------------------------------------------------------------------
    print("\n--- Collecting residual stream activations for Procrustes alignment ---")
    align_dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)
    align_inputs, _, _ = align_dataset.generate_batch(256)
    mx.eval(align_inputs)

    print("  Computing residual stream after L0 for seed 0...")
    X0 = collect_residual_stream_after_layer0(model0, align_inputs)  # (256*64, 128)
    print(f"  Seed 0 activations shape: {X0.shape}")

    print("  Computing residual stream after L0 for seed 1...")
    X1 = collect_residual_stream_after_layer0(model1, align_inputs)  # (256*64, 128)
    print(f"  Seed 1 activations shape: {X1.shape}")

    # -----------------------------------------------------------------------
    # Compute Procrustes rotation: R = argmin ||X0 @ R - X1||_F, R^T R = I
    # -----------------------------------------------------------------------
    print("\n--- Computing Procrustes rotation ---")
    R = compute_procrustes_rotation(X0, X1)
    print(f"  R shape: {R.shape}, orthogonality check ||R^T R - I||: "
          f"{np.linalg.norm(R.T @ R - np.eye(R.shape[0])):.6f}")

    # Alignment quality metrics
    r2 = compute_alignment_r2(X0, X1, R)
    frob_align = compute_frobenius_alignment(X0, X1, R)

    # Baseline: how similar are the raw (unrotated) activations?
    r2_unrotated = compute_alignment_r2(X0, X1, np.eye(MODEL_CONFIG.d_model))
    frob_unrotated = compute_frobenius_alignment(X0, X1, np.eye(MODEL_CONFIG.d_model))

    print(f"  Alignment R^2 (Procrustes): {r2:.4f}")
    print(f"  Alignment R^2 (unrotated):  {r2_unrotated:.4f}")
    print(f"  Frobenius alignment (Procrustes): {frob_align:.4f}")
    print(f"  Frobenius alignment (unrotated):  {frob_unrotated:.4f}")

    results["procrustes_alignment"] = {
        "r2_procrustes": r2,
        "r2_unrotated": r2_unrotated,
        "frobenius_alignment_procrustes": frob_align,
        "frobenius_alignment_unrotated": frob_unrotated,
        "r_orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(R.shape[0]))),
        "note": (
            "R^2 measures how well Procrustes rotation maps X0 onto X1. "
            "LayerNorm is not rotation-invariant, so perfect alignment is impossible. "
            "R^2 < 1 is expected and not a bug."
        ),
    }

    # -----------------------------------------------------------------------
    # Extract donor weights (seed0 L0H3 = critical head)
    # -----------------------------------------------------------------------
    donor_L, donor_H = SEED0_CRITICAL_HEAD
    donor_weights = get_head_weights(model0, donor_L, donor_H)
    print(f"\nExtracted seed0 L{donor_L}H{donor_H} weights "
          f"(Q shape {donor_weights['q'].shape})")

    # Checksum before rotation
    checksum_before = compute_weight_checksum(model0, donor_L, donor_H)
    print(f"  Donor weights Q-norm (checksum): {checksum_before:.6f}")

    # Non-critical donor for control
    ctrl_L, ctrl_H = SEED0_NONCRITICAL_HEAD
    noncritical_weights = get_head_weights(model0, ctrl_L, ctrl_H)

    # -----------------------------------------------------------------------
    # Unaligned transplant (reference — should match circuit_transplant.py result)
    # -----------------------------------------------------------------------
    print("\n--- Unaligned transplant (reference) ---")
    target_L, target_H = SEED1_CRITICAL_HEAD
    model1_unaligned = transplant_head(model1, donor_weights, target_L, target_H)
    changed_unaligned = verify_transplant_changed(model1, model1_unaligned, target_L, target_H)
    print(f"  Weight change verified: {changed_unaligned}")
    acc_unaligned = compute_induction_accuracy(model1_unaligned, eval_dataset)
    print(f"  Unaligned transplant accuracy: {acc_unaligned:.4f}  "
          f"(delta vs seed1 baseline: {acc_unaligned - acc1:+.4f})")
    print(f"  (Prior result from circuit_transplant.py: 0.0899)")

    results["unaligned_transplant"] = {
        "description": "seed0 L0H3 -> seed1 L0H1, no basis alignment (reference)",
        "accuracy": acc_unaligned,
        "delta_vs_seed1_baseline": acc_unaligned - acc1,
        "weight_change_verified": changed_unaligned,
        "prior_result": 0.0899,
    }

    # -----------------------------------------------------------------------
    # Procrustes-aligned transplant (key experiment)
    # -----------------------------------------------------------------------
    print("\n--- Procrustes-aligned transplant (KEY EXPERIMENT) ---")
    aligned_weights = rotate_head_weights(donor_weights, R)

    # Verify rotation actually changed the weights
    q_norm_after = float(np.linalg.norm(aligned_weights["q"]))
    print(f"  Aligned donor Q-norm (should be same magnitude, different direction): "
          f"{q_norm_after:.6f}")
    weights_changed = not np.allclose(donor_weights["q"], aligned_weights["q"])
    print(f"  Rotation changed weights: {weights_changed}")

    model1_aligned = transplant_head(model1, aligned_weights, target_L, target_H)
    changed_aligned = verify_transplant_changed(model1, model1_aligned, target_L, target_H)
    print(f"  Weight change in model verified: {changed_aligned}")
    acc_aligned = compute_induction_accuracy(model1_aligned, eval_dataset)
    print(f"  Aligned transplant accuracy: {acc_aligned:.4f}  "
          f"(delta vs seed1 baseline: {acc_aligned - acc1:+.4f})")
    print(f"  Improvement vs unaligned: {acc_aligned - acc_unaligned:+.4f}")

    results["aligned_transplant"] = {
        "description": "seed0 L0H3 -> seed1 L0H1, Procrustes-aligned weights",
        "accuracy": acc_aligned,
        "delta_vs_seed1_baseline": acc_aligned - acc1,
        "improvement_vs_unaligned": acc_aligned - acc_unaligned,
        "rotation_changed_weights": weights_changed,
        "weight_change_in_model_verified": changed_aligned,
    }

    # -----------------------------------------------------------------------
    # Control 1: Random rotation (should be no better than unaligned)
    # -----------------------------------------------------------------------
    print("\n--- Control 1: Random rotation (should equal unaligned) ---")
    rng = np.random.default_rng(seed=99)
    random_rotated_weights = apply_random_rotation(donor_weights, rng, MODEL_CONFIG.d_model)
    model1_random_rot = transplant_head(model1, random_rotated_weights, target_L, target_H)
    changed_rr = verify_transplant_changed(model1, model1_random_rot, target_L, target_H)
    print(f"  Weight change verified: {changed_rr}")
    acc_random_rot = compute_induction_accuracy(model1_random_rot, eval_dataset)
    print(f"  Random-rotation transplant accuracy: {acc_random_rot:.4f}  "
          f"(delta vs seed1 baseline: {acc_random_rot - acc1:+.4f})")

    results["control_random_rotation"] = {
        "description": "seed0 L0H3 rotated by RANDOM orthogonal matrix -> seed1 L0H1",
        "hypothesis": "should be equivalent to unaligned — random rotation is not informative",
        "accuracy": acc_random_rot,
        "delta_vs_seed1_baseline": acc_random_rot - acc1,
        "weight_change_verified": changed_rr,
    }

    # -----------------------------------------------------------------------
    # Control 2: Procrustes-aligned non-critical head (should NOT preserve function)
    # -----------------------------------------------------------------------
    print("\n--- Control 2: Aligned non-critical donor (should still fail) ---")
    aligned_noncritical = rotate_head_weights(noncritical_weights, R)
    model1_aligned_nc = transplant_head(model1, aligned_noncritical, target_L, target_H)
    changed_anc = verify_transplant_changed(model1, model1_aligned_nc, target_L, target_H)
    print(f"  Weight change verified: {changed_anc}")
    acc_aligned_nc = compute_induction_accuracy(model1_aligned_nc, eval_dataset)
    print(f"  Aligned non-critical donor accuracy: {acc_aligned_nc:.4f}  "
          f"(delta vs seed1 baseline: {acc_aligned_nc - acc1:+.4f})")

    results["control_aligned_noncritical"] = {
        "description": "seed0 L0H0 (non-critical), Procrustes-aligned -> seed1 L0H1",
        "hypothesis": "alignment alone shouldn't rescue a non-functional head",
        "accuracy": acc_aligned_nc,
        "delta_vs_seed1_baseline": acc_aligned_nc - acc1,
        "weight_change_verified": changed_anc,
    }

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    seed1_baseline = acc1
    threshold = 0.10  # within 10% of baseline = "preserved function"

    aligned_preserves = abs(acc_aligned - seed1_baseline) < threshold * seed1_baseline
    random_rot_fails = (seed1_baseline - acc_random_rot) > threshold * seed1_baseline
    aligned_nc_fails = (seed1_baseline - acc_aligned_nc) > threshold * seed1_baseline
    alignment_is_meaningful = (acc_aligned - acc_unaligned) > threshold * seed1_baseline

    print(f"\n  Seed 1 baseline accuracy:             {seed1_baseline:.4f}")
    print(f"  Unaligned transplant:                 {acc_unaligned:.4f}  "
          f"({'PRESERVED' if abs(acc_unaligned - seed1_baseline) < threshold * seed1_baseline else 'FAILED'})")
    print(f"  Procrustes-aligned transplant:        {acc_aligned:.4f}  "
          f"({'PRESERVED' if aligned_preserves else 'FAILED'})")
    print(f"  Random-rotation transplant:           {acc_random_rot:.4f}  "
          f"({'fails as expected' if random_rot_fails else 'did NOT fail as expected'})")
    print(f"  Aligned non-critical donor:           {acc_aligned_nc:.4f}  "
          f"({'fails as expected' if aligned_nc_fails else 'did NOT fail as expected'})")
    print(f"\n  Alignment R^2: {r2:.4f} (raw basis similarity: {r2_unrotated:.4f})")
    print(f"  Improvement from alignment: {acc_aligned - acc_unaligned:+.4f}")

    if aligned_preserves and random_rot_fails and alignment_is_meaningful:
        interpretation = (
            "CIRCUITS ARE BASIS-DEPENDENT BUT ALIGNABLE. "
            "Procrustes rotation successfully recovered transplant function. "
            "Non-portability is correctable with residual stream alignment. "
            "The two networks implement the same computational circuit but in "
            "rotated bases of the residual stream. This reframes 'non-portability' "
            "as a basis mismatch problem, not a fundamental architectural divergence."
        )
    elif alignment_is_meaningful and not aligned_preserves:
        interpretation = (
            "PARTIAL BASIS ALIGNMENT EFFECT. "
            f"Procrustes rotation improved transplant accuracy by {acc_aligned - acc_unaligned:+.4f}, "
            "but did not fully recover function. "
            f"Aligned transplant accuracy {acc_aligned:.4f} still far below baseline {seed1_baseline:.4f}. "
            "Possible causes: (1) LayerNorm breaks rotation invariance, preventing perfect alignment; "
            "(2) Circuits differ in more than just basis orientation; "
            "(3) Intra-layer interactions not captured by post-L0 residual alignment. "
            "Non-portability is partially but not fully explained by basis mismatch."
        )
    elif not alignment_is_meaningful:
        interpretation = (
            "CIRCUITS ARE GENUINELY NON-PORTABLE. "
            f"Procrustes alignment produced no meaningful improvement "
            f"({acc_aligned - acc_unaligned:+.4f}). "
            f"Aligned transplant accuracy ({acc_aligned:.4f}) ≈ unaligned ({acc_unaligned:.4f}) "
            f"≈ random rotation ({acc_random_rot:.4f}). "
            "Non-portability is NOT a basis problem. "
            "The circuits are emergent system-level properties: the induction computation "
            "in seed 0 is inseparable from the full weight context of that network. "
            "No linear alignment of the residual stream can bridge the computational gap, "
            "consistent with the LayerNorm barrier and system-level emergence hypothesis."
        )
    else:
        interpretation = (
            "INCONCLUSIVE: alignment improved transplant but controls are ambiguous. "
            "More experiments needed."
        )

    print(f"\n  CONCLUSION: {interpretation}")

    results["interpretation"] = {
        "aligned_preserves_function": aligned_preserves,
        "random_rotation_fails_as_expected": random_rot_fails,
        "aligned_noncritical_fails_as_expected": aligned_nc_fails,
        "alignment_is_meaningful": alignment_is_meaningful,
        "improvement_from_alignment": acc_aligned - acc_unaligned,
        "conclusion": interpretation,
        "layernorm_caveat": (
            "LayerNorm is not rotation-invariant. Perfect Procrustes alignment of the "
            "post-L0 residual stream does not guarantee that Q/K/V (which read from "
            "LayerNorm(residual)) are perfectly aligned. R^2 < 1 is structural, not a bug."
        ),
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
