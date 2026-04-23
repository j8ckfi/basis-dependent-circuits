"""
Basis-aligned circuit transplant experiment v2.

Fixes two bugs from v1:

BUG 1 FIX: Correct Procrustes solution.
  Objective: minimize ||X_source @ R - X_target||_F, R orthogonal
  Solution: SVD of (X_source.T @ X_target) = U S V.T  =>  R = U @ V.T
  v1 used SVD of (X_target.T @ X_source), which gives the transpose.

BUG 2 FIX: Two-interface alignment.
  L0 attention operates at TWO different residual-stream interfaces:
    - Q/K/V projections READ from ln1(pre_L0_residual)
    - out_proj WRITES into pre_L0_residual (result added as residual)
  One rotation cannot simultaneously align both.
  Fix: collect pre-L0 and post-L0 activations, run two separate Procrustes
  alignments, apply each to the corresponding weight slice.

Weight convention (MLX nn.Linear stores W of shape (out, in), computes x @ W.T):
  qkv_proj.weight shape: (3*d_model, d_model)
    - Q rows: [h*d_head : (h+1)*d_head, :]  -- these are the actual weight rows
    - The forward is:  h = ln1(x) @ W_qkv.T
    - In seed0 basis: h = z0 @ W0.T  where z0 = ln1_seed0(x0)
    - After R_in:     z1 ≈ z0 @ R_in  =>  z0 ≈ z1 @ R_in.T
    - To get same output in seed1 basis: z1 @ R_in.T @ W0.T = z1 @ (W0 @ R_in).T
    - So: W_qkv_new = W0 @ R_in  (right-multiply weight rows by R_in)
    - In matrix form for the stored (out, in) weight: W_new[row, :] = W0[row, :] @ R_in

  out_proj.weight shape: (d_model, d_model)
    - The forward is: y = v_out @ W_out.T  where v_out is concatenated head outputs
    - Head h contributes cols q_start:q_end of out_proj output
    - In seed0 basis: y0 = v_head @ W_out_cols.T  where W_out_cols = out_w[:, q_start:q_end]
    - y0 gets added to pre_L0_residual to produce post_L0_residual
    - We want the rotated output to match seed1's post-L0 residual basis:
        y1 ≈ y0 @ R_out  =>  v_head @ W_out_cols.T @ R_out = v_head @ (R_out.T @ W_out_cols).T
    - So: new_out_cols = R_out.T @ W_out_cols  (left-multiply out columns by R_out.T)

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/basis_aligned_transplant_v2.py
"""

import sys
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

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
RESULTS_PATH = Path(__file__).parent / "basis_aligned_transplant_v2_results.json"

SEED0_CRITICAL_HEAD = (0, 3)    # L0H3: drops 0.81 when ablated
SEED1_CRITICAL_HEAD = (0, 1)    # L0H1: drops 0.85 when ablated
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
# Activation collection: pre-L0 (after LN) and post-L0
# ---------------------------------------------------------------------------

def collect_pre_l0_ln(model: GPT, inputs: mx.array) -> np.ndarray:
    """Collect LayerNorm(pre-L0 residual) — what Q/K/V actually read.

    In the pre-norm architecture: attn input = ln1(wte(idx) + wpe(pos))
    Returns shape (B*T, d_model).
    """
    # residuals[0] = wte(idx) + wpe(pos), shape (B, T, d_model)
    residuals = model.get_residual_stream(inputs)
    pre_l0 = residuals[0]  # (B, T, d_model)

    # Apply L0's LayerNorm (ln1) to get what Q/K/V see
    ln1 = model.blocks[0].ln1
    pre_l0_normed = ln1(pre_l0)  # (B, T, d_model)
    mx.eval(pre_l0_normed)

    arr = np.array(pre_l0_normed)
    B, T, D = arr.shape
    return arr.reshape(B * T, D)


def collect_post_l0(model: GPT, inputs: mx.array) -> np.ndarray:
    """Collect residual stream after block 0 — what out_proj writes into.

    Returns shape (B*T, d_model).
    """
    residuals = model.get_residual_stream(inputs)
    post_l0 = residuals[1]  # (B, T, d_model)
    mx.eval(post_l0)

    arr = np.array(post_l0)
    B, T, D = arr.shape
    return arr.reshape(B * T, D)


# ---------------------------------------------------------------------------
# Procrustes alignment (Bug 1 fix: correct SVD orientation)
# ---------------------------------------------------------------------------

def procrustes_orthogonal(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Find orthogonal R minimizing ||X_source @ R - X_target||_F.

    Solution: SVD of X_source.T @ X_target = U S V.T  =>  R = U @ V.T

    Args:
        X_source: (N, d) activations from source model
        X_target: (N, d) activations from target model

    Returns:
        R: (d, d) orthogonal rotation matrix such that X_source @ R ~= X_target
    """
    M = X_source.T @ X_target  # (d, d)
    U, _S, Vt = np.linalg.svd(M, full_matrices=True)
    R = U @ Vt
    return R.astype(np.float32)


def compute_alignment_r2(X_source: np.ndarray, X_target: np.ndarray, R: np.ndarray) -> float:
    """R^2 of X_source @ R vs X_target."""
    X_rotated = X_source @ R
    residuals = X_target - X_rotated
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((X_target - X_target.mean(axis=0)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_frobenius_alignment(X_source: np.ndarray, X_target: np.ndarray, R: np.ndarray) -> float:
    """1 - ||X_source @ R - X_target||_F / ||X_target||_F."""
    norm_residual = np.linalg.norm(X_source @ R - X_target, "fro")
    norm_target = np.linalg.norm(X_target, "fro")
    if norm_target == 0:
        return 1.0
    return float(1.0 - norm_residual / norm_target)


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def get_head_weights(model: GPT, layer: int, head: int) -> dict:
    """Extract Q/K/V rows and out_proj columns for a single attention head.

    qkv_proj.weight shape: (3*d_model, d_model)
      Q rows: [H*d_head : (H+1)*d_head, :]
      K rows: [d_model + H*d_head : d_model + (H+1)*d_head, :]
      V rows: [2*d_model + H*d_head : 2*d_model + (H+1)*d_head, :]

    out_proj.weight shape: (d_model, d_model)
      columns for head H: [:, H*d_head : (H+1)*d_head]
    """
    d_model = model.config.d_model
    d_head = model.config.d_head

    q_start = head * d_head
    q_end = (head + 1) * d_head

    qkv_w = model.blocks[layer].attn.qkv_proj.weight  # (3*d_model, d_model)
    out_w = model.blocks[layer].attn.out_proj.weight   # (d_model, d_model)

    q_rows = np.array(qkv_w[q_start:q_end, :])                              # (d_head, d_model)
    k_rows = np.array(qkv_w[d_model + q_start : d_model + q_end, :])        # (d_head, d_model)
    v_rows = np.array(qkv_w[2 * d_model + q_start : 2 * d_model + q_end, :])  # (d_head, d_model)
    out_cols = np.array(out_w[:, q_start:q_end])                             # (d_model, d_head)

    return {
        "q": q_rows,      # (d_head, d_model)
        "k": k_rows,      # (d_head, d_model)
        "v": v_rows,      # (d_head, d_model)
        "out": out_cols,  # (d_model, d_head)
    }


# ---------------------------------------------------------------------------
# Weight rotation: two-interface alignment (Bug 2 fix)
# ---------------------------------------------------------------------------

def rotate_head_weights_two_interface(
    weights: dict,
    R_in: np.ndarray,
    R_out: np.ndarray,
) -> dict:
    """Rotate head weights using separate pre-L0 and post-L0 alignments.

    R_in:  (d_model, d_model) aligns pre-L0 LN basis (what Q/K/V read)
           X_source_pre @ R_in ~= X_target_pre
    R_out: (d_model, d_model) aligns post-L0 basis (what out_proj writes into)
           X_source_post @ R_out ~= X_target_post

    Q/K/V rows are (d_head, d_model), the forward is:
        h = z @ W.T  where z = ln1(residual), shape (T, d_model)
        Seed0 basis: h = z0 @ W0.T
        Seed1 basis: z1 ~= z0 @ R_in  =>  z0 ~= z1 @ R_in.T
        Same output: z1 @ R_in.T @ W0.T = z1 @ (W0 @ R_in).T
        => W_new = W0 @ R_in   [shape (d_head, d_model) @ (d_model, d_model) = (d_head, d_model)]

    out_proj columns are (d_model, d_head), the forward contribution is:
        y = v_head @ out_cols.T  where v_head is (T, d_head)
        y gets residual-added into the pre-L0 stream to produce post-L0 stream.
        We want y0 @ R_out to land in seed1's post-L0 basis:
        v_head @ out_cols.T @ R_out = v_head @ (R_out.T @ out_cols).T
        => out_cols_new = R_out.T @ out_cols  [shape (d_model, d_model) @ (d_model, d_head) = (d_model, d_head)]
    """
    q = weights["q"]    # (d_head, d_model)
    k = weights["k"]    # (d_head, d_model)
    v = weights["v"]    # (d_head, d_model)
    out = weights["out"]  # (d_model, d_head)

    # Shape checks
    d_head, d_model = q.shape
    assert R_in.shape == (d_model, d_model), f"R_in shape {R_in.shape} != ({d_model},{d_model})"
    assert R_out.shape == (d_model, d_model), f"R_out shape {R_out.shape} != ({d_model},{d_model})"
    assert out.shape == (d_model, d_head), f"out shape {out.shape} != ({d_model},{d_head})"

    new_q = q @ R_in        # (d_head, d_model) @ (d_model, d_model) = (d_head, d_model)
    new_k = k @ R_in
    new_v = v @ R_in
    new_out = R_out.T @ out  # (d_model, d_model) @ (d_model, d_head) = (d_model, d_head)

    return {
        "q": new_q.astype(np.float32),
        "k": new_k.astype(np.float32),
        "v": new_v.astype(np.float32),
        "out": new_out.astype(np.float32),
    }


def rotate_head_weights_one_interface(weights: dict, R: np.ndarray) -> dict:
    """Single-rotation alignment (v1 approach, for comparison).

    Uses the same R for both Q/K/V and out_proj.
    """
    q = weights["q"]
    k = weights["k"]
    v = weights["v"]
    out = weights["out"]

    new_q = q @ R
    new_k = k @ R
    new_v = v @ R
    new_out = R.T @ out

    return {
        "q": new_q.astype(np.float32),
        "k": new_k.astype(np.float32),
        "v": new_v.astype(np.float32),
        "out": new_out.astype(np.float32),
    }


def apply_random_rotation(weights: dict, rng: np.random.Generator, d_model: int) -> dict:
    """Apply a uniformly random orthogonal rotation (both interfaces same matrix)."""
    A = rng.standard_normal((d_model, d_model)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return rotate_head_weights_one_interface(weights, Q)


# ---------------------------------------------------------------------------
# Transplantation
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
    d_head = original.config.d_head
    q_start = head * d_head
    q_end = (head + 1) * d_head
    orig_qkv = np.array(original.blocks[layer].attn.qkv_proj.weight)
    new_qkv = np.array(transplanted.blocks[layer].attn.qkv_proj.weight)
    return not np.allclose(orig_qkv[q_start:q_end, :], new_qkv[q_start:q_end, :])


# ---------------------------------------------------------------------------
# Accuracy measurement (fixed eval batch, seed=9999)
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
    print("BASIS-ALIGNED CIRCUIT TRANSPLANT v2")
    print("Two bugs fixed: (1) Procrustes SVD orientation, (2) Two-interface alignment")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print(f"\nLoading seed 0 model from: {SEED0_CKPT}")
    model0 = load_model(str(BASE_DIR / SEED0_CKPT), MODEL_CONFIG)

    print(f"Loading seed 1 model from: {SEED1_CKPT}")
    model1 = load_model(str(BASE_DIR / SEED1_CKPT), MODEL_CONFIG)

    # Fixed eval dataset (seed=9999 for fairness across all conditions)
    eval_dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=9999)

    results = {}

    # -----------------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------------
    print("\n--- Baselines ---")
    acc0 = compute_induction_accuracy(model0, eval_dataset)
    acc1 = compute_induction_accuracy(model1, eval_dataset)
    print(f"  Seed 0 baseline accuracy: {acc0:.4f}")
    print(f"  Seed 1 baseline accuracy: {acc1:.4f}")
    results["baselines"] = {"seed0": acc0, "seed1": acc1}

    seed1_baseline = acc1

    # -----------------------------------------------------------------------
    # Collect activations for alignment
    # Use a separate batch (seed=1234) so eval batch (seed=9999) is clean
    # -----------------------------------------------------------------------
    print("\n--- Collecting activations for Procrustes alignment ---")
    align_dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=1234)
    align_inputs, _, _ = align_dataset.generate_batch(256)
    mx.eval(align_inputs)

    print("  Collecting pre-L0 LN activations (what Q/K/V read)...")
    pre_ln_seed0 = collect_pre_l0_ln(model0, align_inputs)   # (256*64, 128)
    pre_ln_seed1 = collect_pre_l0_ln(model1, align_inputs)
    print(f"  Pre-L0 LN shapes: {pre_ln_seed0.shape}, {pre_ln_seed1.shape}")

    print("  Collecting post-L0 activations (what out_proj writes into)...")
    post_l0_seed0 = collect_post_l0(model0, align_inputs)    # (256*64, 128)
    post_l0_seed1 = collect_post_l0(model1, align_inputs)
    print(f"  Post-L0 shapes: {post_l0_seed0.shape}, {post_l0_seed1.shape}")

    # -----------------------------------------------------------------------
    # Compute two Procrustes alignments (corrected SVD orientation)
    # -----------------------------------------------------------------------
    print("\n--- Computing Procrustes alignments ---")
    R_in = procrustes_orthogonal(pre_ln_seed0, pre_ln_seed1)
    R_out = procrustes_orthogonal(post_l0_seed0, post_l0_seed1)

    orth_err_in  = float(np.linalg.norm(R_in.T @ R_in - np.eye(R_in.shape[0])))
    orth_err_out = float(np.linalg.norm(R_out.T @ R_out - np.eye(R_out.shape[0])))
    print(f"  R_in  orthogonality error: {orth_err_in:.2e}")
    print(f"  R_out orthogonality error: {orth_err_out:.2e}")

    r2_in         = compute_alignment_r2(pre_ln_seed0, pre_ln_seed1, R_in)
    frob_in       = compute_frobenius_alignment(pre_ln_seed0, pre_ln_seed1, R_in)
    r2_out        = compute_alignment_r2(post_l0_seed0, post_l0_seed1, R_out)
    frob_out      = compute_frobenius_alignment(post_l0_seed0, post_l0_seed1, R_out)
    r2_in_raw     = compute_alignment_r2(pre_ln_seed0, pre_ln_seed1, np.eye(MODEL_CONFIG.d_model))
    frob_in_raw   = compute_frobenius_alignment(pre_ln_seed0, pre_ln_seed1, np.eye(MODEL_CONFIG.d_model))
    r2_out_raw    = compute_alignment_r2(post_l0_seed0, post_l0_seed1, np.eye(MODEL_CONFIG.d_model))
    frob_out_raw  = compute_frobenius_alignment(post_l0_seed0, post_l0_seed1, np.eye(MODEL_CONFIG.d_model))

    print(f"  Pre-L0 LN  R2 (Procrustes): {r2_in:.4f}  (unrotated: {r2_in_raw:.4f})")
    print(f"  Post-L0    R2 (Procrustes): {r2_out:.4f}  (unrotated: {r2_out_raw:.4f})")
    print(f"  Pre-L0 LN  Frob align:      {frob_in:.4f}  (unrotated: {frob_in_raw:.4f})")
    print(f"  Post-L0    Frob align:      {frob_out:.4f}  (unrotated: {frob_out_raw:.4f})")

    results["alignment_diagnostics"] = {
        "pre_l0_ln": {
            "r2_procrustes":      r2_in,
            "r2_unrotated":       r2_in_raw,
            "frob_procrustes":    frob_in,
            "frob_unrotated":     frob_in_raw,
            "orthogonality_error": orth_err_in,
        },
        "post_l0": {
            "r2_procrustes":      r2_out,
            "r2_unrotated":       r2_out_raw,
            "frob_procrustes":    frob_out,
            "frob_unrotated":     frob_out_raw,
            "orthogonality_error": orth_err_out,
        },
        "note": (
            "LayerNorm is not rotation-invariant, so R2 < 1 is structural. "
            "pre_l0_ln applies ln1 before collecting activations so alignment "
            "is at the exact interface Q/K/V read from."
        ),
    }

    # -----------------------------------------------------------------------
    # Also compute v1-style single-interface rotation (post-L0) for comparison
    # -----------------------------------------------------------------------
    R_single = procrustes_orthogonal(post_l0_seed0, post_l0_seed1)

    # -----------------------------------------------------------------------
    # Extract donor weights
    # -----------------------------------------------------------------------
    donor_L, donor_H = SEED0_CRITICAL_HEAD
    donor_weights = get_head_weights(model0, donor_L, donor_H)
    print(f"\nExtracted seed0 L{donor_L}H{donor_H} weights (Q shape {donor_weights['q'].shape})")

    ctrl_L, ctrl_H = SEED0_NONCRITICAL_HEAD
    noncritical_weights = get_head_weights(model0, ctrl_L, ctrl_H)

    target_L, target_H = SEED1_CRITICAL_HEAD

    # -----------------------------------------------------------------------
    # Experiment 1: Unaligned transplant (baseline)
    # -----------------------------------------------------------------------
    print("\n--- Experiment 1: Unaligned transplant (baseline) ---")
    model1_unaligned = transplant_head(model1, donor_weights, target_L, target_H)
    acc_unaligned = compute_induction_accuracy(model1_unaligned, eval_dataset)
    print(f"  Accuracy: {acc_unaligned:.4f}  (delta vs seed1: {acc_unaligned - seed1_baseline:+.4f})")
    results["unaligned_transplant"] = {
        "description": "seed0 L0H3 -> seed1 L0H1, no alignment",
        "accuracy": acc_unaligned,
        "delta_vs_seed1_baseline": acc_unaligned - seed1_baseline,
    }

    # -----------------------------------------------------------------------
    # Experiment 2: One-interface aligned (v1 method, post-L0 residual only)
    # -----------------------------------------------------------------------
    print("\n--- Experiment 2: One-interface aligned (v1 method, post-L0, corrected SVD) ---")
    single_aligned_weights = rotate_head_weights_one_interface(donor_weights, R_single)
    model1_single = transplant_head(model1, single_aligned_weights, target_L, target_H)
    acc_single = compute_induction_accuracy(model1_single, eval_dataset)
    print(f"  Accuracy: {acc_single:.4f}  (delta vs seed1: {acc_single - seed1_baseline:+.4f})")
    print(f"  Improvement vs unaligned: {acc_single - acc_unaligned:+.4f}")
    results["one_interface_aligned"] = {
        "description": "seed0 L0H3 -> seed1 L0H1, single-rotation aligned (post-L0, corrected SVD)",
        "accuracy": acc_single,
        "delta_vs_seed1_baseline": acc_single - seed1_baseline,
        "improvement_vs_unaligned": acc_single - acc_unaligned,
    }

    # -----------------------------------------------------------------------
    # Experiment 3: Two-interface aligned (correct method)
    # -----------------------------------------------------------------------
    print("\n--- Experiment 3: Two-interface aligned (correct method) ---")
    two_iface_weights = rotate_head_weights_two_interface(donor_weights, R_in, R_out)
    model1_two_iface = transplant_head(model1, two_iface_weights, target_L, target_H)
    acc_two_iface = compute_induction_accuracy(model1_two_iface, eval_dataset)
    print(f"  Accuracy: {acc_two_iface:.4f}  (delta vs seed1: {acc_two_iface - seed1_baseline:+.4f})")
    print(f"  Improvement vs unaligned: {acc_two_iface - acc_unaligned:+.4f}")
    print(f"  Improvement vs one-interface: {acc_two_iface - acc_single:+.4f}")
    results["two_interface_aligned"] = {
        "description": (
            "seed0 L0H3 -> seed1 L0H1, two-rotation aligned "
            "(R_in for Q/K/V, R_out for out_proj)"
        ),
        "accuracy": acc_two_iface,
        "delta_vs_seed1_baseline": acc_two_iface - seed1_baseline,
        "improvement_vs_unaligned": acc_two_iface - acc_unaligned,
        "improvement_vs_one_interface": acc_two_iface - acc_single,
    }

    # -----------------------------------------------------------------------
    # Experiment 4: Random rotation control
    # -----------------------------------------------------------------------
    print("\n--- Experiment 4: Random rotation control ---")
    rng = np.random.default_rng(seed=99)
    random_weights = apply_random_rotation(donor_weights, rng, MODEL_CONFIG.d_model)
    model1_random = transplant_head(model1, random_weights, target_L, target_H)
    acc_random = compute_induction_accuracy(model1_random, eval_dataset)
    print(f"  Accuracy: {acc_random:.4f}  (delta vs seed1: {acc_random - seed1_baseline:+.4f})")
    results["random_rotation_control"] = {
        "description": "seed0 L0H3 rotated by random orthogonal matrix -> seed1 L0H1",
        "hypothesis": "should be equivalent to unaligned — random rotation is not informative",
        "accuracy": acc_random,
        "delta_vs_seed1_baseline": acc_random - seed1_baseline,
    }

    # -----------------------------------------------------------------------
    # Experiment 5: Non-critical donor, two-interface aligned
    # -----------------------------------------------------------------------
    print("\n--- Experiment 5: Non-critical donor (two-interface aligned) ---")
    noncritical_two_iface = rotate_head_weights_two_interface(noncritical_weights, R_in, R_out)
    model1_nc = transplant_head(model1, noncritical_two_iface, target_L, target_H)
    acc_nc = compute_induction_accuracy(model1_nc, eval_dataset)
    print(f"  Accuracy: {acc_nc:.4f}  (delta vs seed1: {acc_nc - seed1_baseline:+.4f})")
    results["noncritical_donor_aligned"] = {
        "description": "seed0 L0H0 (non-critical), two-interface aligned -> seed1 L0H1",
        "hypothesis": "alignment alone should not rescue a non-functional head",
        "accuracy": acc_nc,
        "delta_vs_seed1_baseline": acc_nc - seed1_baseline,
    }

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"{'Configuration':<45} | {'Accuracy':>8} | {'Delta vs 0.964':>14}")
    print("-" * 70)

    def row(label, acc, baseline=seed1_baseline):
        delta = acc - baseline
        return f"{label:<45} | {acc:>8.4f} | {delta:>+14.4f}"

    print(f"{'Seed 1 baseline':<45} | {seed1_baseline:>8.4f} | {'—':>14}")
    print(row("Unaligned transplant",                   acc_unaligned))
    print(row("One-interface (old) aligned",            acc_single))
    print(row("Two-interface (correct) aligned",        acc_two_iface))
    print(row("Random rotation control",                acc_random))
    print(row("Non-critical aligned (correct method)",  acc_nc))
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("\n--- INTERPRETATION ---")
    threshold = 0.10

    two_iface_preserves = abs(acc_two_iface - seed1_baseline) < threshold * seed1_baseline
    random_fails        = (seed1_baseline - acc_random) > threshold * seed1_baseline
    nc_fails            = (seed1_baseline - acc_nc)     > threshold * seed1_baseline
    two_iface_beats_one = (acc_two_iface - acc_single)  > threshold * seed1_baseline * 0.1
    meaningful_gain     = (acc_two_iface - acc_unaligned) > threshold * seed1_baseline

    if two_iface_preserves and random_fails:
        conclusion = (
            "CIRCUITS ARE BASIS-DEPENDENT BUT ALIGNABLE. "
            "Two-interface Procrustes alignment fully rescued the transplant. "
            "Non-portability is a basis mismatch problem, not fundamental architectural divergence."
        )
    elif meaningful_gain and not two_iface_preserves:
        conclusion = (
            "PARTIAL BASIS ALIGNMENT EFFECT. "
            f"Two-interface alignment improved accuracy by {acc_two_iface - acc_unaligned:+.4f} "
            f"but did not reach baseline ({seed1_baseline:.4f}). "
            "Circuits partially share structure but LayerNorm non-linearity prevents perfect alignment."
        )
    else:
        conclusion = (
            "CIRCUITS ARE GENUINELY NON-PORTABLE. "
            f"Two-interface alignment gain: {acc_two_iface - acc_unaligned:+.4f} — not meaningful. "
            "Non-portability is NOT a basis problem. "
            "The induction circuit in seed 0 is inseparable from its full weight context."
        )

    print(f"\n  {conclusion}")
    print(f"\n  Key comparisons:")
    print(f"    Two-interface vs one-interface: {acc_two_iface - acc_single:+.4f}")
    print(f"    Two-interface vs unaligned:     {acc_two_iface - acc_unaligned:+.4f}")
    print(f"    Two-interface vs random:        {acc_two_iface - acc_random:+.4f}")
    print(f"\n  Alignment quality:")
    print(f"    Pre-L0 LN R2:  {r2_in:.4f}  (raw: {r2_in_raw:.4f})")
    print(f"    Post-L0 R2:    {r2_out:.4f}  (raw: {r2_out_raw:.4f})")

    results["interpretation"] = {
        "conclusion": conclusion,
        "two_interface_preserves_function": two_iface_preserves,
        "random_rotation_fails_as_expected": random_fails,
        "noncritical_fails_as_expected": nc_fails,
        "meaningful_gain_over_unaligned": meaningful_gain,
        "two_interface_vs_one_interface_gain": acc_two_iface - acc_single,
        "two_interface_vs_unaligned_gain": acc_two_iface - acc_unaligned,
        "layernorm_caveat": (
            "LayerNorm is not rotation-invariant. We align at ln1(pre_L0_residual) which is the "
            "exact interface Q/K/V read from — this is the theoretically correct alignment point. "
            "R2 < 1 reflects that the two models have genuinely different LN-transformed representations, "
            "not a bug in the alignment procedure."
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
