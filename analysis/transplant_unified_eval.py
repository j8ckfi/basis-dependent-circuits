"""
Unified transplant evaluation with fixed materialized eval set.

REVIEWER BUG FIX:
  Previous scripts called dataset.generate_batch() inside each eval loop,
  advancing the stateful RNG. This meant each condition saw DIFFERENT
  sequences, making accuracy values (e.g. 0.092 vs 0.080) not directly
  comparable.

FIX:
  Generate ONE fixed eval set of 1024 sequences ONCE at script start.
  Materialize as a list of (inputs, targets, mask) tuples.
  Pass this list to every evaluation call — no RNG advances during eval.

Experiments re-run under this protocol:
  Experiment 1: Basic transplant (circuit_transplant.py logic)
  Experiment 2: Procrustes aligned (basis_aligned_transplant_v2.py logic)
  Experiment 3: Whole-layer (whole_layer_transplant.py logic)

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/transplant_unified_eval.py
"""

import sys
import json
import copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import List, Tuple

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
RESULTS_PATH = Path(__file__).parent / "transplant_unified_results.json"

# Key heads identified from ablation analysis
SEED0_CRITICAL_HEAD = (0, 3)    # L0H3: drops 0.81 when ablated
SEED1_CRITICAL_HEAD = (0, 1)    # L0H1: drops 0.85 when ablated
SEED0_NONCRITICAL_HEAD = (0, 0)  # presumed non-critical

# Eval protocol parameters
EVAL_N_SEQUENCES = 1024
EVAL_BATCH_SIZE = 32        # 1024 / 32 = 32 batches
EVAL_DATA_SEED = 9999
EVAL_VOCAB_SIZE = 512
EVAL_SEQ_LEN = 64
EVAL_N_BIGRAMS = 6

# Bootstrap parameters
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Fixed materialized eval set
# ---------------------------------------------------------------------------

FixedEvalSet = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]


def build_fixed_eval_set() -> FixedEvalSet:
    """Generate ONE fixed eval set, materialized as a list of numpy tuples.

    Creates 1024 sequences total (32 batches x 32 sequences).
    The dataset RNG is consumed ONCE here and never advanced again during eval.
    All conditions share exactly the same sequences.

    Returns:
        List of (inputs_np, targets_np, mask_np) tuples, each of shape
        (batch_size, seq_len), (batch_size, seq_len), (batch_size, seq_len).
    """
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
        mx.eval(inputs_mx)
        mx.eval(targets_mx)
        mx.eval(mask_mx)
        fixed_batches.append((
            np.array(inputs_mx),
            np.array(targets_mx),
            np.array(mask_mx),
        ))

    total_seqs = sum(b[0].shape[0] for b in fixed_batches)
    print(f"  [eval set] Materialized {len(fixed_batches)} batches x {EVAL_BATCH_SIZE} = {total_seqs} sequences")
    print(f"  [eval set] Params: vocab={EVAL_VOCAB_SIZE}, seq_len={EVAL_SEQ_LEN}, n_bigrams={EVAL_N_BIGRAMS}, seed={EVAL_DATA_SEED}")
    return fixed_batches


def evaluate_on_fixed_set(model_fn, fixed_eval_set: FixedEvalSet) -> Tuple[float, np.ndarray]:
    """Compute induction accuracy using pre-materialized batches.

    No dataset state is touched. All batches are already in memory as numpy arrays.

    Args:
        model_fn: a GPT model (callable).
        fixed_eval_set: list of (inputs_np, targets_np, mask_np) tuples.

    Returns:
        (accuracy, per_sequence_correct_frac) where per_sequence_correct_frac
        has one entry per sequence (fraction of masked positions correct).
    """
    per_seq_fracs = []

    for inputs_np, targets_np, mask_np in fixed_eval_set:
        inputs_mx = mx.array(inputs_np)
        logits = model_fn(inputs_mx)
        mx.eval(logits)

        logits_np = np.array(logits)
        preds = logits_np.argmax(axis=-1)  # (B, T)
        at_mask = mask_np > 0.5            # (B, T)

        correct = (preds == targets_np) & at_mask  # (B, T)

        # Per-sequence fraction of masked positions correct
        for b in range(inputs_np.shape[0]):
            n_mask = at_mask[b].sum()
            if n_mask > 0:
                per_seq_fracs.append(correct[b].sum() / n_mask)
            else:
                per_seq_fracs.append(0.0)

    per_seq_arr = np.array(per_seq_fracs)
    accuracy = float(per_seq_arr.mean())
    return accuracy, per_seq_arr


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    fracs_a: np.ndarray,
    fracs_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for mean(fracs_b - fracs_a).

    Args:
        fracs_a: per-sequence accuracy fractions for condition A (baseline)
        fracs_b: per-sequence accuracy fractions for condition B (comparison)

    Returns:
        (mean_diff, ci_lo, ci_hi) where mean_diff = mean(b - a)
    """
    assert len(fracs_a) == len(fracs_b), "Must have same number of sequences"
    diff = fracs_b - fracs_a
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    mean_diff = float(diff.mean())
    return mean_diff, ci_lo, ci_hi


def paired_sign_flip_pvalue(a_per_seq, b_per_seq, n_permutations=10000, alternative='two-sided'):
    """
    Paired sign-flip permutation test for H0: mean(a - b) = 0.

    a_per_seq and b_per_seq are arrays of per-sequence accuracy (0 or fraction correct)
    of length N. The observed statistic is mean(a - b). Under H0, the sign of each
    paired difference is exchangeable, so we randomly flip signs and recompute.
    """
    diffs = a_per_seq - b_per_seq
    observed_stat = np.mean(diffs)
    n = len(diffs)

    # Sign-flip permutations
    more_extreme = 0
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_diffs = diffs * signs
        perm_stat = np.mean(perm_diffs)

        if alternative == 'two-sided':
            if abs(perm_stat) >= abs(observed_stat):
                more_extreme += 1
        elif alternative == 'greater':
            if perm_stat >= observed_stat:
                more_extreme += 1
        elif alternative == 'less':
            if perm_stat <= observed_stat:
                more_extreme += 1

    p_value = (more_extreme + 1) / (n_permutations + 1)
    return p_value, observed_stat


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Head-level weight utilities (from circuit_transplant.py)
# ---------------------------------------------------------------------------

def get_head_weights(model: GPT, layer: int, head: int) -> dict:
    d_model = model.config.d_model
    d_head = model.config.d_head
    q_start = head * d_head
    q_end = (head + 1) * d_head

    qkv_w = model.blocks[layer].attn.qkv_proj.weight
    out_w = model.blocks[layer].attn.out_proj.weight

    return {
        "q":   np.array(qkv_w[q_start:q_end, :]),
        "k":   np.array(qkv_w[d_model + q_start : d_model + q_end, :]),
        "v":   np.array(qkv_w[2 * d_model + q_start : 2 * d_model + q_end, :]),
        "out": np.array(out_w[:, q_start:q_end]),
    }


def transplant_head(recipient_model: GPT, donor_weights: dict, target_layer: int, target_head: int) -> GPT:
    config = recipient_model.config
    d_model = config.d_model
    d_head = config.d_head
    q_start = target_head * d_head
    q_end = (target_head + 1) * d_head

    qkv_w = np.array(recipient_model.blocks[target_layer].attn.qkv_proj.weight)
    out_w = np.array(recipient_model.blocks[target_layer].attn.out_proj.weight)

    qkv_w[q_start:q_end, :] = donor_weights["q"]
    qkv_w[d_model + q_start : d_model + q_end, :] = donor_weights["k"]
    qkv_w[2 * d_model + q_start : 2 * d_model + q_end, :] = donor_weights["v"]
    out_w[:, q_start:q_end] = donor_weights["out"]

    flat = dict(nn.utils.tree_flatten(recipient_model.parameters()))
    weights = {k: np.array(v) for k, v in flat.items()}
    weights[f"blocks.{target_layer}.attn.qkv_proj.weight"] = qkv_w
    weights[f"blocks.{target_layer}.attn.out_proj.weight"] = out_w

    new_model = GPT(config)
    new_model.load_weights([(k, mx.array(v)) for k, v in weights.items()])
    mx.eval(new_model.parameters())
    return new_model


def zero_head(model: GPT, target_layer: int, target_head: int) -> GPT:
    d_head = model.config.d_head
    zeros = {
        "q":   np.zeros((d_head, model.config.d_model), dtype=np.float32),
        "k":   np.zeros((d_head, model.config.d_model), dtype=np.float32),
        "v":   np.zeros((d_head, model.config.d_model), dtype=np.float32),
        "out": np.zeros((model.config.d_model, d_head), dtype=np.float32),
    }
    return transplant_head(model, zeros, target_layer, target_head)


def verify_head_changed(original: GPT, transplanted: GPT, layer: int, head: int) -> bool:
    d_head = original.config.d_head
    q_start = head * d_head
    q_end = (head + 1) * d_head
    orig_qkv = np.array(original.blocks[layer].attn.qkv_proj.weight)
    new_qkv = np.array(transplanted.blocks[layer].attn.qkv_proj.weight)
    return not np.allclose(orig_qkv[q_start:q_end, :], new_qkv[q_start:q_end, :])


# ---------------------------------------------------------------------------
# Procrustes alignment utilities (from basis_aligned_transplant_v2.py)
# ---------------------------------------------------------------------------

def collect_pre_l0_ln(model: GPT, inputs: mx.array) -> np.ndarray:
    residuals = model.get_residual_stream(inputs)
    pre_l0 = residuals[0]
    ln1 = model.blocks[0].ln1
    pre_l0_normed = ln1(pre_l0)
    mx.eval(pre_l0_normed)
    arr = np.array(pre_l0_normed)
    B, T, D = arr.shape
    return arr.reshape(B * T, D)


def collect_post_l0(model: GPT, inputs: mx.array) -> np.ndarray:
    residuals = model.get_residual_stream(inputs)
    post_l0 = residuals[1]
    mx.eval(post_l0)
    arr = np.array(post_l0)
    B, T, D = arr.shape
    return arr.reshape(B * T, D)


def procrustes_orthogonal(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Find R minimizing ||X_source @ R - X_target||_F. R = U @ Vt."""
    M = X_source.T @ X_target
    U, _S, Vt = np.linalg.svd(M, full_matrices=True)
    return (U @ Vt).astype(np.float32)


def rotate_head_weights_two_interface(weights: dict, R_in: np.ndarray, R_out: np.ndarray) -> dict:
    return {
        "q":   (weights["q"] @ R_in).astype(np.float32),
        "k":   (weights["k"] @ R_in).astype(np.float32),
        "v":   (weights["v"] @ R_in).astype(np.float32),
        "out": (R_out.T @ weights["out"]).astype(np.float32),
    }


def rotate_head_weights_one_interface(weights: dict, R: np.ndarray) -> dict:
    return {
        "q":   (weights["q"] @ R).astype(np.float32),
        "k":   (weights["k"] @ R).astype(np.float32),
        "v":   (weights["v"] @ R).astype(np.float32),
        "out": (R.T @ weights["out"]).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Whole-layer weight utilities (from whole_layer_transplant.py)
# ---------------------------------------------------------------------------

def get_flat_weights(model: GPT) -> dict:
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    return {k: np.array(v) for k, v in flat.items()}


def build_model_from_weights(weights: dict, config: GPTConfig) -> GPT:
    new_model = GPT(config)
    new_model.load_weights([(k, mx.array(v)) for k, v in weights.items()])
    mx.eval(new_model.parameters())
    return new_model


def splice_keys(recipient: dict, donor: dict, keys: list) -> dict:
    result = dict(recipient)
    for k in keys:
        result[k] = donor[k].copy()
    return result


def verify_keys_changed(orig: dict, modified: dict, keys: list) -> bool:
    return any(not np.allclose(orig[k], modified[k]) for k in keys)


def keys_l0_attention(layer: int = 0) -> list:
    return [f"blocks.{layer}.attn.qkv_proj.weight", f"blocks.{layer}.attn.out_proj.weight"]


def keys_l0_full_block(layer: int = 0) -> list:
    return [
        f"blocks.{layer}.ln1.weight",
        f"blocks.{layer}.attn.qkv_proj.weight",
        f"blocks.{layer}.attn.out_proj.weight",
        f"blocks.{layer}.ln2.weight",
        f"blocks.{layer}.mlp.up_proj.weight",
        f"blocks.{layer}.mlp.down_proj.weight",
    ]


def keys_ln_f() -> list:
    return ["ln_f.weight"]


def keys_embeddings() -> list:
    return ["wte.weight", "wpe.weight"]


# ---------------------------------------------------------------------------
# Experiment 1: Basic transplant
# ---------------------------------------------------------------------------

def run_experiment_1(model0: GPT, model1: GPT, fixed_eval_set: FixedEvalSet) -> Tuple[dict, dict]:
    """Basic single-head transplant experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Basic transplant (circuit_transplant.py logic)")
    print("=" * 70)

    donor_L, donor_H = SEED0_CRITICAL_HEAD
    target_L, target_H = SEED1_CRITICAL_HEAD
    ctrl_L, ctrl_H = SEED0_NONCRITICAL_HEAD

    donor_weights = get_head_weights(model0, donor_L, donor_H)
    ctrl_weights = get_head_weights(model0, ctrl_L, ctrl_H)

    results = []
    fracs_store = {}

    # Seed 0 L0H3 -> seed 1 L0H3 (non-critical slot)
    print("\n  [1a] Seed0 L0H3 -> Seed1 L0H3 (non-critical slot)")
    m = transplant_head(model1, donor_weights, target_layer=0, target_head=3)
    changed = verify_head_changed(model1, m, layer=0, head=3)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["transplant_noncrit_slot"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Seed0 L0H3 -> Seed1 L0H3 (non-critical slot)",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # Seed 0 L0H3 -> seed 1 L0H1 (critical slot) — key experiment
    print("\n  [1b] Seed0 L0H3 -> Seed1 L0H1 (critical slot)")
    m = transplant_head(model1, donor_weights, target_layer=target_L, target_head=target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["transplant_crit_slot"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Seed0 L0H3 -> Seed1 L0H1 (critical slot, unaligned)",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # Control A: non-critical donor -> seed1 L0H1
    print("\n  [1c] Control A: Seed0 L0H0 (non-critical) -> Seed1 L0H1")
    m = transplant_head(model1, ctrl_weights, target_layer=target_L, target_head=target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["ctrl_a_noncrit_donor"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Control A: Seed0 non-critical L0H0 -> Seed1 L0H1",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # Control B: shuffled weights -> seed1 L0H1
    print("\n  [1d] Control B: Seed0 L0H3 shuffled -> Seed1 L0H1")
    rng = np.random.default_rng(seed=0)
    shuffled = {
        k: rng.permutation(donor_weights[k].flatten()).reshape(donor_weights[k].shape).astype(np.float32)
        for k in ("q", "k", "v", "out")
    }
    m = transplant_head(model1, shuffled, target_layer=target_L, target_head=target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["ctrl_b_shuffled"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Control B: Seed0 L0H3 shuffled weights -> Seed1 L0H1",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # Ablate seed1 L0H1 (zero the head)
    print("\n  [1e] Ablate Seed1 L0H1 (zero weights)")
    m = zero_head(model1, target_layer=target_L, target_head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["ablated_l0h1"] = fracs
    print(f"    Accuracy: {acc:.4f}")
    results.append({
        "condition": "Ablate Seed1 L0H1 (no replacement)",
        "accuracy": acc,
        "weight_change_verified": True,
    })

    return {"basic_transplants": results}, fracs_store


# ---------------------------------------------------------------------------
# Experiment 2: Procrustes aligned transplant
# ---------------------------------------------------------------------------

def run_experiment_2(model0: GPT, model1: GPT, fixed_eval_set: FixedEvalSet) -> Tuple[dict, dict]:
    """Procrustes aligned transplant experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Procrustes aligned transplant (basis_aligned_transplant_v2.py logic)")
    print("=" * 70)

    donor_L, donor_H = SEED0_CRITICAL_HEAD
    target_L, target_H = SEED1_CRITICAL_HEAD
    ctrl_L, ctrl_H = SEED0_NONCRITICAL_HEAD

    donor_weights = get_head_weights(model0, donor_L, donor_H)
    ctrl_weights = get_head_weights(model0, ctrl_L, ctrl_H)

    # Collect alignment activations using a separate dataset (not the fixed eval set)
    print("\n  Collecting activations for Procrustes alignment (align_seed=1234)...")
    align_dataset = InductionDataset(vocab_size=EVAL_VOCAB_SIZE, seq_len=EVAL_SEQ_LEN, n_bigrams=EVAL_N_BIGRAMS, seed=1234)
    align_inputs_mx, _, _ = align_dataset.generate_batch(256)
    mx.eval(align_inputs_mx)

    pre_ln_seed0 = collect_pre_l0_ln(model0, align_inputs_mx)
    pre_ln_seed1 = collect_pre_l0_ln(model1, align_inputs_mx)
    post_l0_seed0 = collect_post_l0(model0, align_inputs_mx)
    post_l0_seed1 = collect_post_l0(model1, align_inputs_mx)

    R_in = procrustes_orthogonal(pre_ln_seed0, pre_ln_seed1)
    R_out = procrustes_orthogonal(post_l0_seed0, post_l0_seed1)
    R_single = procrustes_orthogonal(post_l0_seed0, post_l0_seed1)

    orth_err_in = float(np.linalg.norm(R_in.T @ R_in - np.eye(R_in.shape[0])))
    orth_err_out = float(np.linalg.norm(R_out.T @ R_out - np.eye(R_out.shape[0])))
    print(f"  R_in orthogonality error: {orth_err_in:.2e}")
    print(f"  R_out orthogonality error: {orth_err_out:.2e}")

    results = []
    fracs_store = {}

    # Unaligned transplant (baseline for this experiment)
    print("\n  [2a] Unaligned transplant (seed0 L0H3 -> seed1 L0H1, no alignment)")
    m = transplant_head(model1, donor_weights, target_L, target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["unaligned"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Unaligned transplant",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # One-interface aligned (pre-L0 only)
    print("\n  [2b] One-interface aligned (pre-L0 only)")
    one_iface_w = rotate_head_weights_one_interface(donor_weights, R_single)
    m = transplant_head(model1, one_iface_w, target_L, target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["one_iface_aligned"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "One-interface aligned (post-L0 only)",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # Two-interface aligned (pre-L0 for Q/K/V, post-L0 for out_proj)
    print("\n  [2c] Two-interface aligned (pre-L0 for Q/K/V, post-L0 for out_proj)")
    two_iface_w = rotate_head_weights_two_interface(donor_weights, R_in, R_out)
    m = transplant_head(model1, two_iface_w, target_L, target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["two_iface_aligned"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Two-interface aligned (pre-L0 Q/K/V + post-L0 out_proj)",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # Random orthogonal rotation control
    print("\n  [2d] Random orthogonal rotation control")
    rng = np.random.default_rng(seed=99)
    A = rng.standard_normal((MODEL_CONFIG.d_model, MODEL_CONFIG.d_model)).astype(np.float32)
    Q_rand, _ = np.linalg.qr(A)
    rand_w = rotate_head_weights_one_interface(donor_weights, Q_rand)
    m = transplant_head(model1, rand_w, target_L, target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["random_rotation"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Random orthogonal rotation control",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    # Aligned non-critical donor
    print("\n  [2e] Two-interface aligned non-critical donor (Seed0 L0H0)")
    nc_two_iface_w = rotate_head_weights_two_interface(ctrl_weights, R_in, R_out)
    m = transplant_head(model1, nc_two_iface_w, target_L, target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["aligned_noncrit_donor"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({
        "condition": "Aligned non-critical donor (Seed0 L0H0, two-interface)",
        "accuracy": acc,
        "weight_change_verified": changed,
    })

    alignment_diagnostics = {
        "R_in_orthogonality_error": orth_err_in,
        "R_out_orthogonality_error": orth_err_out,
    }

    return {"procrustes_alignment": results, "alignment_diagnostics": alignment_diagnostics}, fracs_store


# ---------------------------------------------------------------------------
# Experiment 3: Whole-layer transplant
# ---------------------------------------------------------------------------

def run_experiment_3(model0: GPT, model1: GPT, fixed_eval_set: FixedEvalSet) -> Tuple[dict, dict]:
    """Whole-layer transplant experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Whole-layer transplant (whole_layer_transplant.py logic)")
    print("=" * 70)

    w0 = get_flat_weights(model0)
    w1 = get_flat_weights(model1)

    # Reference: single-head transplant (from experiment 1 — do NOT re-run, already done)
    donor_weights = get_head_weights(model0, *SEED0_CRITICAL_HEAD)
    target_L, target_H = SEED1_CRITICAL_HEAD

    def run_scope(label: str, keys: list) -> Tuple[dict, np.ndarray]:
        modified = splice_keys(w1, w0, keys)
        changed = verify_keys_changed(w1, modified, keys)
        m = build_model_from_weights(modified, MODEL_CONFIG)
        acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
        print(f"    [{label}]  changed={changed}  acc={acc:.4f}  ({len(keys)} tensors)")
        return {"condition": label, "accuracy": acc, "weight_change_verified": changed, "n_tensors": len(keys)}, fracs

    results = []
    fracs_store = {}

    # Single-head L0H3 -> L0H1 (reference, re-run on fixed set for direct comparison)
    print("\n  [3a] Single-head L0H3 -> L0H1 (reference from Exp1)")
    m = transplant_head(model1, donor_weights, target_L, target_H)
    changed = verify_head_changed(model1, m, layer=target_L, head=target_H)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval_set)
    fracs_store["single_head_ref"] = fracs
    print(f"    Weight changed: {changed}  |  Accuracy: {acc:.4f}")
    results.append({"condition": "Single-head L0H3 -> L0H1 (reference)", "accuracy": acc, "weight_change_verified": changed})

    print("\n  [3b] Whole L0 attention block (all 4 heads)")
    r, fracs = run_scope("Whole L0 attention block (all 4 heads)", keys_l0_attention(0))
    fracs_store["whole_l0_attn"] = fracs
    results.append(r)

    print("\n  [3c] Whole L0 block (attn + MLP + LayerNorms)")
    r, fracs = run_scope("Whole L0 block (attn + MLP + LayerNorms)", keys_l0_full_block(0))
    fracs_store["whole_l0_block"] = fracs
    results.append(r)

    print("\n  [3d] Whole L0 block + final LayerNorm")
    r, fracs = run_scope("Whole L0 block + final LayerNorm", keys_l0_full_block(0) + keys_ln_f())
    fracs_store["whole_l0_block_ln_f"] = fracs
    results.append(r)

    print("\n  [3e] Embeddings only (wte + wpe)")
    r, fracs = run_scope("Embeddings only (wte + wpe)", keys_embeddings())
    fracs_store["embeddings_only"] = fracs
    results.append(r)

    print("\n  [3f] Embeddings + L0 attention")
    r, fracs = run_scope("Embeddings + L0 attention", keys_embeddings() + keys_l0_attention(0))
    fracs_store["embed_l0_attn"] = fracs
    results.append(r)

    print("\n  [3g] Embeddings + L0 full block")
    r, fracs = run_scope("Embeddings + L0 full block", keys_embeddings() + keys_l0_full_block(0))
    fracs_store["embed_l0_block"] = fracs
    results.append(r)

    print("\n  [3h] All except L1 block (embed + L0 + ln_f)")
    r, fracs = run_scope(
        "All except L1 block (embed + L0 + ln_f)",
        keys_embeddings() + keys_l0_full_block(0) + keys_ln_f(),
    )
    fracs_store["all_except_l1"] = fracs
    results.append(r)

    print("\n  [3i] Full transplant (sanity check — should reproduce seed0 accuracy)")
    r, fracs = run_scope("Full transplant (all weights, sanity check)", list(w0.keys()))
    fracs_store["full_transplant"] = fracs
    results.append(r)

    return {"whole_layer": results}, fracs_store


# ---------------------------------------------------------------------------
# Unified results table + paired bootstrap
# ---------------------------------------------------------------------------

def compute_all_bootstrap(
    seed1_fracs: np.ndarray,
    all_fracs: dict,
) -> dict:
    """Compute paired bootstrap CI and paired sign-flip p-value for every condition vs seed1 baseline."""
    bootstrap_results = {}
    for name, fracs in all_fracs.items():
        mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(seed1_fracs, fracs)
        pval, _ = paired_sign_flip_pvalue(seed1_fracs, fracs)
        significant = (ci_lo > 0) or (ci_hi < 0)  # CI excludes 0
        bootstrap_results[name] = {
            "mean_diff_vs_seed1": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": pval,
            "significant_vs_seed1": significant,
        }
    return bootstrap_results


def print_unified_table(
    seed1_acc: float,
    seed0_acc: float,
    all_conditions: list,  # list of (label, accuracy, fracs, key)
    seed1_fracs: np.ndarray,
    bootstrap_results: dict,
):
    print("\n" + "=" * 85)
    print("UNIFIED RESULTS TABLE (all conditions evaluated on same 1024 fixed sequences)")
    print("=" * 85)
    print(f"{'Condition':<52} | {'Accuracy':>8} | {'95% CI (paired bootstrap vs seed1 baseline)':>42}")
    print("-" * 85)
    print(f"{'Seed 0 baseline':<52} | {seed0_acc:>8.4f} | {'(reference)'}")
    print(f"{'Seed 1 baseline':<52} | {seed1_acc:>8.4f} | {'—'}")

    for label, acc, fracs, key in all_conditions:
        if key in bootstrap_results:
            b = bootstrap_results[key]
            ci_str = f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]  p={b['p_value']:.4f}{'*' if b['significant_vs_seed1'] else ' '}"
        else:
            ci_str = "(no bootstrap)"
        print(f"  {label:<50} | {acc:>8.4f} | {ci_str}")

    print("=" * 85)


def print_key_pairwise_comparisons(all_fracs: dict):
    """Robustness concern: are unaligned (0.092), one-iface (0.080), two-iface (0.096) distinguishable?"""
    print("\n" + "=" * 70)
    print("KEY PAIRWISE COMPARISONS (paired bootstrap)")
    print("(Robustness concern: are small accuracy differences statistically distinguishable?)")
    print("=" * 70)

    pairs = [
        ("unaligned", "one_iface_aligned",  "Unaligned vs one-interface aligned"),
        ("unaligned", "two_iface_aligned",  "Unaligned vs two-interface aligned"),
        ("one_iface_aligned", "two_iface_aligned", "One-interface vs two-interface aligned"),
        ("unaligned", "random_rotation",    "Unaligned vs random rotation"),
        ("two_iface_aligned", "random_rotation", "Two-interface aligned vs random rotation"),
        ("unaligned", "aligned_noncrit_donor", "Unaligned vs aligned non-critical donor"),
        ("transplant_crit_slot", "ablated_l0h1", "Unaligned transplant vs ablated"),
        ("whole_l0_attn", "single_head_ref", "Whole-L0 attn block vs single-head"),
    ]

    for key_a, key_b, label in pairs:
        if key_a not in all_fracs or key_b not in all_fracs:
            continue
        mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(all_fracs[key_a], all_fracs[key_b])
        pval, _ = paired_sign_flip_pvalue(all_fracs[key_a], all_fracs[key_b])
        sig = "SIGNIFICANT" if (ci_lo > 0 or ci_hi < 0) else "not significant"
        print(f"\n  {label}")
        print(f"    mean diff: {mean_diff:+.4f}  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  p={pval:.4f}  -> {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TRANSPLANT UNIFIED EVAL")
    print("All conditions evaluated on ONE materialized fixed eval set")
    print(f"Protocol: {EVAL_N_SEQUENCES} sequences, seed={EVAL_DATA_SEED}, "
          f"vocab={EVAL_VOCAB_SIZE}, seq_len={EVAL_SEQ_LEN}, n_bigrams={EVAL_N_BIGRAMS}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Materialize the fixed eval set ONCE
    # -----------------------------------------------------------------------
    print("\n[Step 1] Building fixed eval set...")
    fixed_eval_set = build_fixed_eval_set()

    # -----------------------------------------------------------------------
    # Step 2: Load models
    # -----------------------------------------------------------------------
    print("\n[Step 2] Loading models...")
    print(f"  Seed 0: {SEED0_CKPT}")
    model0 = load_model(str(BASE_DIR / SEED0_CKPT), MODEL_CONFIG)
    print(f"  Seed 1: {SEED1_CKPT}")
    model1 = load_model(str(BASE_DIR / SEED1_CKPT), MODEL_CONFIG)

    # -----------------------------------------------------------------------
    # Step 3: Baselines (same fixed eval set)
    # -----------------------------------------------------------------------
    print("\n[Step 3] Baselines...")
    acc0, fracs0 = evaluate_on_fixed_set(model0, fixed_eval_set)
    acc1, fracs1 = evaluate_on_fixed_set(model1, fixed_eval_set)
    print(f"  Seed 0 baseline: {acc0:.4f}")
    print(f"  Seed 1 baseline: {acc1:.4f}")

    # -----------------------------------------------------------------------
    # Step 4: Run experiments
    # -----------------------------------------------------------------------
    print("\n[Step 4] Running experiments...")

    exp1_results, fracs_exp1 = run_experiment_1(model0, model1, fixed_eval_set)
    exp2_results, fracs_exp2 = run_experiment_2(model0, model1, fixed_eval_set)
    exp3_results, fracs_exp3 = run_experiment_3(model0, model1, fixed_eval_set)

    # Merge all fracs for pairwise comparisons
    all_fracs: dict = {}
    all_fracs.update(fracs_exp1)
    all_fracs.update(fracs_exp2)
    all_fracs.update(fracs_exp3)

    # -----------------------------------------------------------------------
    # Step 5: Bootstrap
    # -----------------------------------------------------------------------
    print(f"\n[Step 5] Computing paired bootstrap CIs vs seed1 baseline ({N_BOOTSTRAP} resamples)...")
    bootstrap_results = compute_all_bootstrap(fracs1, all_fracs)

    # -----------------------------------------------------------------------
    # Step 6: Print unified table
    # -----------------------------------------------------------------------
    # Build ordered list for table display
    all_conditions_ordered = [
        # Exp 1
        ("Exp1 - Seed0 L0H3 -> Seed1 L0H3 (non-crit)",      fracs_exp1.get("transplant_noncrit_slot")),
        ("Exp1 - Seed0 L0H3 -> Seed1 L0H1 (crit, unalign)", fracs_exp1.get("transplant_crit_slot")),
        ("Exp1 - Control A: non-crit donor -> Seed1 L0H1",  fracs_exp1.get("ctrl_a_noncrit_donor")),
        ("Exp1 - Control B: shuffled weights -> Seed1 L0H1",fracs_exp1.get("ctrl_b_shuffled")),
        ("Exp1 - Ablate Seed1 L0H1",                        fracs_exp1.get("ablated_l0h1")),
        # Exp 2
        ("Exp2 - Unaligned transplant",                      fracs_exp2.get("unaligned")),
        ("Exp2 - One-interface aligned",                     fracs_exp2.get("one_iface_aligned")),
        ("Exp2 - Two-interface aligned",                     fracs_exp2.get("two_iface_aligned")),
        ("Exp2 - Random rotation control",                   fracs_exp2.get("random_rotation")),
        ("Exp2 - Aligned non-critical donor",                fracs_exp2.get("aligned_noncrit_donor")),
        # Exp 3
        ("Exp3 - Single-head L0H3->L0H1 (ref)",             fracs_exp3.get("single_head_ref")),
        ("Exp3 - Whole L0 attention block",                  fracs_exp3.get("whole_l0_attn")),
        ("Exp3 - Whole L0 block (attn+MLP+LN)",              fracs_exp3.get("whole_l0_block")),
        ("Exp3 - Whole L0 block + final LN",                 fracs_exp3.get("whole_l0_block_ln_f")),
        ("Exp3 - Embeddings only",                           fracs_exp3.get("embeddings_only")),
        ("Exp3 - Embeddings + L0 attention",                 fracs_exp3.get("embed_l0_attn")),
        ("Exp3 - Embeddings + L0 full block",                fracs_exp3.get("embed_l0_block")),
        ("Exp3 - All except L1 block",                       fracs_exp3.get("all_except_l1")),
        ("Exp3 - Full transplant (sanity check)",            fracs_exp3.get("full_transplant")),
    ]

    # Build display list with accuracy and keys for bootstrap lookup
    key_map = {
        "Exp1 - Seed0 L0H3 -> Seed1 L0H3 (non-crit)":       "transplant_noncrit_slot",
        "Exp1 - Seed0 L0H3 -> Seed1 L0H1 (crit, unalign)":  "transplant_crit_slot",
        "Exp1 - Control A: non-crit donor -> Seed1 L0H1":    "ctrl_a_noncrit_donor",
        "Exp1 - Control B: shuffled weights -> Seed1 L0H1":  "ctrl_b_shuffled",
        "Exp1 - Ablate Seed1 L0H1":                          "ablated_l0h1",
        "Exp2 - Unaligned transplant":                        "unaligned",
        "Exp2 - One-interface aligned":                       "one_iface_aligned",
        "Exp2 - Two-interface aligned":                       "two_iface_aligned",
        "Exp2 - Random rotation control":                     "random_rotation",
        "Exp2 - Aligned non-critical donor":                  "aligned_noncrit_donor",
        "Exp3 - Single-head L0H3->L0H1 (ref)":               "single_head_ref",
        "Exp3 - Whole L0 attention block":                    "whole_l0_attn",
        "Exp3 - Whole L0 block (attn+MLP+LN)":               "whole_l0_block",
        "Exp3 - Whole L0 block + final LN":                   "whole_l0_block_ln_f",
        "Exp3 - Embeddings only":                             "embeddings_only",
        "Exp3 - Embeddings + L0 attention":                   "embed_l0_attn",
        "Exp3 - Embeddings + L0 full block":                  "embed_l0_block",
        "Exp3 - All except L1 block":                         "all_except_l1",
        "Exp3 - Full transplant (sanity check)":              "full_transplant",
    }

    display_list = [
        (label, fracs.mean(), fracs, key_map[label])
        for label, fracs in all_conditions_ordered
        if fracs is not None
    ]

    print_unified_table(acc1, acc0, display_list, fracs1, bootstrap_results)
    print_key_pairwise_comparisons(all_fracs)

    # -----------------------------------------------------------------------
    # Step 7: Build and save JSON results
    # -----------------------------------------------------------------------
    def b_entry(key):
        b = bootstrap_results.get(key, {})
        return {
            "accuracy": float(all_fracs[key].mean()) if key in all_fracs else None,
            "bootstrap_vs_seed1": b,
        }

    interpretation = (
        "All conditions evaluated on identical 1024 fixed sequences (seed=9999). "
        "Paired bootstrap CIs computed over per-sequence accuracy differences vs seed1 baseline. "
        "P-values from paired sign-flip permutation test (10000 permutations, seed=42): "
        "under H0 the sign of each paired difference is exchangeable. "
        "Conditions with CI excluding 0 are statistically distinguishable from seed1 baseline."
    )

    results = {
        "eval_protocol": {
            "n_sequences": EVAL_N_SEQUENCES,
            "data_seed": EVAL_DATA_SEED,
            "vocab_size": EVAL_VOCAB_SIZE,
            "seq_len": EVAL_SEQ_LEN,
            "n_bigrams": EVAL_N_BIGRAMS,
            "batch_size": EVAL_BATCH_SIZE,
            "n_batches": EVAL_N_SEQUENCES // EVAL_BATCH_SIZE,
            "bootstrap_n_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "fix": (
                "Eval set materialized once as list of numpy arrays. "
                "RNG state does NOT advance during evaluation. "
                "All conditions see identical sequences."
            ),
        },
        "baselines": {
            "seed0": acc0,
            "seed1": acc1,
        },
        "experiment_1": {
            **exp1_results,
            "bootstrap": {k: b_entry(k) for k in fracs_exp1},
        },
        "experiment_2": {
            **exp2_results,
            "bootstrap": {k: b_entry(k) for k in fracs_exp2},
        },
        "experiment_3": {
            **exp3_results,
            "bootstrap": {k: b_entry(k) for k in fracs_exp3},
        },
        "pairwise_bootstrap": {},
        "interpretation": interpretation,
    }

    # Pairwise bootstrap for key robustness concern
    pairs_for_json = [
        ("unaligned", "one_iface_aligned"),
        ("unaligned", "two_iface_aligned"),
        ("one_iface_aligned", "two_iface_aligned"),
        ("unaligned", "random_rotation"),
        ("two_iface_aligned", "random_rotation"),
        ("transplant_crit_slot", "ablated_l0h1"),
        ("whole_l0_attn", "single_head_ref"),
    ]
    for key_a, key_b in pairs_for_json:
        if key_a in all_fracs and key_b in all_fracs:
            mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(all_fracs[key_a], all_fracs[key_b])
            pval, _ = paired_sign_flip_pvalue(all_fracs[key_a], all_fracs[key_b])
            results["pairwise_bootstrap"][f"{key_a}_vs_{key_b}"] = {
                "mean_diff_b_minus_a": mean_diff,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_value": pval,
                "significant": (ci_lo > 0 or ci_hi < 0),
            }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Results saved to: {RESULTS_PATH}")
    print(f"  Seed 0 baseline: {acc0:.4f}")
    print(f"  Seed 1 baseline: {acc1:.4f}")


if __name__ == "__main__":
    main()
