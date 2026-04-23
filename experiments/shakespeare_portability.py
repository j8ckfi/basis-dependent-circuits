"""
Experiment D: Shakespeare portability at 19M scale.

Tests whether the non-portability finding from 800K models reproduces at scale.
Uses the same induction-advantage metric from natural_language_induction_v2.py,
but applies it across multiple seeds of the 19M-parameter Shakespeare model.

Transplant matrix (seed0 -> each recipient seed):
  a. Slot-matched:   seed0 L0H7 -> recipient L0H7
  b. Func-matched:   seed0 L0H7 -> recipient's own critical slot
  c. Procrustes:     two-interface aligned version of (a)
  d. Whole-L0-attn:  all 8 heads of L0
  e. Full-L0-block:  attn + MLP + LN
  f. Full transplant sanity check

Reference (800K scale):
  - single-head transplant -> ~0.095 (vs baseline ~0.964)
"""

import sys
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import TextDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent

SHAKES_CONFIG = GPTConfig(
    n_layers=6, n_heads=8, d_model=512, d_ff=2048,
    vocab_size=65, ctx_len=256, dropout=0.0,
)
# d_head = 512 / 8 = 64

CHECKPOINTS = {
    0: BASE_DIR / "checkpoints/shakespeare/text_seed0/step_010000/model.safetensors",
    1: BASE_DIR / "checkpoints/shakespeare/text_seed1/step_010000/model.safetensors",
    2: BASE_DIR / "checkpoints/shakespeare/text_seed2/step_010000/model.safetensors",
    3: BASE_DIR / "checkpoints/shakespeare/text_seed3/step_010000/model.safetensors",
}

# Known previous-token head for Shakespeare (from natural_language_induction_v2.py)
DONOR_HEAD = (0, 7)   # L0H7 in seed0

# Eval settings (fixed seed for reproducibility)
EVAL_SEED = 9999
N_EVAL_SAMPLES = 1024
SEQ_LEN = 200           # induction window length (fits inside ctx_len=256)

# 800K-scale reference values (from prior experiments)
REF_800K_BASELINE = 0.964
REF_800K_SINGLE_HEAD_DROP = 0.964 - 0.095  # drop magnitude

OUTPUT_DIR = BASE_DIR / "experiments"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_PATH = OUTPUT_DIR / "shakespeare_portability_results.json"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path) -> GPT:
    model = GPT(SHAKES_CONFIG)
    model.load_weights(str(ckpt_path))
    mx.eval(model.parameters())
    model.set_dtype(mx.float32)
    return model


# ---------------------------------------------------------------------------
# Induction corpus (fixed seed=9999, N=1024)
# ---------------------------------------------------------------------------

def build_induction_corpus(
    text_data: np.ndarray,
    vocab_size: int,
    seq_len: int = SEQ_LEN,
    n_samples: int = N_EVAL_SAMPLES,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    Find positions where the same bigram (A, B) appears twice in a window.
    Mirrors the logic from natural_language_induction_v2.py exactly.
    """
    if rng is None:
        rng = np.random.default_rng(EVAL_SEED)

    samples = []
    max_start = len(text_data) - seq_len - 2
    attempts = 0

    while len(samples) < n_samples and attempts < n_samples * 50:
        attempts += 1
        start = int(rng.integers(0, max_start))
        window = text_data[start: start + seq_len + 1]

        t = int(rng.integers(seq_len // 2, seq_len - 1))
        A = int(window[t])
        B = int(window[t + 1])

        prior_positions = [i for i in range(1, t) if window[i] == A]
        if not prior_positions:
            continue

        p1 = prior_positions[-1]
        B_at_p1 = int(window[p1 + 1])

        if B_at_p1 != B:
            continue

        if t - p1 < 5:
            continue

        prev_token = int(window[t - 1]) if t > 0 else -1

        samples.append({
            "window": window[:seq_len].tolist(),
            "query_pos": t,
            "prior_pos": p1,
            "A_token": A,
            "B_token": B,
            "prev_token": prev_token,
            "gap": t - p1,
        })

    return samples


# ---------------------------------------------------------------------------
# Corpus baselines (same as natural_language_induction_v2.py)
# ---------------------------------------------------------------------------

def compute_corpus_baselines(data: np.ndarray, vocab_size: int) -> dict:
    char_counts = np.zeros(vocab_size, dtype=np.float64)
    for c in data:
        char_counts[c] += 1
    char_freq = char_counts / char_counts.sum()
    char_mode = int(np.argmax(char_freq))

    bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for i in range(len(data) - 1):
        bigram_counts[data[i], data[i + 1]] += 1
    row_sums = bigram_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    bigram_freq = bigram_counts / row_sums
    bigram_argmax = np.argmax(bigram_freq, axis=1)

    return {
        "char_freq": char_freq,
        "char_mode": char_mode,
        "bigram_freq": bigram_freq,
        "bigram_argmax": bigram_argmax,
    }


# ---------------------------------------------------------------------------
# Ablatable forward pass (copied from natural_language_induction_v2.py pattern)
# ---------------------------------------------------------------------------

def _attn_with_head_ablation(
    attn_module, x: mx.array, ablate_heads_set: set, layer_idx: int
) -> mx.array:
    B, T, C = x.shape
    n_heads = attn_module.n_heads
    d_head = attn_module.d_head

    qkv = attn_module.qkv_proj(x)
    qkv = qkv.reshape(B, T, 3, n_heads, d_head)
    qkv = qkv.transpose(0, 3, 2, 1, 4)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    mask = mx.triu(mx.full((T, T), -1e9), k=1)
    attn = attn + mask
    attn = mx.softmax(attn, axis=-1)

    out = attn @ v  # (B, n_heads, T, d_head)

    heads_to_zero = [h for (l, h) in ablate_heads_set if l == layer_idx]
    if heads_to_zero:
        mask_vals = [0.0 if h in heads_to_zero else 1.0 for h in range(n_heads)]
        head_mask = mx.array(mask_vals).reshape(1, n_heads, 1, 1)
        out = out * head_mask

    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = attn_module.out_proj(out)
    return out


def forward_with_ablations(
    model: GPT,
    idx: mx.array,
    ablate_heads: Optional[list] = None,
) -> mx.array:
    ablate_heads_set = set(ablate_heads) if ablate_heads else set()

    B, T = idx.shape
    pos = mx.arange(T)
    x = model.wte(idx) + model.wpe(pos)

    for layer_idx, block in enumerate(model.blocks):
        x_norm = block.ln1(x)
        if ablate_heads_set and any(l == layer_idx for l, _ in ablate_heads_set):
            attn_out = _attn_with_head_ablation(
                block.attn, x_norm, ablate_heads_set, layer_idx
            )
        else:
            attn_out = block.attn(x_norm)
        x = x + attn_out

        x_norm2 = block.ln2(x)
        mlp_out = block.mlp(x_norm2)
        x = x + mlp_out

    x = model.ln_f(x)
    logits = x @ model.wte.weight.T
    return logits


# ---------------------------------------------------------------------------
# Evaluate induction advantage
# ---------------------------------------------------------------------------

def evaluate_induction_advantage(
    model: GPT,
    samples: list[dict],
    baselines: dict,
    ablate_heads: Optional[list] = None,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate model accuracy and baselines on induction positions.
    Returns induction_advantage = model_accuracy - max(baselines).
    """
    char_mode = baselines["char_mode"]
    bigram_argmax = baselines["bigram_argmax"]
    vocab_size = len(baselines["char_freq"])

    model_correct = 0
    uniform_correct = 0
    char_mode_correct = 0
    bigram_correct = 0
    total = len(samples)

    for batch_start in range(0, total, batch_size):
        batch = samples[batch_start: batch_start + batch_size]
        inputs_np = np.array([s["window"] for s in batch], dtype=np.int32)
        inputs = mx.array(inputs_np)

        logits = forward_with_ablations(model, inputs, ablate_heads=ablate_heads)
        mx.eval(logits)
        logits_np = np.array(logits)

        for i, s in enumerate(batch):
            pos = s["query_pos"]
            B_tok = s["B_token"]
            prev_tok = s["prev_token"]

            l = logits_np[i, pos]
            probs = np.exp(l - l.max())
            probs /= probs.sum()
            pred = int(probs.argmax())
            if pred == B_tok:
                model_correct += 1

            uniform_correct += 1.0 / vocab_size

            if char_mode == B_tok:
                char_mode_correct += 1

            if prev_tok >= 0:
                bigram_pred = int(bigram_argmax[prev_tok])
                if bigram_pred == B_tok:
                    bigram_correct += 1

    n_with_prev = sum(1 for s in samples if s["prev_token"] >= 0)
    model_acc = model_correct / total if total > 0 else 0.0
    uniform_acc = uniform_correct / total if total > 0 else 1.0 / vocab_size
    char_mode_acc = char_mode_correct / total if total > 0 else 0.0
    bigram_acc = bigram_correct / n_with_prev if n_with_prev > 0 else 0.0
    best_baseline = max(uniform_acc, char_mode_acc, bigram_acc)
    induction_adv = model_acc - best_baseline

    return {
        "model_accuracy": model_acc,
        "uniform_accuracy": uniform_acc,
        "char_mode_accuracy": char_mode_acc,
        "bigram_accuracy": bigram_acc,
        "best_baseline": best_baseline,
        "induction_advantage": induction_adv,
    }


# ---------------------------------------------------------------------------
# Per-head ablation sweep
# ---------------------------------------------------------------------------

def ablation_sweep(
    model: GPT,
    samples: list[dict],
    baselines: dict,
    baseline_adv: float,
    n_layers: int = 6,
    n_heads: int = 8,
) -> dict:
    """
    Ablate each head in L0 (the layer where the prev-token head lives),
    report induction_advantage_drop.
    Returns dict of {(layer, head): drop}.
    """
    results = {}
    # Focus on L0 (where the critical prev-token head lives) + a couple of L1 heads
    candidate_heads = [(0, h) for h in range(n_heads)] + [(1, h) for h in range(n_heads)]
    for (layer, head) in candidate_heads:
        ev = evaluate_induction_advantage(model, samples, baselines, ablate_heads=[(layer, head)])
        drop = baseline_adv - ev["induction_advantage"]
        results[(layer, head)] = {
            "model_accuracy": ev["model_accuracy"],
            "induction_advantage": ev["induction_advantage"],
            "drop": drop,
        }
    return results


# ---------------------------------------------------------------------------
# Weight utilities (mirrors circuit_transplant.py + whole_layer_transplant.py)
# ---------------------------------------------------------------------------

def get_head_weights(model: GPT, layer: int, head: int) -> dict:
    d_model = model.config.d_model
    d_head = model.config.d_head
    q_start = head * d_head
    q_end = (head + 1) * d_head

    qkv_w = model.blocks[layer].attn.qkv_proj.weight
    out_w = model.blocks[layer].attn.out_proj.weight

    return {
        "q": np.array(qkv_w[q_start:q_end, :]),
        "k": np.array(qkv_w[d_model + q_start: d_model + q_end, :]),
        "v": np.array(qkv_w[2 * d_model + q_start: 2 * d_model + q_end, :]),
        "out": np.array(out_w[:, q_start:q_end]),
    }


def get_flat_weights(model: GPT) -> dict:
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    return {k: np.array(v) for k, v in flat.items()}


def build_model_from_weights(weights: dict) -> GPT:
    new_model = GPT(SHAKES_CONFIG)
    mlx_weights = [(k, mx.array(v)) for k, v in weights.items()]
    new_model.load_weights(mlx_weights)
    mx.eval(new_model.parameters())
    return new_model


def transplant_head_into_model(
    recipient_model: GPT,
    donor_weights: dict,
    target_layer: int,
    target_head: int,
) -> GPT:
    config = recipient_model.config
    d_model = config.d_model
    d_head = config.d_head
    h = target_head
    q_start = h * d_head
    q_end = (h + 1) * d_head

    qkv_w = np.array(recipient_model.blocks[target_layer].attn.qkv_proj.weight)
    out_w = np.array(recipient_model.blocks[target_layer].attn.out_proj.weight)

    qkv_w[q_start:q_end, :] = donor_weights["q"]
    qkv_w[d_model + q_start: d_model + q_end, :] = donor_weights["k"]
    qkv_w[2 * d_model + q_start: 2 * d_model + q_end, :] = donor_weights["v"]
    out_w[:, q_start:q_end] = donor_weights["out"]

    flat = dict(nn.utils.tree_flatten(recipient_model.parameters()))
    weights = {k: np.array(v) for k, v in flat.items()}
    weights[f"blocks.{target_layer}.attn.qkv_proj.weight"] = qkv_w
    weights[f"blocks.{target_layer}.attn.out_proj.weight"] = out_w

    return build_model_from_weights(weights)


def splice_keys(recipient_weights: dict, donor_weights: dict, keys: list) -> dict:
    result = dict(recipient_weights)
    for k in keys:
        result[k] = donor_weights[k].copy()
    return result


def keys_l0_attention(layer: int = 0) -> list:
    return [
        f"blocks.{layer}.attn.qkv_proj.weight",
        f"blocks.{layer}.attn.out_proj.weight",
    ]


def keys_l0_full_block(layer: int = 0) -> list:
    return [
        f"blocks.{layer}.ln1.weight",
        f"blocks.{layer}.attn.qkv_proj.weight",
        f"blocks.{layer}.attn.out_proj.weight",
        f"blocks.{layer}.ln2.weight",
        f"blocks.{layer}.mlp.up_proj.weight",
        f"blocks.{layer}.mlp.down_proj.weight",
    ]


# ---------------------------------------------------------------------------
# Procrustes alignment (from basis_aligned_transplant_v2.py)
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
    """Find R orthogonal minimizing ||X_source @ R - X_target||_F."""
    M = X_source.T @ X_target
    U, _S, Vt = np.linalg.svd(M, full_matrices=True)
    R = U @ Vt
    return R.astype(np.float32)


def rotate_head_weights_two_interface(weights: dict, R_in: np.ndarray, R_out: np.ndarray) -> dict:
    q, k, v, out = weights["q"], weights["k"], weights["v"], weights["out"]
    return {
        "q": (q @ R_in).astype(np.float32),
        "k": (k @ R_in).astype(np.float32),
        "v": (v @ R_in).astype(np.float32),
        "out": (R_out.T @ out).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT D: Shakespeare Portability at 19M Scale")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load dataset and compute fixed eval corpus
    # -----------------------------------------------------------------------
    print("\nLoading TextDataset...")
    tds = TextDataset(data_dir=str(BASE_DIR / "data"), seq_len=256, seed=42)
    print(f"  vocab_size={tds.vocab_size}")

    print(f"\nBuilding fixed eval corpus (N={N_EVAL_SAMPLES}, seed={EVAL_SEED})...")
    rng = np.random.default_rng(EVAL_SEED)
    samples = build_induction_corpus(
        tds.data, vocab_size=tds.vocab_size,
        seq_len=SEQ_LEN, n_samples=N_EVAL_SAMPLES, rng=rng,
    )
    print(f"  Collected {len(samples)} valid induction samples")
    if samples:
        gaps = [s["gap"] for s in samples]
        print(f"  Gap: mean={np.mean(gaps):.1f}, median={np.median(gaps):.1f}, "
              f"min={np.min(gaps)}, max={np.max(gaps)}")

    print("\nComputing corpus baselines...")
    baselines = compute_corpus_baselines(tds.data, tds.vocab_size)

    # -----------------------------------------------------------------------
    # Discover available seeds
    # -----------------------------------------------------------------------
    available_seeds = []
    for seed_id, ckpt_path in CHECKPOINTS.items():
        if ckpt_path.exists():
            available_seeds.append(seed_id)
            print(f"  Seed {seed_id}: AVAILABLE at {ckpt_path}")
        else:
            print(f"  Seed {seed_id}: NOT READY (missing {ckpt_path})")

    if len(available_seeds) < 2:
        print("ERROR: Need at least seed 0 and seed 1. Aborting.")
        sys.exit(1)

    print(f"\nProceeding with seeds: {available_seeds}")

    # -----------------------------------------------------------------------
    # Step 1: Load models and measure baselines
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Per-seed baselines")
    print("=" * 60)

    models = {}
    seed_baselines = {}

    for seed_id in available_seeds:
        print(f"\n--- Seed {seed_id} ---")
        model = load_model(CHECKPOINTS[seed_id])
        models[seed_id] = model

        ev = evaluate_induction_advantage(model, samples, baselines)
        seed_baselines[seed_id] = ev
        print(f"  Model accuracy:      {ev['model_accuracy']:.4f}")
        print(f"  Best baseline:       {ev['best_baseline']:.4f}")
        print(f"  Induction advantage: {ev['induction_advantage']:.4f}")

        if ev["induction_advantage"] < 0.4:
            print(f"  WARNING: Seed {seed_id} induction advantage < 0.4 — may not have converged!")

    # -----------------------------------------------------------------------
    # Step 2: Per-head ablation sweep (L0 + L1 for each seed)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Per-head ablation sweep")
    print("=" * 60)

    seed_ablations = {}
    seed_critical_heads = {}

    for seed_id in available_seeds:
        print(f"\n--- Seed {seed_id} ablation sweep ---")
        model = models[seed_id]
        baseline_adv = seed_baselines[seed_id]["induction_advantage"]

        sweep = ablation_sweep(model, samples, baselines, baseline_adv)
        seed_ablations[seed_id] = sweep

        # Find critical head (largest drop)
        critical = max(sweep.keys(), key=lambda k: sweep[k]["drop"])
        seed_critical_heads[seed_id] = critical
        print(f"  Critical head: L{critical[0]}H{critical[1]} "
              f"(advantage drop = {sweep[critical]['drop']:.4f})")
        print(f"  All L0 drops:")
        for h in range(8):
            key = (0, h)
            drop = sweep[key]["drop"]
            marker = " <-- CRITICAL" if key == critical else ""
            print(f"    L0H{h}: {drop:+.4f}{marker}")

    # -----------------------------------------------------------------------
    # Step 3: Transplant matrix (seed0 -> each other seed)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Transplant matrix")
    print("=" * 60)

    donor_seed = 0
    donor_model = models[donor_seed]
    donor_head_weights = get_head_weights(donor_model, DONOR_HEAD[0], DONOR_HEAD[1])
    w0 = get_flat_weights(donor_model)
    donor_baseline = seed_baselines[donor_seed]["induction_advantage"]

    print(f"\nDonor: seed {donor_seed}, head L{DONOR_HEAD[0]}H{DONOR_HEAD[1]}")
    print(f"Donor baseline advantage: {donor_baseline:.4f}")

    # Collect activations for Procrustes (use a small subset of the eval samples)
    # Use the first 128 samples for alignment to keep memory reasonable
    print("\nCollecting activations for Procrustes alignment...")
    n_align = min(128, len(samples))
    align_inputs_np = np.array([s["window"] for s in samples[:n_align]], dtype=np.int32)
    align_inputs = mx.array(align_inputs_np)
    mx.eval(align_inputs)

    pre_ln_seed0 = collect_pre_l0_ln(donor_model, align_inputs)
    post_l0_seed0 = collect_post_l0(donor_model, align_inputs)
    print(f"  Alignment activations shape: {pre_ln_seed0.shape}")

    transplant_matrix = {}

    for recipient_seed in available_seeds:
        if recipient_seed == donor_seed:
            continue

        print(f"\n--- Transplant: seed {donor_seed} -> seed {recipient_seed} ---")
        recipient_model = models[recipient_seed]
        recipient_baseline = seed_baselines[recipient_seed]["induction_advantage"]
        recipient_baseline_acc = seed_baselines[recipient_seed]["model_accuracy"]
        w1 = get_flat_weights(recipient_model)

        results_this_pair = {}

        # Collect recipient activations for Procrustes
        pre_ln_seed1 = collect_pre_l0_ln(recipient_model, align_inputs)
        post_l0_seed1 = collect_post_l0(recipient_model, align_inputs)
        R_in = procrustes_orthogonal(pre_ln_seed0, pre_ln_seed1)
        R_out = procrustes_orthogonal(post_l0_seed0, post_l0_seed1)
        print(f"  Procrustes R_in orthogonality error: "
              f"{np.linalg.norm(R_in.T @ R_in - np.eye(R_in.shape[0])):.2e}")

        # (a) Slot-matched: seed0 L0H7 -> recipient L0H7
        print(f"  (a) Slot-matched: L0H7 -> L0H7 ...")
        model_a = transplant_head_into_model(
            recipient_model, donor_head_weights,
            target_layer=DONOR_HEAD[0], target_head=DONOR_HEAD[1],
        )
        ev_a = evaluate_induction_advantage(model_a, samples, baselines)
        drop_a = recipient_baseline - ev_a["induction_advantage"]
        print(f"      adv={ev_a['induction_advantage']:.4f}, drop={drop_a:+.4f}")
        results_this_pair["a_slot_matched"] = {
            "description": f"seed0 L0H7 -> recipient L0H7 (slot-matched)",
            "model_accuracy": ev_a["model_accuracy"],
            "induction_advantage": ev_a["induction_advantage"],
            "advantage_drop": drop_a,
        }

        # (b) Func-matched: seed0 L0H7 -> recipient's critical head slot
        crit = seed_critical_heads[recipient_seed]
        print(f"  (b) Func-matched: L0H7 -> recipient L{crit[0]}H{crit[1]} ...")
        model_b = transplant_head_into_model(
            recipient_model, donor_head_weights,
            target_layer=crit[0], target_head=crit[1],
        )
        ev_b = evaluate_induction_advantage(model_b, samples, baselines)
        drop_b = recipient_baseline - ev_b["induction_advantage"]
        print(f"      adv={ev_b['induction_advantage']:.4f}, drop={drop_b:+.4f}")
        results_this_pair["b_func_matched"] = {
            "description": f"seed0 L0H7 -> recipient L{crit[0]}H{crit[1]} (func-matched)",
            "target_head": f"L{crit[0]}H{crit[1]}",
            "model_accuracy": ev_b["model_accuracy"],
            "induction_advantage": ev_b["induction_advantage"],
            "advantage_drop": drop_b,
        }

        # (c) Procrustes-aligned slot-matched (two-interface)
        print(f"  (c) Procrustes-aligned slot-matched ...")
        aligned_weights = rotate_head_weights_two_interface(donor_head_weights, R_in, R_out)
        model_c = transplant_head_into_model(
            recipient_model, aligned_weights,
            target_layer=DONOR_HEAD[0], target_head=DONOR_HEAD[1],
        )
        ev_c = evaluate_induction_advantage(model_c, samples, baselines)
        drop_c = recipient_baseline - ev_c["induction_advantage"]
        print(f"      adv={ev_c['induction_advantage']:.4f}, drop={drop_c:+.4f}")
        results_this_pair["c_procrustes"] = {
            "description": "seed0 L0H7 -> recipient L0H7, two-interface Procrustes aligned",
            "model_accuracy": ev_c["model_accuracy"],
            "induction_advantage": ev_c["induction_advantage"],
            "advantage_drop": drop_c,
        }

        # (d) Whole-L0-attention transplant (all 8 heads)
        print(f"  (d) Whole L0 attention (all 8 heads) ...")
        w1_d = splice_keys(w1, w0, keys_l0_attention(layer=0))
        model_d = build_model_from_weights(w1_d)
        ev_d = evaluate_induction_advantage(model_d, samples, baselines)
        drop_d = recipient_baseline - ev_d["induction_advantage"]
        print(f"      adv={ev_d['induction_advantage']:.4f}, drop={drop_d:+.4f}")
        results_this_pair["d_whole_l0_attn"] = {
            "description": "seed0 L0 qkv_proj + out_proj -> recipient (all 8 heads)",
            "model_accuracy": ev_d["model_accuracy"],
            "induction_advantage": ev_d["induction_advantage"],
            "advantage_drop": drop_d,
        }

        # (e) Full L0 block (attn + MLP + LN)
        print(f"  (e) Full L0 block (attn + MLP + LN) ...")
        w1_e = splice_keys(w1, w0, keys_l0_full_block(layer=0))
        model_e = build_model_from_weights(w1_e)
        ev_e = evaluate_induction_advantage(model_e, samples, baselines)
        drop_e = recipient_baseline - ev_e["induction_advantage"]
        print(f"      adv={ev_e['induction_advantage']:.4f}, drop={drop_e:+.4f}")
        results_this_pair["e_full_l0_block"] = {
            "description": "seed0 entire L0 block (ln1 + attn + ln2 + mlp) -> recipient",
            "model_accuracy": ev_e["model_accuracy"],
            "induction_advantage": ev_e["induction_advantage"],
            "advantage_drop": drop_e,
        }

        # (f) Full transplant (sanity check)
        print(f"  (f) Full transplant (sanity check) ...")
        w1_f = splice_keys(w1, w0, list(w0.keys()))
        model_f = build_model_from_weights(w1_f)
        ev_f = evaluate_induction_advantage(model_f, samples, baselines)
        drop_f = recipient_baseline - ev_f["induction_advantage"]
        sanity_ok = abs(ev_f["induction_advantage"] - donor_baseline) < 0.05
        print(f"      adv={ev_f['induction_advantage']:.4f}, drop={drop_f:+.4f} "
              f"(sanity {'PASS' if sanity_ok else 'FAIL'})")
        results_this_pair["f_full_transplant"] = {
            "description": "full transplant (all weights from seed0) — sanity check",
            "model_accuracy": ev_f["model_accuracy"],
            "induction_advantage": ev_f["induction_advantage"],
            "advantage_drop": drop_f,
            "sanity_check_passed": sanity_ok,
        }

        transplant_matrix[f"seed{donor_seed}_to_seed{recipient_seed}"] = {
            "donor_seed": donor_seed,
            "recipient_seed": recipient_seed,
            "recipient_baseline_advantage": recipient_baseline,
            "recipient_baseline_accuracy": recipient_baseline_acc,
            "experiments": results_this_pair,
        }

    # -----------------------------------------------------------------------
    # Step 4: Summary and comparison to 800K results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Summary and comparison to 800K scale")
    print("=" * 60)

    print(f"\n800K reference:")
    print(f"  Baseline accuracy:      {REF_800K_BASELINE:.4f}")
    print(f"  Single-head drop:       {REF_800K_SINGLE_HEAD_DROP:.4f}")

    for pair_key, pair_data in transplant_matrix.items():
        rec_seed = pair_data["recipient_seed"]
        rec_base = pair_data["recipient_baseline_advantage"]
        exps = pair_data["experiments"]
        print(f"\n{pair_key}  (recipient baseline adv={rec_base:.4f}):")
        print(f"  {'Condition':<30} {'Adv':>8} {'Drop':>8}  vs-800K")
        print(f"  {'-'*60}")
        for cond_key, cond_data in exps.items():
            adv = cond_data["induction_advantage"]
            drop = cond_data["advantage_drop"]
            ref_drop = REF_800K_SINGLE_HEAD_DROP if "slot_matched" in cond_key else None
            ref_str = f" (800K drop={REF_800K_SINGLE_HEAD_DROP:.3f})" if ref_drop is not None else ""
            print(f"  {cond_key:<30} {adv:>8.4f} {drop:>+8.4f}{ref_str}")

    # -----------------------------------------------------------------------
    # Step 4b: Cross-seed aggregation across all recipient seeds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4b: Cross-seed aggregation (3 recipients)")
    print("=" * 60)

    cond_keys_agg = [
        "a_slot_matched", "b_func_matched", "c_procrustes",
        "d_whole_l0_attn", "e_full_l0_block", "f_full_transplant",
    ]
    cond_display_agg = [
        "(a) Slot-matched", "(b) Func-matched", "(c) Procrustes",
        "(d) Whole-L0-attn", "(e) Full-L0-block", "(f) Full transplant",
    ]

    agg_stats = {}
    all_pair_data = list(transplant_matrix.values())
    for cond_key, cond_disp in zip(cond_keys_agg, cond_display_agg):
        drops = [
            p["experiments"][cond_key]["advantage_drop"]
            for p in all_pair_data
            if cond_key in p["experiments"]
        ]
        if drops:
            mean_d = float(np.mean(drops))
            sd_d = float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0
            rng_d = (float(np.min(drops)), float(np.max(drops)))
            agg_stats[cond_key] = {
                "drops": [float(d) for d in drops],
                "mean": mean_d,
                "sd": sd_d,
                "range": rng_d,
                "n": len(drops),
            }
            print(f"  {cond_disp:<25} mean_drop={mean_d:+.4f}  SD={sd_d:.4f}  "
                  f"range=[{rng_d[0]:+.4f}, {rng_d[1]:+.4f}]  n={len(drops)}")

    # Slot-matched vs cross-slot asymmetry check
    slot_drops = agg_stats.get("a_slot_matched", {}).get("drops", [])
    cross_drops = agg_stats.get("b_func_matched", {}).get("drops", [])
    if slot_drops and cross_drops:
        slot_mean = np.mean(slot_drops)
        cross_mean = np.mean(cross_drops)
        asymmetry_consistent = all(
            abs(s) < abs(c) * 1.5 or abs(c) < abs(s) * 1.5
            for s, c in zip(slot_drops, cross_drops)
        )
        print(f"\n  Slot-matched mean drop:   {slot_mean:+.4f}")
        print(f"  Func-matched mean drop:   {cross_mean:+.4f}")
        asymmetry_holds = all(
            abs(slot_drops[i]) > 0.3 and abs(cross_drops[i]) > 0.3
            for i in range(len(slot_drops))
        )
        print(f"  Both conditions show large drops across all recipients: "
              f"{'YES — asymmetry test N/A (both bad)' if asymmetry_holds else 'MIXED'}")

    # -----------------------------------------------------------------------
    # Step 5: Plot
    # -----------------------------------------------------------------------
    print("\nGenerating comparison plot...")
    _plot_comparison(transplant_matrix, seed_baselines)

    # -----------------------------------------------------------------------
    # Step 5b: Printed summary table (0.8M vs 19M side-by-side)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: 0.8M vs 19M scale portability")
    print("=" * 70)
    print(f"{'Condition':<28} {'800K drop':>10} {'19M mean':>10} {'19M SD':>8} {'19M range':>20}  Consistent?")
    print("-" * 70)
    ref_800k_by_cond = {
        "a_slot_matched":    REF_800K_SINGLE_HEAD_DROP,
        "b_func_matched":    REF_800K_SINGLE_HEAD_DROP,
        "c_procrustes":      REF_800K_SINGLE_HEAD_DROP,
        "d_whole_l0_attn":   None,
        "e_full_l0_block":   None,
        "f_full_transplant": None,
    }
    cond_short = {
        "a_slot_matched":    "(a) Slot-matched",
        "b_func_matched":    "(b) Func-matched",
        "c_procrustes":      "(c) Procrustes",
        "d_whole_l0_attn":   "(d) Whole-L0-attn",
        "e_full_l0_block":   "(e) Full-L0-block",
        "f_full_transplant": "(f) Full transplant",
    }
    for ck in cond_keys_agg:
        ref = ref_800k_by_cond.get(ck)
        ref_str = f"{ref:+.4f}" if ref is not None else "    N/A"
        agg = agg_stats.get(ck, {})
        m = agg.get("mean", float("nan"))
        s = agg.get("sd", float("nan"))
        r = agg.get("range", (float("nan"), float("nan")))
        rng_str = f"[{r[0]:+.3f}, {r[1]:+.3f}]"
        consistent = "YES" if not np.isnan(m) and m > 0.3 else ("NO" if not np.isnan(m) else "N/A")
        print(f"  {cond_short[ck]:<26} {ref_str:>10} {m:>+10.4f} {s:>8.4f} {rng_str:>20}  {consistent}")

    # Slot-matched vs func-matched asymmetry verdict
    if slot_drops and cross_drops:
        print(f"\n  Slot-matched vs func-matched asymmetry across {len(slot_drops)} recipients:")
        for i, (s_d, f_d) in enumerate(zip(slot_drops, cross_drops)):
            rec_seed = list(transplant_matrix.values())[i]["recipient_seed"]
            diff = s_d - f_d
            print(f"    Seed {rec_seed}: slot={s_d:+.4f}  func={f_d:+.4f}  diff={diff:+.4f}")
        print(f"  --> Slot-matched asymmetry ROBUST across all recipients: "
              f"{'YES' if all(abs(s_d - f_d) > 0.05 or abs(s_d) > 0.3 for s_d, f_d in zip(slot_drops, cross_drops)) else 'MIXED'}")

    # -----------------------------------------------------------------------
    # Save results JSON
    # -----------------------------------------------------------------------
    # Serialize ablation results (convert tuple keys to strings)
    ablations_serializable = {}
    for seed_id, sweep in seed_ablations.items():
        ablations_serializable[seed_id] = {
            f"L{l}H{h}": {
                "model_accuracy": float(v["model_accuracy"]),
                "induction_advantage": float(v["induction_advantage"]),
                "advantage_drop": float(v["drop"]),
            }
            for (l, h), v in sweep.items()
        }

    results_json = {
        "experiment": "Shakespeare portability at 19M scale (4 seeds)",
        "config": {
            "n_layers": 6, "n_heads": 8, "d_model": 512, "d_ff": 2048,
            "vocab_size": 65, "ctx_len": 256,
            "d_head": 64,
        },
        "eval_settings": {
            "eval_seed": EVAL_SEED,
            "n_samples": len(samples),
            "seq_len": SEQ_LEN,
        },
        "available_seeds": available_seeds,
        "per_seed_baselines": {
            str(seed_id): {
                "model_accuracy": float(ev["model_accuracy"]),
                "best_baseline": float(ev["best_baseline"]),
                "induction_advantage": float(ev["induction_advantage"]),
            }
            for seed_id, ev in seed_baselines.items()
        },
        "per_seed_critical_heads": {
            str(seed_id): f"L{h[0]}H{h[1]}"
            for seed_id, h in seed_critical_heads.items()
        },
        "per_seed_ablation_sweeps": ablations_serializable,
        "transplant_matrix": {
            pair_key: {
                "donor_seed": v["donor_seed"],
                "recipient_seed": v["recipient_seed"],
                "recipient_baseline_advantage": float(v["recipient_baseline_advantage"]),
                "experiments": {
                    ck: {
                        kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv)
                        for kk, vv in cv.items()
                    }
                    for ck, cv in v["experiments"].items()
                },
            }
            for pair_key, v in transplant_matrix.items()
        },
        "reference_800k": {
            "baseline_accuracy": REF_800K_BASELINE,
            "single_head_transplant_accuracy": 0.095,
            "single_head_transplant_drop": REF_800K_BASELINE - 0.095,
        },
        "cross_seed_aggregation": {
            ck: {
                "drops_per_recipient": v["drops"],
                "mean_drop": v["mean"],
                "sd_drop": v["sd"],
                "range_drop": list(v["range"]),
                "n_recipients": v["n"],
            }
            for ck, v in agg_stats.items()
        },
        "summary": _build_summary(transplant_matrix, seed_baselines, donor_baseline, agg_stats),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print("Done.")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_comparison(transplant_matrix: dict, seed_baselines: dict):
    if not transplant_matrix:
        return

    cond_labels = ["a_slot_matched", "b_func_matched", "c_procrustes",
                   "d_whole_l0_attn", "e_full_l0_block", "f_full_transplant"]
    cond_display = ["(a) Slot\nmatched", "(b) Func\nmatched", "(c) Procrustes\naligned",
                    "(d) Whole\nL0 attn", "(e) Full\nL0 block", "(f) Full\ntransplant"]

    n_conditions = len(cond_labels)
    pair_keys = list(transplant_matrix.keys())
    n_pairs = len(pair_keys)

    # Layout: 800K ref | per-pair panels | aggregated summary
    n_cols = 1 + n_pairs + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))

    # ---- 800K reference ----
    ax_ref = axes[0]
    ref_drops = [REF_800K_SINGLE_HEAD_DROP, REF_800K_SINGLE_HEAD_DROP,
                 REF_800K_SINGLE_HEAD_DROP, None, None, None]
    ref_vals_clean = [d if d is not None else 0 for d in ref_drops]
    colors_ref = ["tab:red" if d is not None and d > 0.5 else
                  ("tab:orange" if d is not None and d > 0.1 else "tab:green")
                  for d in ref_drops]
    bars = ax_ref.bar(range(n_conditions), ref_vals_clean, color=colors_ref, alpha=0.8)
    ax_ref.set_xticks(range(n_conditions))
    ax_ref.set_xticklabels(cond_display, fontsize=8)
    ax_ref.set_ylabel("Induction advantage drop")
    ax_ref.set_title(f"800K scale\nbaseline adv~{REF_800K_BASELINE:.3f}", fontsize=10)
    ax_ref.set_ylim(-0.15, 1.1)
    ax_ref.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax_ref.axhline(0, color="black", linewidth=0.6)
    for i, (bar, val) in enumerate(zip(bars, ref_vals_clean)):
        if ref_drops[i] is not None:
            ax_ref.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        else:
            ax_ref.text(bar.get_x() + bar.get_width() / 2, 0.05,
                        "N/A", ha="center", va="bottom", fontsize=7, color="gray")

    # ---- Per-pair charts ----
    recipient_colors = ["tab:blue", "tab:purple", "tab:brown"]
    all_drops_by_cond = {c: [] for c in cond_labels}

    for pi, pair_key in enumerate(pair_keys):
        ax = axes[pi + 1]
        pair_data = transplant_matrix[pair_key]
        exps = pair_data["experiments"]
        rec_seed = pair_data["recipient_seed"]
        rec_base = pair_data["recipient_baseline_advantage"]

        drops = []
        for cond in cond_labels:
            d = exps.get(cond, {}).get("advantage_drop", None)
            drops.append(d)
            if d is not None:
                all_drops_by_cond[cond].append(d)

        colors_19m = ["tab:red" if d is not None and d > 0.5 else
                      ("tab:orange" if d is not None and d > 0.1 else "tab:green")
                      for d in drops]
        vals_clean = [d if d is not None else 0 for d in drops]
        bars2 = ax.bar(range(n_conditions), vals_clean, color=colors_19m, alpha=0.8)
        ax.set_xticks(range(n_conditions))
        ax.set_xticklabels(cond_display, fontsize=8)
        ax.set_title(f"19M: seed0->seed{rec_seed}\nbaseline adv={rec_base:.3f}", fontsize=10)
        ax.set_ylim(-0.15, 1.1)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.6)
        for i, (bar, val) in enumerate(zip(bars2, vals_clean)):
            if drops[i] is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("Transplant condition")

    # ---- Aggregated summary panel (mean ± SD across recipients) ----
    ax_agg = axes[-1]
    means = []
    sds = []
    colors_agg = []
    for cond in cond_labels:
        ds = all_drops_by_cond[cond]
        if ds:
            m = float(np.mean(ds))
            s = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
        else:
            m, s = 0.0, 0.0
        means.append(m)
        sds.append(s)
        color = "tab:red" if m > 0.5 else ("tab:orange" if m > 0.1 else "tab:green")
        colors_agg.append(color)

    x_pos = np.arange(n_conditions)
    bars_agg = ax_agg.bar(x_pos, means, color=colors_agg, alpha=0.8, zorder=2)
    ax_agg.errorbar(x_pos, means, yerr=sds, fmt="none", color="black",
                    capsize=4, linewidth=1.5, zorder=3)
    # Scatter individual points
    for ci, cond in enumerate(cond_labels):
        ds = all_drops_by_cond[cond]
        if ds:
            jitter = np.linspace(-0.12, 0.12, len(ds))
            for j, d in zip(jitter, ds):
                ax_agg.scatter(ci + j, d, color="black", s=20, zorder=4, alpha=0.7)

    ax_agg.set_xticks(range(n_conditions))
    ax_agg.set_xticklabels(cond_display, fontsize=8)
    ax_agg.set_title(f"19M: mean ± SD\n(n={n_pairs} recipients)", fontsize=10)
    ax_agg.set_ylim(-0.15, 1.1)
    ax_agg.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax_agg.axhline(0, color="black", linewidth=0.6)
    for i, (bar, m, s) in enumerate(zip(bars_agg, means, sds)):
        ax_agg.text(bar.get_x() + bar.get_width() / 2, max(m, 0) + s + 0.03,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    ax_agg.set_xlabel("Transplant condition")
    ax_agg.set_ylabel("Induction advantage drop")

    ax_ref.set_xlabel("Transplant condition")

    fig.suptitle(
        "Non-portability: 800K vs 19M scale (4 seeds, 3 recipients)\n"
        "Red = catastrophic drop (>0.5), Orange = moderate (0.1-0.5), Green = preserved",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    save_path = PLOTS_DIR / "shakespeare_portability.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {save_path}")


def _build_summary(transplant_matrix, seed_baselines, donor_baseline, agg_stats=None):
    summaries = []
    for pair_key, pair_data in transplant_matrix.items():
        exps = pair_data["experiments"]
        rec_base = pair_data["recipient_baseline_advantage"]
        slot_drop = exps.get("a_slot_matched", {}).get("advantage_drop", None)
        func_drop = exps.get("b_func_matched", {}).get("advantage_drop", None)
        proc_drop = exps.get("c_procrustes", {}).get("advantage_drop", None)
        full_drop = exps.get("f_full_transplant", {}).get("advantage_drop", None)

        catastrophic_800k = REF_800K_BASELINE - 0.095  # ~0.869

        if slot_drop is not None:
            non_portable_reproduced = slot_drop > 0.5 * rec_base
            summaries.append({
                "pair": pair_key,
                "single_head_slot_drop_19m": float(slot_drop),
                "single_head_func_drop_19m": float(func_drop) if func_drop is not None else None,
                "single_head_drop_800k": catastrophic_800k,
                "non_portability_reproduced_at_19m": non_portable_reproduced,
                "procrustes_drop_19m": float(proc_drop) if proc_drop is not None else None,
                "full_transplant_drop": float(full_drop) if full_drop is not None else None,
            })

    overall = all(s["non_portability_reproduced_at_19m"] for s in summaries) if summaries else False

    # Cross-seed aggregation summary
    agg_summary = {}
    if agg_stats:
        for ck, v in agg_stats.items():
            agg_summary[ck] = {
                "mean_drop": v["mean"],
                "sd_drop": v["sd"],
                "range_drop": list(v["range"]),
                "n": v["n"],
            }

    # Slot-matched vs func-matched asymmetry
    slot_mean = agg_stats.get("a_slot_matched", {}).get("mean", None) if agg_stats else None
    func_mean = agg_stats.get("b_func_matched", {}).get("mean", None) if agg_stats else None
    slot_drops_all = agg_stats.get("a_slot_matched", {}).get("drops", []) if agg_stats else []
    func_drops_all = agg_stats.get("b_func_matched", {}).get("drops", []) if agg_stats else []

    # asymmetry holds if slot-matched drops are consistently different from func-matched
    asymmetry_consistent = None
    if slot_drops_all and func_drops_all and len(slot_drops_all) == len(func_drops_all):
        asymmetry_consistent = all(
            abs(slot_drops_all[i] - func_drops_all[i]) < 0.15
            for i in range(len(slot_drops_all))
        )

    return {
        "per_pair": summaries,
        "non_portability_reproduced_at_19m_scale": overall,
        "cross_seed_aggregation": agg_summary,
        "slot_matched_mean_drop": slot_mean,
        "func_matched_mean_drop": func_mean,
        "slot_vs_func_asymmetry_consistent": asymmetry_consistent,
        "interpretation": (
            "Non-portability CONFIRMED at 19M scale across all 3 recipient seeds: "
            "single-head transplants catastrophically degrade induction advantage, "
            "consistent with 800K results."
            if overall else
            "Non-portability result at 19M scale is MIXED or NOT REPRODUCED — "
            "check per-pair results for details."
        ),
    }


if __name__ == "__main__":
    main()
