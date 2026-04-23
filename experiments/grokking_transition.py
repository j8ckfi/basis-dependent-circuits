"""
Mechanistic analysis of the grokking transition in the content-matching induction model.

2-layer, 4-head, 128-dim model trained on random-position bigram induction.
Analyzes checkpoints from step 10000-15000 (spanning the sharp phase transition).

Key question: Does L0 develop first (writing token identity) and L1 follow?
Or do they co-develop? Is there a detectable precursor signal?
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

from src.model import GPT, GPTConfig
from src.data import InductionDataset


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CKPT_BASE = Path("checkpoints/induction_content_match/induction_seed0")
OUTPUT_PATH = Path("checkpoints/induction_content_match/grokking_analysis.json")

MODEL_CONFIG = GPTConfig(
    n_layers=2,
    n_heads=4,
    d_model=128,
    d_ff=512,
    vocab_size=512,
    ctx_len=64,
)

# Steps spanning the transition (10000–15000, every 500)
STEPS = list(range(10000, 15001, 500))  # 11 checkpoints

EVAL_BATCH_SIZE = 256
EVAL_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(step: int) -> GPT:
    """Load model weights from a checkpoint step."""
    ckpt_dir = CKPT_BASE / f"step_{step:06d}"
    weights_path = ckpt_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    model = GPT(MODEL_CONFIG)
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())
    return model


def measure_task_accuracy(model: GPT, dataset: InductionDataset, n_batches: int = 4) -> float:
    """Measure accuracy on induction positions only (the real task metric)."""
    correct = 0
    total = 0
    for _ in range(n_batches):
        inputs, targets, mask = dataset.generate_batch(EVAL_BATCH_SIZE)
        logits = model(inputs)  # (B, T, V)
        mx.eval(logits)
        logits_np = np.array(logits)
        targets_np = np.array(targets)
        mask_np = np.array(mask)

        preds = logits_np.argmax(axis=-1)  # (B, T)
        induction_correct = ((preds == targets_np) * mask_np).sum()
        induction_total = mask_np.sum()
        correct += induction_correct
        total += induction_total

    return float(correct / total) if total > 0 else 0.0


def get_attention_patterns(model: GPT, inputs: mx.array) -> list[np.ndarray]:
    """Run forward pass and return attention patterns for all layers.

    Returns list of length n_layers, each shape (B, n_heads, T, T).
    """
    _ = model(inputs)
    patterns = model.get_attention_patterns()
    mx.eval(patterns)
    return [np.array(p) for p in patterns]


def positional_dependence(attn: np.ndarray) -> float:
    """Measure how positional (vs content-dependent) an attention pattern is.

    High score = attention pattern is almost the same across all batch items
    (positional head). Low score = attention varies by content (content-matching head).

    attn shape: (B, T, T)
    Returns scalar in [0, 1]: 1 = purely positional, 0 = purely content-dependent.
    """
    # Variance across batch at each (query_pos, key_pos) cell
    pattern_variance = attn.var(axis=0)  # (T, T)
    pattern_mean = attn.mean(axis=0)     # (T, T)
    # Coefficient of variation per cell; low CV = positional
    # We want: 1 - normalized_variance
    # Use: 1 - mean(var) / (mean(mean_abs) + eps)
    pos_dep = 1.0 - min(1.0, pattern_variance.mean() / (np.abs(pattern_mean).mean() + 1e-8))
    return float(pos_dep)


def copy_score(model: GPT, layer_idx: int, head_idx: int) -> float:
    """OV diagonal dominance: measures whether the head copies token identity.

    Computes E^T @ O_h @ V_h @ E and checks diagonal vs off-diagonal ratio.
    Returns a positive float; higher = more copying.
    """
    block = model.blocks[layer_idx]
    d_model = MODEL_CONFIG.d_model
    d_head = MODEL_CONFIG.d_head

    qkv_weight = np.array(block.attn.qkv_proj.weight)  # (3*d_model, d_model)
    out_weight = np.array(block.attn.out_proj.weight)   # (d_model, d_model)

    # V projection for this head: rows [2*d_model + head*d_head : 2*d_model + (head+1)*d_head]
    v_start = 2 * d_model + head_idx * d_head
    v_head = qkv_weight[v_start: v_start + d_head, :]  # (d_head, d_model)

    # O projection for this head: columns [head*d_head : (head+1)*d_head]
    o_head = out_weight[:, head_idx * d_head:(head_idx + 1) * d_head]  # (d_model, d_head)

    # OV in residual stream space
    ov_matrix = o_head @ v_head  # (d_model, d_model)

    embed = np.array(model.wte.weight)  # (vocab, d_model)

    # Token-space OV: project only a random subset of vocab for speed (vocab=512)
    token_ov = embed @ ov_matrix @ embed.T  # (vocab, vocab)

    diag = np.diag(token_ov)
    off_diag_mean = (token_ov.sum() - diag.sum()) / (token_ov.size - len(diag))
    score = float((diag.mean() - off_diag_mean) / (np.abs(token_ov).mean() + 1e-8))
    return score


def qk_content_score(
    model: GPT,
    layer_idx: int,
    head_idx: int,
    inputs: mx.array,
) -> float:
    """QK content-matching score: how much does this head attend to same-token positions?

    For each query position q with token t, compare:
      - mean attention to positions k where input[k] == t  (same token)
      - mean attention to positions k where input[k] != t  (different token)

    Returns (same_token_attn - diff_token_attn), higher = stronger content matching.
    """
    _ = model(inputs)
    patterns = model.get_attention_patterns()
    mx.eval(patterns)
    attn = np.array(patterns[layer_idx])  # (B, n_heads, T, T)
    head_attn = attn[:, head_idx, :, :]  # (B, T, T)

    inputs_np = np.array(inputs)  # (B, T)
    B, T = inputs_np.shape

    same_attn_vals = []
    diff_attn_vals = []

    for b in range(B):
        seq = inputs_np[b]  # (T,)
        for q in range(1, T):  # skip position 0 (no keys before it worth checking)
            q_token = seq[q]
            for k in range(q):  # causal: only attend to past
                if seq[k] == q_token:
                    same_attn_vals.append(head_attn[b, q, k])
                else:
                    diff_attn_vals.append(head_attn[b, q, k])

    if not same_attn_vals or not diff_attn_vals:
        return 0.0

    same_mean = float(np.mean(same_attn_vals))
    diff_mean = float(np.mean(diff_attn_vals))
    return same_mean - diff_mean


def logit_attribution_induction(
    model: GPT,
    inputs: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> dict[str, float]:
    """Compute per-component direct logit attribution at induction positions.

    For each induction position, projects each component's residual stream
    contribution onto the unembedding direction of the correct token.
    Returns average across all induction positions.
    """
    inputs_np = np.array(inputs)
    targets_np = np.array(targets)
    mask_np = np.array(mask)
    embed_weight = np.array(model.wte.weight)  # (vocab, d_model)

    # Run a decomposed forward pass to get per-component contributions
    B, T = inputs_np.shape
    pos = mx.arange(T)

    with mx.no_grad() if hasattr(mx, 'no_grad') else _noop_ctx():
        tok_embed = model.wte(inputs)
        pos_embed = model.wpe(pos)
        x = tok_embed + pos_embed

    # Manually collect component outputs (same pattern as circuit_mapping.py)
    component_outputs = {}
    x_running = np.array(tok_embed) + np.array(pos_embed)

    for i, block in enumerate(model.blocks):
        # Need MLX arrays for the block forward pass
        x_mx = mx.array(x_running)
        attn_in = block.ln1(x_mx)
        attn_out = block.attn(attn_in)
        mx.eval(attn_out)
        attn_out_np = np.array(attn_out)
        component_outputs[f"L{i}_attn"] = attn_out_np
        x_running = x_running + attn_out_np

        x_mx2 = mx.array(x_running)
        mlp_in = block.ln2(x_mx2)
        mlp_out = block.mlp(mlp_in)
        mx.eval(mlp_out)
        mlp_out_np = np.array(mlp_out)
        component_outputs[f"L{i}_mlp"] = mlp_out_np
        x_running = x_running + mlp_out_np

    # Also include embedding
    component_outputs["embed"] = np.array(tok_embed) + np.array(pos_embed)

    # Average attribution across induction positions
    attr_accum = {k: 0.0 for k in component_outputs}
    n_positions = 0

    for b in range(B):
        induction_positions = np.where(mask_np[b] > 0)[0]
        for pos_idx in induction_positions:
            target_tok = int(targets_np[b, pos_idx])
            unembed_dir = embed_weight[target_tok]  # (d_model,)
            for name, comp in component_outputs.items():
                contrib = comp[b, pos_idx]  # (d_model,)
                attr_accum[name] += float(np.dot(contrib, unembed_dir))
            n_positions += 1

    if n_positions == 0:
        return {k: 0.0 for k in attr_accum}

    return {k: v / n_positions for k, v in attr_accum.items()}


class _noop_ctx:
    """No-op context manager (MLX doesn't have no_grad)."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ─────────────────────────────────────────────────────────────────────────────
# Per-checkpoint analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_checkpoint(step: int, dataset: InductionDataset) -> dict:
    """Run all measurements on a single checkpoint."""
    print(f"\n--- Step {step:6d} ---")

    model = load_checkpoint(step)

    # Shared batch for attention analysis (keep small for qk_content_score speed)
    inputs, targets, mask = dataset.generate_batch(64)

    # 1. Task accuracy (uses larger batch for stability)
    accuracy = measure_task_accuracy(model, dataset, n_batches=4)
    print(f"  accuracy: {accuracy:.3f}")

    # 2. Attention patterns for positional dependence
    attn_patterns = get_attention_patterns(model, inputs)

    head_metrics = []
    n_layers = MODEL_CONFIG.n_layers
    n_heads = MODEL_CONFIG.n_heads

    for layer in range(n_layers):
        for head in range(n_heads):
            attn = attn_patterns[layer][:, head, :, :]  # (B, T, T)

            # Positional dependence
            pos_dep = positional_dependence(attn)

            # Copy score (weight-space)
            copy_sc = copy_score(model, layer, head)

            # QK content-matching score (activation-space, over the shared batch)
            qk_score = qk_content_score(model, layer, head, inputs)

            # Previous-token score (mean attention to i-1)
            B, T, _ = attn.shape
            prev_tok = float(np.mean([attn[:, pos, pos - 1].mean() for pos in range(1, T)]))

            head_metrics.append({
                "layer": layer,
                "head": head,
                "pos_dep": round(pos_dep, 4),
                "copy_score": round(copy_sc, 4),
                "qk_content_score": round(qk_score, 4),
                "prev_token_score": round(prev_tok, 4),
            })

            print(f"  L{layer}H{head}: pos_dep={pos_dep:.3f}  copy={copy_sc:.3f}  "
                  f"qk_content={qk_score:.4f}  prev_tok={prev_tok:.3f}")

    # 3. Logit attribution at induction positions
    attr = logit_attribution_induction(model, inputs, targets, mask)
    attr_rounded = {k: round(float(v), 4) for k, v in attr.items()}
    print(f"  logit_attr: {attr_rounded}")

    return {
        "step": step,
        "accuracy": round(accuracy, 4),
        "heads": head_metrics,
        "logit_attribution": attr_rounded,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Timeline analysis
# ─────────────────────────────────────────────────────────────────────────────

def print_timeline(results: list[dict]) -> None:
    """Print a clear human-readable timeline of what changes first."""

    print("\n" + "=" * 72)
    print("GROKKING TRANSITION TIMELINE")
    print("=" * 72)

    steps = [r["step"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    # Print accuracy curve
    print(f"\n{'Step':>8}  {'Accuracy':>9}  {'Change':>8}")
    print("-" * 32)
    for i, (s, a) in enumerate(zip(steps, accuracies)):
        delta = a - accuracies[i - 1] if i > 0 else 0.0
        marker = " <-- TRANSITION" if delta > 0.1 else ""
        print(f"{s:>8}  {a:>9.3f}  {delta:>+8.3f}{marker}")

    # Find transition step (largest single-step accuracy jump)
    deltas = [accuracies[i] - accuracies[i - 1] for i in range(1, len(accuracies))]
    transition_idx = int(np.argmax(deltas)) + 1  # index in results
    transition_step = steps[transition_idx]
    print(f"\nSharpest accuracy jump: step {steps[transition_idx - 1]} -> {transition_step} "
          f"(+{deltas[transition_idx - 1]:.3f})")

    # Per-head metric trajectories
    n_layers = MODEL_CONFIG.n_layers
    n_heads = MODEL_CONFIG.n_heads

    print(f"\n{'':=<72}")
    print("PER-HEAD METRIC TRAJECTORIES")
    print(f"{'':=<72}")

    for layer in range(n_layers):
        for head in range(n_heads):
            label = f"L{layer}H{head}"
            pos_deps     = [r["heads"][layer * n_heads + head]["pos_dep"]        for r in results]
            copy_scores  = [r["heads"][layer * n_heads + head]["copy_score"]     for r in results]
            qk_scores    = [r["heads"][layer * n_heads + head]["qk_content_score"] for r in results]
            prev_toks    = [r["heads"][layer * n_heads + head]["prev_token_score"] for r in results]

            print(f"\n  {label}:")
            print(f"    {'step':>7}  {'pos_dep':>8}  {'copy':>7}  {'qk_content':>10}  {'prev_tok':>8}")
            for i, s in enumerate(steps):
                print(f"    {s:>7}  {pos_deps[i]:>8.3f}  {copy_scores[i]:>7.3f}  "
                      f"{qk_scores[i]:>10.4f}  {prev_toks[i]:>8.3f}")

    # Logit attribution trajectories
    print(f"\n{'':=<72}")
    print("LOGIT ATTRIBUTION AT INDUCTION POSITIONS (mean over batch)")
    print(f"{'':=<72}")
    attr_keys = list(results[0]["logit_attribution"].keys())
    print(f"  {'step':>7}  " + "  ".join(f"{k:>12}" for k in attr_keys))
    for r in results:
        vals = [r["logit_attribution"].get(k, 0.0) for k in attr_keys]
        print(f"  {r['step']:>7}  " + "  ".join(f"{v:>12.3f}" for v in vals))

    # Mechanistic ordering analysis
    print(f"\n{'':=<72}")
    print("MECHANISTIC ORDERING: WHAT DEVELOPS FIRST?")
    print(f"{'':=<72}")

    # Find first step where each signal crosses a meaningful threshold
    signals = {}

    # Accuracy threshold: first step above 20% (pre-transition would be ~random)
    for i, (s, a) in enumerate(zip(steps, accuracies)):
        if a > 0.20:
            signals["accuracy_>20%"] = s
            break

    # Per-head: first step where qk_content_score > 0.001 (content-matching emerging)
    for layer in range(n_layers):
        for head in range(n_heads):
            label = f"L{layer}H{head}"
            qk_scores = [r["heads"][layer * n_heads + head]["qk_content_score"] for r in results]
            copy_scores = [r["heads"][layer * n_heads + head]["copy_score"] for r in results]
            pos_deps = [r["heads"][layer * n_heads + head]["pos_dep"] for r in results]

            baseline_qk = qk_scores[0]
            baseline_copy = copy_scores[0]

            for i, s in enumerate(steps):
                if qk_scores[i] > baseline_qk + 0.001:
                    signals[f"{label}_qk_emerges"] = s
                    break

            for i, s in enumerate(steps):
                if copy_scores[i] > baseline_copy + 0.05:
                    signals[f"{label}_copy_grows"] = s
                    break

            # Pos-dep drop: positional -> content transition
            for i, s in enumerate(steps):
                if pos_deps[i] < pos_deps[0] - 0.05:
                    signals[f"{label}_pos_dep_drops"] = s
                    break

    # Sort by step
    ordered = sorted(signals.items(), key=lambda x: x[1])

    print(f"\n  {'Signal':<35}  {'First at step':>13}")
    print(f"  {'-' * 35}  {'-' * 13}")
    for sig, step_val in ordered:
        marker = " <-- PRECURSOR" if step_val < transition_step else ""
        print(f"  {sig:<35}  {step_val:>13}{marker}")

    # L0 vs L1 interpretation
    print(f"\n  TRANSITION STEP: {transition_step}")
    print()
    l0_signals = {k: v for k, v in ordered if k.startswith("L0")}
    l1_signals = {k: v for k, v in ordered if k.startswith("L1")}

    l0_first = min(l0_signals.values()) if l0_signals else None
    l1_first = min(l1_signals.values()) if l1_signals else None

    if l0_first and l1_first:
        if l0_first < l1_first:
            print(f"  CONCLUSION: L0 develops FIRST (step {l0_first}) before L1 (step {l1_first})")
            print(f"  -> L0 writes token identity information; L1 reads it to do induction.")
        elif l1_first < l0_first:
            print(f"  CONCLUSION: L1 develops FIRST (step {l1_first}) before L0 (step {l0_first})")
            print(f"  -> Unexpected ordering. Check for positional shortcuts in L1.")
        else:
            print(f"  CONCLUSION: L0 and L1 develop CO-CURRENTLY (both at step {l0_first})")
            print(f"  -> May indicate a coupled phase transition rather than sequential assembly.")
    else:
        print(f"  CONCLUSION: Insufficient signal changes detected in the measured window.")

    # Precursor signals
    precursors = [(k, v) for k, v in ordered if v < transition_step]
    if precursors:
        print(f"\n  PRECURSOR SIGNALS (before accuracy jump at step {transition_step}):")
        for k, v in precursors:
            print(f"    step {v:6d}: {k}")
    else:
        print(f"\n  No precursor signals detected before accuracy jump.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("GROKKING TRANSITION ANALYSIS")
    print(f"Model: 2L 4H 128d | Vocab: 512 | SeqLen: 64")
    print(f"Analyzing steps: {STEPS[0]} to {STEPS[-1]} (every 500)")
    print("=" * 72)

    # Fixed evaluation dataset
    dataset = InductionDataset(
        vocab_size=512,
        seq_len=64,
        n_bigrams=6,
        seed=EVAL_SEED,
    )

    results = []
    for step in STEPS:
        try:
            result = analyze_checkpoint(step, dataset)
            results.append(result)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

    # Save JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    # Print timeline
    print_timeline(results)


if __name__ == "__main__":
    main()
