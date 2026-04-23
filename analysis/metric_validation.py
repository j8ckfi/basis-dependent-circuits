"""
Metric Validation: pos_dep (positional_score) vs permutation_sensitivity

RESEARCH QUESTION:
Does our existing `positional_score` metric (cross-batch variance of attention
patterns) correctly identify positional heads vs content-based heads?

THE PROBLEM WITH pos_dep:
  positional_score = 1 - variance(attn_pattern) / mean(attn_pattern)

This is LOW variance when:
  1. The head attends to the same POSITION regardless of content (truly positional)
  2. The head attends to the same CONTENT regardless of sequence (e.g., always
     attends to self, i.e., the diagonal) -- this looks positional by this metric
     but is actually content-based: the QUERY matches the KEY based on token identity

THE FIX -- permutation sensitivity:
  Take a batch of inputs.
  Apply a random TOKEN PERMUTATION (reassign token identities) to the inputs,
  keeping positions unchanged.
  Measure L2 distance between original and permuted attention patterns.

  Truly positional head:  pattern unchanged  -> low sensitivity
  Content-based head:     pattern changes     -> high sensitivity

This distinguishes the two failure modes above.

PRIORITY QUESTION:
At entropy=0 (deterministic Markov), are "positional" heads really positional,
or are they content-based (diagonal attention + MLP lookup)?
"""

import sys
import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig, create_model
from src.data import InductionDataset, TextDataset
from src.circuit_mapping import profile_all_heads

# Import MarkovDataset from experiments (it lives there, not in src/)
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from entropy_continuum import MarkovDataset, train_model as train_markov_model


# ============================================================
# Core metric: permutation sensitivity
# ============================================================

def compute_permutation_sensitivity(
    model: GPT,
    dataset,
    n_samples: int = 64,
    n_permutations: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Compute permutation sensitivity for every attention head.

    Strategy:
      For each of n_permutations random token-identity permutations pi:
        1. Take the original batch of inputs X.
        2. Build permuted inputs X' = pi[X]  (replace token ids, keep positions).
        3. Run both X and X' through the model and grab attention patterns.
        4. For each head: compute mean L2 distance between original and permuted
           attention rows across the batch.
      Average L2 distances over permutations.

    A truly POSITIONAL head: attention depends only on position, so the pattern
    stays the same when we swap token identities -> low sensitivity.

    A CONTENT-BASED head: attention depends on which tokens are present, so
    the pattern changes when tokens are permuted -> high sensitivity.

    Args:
        model:          Trained GPT model.
        dataset:        Dataset with generate_batch() method.
        n_samples:      Batch size for computing statistics.
        n_permutations: How many random token permutations to average over.
        seed:           RNG seed for reproducibility.

    Returns:
        sensitivity: np.ndarray of shape (n_layers, n_heads)
                     Higher = more content-dependent.
    """
    rng = np.random.default_rng(seed)
    vocab_size = model.config.vocab_size
    n_layers = len(model.blocks)
    n_heads = model.config.n_heads

    # Get a fixed batch of original inputs (we reuse across permutations)
    batch = dataset.generate_batch(n_samples)
    inputs_orig = np.array(batch[0])  # (B, T)

    # Run original forward pass once to get baseline patterns
    _ = model(mx.array(inputs_orig))
    orig_patterns = [np.array(p) for p in model.get_attention_patterns()]
    # orig_patterns[l]: (B, n_heads, T, T)

    # Accumulate L2 distances: shape (n_layers, n_heads)
    total_l2 = np.zeros((n_layers, n_heads), dtype=np.float64)

    for perm_idx in range(n_permutations):
        # Build a random bijection on the token vocabulary: pi[old_token] = new_token
        # This replaces every occurrence of token X with a different token Y
        # across the entire batch simultaneously, keeping positions unchanged.
        pi = rng.permutation(vocab_size).astype(np.int32)  # (vocab_size,)

        # Apply permutation to inputs
        inputs_perm = pi[inputs_orig]  # fancy indexing: (B, T)

        # Forward pass on permuted inputs
        _ = model(mx.array(inputs_perm))
        perm_patterns = [np.array(p) for p in model.get_attention_patterns()]

        for layer_idx in range(n_layers):
            orig = orig_patterns[layer_idx]  # (B, n_heads, T, T)
            perm = perm_patterns[layer_idx]  # (B, n_heads, T, T)

            # L2 distance between attention rows, averaged over batch and positions
            # diff: (B, n_heads, T, T) -> L2 over last axis -> (B, n_heads, T)
            diff = orig - perm
            # Per-row L2: sqrt(sum of squares over T dim)
            l2_per_row = np.sqrt((diff ** 2).sum(axis=-1))  # (B, n_heads, T)
            # Average over batch and positions
            mean_l2 = l2_per_row.mean(axis=(0, 2))  # (n_heads,)
            total_l2[layer_idx] += mean_l2

    # Average over permutations
    sensitivity = total_l2 / n_permutations
    return sensitivity  # (n_layers, n_heads)


# ============================================================
# Compare old metric vs new metric
# ============================================================

def compute_old_metric(
    model: GPT,
    dataset,
    n_samples: int = 128,
) -> np.ndarray:
    """Compute the existing positional_score metric for all heads.

    Extracts the `positional_score` values from profile_all_heads().

    Returns:
        scores: np.ndarray of shape (n_layers, n_heads)
                Higher = more positional by old metric.
    """
    profiles = profile_all_heads(model, dataset, n_samples=n_samples)
    n_layers = len(model.blocks)
    n_heads = model.config.n_heads
    scores = np.zeros((n_layers, n_heads))
    for p in profiles:
        scores[p.layer, p.head] = p.positional_score
    return scores


def classify_metric(value: float, threshold: float) -> str:
    """Classify a head as positional or content-based given a threshold."""
    return "positional" if value > threshold else "content"


def compare_metrics(
    model_name: str,
    model: GPT,
    dataset,
    n_samples: int = 64,
    n_permutations: int = 8,
    pos_dep_threshold: float = 0.6,    # old metric: >0.6 = positional
    perm_sens_threshold: float = 0.3,  # new metric: >0.3 = content-based
) -> dict:
    """Run both metrics on a model and produce a comparison table.

    pos_dep (old):         high -> positional
    perm_sensitivity (new): high -> content-based

    For agreement: they agree when both say positional or both say content-based.
    Disagreement is our signal that old metric may be wrong.

    Returns:
        results dict with per-head rows plus summary statistics.
    """
    n_layers = len(model.blocks)
    n_heads = model.config.n_heads

    print(f"\n  Computing old pos_dep metric (n_samples={n_samples})...")
    t0 = time.time()
    pos_dep = compute_old_metric(model, dataset, n_samples=n_samples)
    print(f"    Done in {time.time()-t0:.1f}s")

    print(f"  Computing permutation sensitivity (n_samples={n_samples}, n_perms={n_permutations})...")
    t0 = time.time()
    perm_sens = compute_permutation_sensitivity(
        model, dataset, n_samples=n_samples, n_permutations=n_permutations
    )
    print(f"    Done in {time.time()-t0:.1f}s")

    rows = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            pd_val = float(pos_dep[layer_idx, head_idx])
            ps_val = float(perm_sens[layer_idx, head_idx])

            # Old metric: high pos_dep -> "positional"
            old_class = classify_metric(pd_val, pos_dep_threshold)
            # New metric: HIGH perm_sens -> "content" (inverted sense)
            new_class = "positional" if ps_val < perm_sens_threshold else "content"

            agree = (old_class == new_class)
            agreement_str = "agree" if agree else "DISAGREE"

            rows.append({
                "head_id": f"L{layer_idx}H{head_idx}",
                "layer": layer_idx,
                "head": head_idx,
                "pos_dep": pd_val,
                "perm_sensitivity": ps_val,
                "old_classification": old_class,
                "new_classification": new_class,
                "agreement": agree,
            })

    n_disagree = sum(1 for r in rows if not r["agreement"])
    n_total = len(rows)

    # Identify heads where old metric says "positional" but new says "content"
    false_positional = [r for r in rows
                        if r["old_classification"] == "positional"
                        and r["new_classification"] == "content"]

    # Heads where old says "content" but new says "positional"
    false_content = [r for r in rows
                     if r["old_classification"] == "content"
                     and r["new_classification"] == "positional"]

    return {
        "model_name": model_name,
        "n_heads_total": n_total,
        "n_disagree": n_disagree,
        "n_agree": n_total - n_disagree,
        "false_positional_by_old": [r["head_id"] for r in false_positional],
        "false_content_by_old": [r["head_id"] for r in false_content],
        "pos_dep_threshold": pos_dep_threshold,
        "perm_sens_threshold": perm_sens_threshold,
        "rows": rows,
    }


def print_comparison_table(results: dict):
    """Print a human-readable comparison table for one model."""
    print(f"\n{'='*75}")
    print(f"  MODEL: {results['model_name']}")
    print(f"{'='*75}")
    print(f"  Thresholds: pos_dep>{results['pos_dep_threshold']:.2f} = positional | "
          f"perm_sens>{results['perm_sens_threshold']:.2f} = content-based")
    print(f"{'='*75}")
    header = f"  {'Head':<6} | {'pos_dep (old)':>18} | {'perm_sens (new)':>18} | {'Status':>22}"
    print(header)
    print(f"  {'-'*6}-+-{'-'*18}-+-{'-'*18}-+-{'-'*22}")

    for row in results["rows"]:
        pd = row["pos_dep"]
        ps = row["perm_sensitivity"]
        old_tag = f"{pd:.3f} ({row['old_classification']})"
        new_tag = f"{ps:.3f} ({row['new_classification']})"

        if row["agreement"]:
            status = f"✓ {row['old_classification']}"
        else:
            status = f"✗ old={row['old_classification']} new={row['new_classification']}"

        print(f"  {row['head_id']:<6} | {old_tag:>18} | {new_tag:>18} | {status:>22}")

    print(f"{'='*75}")
    print(f"  Agreement: {results['n_agree']}/{results['n_heads_total']} heads")
    print(f"  Disagreements: {results['n_disagree']}")
    if results["false_positional_by_old"]:
        print(f"  OLD METRIC WRONG (calls positional, actually content): "
              f"{results['false_positional_by_old']}")
    if results["false_content_by_old"]:
        print(f"  OLD METRIC WRONG (calls content, actually positional): "
              f"{results['false_content_by_old']}")


# ============================================================
# Model loading helpers
# ============================================================

def load_induction_model(ckpt_path: str) -> GPT:
    """Load content-matching induction model (2L4H, vocab=512, ctx=64)."""
    config = GPTConfig(
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=512,
        ctx_len=64,
    )
    model = GPT(config)
    model.load_weights(ckpt_path)
    mx.eval(model.parameters())
    return model


def load_shakespeare_model(ckpt_dir: str) -> GPT:
    """Load Shakespeare model (6L8H, vocab=65, ctx=256).

    Finds the latest step_* checkpoint in the directory.
    """
    ckpt_path = Path(ckpt_dir)
    checkpoints = sorted(ckpt_path.glob("step_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    latest = checkpoints[-1]
    print(f"    Loading Shakespeare checkpoint: {latest.name}")

    config = GPTConfig(
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        vocab_size=65,
        ctx_len=256,
    )
    model = GPT(config)
    model.load_weights(str(latest / "model.safetensors"))
    mx.eval(model.parameters())
    return model


def get_or_train_markov_entropy0(save_dir: Path) -> GPT:
    """Load entropy=0 Markov model if checkpoint exists, else train one.

    Uses the same config as entropy_continuum.py (4L4H, vocab=64, ctx=128)
    but only 3000 steps (enough for entropy=0 which is easy).
    """
    ckpt_file = save_dir / "model_entropy0.safetensors"
    config = GPTConfig(
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=64,
        ctx_len=128,
        dropout=0.0,
    )

    if ckpt_file.exists():
        print(f"    Loading saved entropy=0 model from {ckpt_file}")
        model = GPT(config)
        model.load_weights(str(ckpt_file))
        mx.eval(model.parameters())
        return model

    print("    No saved entropy=0 checkpoint found. Training 3000-step model...")
    dataset = MarkovDataset(entropy_level=0.0, vocab_size=64, seq_len=128, seed=42)
    print(f"    Dataset measured_entropy = {dataset.measured_entropy:.4f} nats")

    # Train using the same infrastructure as entropy_continuum.py
    # but with 3000 steps instead of 5000
    lr = 1e-3
    n_steps = 3000
    batch_size = 64
    model = create_model(config, seed=0)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.1)

    def compute_loss(model, inputs, targets, mask):
        logits = model(inputs)
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        mask_flat = mask.reshape(B * T)
        per_tok = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (per_tok * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, compute_loss)
    import mlx.utils

    t0 = time.time()
    data_iter = dataset.iter_batches(batch_size, n_steps)
    for step, (inputs, targets, mask) in enumerate(data_iter):
        # Linear warmup + cosine decay
        warmup = 200
        if step < warmup:
            current_lr = lr * step / max(1, warmup)
        else:
            progress = (step - warmup) / max(1, n_steps - warmup)
            import math
            current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.learning_rate = current_lr

        loss, grads = loss_and_grad(model, inputs, targets, mask)
        grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"      step {step:4d}/{n_steps} | loss={loss.item():.4f} | {elapsed:.1f}s")

    print(f"    Training done in {time.time()-t0:.1f}s")

    # Save for future use
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(ckpt_file))
    print(f"    Saved to {ckpt_file}")

    return model


# ============================================================
# Main experiment
# ============================================================

def run_validation():
    """Run the full metric validation across all three models."""
    print("=" * 75)
    print("METRIC VALIDATION: pos_dep vs permutation_sensitivity")
    print("=" * 75)
    print()
    print("Comparing two ways to classify attention heads as positional vs content-based:")
    print("  OLD: pos_dep  = 1 - variance(attn_pattern) / mean (low variance -> positional)")
    print("  NEW: perm_sens = L2(attn_orig, attn_permuted) (insensitive to permutation -> positional)")
    print()

    all_results = {}
    base_dir = Path(__file__).parent.parent

    # ----------------------------------------------------------------
    # MODEL 1: Content-matching induction (2L4H)
    # ----------------------------------------------------------------
    print("=" * 75)
    print("MODEL 1: Content-matching induction (2L, 4H, vocab=512, ctx=64)")
    print("=" * 75)

    induction_ckpt = (base_dir / "checkpoints/induction_content_match/induction_seed0"
                      / "step_020000/model.safetensors")
    print(f"  Checkpoint: {induction_ckpt}")
    model_induction = load_induction_model(str(induction_ckpt))

    # InductionDataset vocab must match model (512)
    ds_induction = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=8, seed=42)

    results_induction = compare_metrics(
        model_name="Induction (2L4H)",
        model=model_induction,
        dataset=ds_induction,
        n_samples=64,
        n_permutations=8,
    )
    print_comparison_table(results_induction)
    all_results["induction"] = results_induction

    # ----------------------------------------------------------------
    # MODEL 2: Shakespeare (6L8H)
    # ----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("MODEL 2: Shakespeare (6L, 8H, vocab=65, ctx=256)")
    print("=" * 75)

    shakespeare_dir = base_dir / "checkpoints/shakespeare/text_seed0"
    print(f"  Checkpoint dir: {shakespeare_dir}")
    model_shakespeare = load_shakespeare_model(str(shakespeare_dir))

    # TextDataset for Shakespeare (character-level vocab=65)
    ds_shakespeare = TextDataset(
        data_dir=str(base_dir / "data"),
        seq_len=256,
        seed=42,
        corpus="shakespeare",
    )

    results_shakespeare = compare_metrics(
        model_name="Shakespeare (6L8H)",
        model=model_shakespeare,
        dataset=ds_shakespeare,
        n_samples=64,
        n_permutations=8,
    )
    print_comparison_table(results_shakespeare)
    all_results["shakespeare"] = results_shakespeare

    # ----------------------------------------------------------------
    # MODEL 3: Entropy=0 Markov (4L4H)
    # ----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("MODEL 3: Entropy=0 Markov chain (4L, 4H, vocab=64, ctx=128)")
    print("=" * 75)
    print("  NOTE: entropy=0 means each token has EXACTLY ONE successor.")
    print("  Priority question: are 'positional' heads really positional,")
    print("  or are they doing diagonal (self) attention + MLP lookup?")

    entropy0_save_dir = base_dir / "checkpoints" / "entropy0_validation"
    model_entropy0 = get_or_train_markov_entropy0(entropy0_save_dir)

    ds_entropy0 = MarkovDataset(entropy_level=0.0, vocab_size=64, seq_len=128, seed=42)
    print(f"  Dataset measured_entropy = {ds_entropy0.measured_entropy:.4f} nats")

    results_entropy0 = compare_metrics(
        model_name="Entropy=0 Markov (4L4H)",
        model=model_entropy0,
        dataset=ds_entropy0,
        n_samples=64,
        n_permutations=8,
    )
    print_comparison_table(results_entropy0)
    all_results["entropy0_markov"] = results_entropy0

    # ----------------------------------------------------------------
    # Cross-model summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CROSS-MODEL SUMMARY")
    print("=" * 75)
    print(f"  {'Model':<30} | {'Heads':>6} | {'Agree':>6} | {'Disagree':>8} | {'False-pos by old':>18}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*18}")
    for key, res in all_results.items():
        fp = ", ".join(res["false_positional_by_old"]) if res["false_positional_by_old"] else "none"
        print(f"  {res['model_name']:<30} | {res['n_heads_total']:>6} | "
              f"{res['n_agree']:>6} | {res['n_disagree']:>8} | {fp:>18}")

    # ----------------------------------------------------------------
    # Answer the priority question about entropy=0
    # ----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PRIORITY QUESTION: Are entropy=0 'positional' heads truly positional?")
    print("=" * 75)

    e0_rows = results_entropy0["rows"]
    pos_by_old = [r for r in e0_rows if r["old_classification"] == "positional"]
    pos_by_new = [r for r in e0_rows if r["new_classification"] == "positional"]
    content_by_new_but_pos_by_old = [r for r in e0_rows
                                      if r["old_classification"] == "positional"
                                      and r["new_classification"] == "content"]

    print(f"\n  Heads called 'positional' by OLD metric (pos_dep > 0.6):")
    for r in pos_by_old:
        print(f"    {r['head_id']}: pos_dep={r['pos_dep']:.3f}, "
              f"perm_sens={r['perm_sensitivity']:.3f} -> NEW says: {r['new_classification'].upper()}")

    if content_by_new_but_pos_by_old:
        print(f"\n  FINDING: {len(content_by_new_but_pos_by_old)} heads called positional by old metric")
        print(f"  are actually CONTENT-BASED by permutation sensitivity!")
        print(f"  Interpretation: these heads likely attend to 'self' (diagonal) because")
        print(f"  in a deterministic Markov chain, the optimal strategy is to look up")
        print(f"  the current token and use the MLP as a lookup table for its successor.")
        print(f"  The attention pattern looks stable (low variance across batch) because")
        print(f"  every token always attends to itself — not because it's positional.")
        print(f"  This INVALIDATES the 'positional' classification for these heads.")
    else:
        print(f"\n  FINDING: All heads that old metric calls 'positional' are CONFIRMED")
        print(f"  positional by the permutation sensitivity metric too.")
        print(f"  Old metric findings appear ROBUST for entropy=0 Markov model.")

    print(f"\n  Heads confirmed positional by BOTH metrics:")
    both_pos = [r for r in e0_rows
                if r["old_classification"] == "positional" and r["new_classification"] == "positional"]
    for r in both_pos:
        print(f"    {r['head_id']}: pos_dep={r['pos_dep']:.3f}, perm_sens={r['perm_sensitivity']:.3f}")
    if not both_pos:
        print("    (none)")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    # Convert to JSON-serializable form
    output = {}
    for key, res in all_results.items():
        output[key] = {
            "model_name": res["model_name"],
            "n_heads_total": res["n_heads_total"],
            "n_agree": res["n_agree"],
            "n_disagree": res["n_disagree"],
            "false_positional_by_old": res["false_positional_by_old"],
            "false_content_by_old": res["false_content_by_old"],
            "pos_dep_threshold": res["pos_dep_threshold"],
            "perm_sens_threshold": res["perm_sens_threshold"],
            "heads": res["rows"],
        }

    output_path = Path(__file__).parent / "metric_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_validation()
