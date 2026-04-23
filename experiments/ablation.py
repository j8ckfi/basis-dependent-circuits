"""
Surgical ablation experiments to causally validate circuit maps.

For each model, we:
1. Measure baseline performance
2. Zero out individual components (attention heads, MLP layers, attention blocks)
3. Measure performance drop => causal importance

Ablation approach: zero the output of a component during the forward pass.
For per-head ablation: run attention normally, zero specific head slice before out_proj.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

from src.model import GPT, GPTConfig
from src.data import InductionDataset, TextDataset


# ============================================================
# Ablation forward pass utilities
# ============================================================

def forward_with_ablations(
    model: GPT,
    inputs: mx.array,
    ablate_heads: list[tuple[int, int]] | None = None,
    ablate_attn_layers: list[int] | None = None,
    ablate_mlp_layers: list[int] | None = None,
) -> mx.array:
    """
    Forward pass with specified components zeroed out.

    Args:
        ablate_heads: List of (layer, head) tuples to zero out
        ablate_attn_layers: List of layer indices to zero entire attention output
        ablate_mlp_layers: List of layer indices to zero entire MLP output
    """
    ablate_heads = ablate_heads or []
    ablate_attn_layers = ablate_attn_layers or []
    ablate_mlp_layers = ablate_mlp_layers or []

    # Build sets for fast lookup
    head_set = set(ablate_heads)
    attn_layer_set = set(ablate_attn_layers)
    mlp_layer_set = set(ablate_mlp_layers)

    B, T = inputs.shape
    pos = mx.arange(T)
    x = model.wte(inputs) + model.wpe(pos)

    for i, block in enumerate(model.blocks):
        # --- Attention ---
        attn_in = block.ln1(x)

        if i in attn_layer_set:
            # Zero entire attention output for this layer
            attn_out = mx.zeros_like(attn_in)
        else:
            # Check if we need per-head ablation in this layer
            heads_to_ablate_this_layer = [h for (l, h) in head_set if l == i]

            if heads_to_ablate_this_layer:
                attn_out = _attn_with_head_ablation(
                    block.attn, attn_in, heads_to_ablate_this_layer
                )
            else:
                attn_out = block.attn(attn_in)

        x = x + attn_out

        # --- MLP ---
        mlp_in = block.ln2(x)
        if i in mlp_layer_set:
            mlp_out = mx.zeros_like(mlp_in)
        else:
            mlp_out = block.mlp(mlp_in)

        x = x + mlp_out

    x = model.ln_f(x)
    logits = x @ model.wte.weight.T
    return logits


def _attn_with_head_ablation(
    attn_module,
    x: mx.array,
    heads_to_zero: list[int],
) -> mx.array:
    """
    Run attention but zero specified heads before the output projection.

    The output of each head has shape (B, T, d_head).
    We zero those slices, then run out_proj.
    """
    B, T, C = x.shape
    n_heads = attn_module.n_heads
    d_head = attn_module.d_head

    # QKV projection
    qkv = attn_module.qkv_proj(x)
    qkv = qkv.reshape(B, T, 3, n_heads, d_head)
    qkv = qkv.transpose(0, 3, 2, 1, 4)  # (B, n_heads, 3, T, d_head)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    # Scaled dot-product with causal mask
    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    mask = mx.triu(mx.full((T, T), -1e9), k=1)
    attn = mx.softmax(attn + mask, axis=-1)

    # Values: (B, n_heads, T, d_head)
    out = attn @ v

    # Zero ablated heads — build a mask over heads dim
    # out shape: (B, n_heads, T, d_head)
    head_mask = np.ones((1, n_heads, 1, 1), dtype=np.float32)
    for h in heads_to_zero:
        head_mask[0, h, 0, 0] = 0.0
    head_mask_mx = mx.array(head_mask)
    out = out * head_mask_mx

    # Reshape and project
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = attn_module.out_proj(out)
    return out


# ============================================================
# Metric helpers
# ============================================================

def compute_induction_accuracy(
    model: GPT,
    dataset: InductionDataset,
    n_batches: int = 20,
    batch_size: int = 64,
    ablate_heads: list[tuple[int, int]] | None = None,
    ablate_attn_layers: list[int] | None = None,
    ablate_mlp_layers: list[int] | None = None,
) -> float:
    """Compute accuracy on induction positions (where induction_mask == 1)."""
    correct = 0
    total = 0
    for _ in range(n_batches):
        inputs, targets, mask = dataset.generate_batch(batch_size)
        logits = forward_with_ablations(
            model, inputs,
            ablate_heads=ablate_heads,
            ablate_attn_layers=ablate_attn_layers,
            ablate_mlp_layers=ablate_mlp_layers,
        )
        mx.eval(logits)
        preds = np.array(mx.argmax(logits, axis=-1))
        tgts = np.array(targets)
        msk = np.array(mask) > 0
        correct += int((preds == tgts)[msk].sum())
        total += int(msk.sum())
    return correct / max(total, 1)


def compute_lm_loss(
    model: GPT,
    dataset: TextDataset,
    n_batches: int = 20,
    batch_size: int = 32,
    ablate_heads: list[tuple[int, int]] | None = None,
    ablate_attn_layers: list[int] | None = None,
    ablate_mlp_layers: list[int] | None = None,
) -> float:
    """Compute average cross-entropy loss over text sequences."""
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        inputs, targets = dataset.generate_batch(batch_size)
        logits = forward_with_ablations(
            model, inputs,
            ablate_heads=ablate_heads,
            ablate_attn_layers=ablate_attn_layers,
            ablate_mlp_layers=ablate_mlp_layers,
        )
        mx.eval(logits)
        logits_np = np.array(logits)        # (B, T, V)
        targets_np = np.array(targets)      # (B, T)
        B, T, V = logits_np.shape
        # Cross-entropy: log softmax then gather target log-prob
        # Use logsumexp for numerical stability
        log_sum_exp = np.log(np.exp(logits_np - logits_np.max(-1, keepdims=True)).sum(-1) + 1e-10) + logits_np.max(-1)
        target_logits = logits_np[np.arange(B)[:, None], np.arange(T)[None, :], targets_np]
        loss = -(target_logits - log_sum_exp)
        total_loss += float(loss.sum())
        total_tokens += B * T
    return total_loss / max(total_tokens, 1)


def load_model(weights_path: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


# ============================================================
# Model 1: Content-matching induction (2-layer)
# ============================================================

def run_model1(results: dict):
    print("\n" + "="*60)
    print("MODEL 1: Content-matching induction (2-layer, vocab=512)")
    print("="*60)

    weights = "checkpoints/induction_content_match/induction_seed0/step_020000/model.safetensors"
    config = GPTConfig(n_layers=2, n_heads=4, d_model=128, d_ff=512, vocab_size=512, ctx_len=64)
    dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)

    model = load_model(weights, config)
    print(f"Loaded: {weights}")

    # Baseline
    baseline = compute_induction_accuracy(model, dataset)
    print(f"\nBaseline induction accuracy: {baseline:.4f}")

    # Per-head ablation: zero one head at a time
    head_results = {}
    for layer in range(config.n_layers):
        for head in range(config.n_heads):
            acc = compute_induction_accuracy(model, dataset, ablate_heads=[(layer, head)])
            drop = baseline - acc
            key = f"L{layer}H{head}"
            head_results[key] = {"accuracy": acc, "drop": drop}
            print(f"  Ablate {key}: acc={acc:.4f}  drop={drop:+.4f}")

    # Summarize
    print("\nRanked by importance (drop):")
    ranked = sorted(head_results.items(), key=lambda x: x[1]["drop"], reverse=True)
    for k, v in ranked:
        bar = "#" * int(abs(v["drop"]) * 40)
        print(f"  {k}: drop={v['drop']:+.4f}  {bar}")

    results["model1"] = {
        "description": "Content-matching induction 2-layer vocab=512",
        "weights": weights,
        "baseline_accuracy": baseline,
        "head_ablations": head_results,
    }


# ============================================================
# Model 2: Shakespeare char-level (6-layer)
# ============================================================

def run_model2(results: dict):
    print("\n" + "="*60)
    print("MODEL 2: Shakespeare char-level (6-layer)")
    print("="*60)

    # Use latest available step
    weights = "checkpoints/shakespeare/text_seed0/step_010000/model.safetensors"
    config = GPTConfig(n_layers=6, n_heads=8, d_model=512, d_ff=2048, vocab_size=65, ctx_len=256)
    dataset = TextDataset(
        data_dir="data",
        seq_len=256,
        seed=42,
    )

    model = load_model(weights, config)
    print(f"Loaded: {weights}")

    # Baseline
    baseline_loss = compute_lm_loss(model, dataset)
    print(f"\nBaseline LM loss: {baseline_loss:.4f}")

    ablation_results = {}

    # Test 1: Zero out L0H7 (previous-token head with 0.908 score)
    loss_no_L0H7 = compute_lm_loss(model, dataset, ablate_heads=[(0, 7)])
    delta_L0H7 = loss_no_L0H7 - baseline_loss
    ablation_results["L0H7"] = {"loss": loss_no_L0H7, "delta": delta_L0H7}
    print(f"\nAblate L0H7 (prev-token head): loss={loss_no_L0H7:.4f}  delta={delta_L0H7:+.4f}")

    # Test 2: Zero ALL attention in layers 2-5 (suppression cascade)
    loss_no_L2_5 = compute_lm_loss(model, dataset, ablate_attn_layers=[2, 3, 4, 5])
    delta_L2_5 = loss_no_L2_5 - baseline_loss
    ablation_results["attn_L2-L5"] = {"loss": loss_no_L2_5, "delta": delta_L2_5}
    print(f"Ablate ALL attn L2-L5 (suppression cascade): loss={loss_no_L2_5:.4f}  delta={delta_L2_5:+.4f}")

    # Test 3: Zero L5 MLP (highest logit attribution)
    loss_no_L5mlp = compute_lm_loss(model, dataset, ablate_mlp_layers=[5])
    delta_L5mlp = loss_no_L5mlp - baseline_loss
    ablation_results["L5_mlp"] = {"loss": loss_no_L5mlp, "delta": delta_L5mlp}
    print(f"Ablate L5_mlp (top logit attr): loss={loss_no_L5mlp:.4f}  delta={delta_L5mlp:+.4f}")

    print("\nRanked by causal impact (loss increase):")
    ranked = sorted(ablation_results.items(), key=lambda x: x[1]["delta"], reverse=True)
    for k, v in ranked:
        bar = "#" * int(abs(v["delta"]) * 10)
        print(f"  {k}: delta={v['delta']:+.4f}  {bar}")

    results["model2"] = {
        "description": "Shakespeare char-level 6-layer",
        "weights": weights,
        "baseline_loss": baseline_loss,
        "ablations": ablation_results,
    }


# ============================================================
# Model 3: Positional induction from seed divergence (4-layer)
# ============================================================

def run_model3(results: dict):
    print("\n" + "="*60)
    print("MODEL 3: Positional induction seed_divergence (4-layer, vocab=50)")
    print("="*60)

    # Latest step in seed_divergence/induction_seed0
    weights = "checkpoints/seed_divergence/induction_seed0/step_002000/model.safetensors"
    config = GPTConfig(n_layers=4, n_heads=4, d_model=128, d_ff=512, vocab_size=50, ctx_len=64)
    dataset = InductionDataset(vocab_size=50, seq_len=64, n_bigrams=6, seed=42)

    model = load_model(weights, config)
    print(f"Loaded: {weights}")

    # Check baseline accuracy
    baseline = compute_induction_accuracy(model, dataset)
    print(f"\nBaseline induction accuracy: {baseline:.4f}")

    if baseline < 0.05:
        print("WARNING: Baseline accuracy very low — model likely trained on different data format.")
        print("Testing with repeated-sequence data manually...")
        baseline_rep = _eval_repeated_seq(model, config)
        print(f"Repeated-sequence accuracy: {baseline_rep:.4f}")

        if baseline_rep < 0.05:
            print("SKIP: Model does not exhibit induction behavior on either format.")
            results["model3"] = {
                "description": "Seed divergence 4-layer vocab=50 (SKIPPED - low accuracy)",
                "weights": weights,
                "baseline_accuracy": baseline,
                "baseline_repeated_seq_accuracy": baseline_rep,
                "skipped": True,
            }
            return

        # Use repeated-seq baseline going forward
        baseline = baseline_rep
        use_repeated = True
    else:
        use_repeated = False

    # Ablate L0 attention (dominant component per analysis)
    if use_repeated:
        acc_no_L0attn = _eval_repeated_seq(model, config, ablate_attn_layers=[0])
    else:
        acc_no_L0attn = compute_induction_accuracy(model, dataset, ablate_attn_layers=[0])
    drop_L0attn = baseline - acc_no_L0attn
    print(f"\nAblate L0_attn (dominant component): acc={acc_no_L0attn:.4f}  drop={drop_L0attn:+.4f}")

    # Also test per-head ablation in L0
    head_results = {}
    for head in range(config.n_heads):
        if use_repeated:
            acc = _eval_repeated_seq(model, config, ablate_heads=[(0, head)])
        else:
            acc = compute_induction_accuracy(model, dataset, ablate_heads=[(0, head)])
        drop = baseline - acc
        key = f"L0H{head}"
        head_results[key] = {"accuracy": acc, "drop": drop}
        print(f"  Ablate {key}: acc={acc:.4f}  drop={drop:+.4f}")

    print("\nRanked by importance (drop):")
    ranked = sorted(head_results.items(), key=lambda x: x[1]["drop"], reverse=True)
    for k, v in ranked:
        bar = "#" * int(abs(v["drop"]) * 40)
        print(f"  {k}: drop={v['drop']:+.4f}  {bar}")

    results["model3"] = {
        "description": "Seed divergence 4-layer vocab=50",
        "weights": weights,
        "baseline_accuracy": baseline,
        "used_repeated_seq": use_repeated,
        "L0_attn_ablation": {"accuracy": acc_no_L0attn, "drop": drop_L0attn},
        "L0_head_ablations": head_results,
    }


def _eval_repeated_seq(
    model: GPT,
    config: GPTConfig,
    n_batches: int = 20,
    batch_size: int = 64,
    ablate_heads: list[tuple[int, int]] | None = None,
    ablate_attn_layers: list[int] | None = None,
) -> float:
    """
    Evaluate induction on manually constructed repeated sequences:
        full_seq = [first_half, first_half]
    The model should predict second half given first half context.
    """
    rng = np.random.default_rng(99)
    half = config.ctx_len // 2
    correct = 0
    total = 0

    for _ in range(n_batches):
        # Build batch of repeated sequences
        first_halves = rng.integers(0, config.vocab_size, size=(batch_size, half))
        seqs = np.concatenate([first_halves, first_halves], axis=1).astype(np.int32)
        inputs = mx.array(seqs[:, :-1])   # (B, ctx_len-1)
        targets_np = seqs[:, 1:]           # (B, ctx_len-1)

        logits = forward_with_ablations(
            model, inputs,
            ablate_heads=ablate_heads,
            ablate_attn_layers=ablate_attn_layers,
        )
        mx.eval(logits)
        preds = np.array(mx.argmax(logits, axis=-1))

        # Only measure accuracy on second-half positions (positions half..ctx_len-2)
        # In the shifted view: positions (half-1)..(ctx_len-2)
        for b in range(batch_size):
            for t in range(half - 1, config.ctx_len - 1):
                correct += int(preds[b, t] == targets_np[b, t])
                total += 1

    return correct / max(total, 1)


# ============================================================
# Main
# ============================================================

def print_summary(results: dict):
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)

    m1 = results.get("model1", {})
    if m1:
        print(f"\nModel 1 (Induction 2L): baseline accuracy = {m1['baseline_accuracy']:.4f}")
        ranked = sorted(m1["head_ablations"].items(), key=lambda x: x[1]["drop"], reverse=True)
        critical = [(k, v) for k, v in ranked if v["drop"] > 0.05]
        expendable = [(k, v) for k, v in ranked if abs(v["drop"]) <= 0.05]
        print(f"  Causally critical (drop > 5%): {[k for k, _ in critical]}")
        print(f"  Expendable (drop <= 5%):        {[k for k, _ in expendable]}")
        if ranked:
            top = ranked[0]
            print(f"  Most critical: {top[0]} (drop={top[1]['drop']:+.4f})")

    m2 = results.get("model2", {})
    if m2:
        print(f"\nModel 2 (Shakespeare 6L): baseline loss = {m2['baseline_loss']:.4f}")
        abl = m2.get("ablations", {})
        ranked = sorted(abl.items(), key=lambda x: x[1]["delta"], reverse=True)
        for k, v in ranked:
            impact = "CRITICAL" if v["delta"] > 0.10 else ("moderate" if v["delta"] > 0.02 else "minimal")
            print(f"  {k}: delta={v['delta']:+.4f}  [{impact}]")

    m3 = results.get("model3", {})
    if m3:
        if m3.get("skipped"):
            print(f"\nModel 3 (Seed divergence 4L): SKIPPED (baseline acc too low)")
        else:
            print(f"\nModel 3 (Seed divergence 4L): baseline accuracy = {m3['baseline_accuracy']:.4f}")
            l0 = m3.get("L0_attn_ablation", {})
            print(f"  L0_attn full ablation: drop={l0.get('drop', 0):+.4f}")
            heads = m3.get("L0_head_ablations", {})
            if heads:
                ranked_h = sorted(heads.items(), key=lambda x: x[1]["drop"], reverse=True)
                print(f"  Most critical L0 head: {ranked_h[0][0]} (drop={ranked_h[0][1]['drop']:+.4f})")


def main():
    results = {}

    run_model1(results)
    run_model2(results)
    run_model3(results)

    print_summary(results)

    # Save results
    out_path = Path("checkpoints/ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
