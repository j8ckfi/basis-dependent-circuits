"""
Surgical ablation experiments to causally validate circuit maps.

For each model:
1. Per-head ablation: zero out each head's output one at a time
2. Per-layer-type ablation: zero out all attention OR all MLP in a layer
3. Role-specific ablations: targeted knockouts based on circuit hypotheses

Models tested:
- Induction (2L4H): claims L1 heads are induction heads, L0 heads encode token identity
- Shakespeare (6L8H): claims L0H7 is prev-token head, L1-L5 are suppression cascade, L5 MLP predicts
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
from src.data import InductionDataset, TextDataset


# ---------------------------------------------------------------------------
# Custom ablatable forward pass
# ---------------------------------------------------------------------------

def forward_with_ablations(
    model: GPT,
    idx: mx.array,
    ablate_heads: Optional[list[tuple[int, int]]] = None,
    ablate_attn_layers: Optional[list[int]] = None,
    ablate_mlp_layers: Optional[list[int]] = None,
) -> mx.array:
    """Forward pass with selective zeroing of components.

    Args:
        model: GPT model
        idx: (B, T) token indices
        ablate_heads: list of (layer, head) tuples to zero out
        ablate_attn_layers: list of layer indices to zero ALL attention output
        ablate_mlp_layers: list of layer indices to zero ALL MLP output

    Returns:
        logits: (B, T, vocab_size)
    """
    ablate_heads = ablate_heads or []
    ablate_attn_layers = ablate_attn_layers or []
    ablate_mlp_layers = ablate_mlp_layers or []

    # Build sets for O(1) lookup
    ablate_heads_set = set(ablate_heads)
    ablate_attn_set = set(ablate_attn_layers)
    ablate_mlp_set = set(ablate_mlp_layers)

    B, T = idx.shape
    pos = mx.arange(T)
    x = model.wte(idx) + model.wpe(pos)

    for layer_idx, block in enumerate(model.blocks):
        # --- Attention sub-block ---
        x_norm = block.ln1(x)

        if layer_idx in ablate_attn_set:
            # Zero entire attention output for this layer
            attn_out = mx.zeros_like(x_norm)
        elif ablate_heads_set and any(l == layer_idx for l, _ in ablate_heads_set):
            # Need to ablate specific heads: manual per-head computation
            attn_out = _attn_with_head_ablation(
                block.attn, x_norm, ablate_heads_set, layer_idx
            )
        else:
            attn_out = block.attn(x_norm)

        x = x + attn_out

        # --- MLP sub-block ---
        x_norm2 = block.ln2(x)
        if layer_idx in ablate_mlp_set:
            mlp_out = mx.zeros_like(x_norm2)
        else:
            mlp_out = block.mlp(x_norm2)

        x = x + mlp_out

    x = model.ln_f(x)
    logits = x @ model.wte.weight.T
    return logits


def _attn_with_head_ablation(
    attn_module,
    x: mx.array,
    ablate_heads_set: set,
    layer_idx: int,
) -> mx.array:
    """Run attention with specific heads zeroed.

    Replicates CausalSelfAttention.__call__ but zeros specified heads'
    value-weighted output before the out_proj.
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
    attn = attn + mask
    attn = mx.softmax(attn, axis=-1)

    # (B, n_heads, T, d_head)
    out = attn @ v

    # Zero out ablated heads
    for _, head_idx in [(l, h) for (l, h) in ablate_heads_set if l == layer_idx]:
        # out[:, head_idx, :, :] = 0  -- MLX needs explicit construction
        zeros = mx.zeros((B, T, d_head))
        # Build new out by slicing and reassembling
        # MLX doesn't support item assignment; use masked multiply
        head_mask = mx.ones((n_heads,))
        head_mask_list = [0.0 if h == head_idx else 1.0 for h in range(n_heads)]
        head_mask = mx.array(head_mask_list).reshape(1, n_heads, 1, 1)
        out = out * head_mask

    # Reshape and project
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = attn_module.out_proj(out)
    return out


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_induction_accuracy(
    model: GPT,
    dataset: InductionDataset,
    n_batches: int = 10,
    batch_size: int = 32,
    ablate_heads: Optional[list] = None,
    ablate_attn_layers: Optional[list] = None,
    ablate_mlp_layers: Optional[list] = None,
) -> float:
    """Compute top-1 accuracy at induction-masked positions."""
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

        logits_np = np.array(logits)      # (B, T, V)
        targets_np = np.array(targets)    # (B, T)
        mask_np = np.array(mask)          # (B, T)

        preds = logits_np.argmax(axis=-1)  # (B, T)
        at_mask = mask_np > 0.5

        correct += int((preds == targets_np)[at_mask].sum())
        total += int(at_mask.sum())

    return correct / max(total, 1)


def compute_shakespeare_loss(
    model: GPT,
    dataset: TextDataset,
    n_batches: int = 10,
    batch_size: int = 16,
    ablate_heads: Optional[list] = None,
    ablate_attn_layers: Optional[list] = None,
    ablate_mlp_layers: Optional[list] = None,
) -> float:
    """Compute mean cross-entropy loss over all positions."""
    total_loss = 0.0
    n_seen = 0

    for _ in range(n_batches):
        inputs, targets = dataset.generate_batch(batch_size)
        logits = forward_with_ablations(
            model, inputs,
            ablate_heads=ablate_heads,
            ablate_attn_layers=ablate_attn_layers,
            ablate_mlp_layers=ablate_mlp_layers,
        )
        mx.eval(logits)

        logits_np = np.array(logits)   # (B, T, V)
        targets_np = np.array(targets) # (B, T)
        B, T, V = logits_np.shape

        # Cross-entropy via numerically stable log-softmax.
        max_logits = logits_np.max(axis=-1, keepdims=True)
        log_probs = logits_np - max_logits - np.log(
            np.exp(logits_np - max_logits).sum(axis=-1, keepdims=True) + 1e-10
        )
        flat_lp = log_probs.reshape(B * T, V)
        flat_tgt = targets_np.reshape(B * T)
        loss = -flat_lp[np.arange(B * T), flat_tgt].mean()

        total_loss += float(loss)
        n_seen += 1

    return total_loss / max(n_seen, 1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_induction_ablations(model: GPT, dataset: InductionDataset) -> dict:
    """Run all ablation experiments for the induction model."""
    config = model.config
    n_layers = config.n_layers
    n_heads = config.n_heads

    print("\n" + "=" * 60)
    print("INDUCTION MODEL ABLATIONS")
    print("=" * 60)

    results = {}

    # Baseline
    print("\nComputing baseline accuracy...")
    baseline = compute_induction_accuracy(model, dataset)
    results["baseline"] = baseline
    print(f"  Baseline accuracy: {baseline:.4f}")

    # 1. Per-head ablation
    print("\n--- Per-head ablation ---")
    per_head = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            acc = compute_induction_accuracy(model, dataset, ablate_heads=[(layer, head)])
            drop = baseline - acc
            key = f"L{layer}H{head}"
            per_head[key] = {"accuracy": acc, "drop": drop}
            print(f"  Ablate {key}: acc={acc:.4f}  drop={drop:+.4f}")
    results["per_head"] = per_head

    # 2. Per-layer-type ablation
    print("\n--- Per-layer-type ablation ---")
    per_layer_type = {}
    for layer in range(n_layers):
        # Zero all attention
        acc_attn = compute_induction_accuracy(model, dataset, ablate_attn_layers=[layer])
        drop_attn = baseline - acc_attn
        key_attn = f"L{layer}_attn"
        per_layer_type[key_attn] = {"accuracy": acc_attn, "drop": drop_attn}
        print(f"  Ablate {key_attn}: acc={acc_attn:.4f}  drop={drop_attn:+.4f}")

        # Zero MLP
        acc_mlp = compute_induction_accuracy(model, dataset, ablate_mlp_layers=[layer])
        drop_mlp = baseline - acc_mlp
        key_mlp = f"L{layer}_mlp"
        per_layer_type[key_mlp] = {"accuracy": acc_mlp, "drop": drop_mlp}
        print(f"  Ablate {key_mlp}: acc={acc_mlp:.4f}  drop={drop_mlp:+.4f}")
    results["per_layer_type"] = per_layer_type

    # 3. Role-specific ablations
    print("\n--- Role-specific ablations ---")
    role_specific = {}

    # Ablate ALL L1 heads simultaneously
    all_l1_heads = [(1, h) for h in range(n_heads)]
    acc = compute_induction_accuracy(model, dataset, ablate_heads=all_l1_heads)
    role_specific["all_L1_heads"] = {
        "accuracy": acc,
        "drop": baseline - acc,
        "hypothesis": "Should drop to chance: L1 heads are induction heads",
    }
    print(f"  Ablate all L1 heads: acc={acc:.4f}  drop={baseline-acc:+.4f}")

    # Ablate ALL L0 heads simultaneously
    all_l0_heads = [(0, h) for h in range(n_heads)]
    acc = compute_induction_accuracy(model, dataset, ablate_heads=all_l0_heads)
    role_specific["all_L0_heads"] = {
        "accuracy": acc,
        "drop": baseline - acc,
        "hypothesis": "Should also drop: L0 heads encode token identity",
    }
    print(f"  Ablate all L0 heads: acc={acc:.4f}  drop={baseline-acc:+.4f}")

    results["role_specific"] = role_specific

    return results


def run_shakespeare_ablations(model: GPT, dataset: TextDataset) -> dict:
    """Run all ablation experiments for the Shakespeare model."""
    config = model.config
    n_layers = config.n_layers
    n_heads = config.n_heads

    print("\n" + "=" * 60)
    print("SHAKESPEARE MODEL ABLATIONS")
    print("=" * 60)

    results = {}

    # Baseline
    print("\nComputing baseline loss...")
    baseline = compute_shakespeare_loss(model, dataset)
    results["baseline"] = baseline
    print(f"  Baseline loss: {baseline:.4f}")

    # 1. Per-head ablation
    print("\n--- Per-head ablation ---")
    per_head = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            loss = compute_shakespeare_loss(model, dataset, ablate_heads=[(layer, head)])
            increase = loss - baseline
            key = f"L{layer}H{head}"
            per_head[key] = {"loss": loss, "increase": increase}
            print(f"  Ablate {key}: loss={loss:.4f}  increase={increase:+.4f}")
    results["per_head"] = per_head

    # 2. Per-layer-type ablation
    print("\n--- Per-layer-type ablation ---")
    per_layer_type = {}
    for layer in range(n_layers):
        loss_attn = compute_shakespeare_loss(model, dataset, ablate_attn_layers=[layer])
        inc_attn = loss_attn - baseline
        key_attn = f"L{layer}_attn"
        per_layer_type[key_attn] = {"loss": loss_attn, "increase": inc_attn}
        print(f"  Ablate {key_attn}: loss={loss_attn:.4f}  increase={inc_attn:+.4f}")

        loss_mlp = compute_shakespeare_loss(model, dataset, ablate_mlp_layers=[layer])
        inc_mlp = loss_mlp - baseline
        key_mlp = f"L{layer}_mlp"
        per_layer_type[key_mlp] = {"loss": loss_mlp, "increase": inc_mlp}
        print(f"  Ablate {key_mlp}: loss={loss_mlp:.4f}  increase={inc_mlp:+.4f}")
    results["per_layer_type"] = per_layer_type

    # 3. Role-specific ablations
    print("\n--- Role-specific ablations ---")
    role_specific = {}

    # Ablate L0H7 specifically
    loss = compute_shakespeare_loss(model, dataset, ablate_heads=[(0, 7)])
    role_specific["L0H7"] = {
        "loss": loss,
        "increase": loss - baseline,
        "hypothesis": "Should hurt sequential prediction: L0H7 is prev-token head",
    }
    print(f"  Ablate L0H7: loss={loss:.4f}  increase={loss-baseline:+.4f}")

    # Ablate all L1-L5 attention (suppression cascade)
    l1_to_l5_attn = list(range(1, 6))
    loss = compute_shakespeare_loss(model, dataset, ablate_attn_layers=l1_to_l5_attn)
    role_specific["all_L1_L5_attn"] = {
        "loss": loss,
        "increase": loss - baseline,
        "hypothesis": "Tests if suppression cascade (L1-L5 attn) is needed",
    }
    print(f"  Ablate all L1-L5 attention: loss={loss:.4f}  increase={loss-baseline:+.4f}")

    # Ablate L5 MLP
    loss = compute_shakespeare_loss(model, dataset, ablate_mlp_layers=[5])
    role_specific["L5_mlp"] = {
        "loss": loss,
        "increase": loss - baseline,
        "hypothesis": "Tests if L5 MLP is critical for final prediction",
    }
    print(f"  Ablate L5 MLP: loss={loss:.4f}  increase={loss-baseline:+.4f}")

    results["role_specific"] = role_specific

    return results


# ---------------------------------------------------------------------------
# Pretty table printing
# ---------------------------------------------------------------------------

def print_induction_summary(results: dict):
    baseline = results["baseline"]
    print("\n" + "=" * 60)
    print("INDUCTION MODEL — SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Component':<20} {'Accuracy':>10} {'Drop':>10} {'% Drop':>10}")
    print("-" * 55)
    print(f"{'Baseline':<20} {baseline:>10.4f} {'—':>10} {'—':>10}")

    print("\n[Per-head ablation]")
    for key, v in results["per_head"].items():
        pct = 100 * v["drop"] / max(baseline, 1e-9)
        print(f"  {key:<18} {v['accuracy']:>10.4f} {v['drop']:>+10.4f} {pct:>9.1f}%")

    print("\n[Per-layer-type ablation]")
    for key, v in results["per_layer_type"].items():
        pct = 100 * v["drop"] / max(baseline, 1e-9)
        print(f"  {key:<18} {v['accuracy']:>10.4f} {v['drop']:>+10.4f} {pct:>9.1f}%")

    print("\n[Role-specific ablations]")
    for key, v in results["role_specific"].items():
        pct = 100 * v["drop"] / max(baseline, 1e-9)
        print(f"  {key:<18} {v['accuracy']:>10.4f} {v['drop']:>+10.4f} {pct:>9.1f}%")
        print(f"    -> {v['hypothesis']}")


def print_shakespeare_summary(results: dict):
    baseline = results["baseline"]
    print("\n" + "=" * 60)
    print("SHAKESPEARE MODEL — SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Component':<20} {'Loss':>10} {'Increase':>10} {'% Incr':>10}")
    print("-" * 55)
    print(f"{'Baseline':<20} {baseline:>10.4f} {'—':>10} {'—':>10}")

    print("\n[Per-head ablation]")
    for key, v in results["per_head"].items():
        pct = 100 * v["increase"] / max(baseline, 1e-9)
        print(f"  {key:<18} {v['loss']:>10.4f} {v['increase']:>+10.4f} {pct:>9.1f}%")

    print("\n[Per-layer-type ablation]")
    for key, v in results["per_layer_type"].items():
        pct = 100 * v["increase"] / max(baseline, 1e-9)
        print(f"  {key:<18} {v['loss']:>10.4f} {v['increase']:>+10.4f} {pct:>9.1f}%")

    print("\n[Role-specific ablations]")
    for key, v in results["role_specific"].items():
        pct = 100 * v["increase"] / max(baseline, 1e-9)
        print(f"  {key:<18} {v['loss']:>10.4f} {v['increase']:>+10.4f} {pct:>9.1f}%")
        print(f"    -> {v['hypothesis']}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_induction_ablations(results: dict, save_path: str):
    baseline = results["baseline"]
    per_head = results["per_head"]
    per_layer_type = results["per_layer_type"]
    role_specific = results["role_specific"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Induction Model Ablations\n(Higher drop = more important component)", fontsize=13)

    # Panel 1: Per-head drops
    ax = axes[0]
    keys = list(per_head.keys())
    drops = [per_head[k]["drop"] for k in keys]
    colors = ["#d62728" if d > 0.05 else "#1f77b4" for d in drops]
    ax.barh(keys, drops, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Drop")
    ax.set_title("Per-Head Ablation")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 2: Per-layer-type drops
    ax = axes[1]
    keys2 = list(per_layer_type.keys())
    drops2 = [per_layer_type[k]["drop"] for k in keys2]
    colors2 = ["#d62728" if d > 0.05 else "#1f77b4" for d in drops2]
    ax.barh(keys2, drops2, color=colors2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Drop")
    ax.set_title("Per-Layer-Type Ablation")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 3: Role-specific
    ax = axes[2]
    rs_keys = list(role_specific.keys())
    rs_drops = [role_specific[k]["drop"] for k in rs_keys]
    ax.bar(rs_keys, rs_drops, color=["#d62728" if d > 0.05 else "#1f77b4" for d in rs_drops])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(baseline, color="green", linestyle="--", linewidth=1, label=f"baseline={baseline:.3f}")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Role-Specific Ablations")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shakespeare_ablations(results: dict, save_path: str):
    baseline = results["baseline"]
    per_head = results["per_head"]
    per_layer_type = results["per_layer_type"]
    role_specific = results["role_specific"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Shakespeare Model Ablations\n(Higher increase = more important component)", fontsize=13)

    # Panel 1: Per-head loss increase (grouped by layer)
    ax = axes[0]
    keys = list(per_head.keys())
    increases = [per_head[k]["increase"] for k in keys]
    colors = ["#d62728" if i > 0.05 else "#1f77b4" for i in increases]
    y_pos = range(len(keys))
    ax.barh(list(y_pos), increases, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(keys, fontsize=6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Loss Increase")
    ax.set_title("Per-Head Ablation")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 2: Per-layer-type loss increase
    ax = axes[1]
    keys2 = list(per_layer_type.keys())
    increases2 = [per_layer_type[k]["increase"] for k in keys2]
    colors2 = ["#d62728" if i > 0.05 else "#1f77b4" for i in increases2]
    ax.barh(keys2, increases2, color=colors2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Loss Increase")
    ax.set_title("Per-Layer-Type Ablation")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 3: Role-specific
    ax = axes[2]
    rs_keys = list(role_specific.keys())
    rs_inc = [role_specific[k]["increase"] for k in rs_keys]
    ax.bar(rs_keys, rs_inc, color=["#d62728" if i > 0.05 else "#1f77b4" for i in rs_inc])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Loss Increase")
    ax.set_title("Role-Specific Ablations")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_results = {}

    # -----------------------------------------------------------------------
    # MODEL 1: Induction content-match (2L, 4H)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# MODEL 1: Induction content-match (2L, 4H)")
    print("#" * 70)

    induction_ckpt = (
        "checkpoints/induction_content_match/induction_seed0/"
        "step_020000/model.safetensors"
    )
    induction_config = GPTConfig(
        n_layers=2, n_heads=4, d_model=128, d_ff=512,
        vocab_size=512, ctx_len=64,
    )

    print(f"Loading model from: {induction_ckpt}")
    induction_model = load_model(induction_ckpt, induction_config)

    induction_dataset = InductionDataset(
        vocab_size=512, seq_len=64, n_bigrams=6, seed=42
    )

    induction_results = run_induction_ablations(induction_model, induction_dataset)
    print_induction_summary(induction_results)
    all_results["induction"] = induction_results

    plot_induction_ablations(
        induction_results,
        str(plots_dir / "ablation_induction.png"),
    )

    # -----------------------------------------------------------------------
    # MODEL 2: Shakespeare char-level (6L, 8H)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# MODEL 2: Shakespeare char-level (6L, 8H)")
    print("#" * 70)

    # Find latest Shakespeare checkpoint
    shakes_run_dir = Path("checkpoints/shakespeare/text_seed0")
    shakes_ckpts = sorted(shakes_run_dir.glob("step_*"))
    if not shakes_ckpts:
        raise FileNotFoundError(f"No checkpoints found in {shakes_run_dir}")
    shakes_ckpt = shakes_ckpts[-1] / "model.safetensors"
    print(f"Loading Shakespeare checkpoint: {shakes_ckpt}")

    # Load dataset first to get vocab_size
    print("Loading TextDataset (Shakespeare)...")
    shakes_dataset = TextDataset(data_dir="data", seq_len=256, seed=42)

    shakes_config = GPTConfig(
        n_layers=6, n_heads=8, d_model=512, d_ff=2048,
        vocab_size=shakes_dataset.vocab_size, ctx_len=256,
    )
    print(f"vocab_size={shakes_dataset.vocab_size}")

    shakes_model = load_model(str(shakes_ckpt), shakes_config)

    shakes_results = run_shakespeare_ablations(shakes_model, shakes_dataset)
    print_shakespeare_summary(shakes_results)
    all_results["shakespeare"] = shakes_results

    plot_shakespeare_ablations(
        shakes_results,
        str(plots_dir / "ablation_shakespeare.png"),
    )

    # -----------------------------------------------------------------------
    # Save combined results
    # -----------------------------------------------------------------------
    out_path = Path(__file__).parent / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
