"""
Experiment: Suppression Cascade vs Model Depth

RESEARCH QUESTION:
The Shakespeare model (6 layers, 8 heads, 512 dim) shows a suppression cascade —
39/48 heads are suppressors with monotonically increasing strength from L1 to L5.
Does this generalise across depths?

DESIGN:
Train 4 Shakespeare char-level models at depths 2, 4, 6, 8 (all else equal).
Measure how circuit architecture (suppression cascade, head roles, logit attribution)
changes with depth.

RUNS:
- 4 models: n_layers ∈ {2, 4, 6, 8}
- All: 8 heads, 512 d_model, 2048 d_ff, vocab=65, ctx_len=128
- 3000 steps, batch_size=64, lr=3e-4, AdamW wd=0.1, warmup=200, cosine decay
- Save to checkpoints/depth_scaling/depth_{N}/model.safetensors

OUTPUT:
- experiments/depth_plots/depth_vs_loss.png
- experiments/depth_plots/depth_vs_suppression.png
- experiments/depth_plots/depth_vs_attribution.png
- experiments/depth_plots/cascade_pattern.png
- experiments/depth_scaling_results.json

FINDINGS section printed at end.
"""

import sys
import json
import time
import subprocess
import tempfile
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig, create_model
from src.data import TextDataset
from src.train import TrainConfig, get_lr, compute_loss
from src.circuit_mapping import profile_all_heads, decompose_residual_stream


# ============================================================
# Configuration
# ============================================================

DEPTHS = [2, 4, 6, 8]
N_HEADS = 8
D_MODEL = 512
D_FF = 2048
VOCAB_SIZE = 65          # Shakespeare char-level
CTX_LEN = 128            # Reduced from 256 for speed
N_STEPS = 3000           # Reduced from 5000 for speed
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 200
SEED = 42
DATA_DIR = "data"

CHECKPOINT_BASE = Path("checkpoints/depth_scaling")
PLOT_DIR = Path("experiments/depth_plots")
RESULTS_FILE = Path("experiments/depth_scaling_results.json")


# ============================================================
# Training helper (runs in-process)
# ============================================================

def train_depth_model(n_layers: int) -> dict:
    """Train a single depth model and return final metrics."""
    run_name = f"depth_{n_layers}"
    run_dir = CHECKPOINT_BASE / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = GPTConfig(
        n_layers=n_layers,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
        ctx_len=CTX_LEN,
        dropout=0.0,
    )

    print(f"\n{'='*60}")
    print(f"Training depth={n_layers} model (~{model_config.param_count():,} params)")
    print(f"{'='*60}")

    mx.random.seed(SEED)
    model = GPT(model_config)
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    dataset = TextDataset(data_dir=DATA_DIR, seq_len=CTX_LEN, seed=SEED)

    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    # Save config
    config_dict = {
        "model": asdict(model_config),
        "train": {
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "seed": SEED,
            "data_seed": SEED,
            "task": "text",
        },
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    t0 = time.time()
    loss_history = []

    for step, batch in enumerate(dataset.iter_batches(BATCH_SIZE, N_STEPS)):
        inputs, targets = batch

        # LR schedule: linear warmup + cosine decay
        if step < WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / max(1, N_STEPS - WARMUP_STEPS)
            lr = LR * 0.5 * (1.0 + np.cos(np.pi * progress))
        optimizer.learning_rate = lr

        loss, grads = loss_and_grad_fn(model, inputs, targets)

        # Elementwise gradient clipping.
        import mlx.utils
        grads = mlx.utils.tree_map(
            lambda g: mx.clip(g, -1.0, 1.0),
            grads,
        )

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % 100 == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            print(f"  depth={n_layers} step {step:4d}/{N_STEPS} | loss={loss_val:.4f} | "
                  f"lr={lr:.2e} | {steps_per_sec:.1f} step/s")

    elapsed = time.time() - t0
    print(f"  depth={n_layers} done in {elapsed:.1f}s")

    # Save final model
    model_path = run_dir / "model.safetensors"
    model.save_weights(str(model_path))
    print(f"  Saved to {model_path}")

    # Final loss: average of last 100 steps
    final_loss = float(np.mean(loss_history[-100:]))

    return {
        "n_layers": n_layers,
        "model_path": str(model_path),
        "final_loss": final_loss,
        "loss_history": loss_history,
        "elapsed": elapsed,
    }


# ============================================================
# Analysis helpers
# ============================================================

def compute_final_loss(model: GPT, dataset: TextDataset, n_batches: int = 20) -> float:
    """Evaluate model on held-out batches."""
    losses = []
    # Use a fresh dataset with different seed for held-out evaluation
    eval_dataset = TextDataset(data_dir=DATA_DIR, seq_len=CTX_LEN, seed=999)
    for batch in eval_dataset.iter_batches(32, n_batches):
        inputs, targets = batch
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        losses.append(loss.item())
    return float(np.mean(losses))


def analyze_model(model: GPT, dataset: TextDataset) -> dict:
    """Run full circuit analysis on a trained model."""
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads

    # Profile all heads
    profiles = profile_all_heads(model, dataset, n_samples=128)

    # Head role distribution
    role_counts = {}
    for p in profiles:
        role_counts[p.role] = role_counts.get(p.role, 0) + 1

    # Suppression heads: copy_score < -0.3  (negative OV diagonal)
    # Note: the HeadProfile uses suppression_score = max(0, -diag_mean / abs_mean)
    # A head is a suppressor if suppression_score > 0.3
    suppression_heads = [p for p in profiles if p.suppression_score > 0.3]
    n_suppression = len(suppression_heads)
    frac_suppression = n_suppression / len(profiles)

    # Per-layer mean suppression score (cascade pattern)
    layer_suppression = []
    for layer_idx in range(n_layers):
        layer_profiles = [p for p in profiles if p.layer == layer_idx]
        mean_supp = float(np.mean([p.suppression_score for p in layer_profiles]))
        layer_suppression.append(mean_supp)

    # Per-layer mean copy score
    layer_copy = []
    for layer_idx in range(n_layers):
        layer_profiles = [p for p in profiles if p.layer == layer_idx]
        mean_copy = float(np.mean([p.copy_score for p in layer_profiles]))
        layer_copy.append(mean_copy)

    # Logit attribution: attn vs MLP per layer
    # Use a batch of samples and decompose residual stream
    batch = dataset.generate_batch(32)
    inputs = batch[0]

    # Decompose residual stream into per-component contributions
    contributions = decompose_residual_stream(model, inputs)

    # Project each component through unembedding direction
    # Use mean logit magnitude per component as attribution strength
    embed_weight = np.array(model.wte.weight)  # (vocab, d_model)

    # Apply final layer norm approximation: just use L2 norms of contributions
    # as a proxy for logit attribution magnitude
    layer_attn_attr = []
    layer_mlp_attr = []

    for layer_idx in range(n_layers):
        attn_key = f"L{layer_idx}_attn"
        mlp_key = f"L{layer_idx}_mlp"

        attn_contrib = np.array(contributions[attn_key])  # (B, T, d_model)
        mlp_contrib = np.array(contributions[mlp_key])

        # Mean L2 norm as proxy for contribution magnitude
        attn_mag = float(np.mean(np.linalg.norm(attn_contrib, axis=-1)))
        mlp_mag = float(np.mean(np.linalg.norm(mlp_contrib, axis=-1)))

        layer_attn_attr.append(attn_mag)
        layer_mlp_attr.append(mlp_mag)

    # All head profiles as dicts
    head_profiles = [
        {
            "layer": p.layer,
            "head": p.head,
            "role": p.role,
            "copy_score": p.copy_score,
            "suppression_score": p.suppression_score,
            "prev_token_score": p.prev_token_score,
            "entropy": p.entropy,
            "positional_score": p.positional_score,
        }
        for p in profiles
    ]

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "total_heads": n_layers * n_heads,
        "role_counts": role_counts,
        "n_suppression": n_suppression,
        "frac_suppression": frac_suppression,
        "layer_suppression": layer_suppression,
        "layer_copy": layer_copy,
        "layer_attn_attr": layer_attn_attr,
        "layer_mlp_attr": layer_mlp_attr,
        "head_profiles": head_profiles,
    }


def check_cascade_monotonic(layer_suppression: list) -> bool:
    """Return True if suppression scores are monotonically increasing across layers."""
    for i in range(len(layer_suppression) - 1):
        if layer_suppression[i] > layer_suppression[i + 1]:
            return False
    return True


# ============================================================
# Plotting
# ============================================================

def make_plots(all_results: list):
    """Generate all depth scaling plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    depths = [r["n_layers"] for r in all_results]
    final_losses = [r["final_loss"] for r in all_results]
    eval_losses = [r.get("eval_loss", r["final_loss"]) for r in all_results]
    frac_supp = [r["analysis"]["frac_suppression"] for r in all_results]
    n_supp = [r["analysis"]["n_suppression"] for r in all_results]

    # ---- Plot 1: depth vs loss ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(depths, final_losses, "o-", color="steelblue", label="Train loss (last 100 steps)")
    ax.plot(depths, eval_losses, "s--", color="tomato", label="Eval loss (held-out)")
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Cross-entropy Loss")
    ax.set_title("Prediction Quality vs Depth")
    ax.legend()
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # bits per char = loss / ln(2)
    bpc_train = [l / np.log(2) for l in final_losses]
    bpc_eval = [l / np.log(2) for l in eval_losses]
    ax.plot(depths, bpc_train, "o-", color="steelblue", label="Train BPC")
    ax.plot(depths, bpc_eval, "s--", color="tomato", label="Eval BPC")
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Bits per Character")
    ax.set_title("Bits per Character vs Depth")
    ax.legend()
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "depth_vs_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / 'depth_vs_loss.png'}")

    # ---- Plot 2: suppression fraction + count vs depth ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(depths, frac_supp, "o-", color="darkorange", linewidth=2)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Fraction of Suppression Heads")
    ax.set_title("Suppression Head Fraction vs Depth")
    ax.set_xticks(depths)
    ax.set_ylim(0, 1)
    ax.axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="threshold=0.3")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    total_heads = [d * N_HEADS for d in depths]
    ax.bar(depths, n_supp, color="darkorange", alpha=0.7, label="Suppression heads")
    ax.bar(depths, [t - n for t, n in zip(total_heads, n_supp)],
           bottom=n_supp, color="steelblue", alpha=0.7, label="Other heads")
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Number of Heads")
    ax.set_title("Head Count by Role vs Depth")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "depth_vs_suppression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / 'depth_vs_suppression.png'}")

    # ---- Plot 3: MLP vs attn attribution per depth ----
    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 4), sharey=True)
    if len(all_results) == 1:
        axes = [axes]

    for ax, result in zip(axes, all_results):
        n_layers = result["n_layers"]
        layer_ids = list(range(n_layers))
        attn_attr = result["analysis"]["layer_attn_attr"]
        mlp_attr = result["analysis"]["layer_mlp_attr"]

        x = np.arange(n_layers)
        width = 0.35
        ax.bar(x - width/2, attn_attr, width, label="Attn", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, mlp_attr, width, label="MLP", color="tomato", alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_title(f"Depth={n_layers}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in layer_ids])
        if ax == axes[0]:
            ax.set_ylabel("Mean Contribution Magnitude (L2)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Attn vs MLP Contribution per Layer", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "depth_vs_attribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / 'depth_vs_attribution.png'}")

    # ---- Plot 4: cascade pattern (per-layer suppression, one line per depth) ----
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["steelblue", "darkorange", "green", "crimson"]
    markers = ["o", "s", "^", "D"]

    for i, result in enumerate(all_results):
        n_layers = result["n_layers"]
        layer_supp = result["analysis"]["layer_suppression"]
        layer_ids = list(range(n_layers))
        ax.plot(
            layer_ids,
            layer_supp,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"Depth={n_layers}",
        )

    ax.axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="Suppressor threshold")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Suppression Score")
    ax.set_title("Suppression Cascade Pattern by Depth\n(Higher = stronger suppression)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cascade_pattern.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / 'cascade_pattern.png'}")


# ============================================================
# Analysis & Findings
# ============================================================

def print_findings(all_results: list):
    """Print a human-readable summary of findings."""
    print("\n" + "=" * 70)
    print("FINDINGS: Suppression Cascade vs Depth")
    print("=" * 70)

    for r in all_results:
        n = r["n_layers"]
        loss = r["final_loss"]
        eval_loss = r.get("eval_loss", loss)
        ana = r["analysis"]
        frac = ana["frac_suppression"]
        n_supp = ana["n_suppression"]
        total = ana["total_heads"]
        layer_supp = ana["layer_suppression"]
        is_mono = check_cascade_monotonic(layer_supp)

        print(f"\nDepth={n}:")
        print(f"  Train loss (last 100 steps): {loss:.4f}")
        print(f"  Eval loss (held-out):         {eval_loss:.4f}")
        print(f"  Suppression heads: {n_supp}/{total} ({frac*100:.1f}%)")
        print(f"  Layer suppression scores: {[f'{s:.3f}' for s in layer_supp]}")
        print(f"  Cascade is monotonic: {is_mono}")
        print(f"  Head roles: {ana['role_counts']}")

    print("\n--- Summary ---")

    # Q1: Does the cascade appear at all depths?
    depths_with_cascade = [r["n_layers"] for r in all_results
                           if r["analysis"]["frac_suppression"] > 0.3]
    print(f"\nQ1: Suppression cascade (>30% suppressor heads) at depths: {depths_with_cascade}")
    print(f"    -> {'YES, appears at all tested depths' if len(depths_with_cascade) == len(all_results) else 'NO, only at some depths'}")

    # Q2: Does it get steeper with depth?
    fracs = [r["analysis"]["frac_suppression"] for r in all_results]
    depths = [r["n_layers"] for r in all_results]
    is_increasing = all(fracs[i] <= fracs[i+1] for i in range(len(fracs)-1))
    print(f"\nQ2: Suppression fraction by depth: {dict(zip(depths, [f'{f:.2f}' for f in fracs]))}")
    print(f"    -> {'YES, steeper with depth' if is_increasing else 'NO, non-monotonic'}")

    # Q3: Prediction quality vs depth
    losses = [r.get("eval_loss", r["final_loss"]) for r in all_results]
    is_improving = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    print(f"\nQ3: Eval loss by depth: {dict(zip(depths, [f'{l:.4f}' for l in losses]))}")
    print(f"    -> {'YES, quality improves with depth' if is_improving else 'NO, non-monotonic'}")

    # Q4: Critical depth below which cascade doesn't form?
    no_cascade = [r["n_layers"] for r in all_results
                  if r["analysis"]["frac_suppression"] <= 0.3]
    if no_cascade:
        print(f"\nQ4: No suppression cascade at depths: {no_cascade}")
        print(f"    -> Critical depth threshold appears to be above {max(no_cascade)}")
    else:
        print(f"\nQ4: Cascade forms at all tested depths (min depth={min(depths)})")
        print(f"    -> Critical depth (if any) is below {min(depths)} layers")

    print()


# ============================================================
# Main
# ============================================================

def main():
    print("Depth Scaling Experiment: Suppression Cascade")
    print(f"Depths: {DEPTHS}, Steps: {N_STEPS}, ctx_len: {CTX_LEN}, batch: {BATCH_SIZE}")
    print(f"Checkpoints: {CHECKPOINT_BASE}")
    print(f"Plots: {PLOT_DIR}")

    CHECKPOINT_BASE.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load dataset once (downloads shakespeare.txt if needed)
    print("\nLoading dataset...")
    dataset = TextDataset(data_dir=DATA_DIR, seq_len=CTX_LEN, seed=SEED)

    all_results = []

    # Train models sequentially (MLX on Apple Silicon doesn't benefit from
    # process-level parallelism due to shared GPU memory)
    for n_layers in DEPTHS:
        model_path = CHECKPOINT_BASE / f"depth_{n_layers}" / "model.safetensors"

        if model_path.exists():
            print(f"\nFound existing checkpoint for depth={n_layers}, skipping training.")
            # Load config to get final_loss from metrics if available
            run_dir = CHECKPOINT_BASE / f"depth_{n_layers}"
            # We don't have stored final loss, so we'll compute it during analysis
            train_result = {
                "n_layers": n_layers,
                "model_path": str(model_path),
                "final_loss": None,
                "loss_history": [],
                "elapsed": 0.0,
            }
        else:
            train_result = train_depth_model(n_layers)

        all_results.append(train_result)

    # Analyze each model
    print("\n" + "=" * 60)
    print("Analysis phase")
    print("=" * 60)

    for result in all_results:
        n_layers = result["n_layers"]
        model_path = Path(result["model_path"])

        print(f"\nAnalyzing depth={n_layers} model...")

        model_config = GPTConfig(
            n_layers=n_layers,
            n_heads=N_HEADS,
            d_model=D_MODEL,
            d_ff=D_FF,
            vocab_size=VOCAB_SIZE,
            ctx_len=CTX_LEN,
            dropout=0.0,
        )
        model = GPT(model_config)
        model.load_weights(str(model_path))
        mx.eval(model.parameters())

        # Final loss if not set (model was loaded from existing checkpoint)
        if result["final_loss"] is None:
            result["final_loss"] = compute_final_loss(model, dataset)
            print(f"  Computed train loss: {result['final_loss']:.4f}")

        # Held-out eval loss
        result["eval_loss"] = compute_final_loss(model, dataset)
        print(f"  Eval loss: {result['eval_loss']:.4f}")

        # Circuit analysis
        result["analysis"] = analyze_model(model, dataset)
        print(f"  Suppression heads: {result['analysis']['n_suppression']}/"
              f"{result['analysis']['total_heads']} "
              f"({result['analysis']['frac_suppression']*100:.1f}%)")
        print(f"  Layer suppression: {[f'{s:.3f}' for s in result['analysis']['layer_suppression']]}")
        print(f"  Monotonic cascade: {check_cascade_monotonic(result['analysis']['layer_suppression'])}")

    # Print findings
    print_findings(all_results)

    # Generate plots
    print("Generating plots...")
    make_plots(all_results)

    # Save results to JSON (exclude bulky loss_history, keep head_profiles)
    results_to_save = []
    for r in all_results:
        entry = {
            "n_layers": r["n_layers"],
            "model_path": r["model_path"],
            "final_loss": r["final_loss"],
            "eval_loss": r.get("eval_loss"),
            "elapsed_s": r.get("elapsed", 0.0),
            "analysis": {k: v for k, v in r["analysis"].items()
                         if k != "head_profiles"},
            "head_profiles": r["analysis"]["head_profiles"],
            "cascade_monotonic": check_cascade_monotonic(r["analysis"]["layer_suppression"]),
        }
        results_to_save.append(entry)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
