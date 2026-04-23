"""
Ablation robustness analysis across random seeds.

Tests whether the "single catastrophically critical head in Layer 0" finding
from seed 0 reproduces across seeds 1, 2, 3.

Pipeline:
1. Train content-matching induction models for seeds 1, 2, 3 (seed 0 already exists)
2. For each of the 4 seeds, run per-head ablation (8 heads: 2 layers x 4 heads)
3. Identify the "critical head" per seed (largest accuracy drop)
4. Report statistics and generate a bar-chart comparison

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/ablation_robustness.py
"""

import sys
import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Make src importable from analysis/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig, create_model
from src.data import InductionDataset
from src.train import TrainConfig, get_lr, compute_masked_loss, compute_task_accuracy


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

TRAIN_STEPS = 20000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
DATA_SEED = 42   # fixed across all seeds

SEEDS = [0, 1, 2, 3]

BASE_DIR = Path(__file__).parent.parent
CKPT_BASE = BASE_DIR / "checkpoints" / "induction_content_match"
RESULTS_PATH = Path(__file__).parent / "ablation_robustness_results.json"
PLOT_PATH = Path(__file__).parent / "plots" / "ablation_robustness.png"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seed(seed: int) -> Path:
    """Train a content-matching induction model for a given seed.

    Returns path to the saved model weights file.
    """
    run_name = f"induction_seed{seed}"
    run_dir = CKPT_BASE / run_name
    final_ckpt = run_dir / "step_020000" / "model.safetensors"

    if final_ckpt.exists():
        print(f"[seed {seed}] Checkpoint already exists at {final_ckpt}, skipping training.")
        return final_ckpt

    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[seed {seed}] Creating model (~{MODEL_CONFIG.param_count():,} params)...")
    model = create_model(MODEL_CONFIG, seed=seed)

    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    dataset = InductionDataset(
        vocab_size=512,
        seq_len=64,
        n_bigrams=6,
        seed=DATA_SEED,
    )

    loss_and_grad_fn = nn.value_and_grad(model, compute_masked_loss)

    print(f"[seed {seed}] Training for {TRAIN_STEPS} steps...")
    t0 = time.time()
    last_metrics: dict = {}

    for step, (inputs, targets, mask) in enumerate(
        dataset.iter_batches(BATCH_SIZE, TRAIN_STEPS)
    ):
        lr = _get_lr(step)
        optimizer.learning_rate = lr

        loss, grads = loss_and_grad_fn(model, inputs, targets, mask)

        if GRAD_CLIP > 0:
            import mlx.utils
            grads = mlx.utils.tree_map(
                lambda g: mx.clip(g, -GRAD_CLIP, GRAD_CLIP),
                grads,
            )

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        last_metrics = {"step": step, "loss": loss.item(), "lr": lr}

        if step % 1000 == 0:
            acc = compute_task_accuracy(model, inputs, targets, mask)
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (TRAIN_STEPS - step) / max(rate, 1e-6)
            print(
                f"  [seed {seed}] step {step:5d}/{TRAIN_STEPS} | "
                f"loss={loss.item():.4f} | acc={acc:.3f} | "
                f"{rate:.0f} steps/s | ETA {eta/60:.1f}min"
            )

    # Save final checkpoint
    ckpt_dir = run_dir / "step_020000"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(ckpt_dir / "model.safetensors"))

    elapsed = time.time() - t0
    print(f"[seed {seed}] Done in {elapsed:.1f}s ({TRAIN_STEPS / elapsed:.0f} steps/s)")
    print(f"[seed {seed}] Saved to {ckpt_dir / 'model.safetensors'}")
    return ckpt_dir / "model.safetensors"


def _get_lr(step: int) -> float:
    """Linear warmup + cosine decay (mirrors src/train.py)."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: Path) -> GPT:
    model = GPT(MODEL_CONFIG)
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Ablation (re-uses the approach from analysis/ablation.py)
# ---------------------------------------------------------------------------

def forward_with_ablations(
    model: GPT,
    idx: mx.array,
    ablate_heads: Optional[list] = None,
) -> mx.array:
    """Forward pass zeroing out specified (layer, head) pairs."""
    ablate_heads_set = set(ablate_heads or [])

    B, T = idx.shape
    pos = mx.arange(T)
    x = model.wte(idx) + model.wpe(pos)

    for layer_idx, block in enumerate(model.blocks):
        x_norm = block.ln1(x)

        if any(l == layer_idx for l, _ in ablate_heads_set):
            attn_out = _attn_with_head_ablation(
                block.attn, x_norm, ablate_heads_set, layer_idx
            )
        else:
            attn_out = block.attn(x_norm)

        x = x + attn_out
        x = x + block.mlp(block.ln2(x))

    x = model.ln_f(x)
    return x @ model.wte.weight.T


def _attn_with_head_ablation(attn_module, x, ablate_heads_set, layer_idx):
    """Run attention with specific heads zeroed (mirrors analysis/ablation.py)."""
    B, T, C = x.shape
    n_heads = attn_module.n_heads
    d_head = attn_module.d_head

    qkv = attn_module.qkv_proj(x)
    qkv = qkv.reshape(B, T, 3, n_heads, d_head)
    qkv = qkv.transpose(0, 3, 2, 1, 4)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    causal_mask = mx.triu(mx.full((T, T), -1e9), k=1)
    attn = mx.softmax(attn + causal_mask, axis=-1)

    out = attn @ v  # (B, n_heads, T, d_head)

    for _, head_idx in [(l, h) for (l, h) in ablate_heads_set if l == layer_idx]:
        head_mask = mx.array(
            [0.0 if h == head_idx else 1.0 for h in range(n_heads)]
        ).reshape(1, n_heads, 1, 1)
        out = out * head_mask

    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    return attn_module.out_proj(out)


def compute_induction_accuracy(
    model: GPT,
    dataset: InductionDataset,
    n_batches: int = 20,
    batch_size: int = 64,
    ablate_heads: Optional[list] = None,
) -> float:
    """Top-1 accuracy at induction-masked positions."""
    correct = 0
    total = 0
    for _ in range(n_batches):
        inputs, targets, mask = dataset.generate_batch(batch_size)
        logits = forward_with_ablations(model, inputs, ablate_heads=ablate_heads)
        mx.eval(logits)

        preds = np.array(logits).argmax(axis=-1)
        tgt_np = np.array(targets)
        mask_np = np.array(mask) > 0.5

        correct += int((preds == tgt_np)[mask_np].sum())
        total += int(mask_np.sum())
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Per-seed ablation sweep
# ---------------------------------------------------------------------------

def run_ablation_for_seed(seed: int, dataset: InductionDataset) -> dict:
    """Load model for seed, compute baseline + per-head ablation results."""
    weights_path = CKPT_BASE / f"induction_seed{seed}" / "step_020000" / "model.safetensors"
    print(f"\n[seed {seed}] Loading model from {weights_path}")
    model = load_model(weights_path)

    print(f"[seed {seed}] Computing baseline accuracy...")
    baseline = compute_induction_accuracy(model, dataset)
    print(f"[seed {seed}] Baseline: {baseline:.4f}")

    n_layers = MODEL_CONFIG.n_layers
    n_heads = MODEL_CONFIG.n_heads

    per_head: dict = {}
    print(f"[seed {seed}] Per-head ablation ({n_layers * n_heads} heads)...")
    for layer in range(n_layers):
        for head in range(n_heads):
            acc = compute_induction_accuracy(
                model, dataset, ablate_heads=[(layer, head)]
            )
            drop = baseline - acc
            key = f"L{layer}H{head}"
            per_head[key] = {"accuracy": acc, "drop": drop, "layer": layer, "head": head}
            print(f"  Ablate {key}: acc={acc:.4f}  drop={drop:+.4f}")

    # Identify critical head
    critical_key = max(per_head, key=lambda k: per_head[k]["drop"])
    critical = per_head[critical_key]

    return {
        "seed": seed,
        "baseline": baseline,
        "per_head": per_head,
        "critical_head": {
            "key": critical_key,
            "layer": critical["layer"],
            "head": critical["head"],
            "accuracy_after_ablation": critical["accuracy"],
            "accuracy_drop": critical["drop"],
        },
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(seed_results: list[dict]) -> dict:
    """Aggregate findings across seeds."""
    baselines = [r["baseline"] for r in seed_results]
    critical_heads = [r["critical_head"] for r in seed_results]
    drops = [c["accuracy_drop"] for c in critical_heads]
    layers = [c["layer"] for c in critical_heads]

    all_in_layer0 = all(l == 0 for l in layers)
    single_head_criticality = all(d > 0.5 for d in drops)

    return {
        "baseline_mean": float(np.mean(baselines)),
        "baseline_std": float(np.std(baselines)),
        "critical_head_drop_mean": float(np.mean(drops)),
        "critical_head_drop_std": float(np.std(drops)),
        "critical_heads_always_in_layer0": all_in_layer0,
        "layer_distribution": {f"seed{r['seed']}": r["critical_head"]["layer"] for r in seed_results},
        "head_distribution": {f"seed{r['seed']}": r["critical_head"]["head"] for r in seed_results},
        "all_heads_same": len(set(c["key"] for c in critical_heads)) == 1,
        "single_head_criticality_reproduces": single_head_criticality,
        "interpretation": _interpret(seed_results, all_in_layer0, single_head_criticality, drops),
    }


def _interpret(seed_results, all_in_layer0, single_head_criticality, drops) -> str:
    lines = []
    lines.append(
        f"Baseline accuracy: {np.mean([r['baseline'] for r in seed_results]):.3f} "
        f"+/- {np.std([r['baseline'] for r in seed_results]):.3f} across {len(seed_results)} seeds."
    )
    if all_in_layer0:
        lines.append(
            "Critical head is always in Layer 0, confirming the L0 'previous-token / "
            "content-matching' role is universal."
        )
    else:
        layers = [r["critical_head"]["layer"] for r in seed_results]
        lines.append(
            f"Critical head layer varies across seeds: {layers}. "
            "The circuit may not be layer-invariant."
        )
    if single_head_criticality:
        lines.append(
            f"Single-head criticality reproduces in all seeds "
            f"(mean drop={np.mean(drops):.3f} +/- {np.std(drops):.3f}). "
            "This is a robust finding, not a seed-0 artifact."
        )
    else:
        lines.append(
            "Single-head criticality does NOT reproduce consistently across seeds."
        )
    keys = [r["critical_head"]["key"] for r in seed_results]
    if len(set(keys)) == len(keys):
        lines.append(
            f"The specific critical head differs per seed ({', '.join(keys)}), "
            "consistent with the contingency finding from seed-divergence experiments "
            "(only ~24% head-assignment agreement)."
        )
    else:
        counts = {k: keys.count(k) for k in set(keys)}
        lines.append(
            f"Head assignment partially shared across seeds: {counts}."
        )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(seed_results: list[dict], save_path: Path):
    """Bar chart of per-head ablation accuracy drops for all seeds side by side."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_seeds = len(seed_results)
    n_layers = MODEL_CONFIG.n_layers
    n_heads = MODEL_CONFIG.n_heads
    head_keys = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    n_heads_total = len(head_keys)

    fig, axes = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 5), sharey=True)
    if n_seeds == 1:
        axes = [axes]

    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, result, color in zip(axes, seed_results, seed_colors):
        seed = result["seed"]
        baseline = result["baseline"]
        per_head = result["per_head"]
        critical_key = result["critical_head"]["key"]

        drops = [per_head[k]["drop"] for k in head_keys]
        bar_colors = [
            "#d62728" if k == critical_key else color
            for k in head_keys
        ]

        x = np.arange(n_heads_total)
        bars = ax.bar(x, drops, color=bar_colors, edgecolor="black", linewidth=0.5)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(head_keys, rotation=45, ha="right", fontsize=9)
        ax.set_title(
            f"Seed {seed}\nbaseline={baseline:.3f}\ncritical={critical_key}",
            fontsize=10,
        )
        ax.set_xlabel("Head")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy Drop (baseline - ablated)")
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate the critical head bar
        crit_idx = head_keys.index(critical_key)
        ax.annotate(
            f"{drops[crit_idx]:.3f}",
            xy=(crit_idx, drops[crit_idx]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#d62728",
            fontweight="bold",
        )

    fig.suptitle(
        "Per-Head Ablation Accuracy Drops — Content-Matching Induction (Seeds 0–3)\n"
        "Red bars = critical head per seed",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ABLATION ROBUSTNESS ACROSS SEEDS")
    print("=" * 70)

    # Step 1: Train seeds 1, 2, 3 (seed 0 already trained)
    print("\n--- Phase 1: Training ---")
    for seed in SEEDS:
        train_seed(seed)

    # Step 2: Evaluation dataset (fixed seed so measurements are comparable)
    eval_dataset = InductionDataset(
        vocab_size=512,
        seq_len=64,
        n_bigrams=6,
        seed=DATA_SEED,
    )

    # Step 3: Run ablations for all seeds
    print("\n--- Phase 2: Ablation experiments ---")
    seed_results = []
    for seed in SEEDS:
        result = run_ablation_for_seed(seed, eval_dataset)
        seed_results.append(result)

    # Step 4: Summary statistics
    print("\n--- Phase 3: Summary ---")
    summary = compute_summary(seed_results)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Baseline accuracy:  mean={summary['baseline_mean']:.4f}  std={summary['baseline_std']:.4f}")
    print(f"Critical-head drop: mean={summary['critical_head_drop_mean']:.4f}  std={summary['critical_head_drop_std']:.4f}")
    print(f"Critical heads always in Layer 0: {summary['critical_heads_always_in_layer0']}")
    print(f"Single-head criticality reproduces: {summary['single_head_criticality_reproduces']}")
    print(f"All seeds share the same head: {summary['all_heads_same']}")
    print()
    print("Per-seed critical heads:")
    for result in seed_results:
        ch = result["critical_head"]
        print(
            f"  Seed {result['seed']}: {ch['key']}  "
            f"drop={ch['accuracy_drop']:.4f}  "
            f"acc_after={ch['accuracy_after_ablation']:.4f}"
        )
    print()
    print("Interpretation:")
    print(summary["interpretation"])

    # Step 5: Save results
    output = {
        "seeds": seed_results,
        "summary": summary,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Step 6: Plot
    plot_results(seed_results, PLOT_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
