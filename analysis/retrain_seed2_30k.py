"""
Retrain seed 2 for 30000 steps to let it grok.

One audit noted seed 2 did NOT converge at 20k steps (baseline_accuracy=0.193,
critical_head_drop=0.064). This script extends training to 30k steps.

If 30k with data_seed=42 still fails, falls back to model_seed=2, data_seed=43.

Saves to:
  checkpoints/induction_content_match/induction_seed2/step_030000/model.safetensors

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/retrain_seed2_30k.py
"""

import sys
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig, create_model
from src.data import InductionDataset
from src.train import compute_masked_loss, compute_task_accuracy

MODEL_CONFIG = GPTConfig(
    n_layers=2,
    n_heads=4,
    d_model=128,
    d_ff=512,
    vocab_size=512,
    ctx_len=64,
    dropout=0.0,
)

TRAIN_STEPS = 30000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0

BASE_DIR = Path(__file__).parent.parent
CKPT_BASE = BASE_DIR / "checkpoints" / "induction_content_match"

CONVERGENCE_THRESHOLD = 0.5  # baseline_accuracy must exceed this


def get_lr(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_attempt(model_seed: int, data_seed: int, n_steps: int, ckpt_dir: Path) -> tuple[float, Path]:
    """Train one attempt. Returns (final_accuracy, checkpoint_path)."""

    if ckpt_dir.exists():
        weights_path = ckpt_dir / "model.safetensors"
        if weights_path.exists():
            print(f"  Checkpoint already exists at {weights_path}, loading to check accuracy...")
            model = GPT(MODEL_CONFIG)
            model.load_weights(str(weights_path))
            mx.eval(model.parameters())
            # Quick accuracy check on a fresh batch
            check_ds = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=9999)
            inputs, targets, mask = check_ds.generate_batch(256)
            acc = compute_task_accuracy(model, inputs, targets, mask)
            print(f"  Loaded checkpoint accuracy: {acc:.4f}")
            return float(np.array(acc)), weights_path

    print(f"\n  model_seed={model_seed}, data_seed={data_seed}, steps={n_steps}")
    model = create_model(MODEL_CONFIG, seed=model_seed)
    print(f"  Parameters: {MODEL_CONFIG.param_count():,}")

    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=data_seed)
    loss_and_grad_fn = nn.value_and_grad(model, compute_masked_loss)

    t0 = time.time()
    inputs_last = targets_last = mask_last = None

    for step, (inputs, targets, mask) in enumerate(dataset.iter_batches(BATCH_SIZE, n_steps)):
        lr = get_lr(step, n_steps)
        optimizer.learning_rate = lr

        loss, grads = loss_and_grad_fn(model, inputs, targets, mask)
        grads = mlx.utils.tree_map(lambda g: mx.clip(g, -GRAD_CLIP, GRAD_CLIP), grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        inputs_last, targets_last, mask_last = inputs, targets, mask

        if step % 2000 == 0:
            acc = compute_task_accuracy(model, inputs, targets, mask)
            elapsed = time.time() - t0
            rate = (step + 1) / max(elapsed, 1e-6)
            eta = (n_steps - step) / max(rate, 1e-6)
            print(
                f"  step {step:5d}/{n_steps} | loss={loss.item():.4f} | acc={acc:.3f} | "
                f"{rate:.0f} steps/s | ETA {eta/60:.1f}min",
                flush=True,
            )

    # Final accuracy
    final_acc = compute_task_accuracy(model, inputs_last, targets_last, mask_last)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s | final_acc={final_acc:.4f}", flush=True)

    # Save checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / "model.safetensors"
    model.save_weights(str(weights_path))
    print(f"  Saved: {weights_path}")

    return float(np.array(final_acc)), weights_path


def main():
    print("=" * 70)
    print("SEED 2 RETRAINING: 30000 STEPS")
    print("=" * 70)
    print()
    print("Background: seed 2 did NOT converge at 20k steps.")
    print("  baseline_accuracy = 0.193  (threshold: > 0.5)")
    print("  critical_head_drop = 0.064  (very low — model is near-random)")
    print()
    print("Strategy: extend to 30000 steps (model_seed=2, data_seed=42).")
    print("Fallback: model_seed=2, data_seed=43 if still stuck.")
    print()

    run_dir = CKPT_BASE / "induction_seed2"

    # -----------------------------------------------------------------------
    # Attempt 1: 30k steps, data_seed=42
    # -----------------------------------------------------------------------
    print("=== Attempt 1: 30k steps, data_seed=42 ===")
    ckpt_dir_30k = run_dir / "step_030000"
    acc_30k, path_30k = train_attempt(
        model_seed=2, data_seed=42, n_steps=TRAIN_STEPS, ckpt_dir=ckpt_dir_30k
    )

    if acc_30k >= CONVERGENCE_THRESHOLD:
        print(f"\n[SUCCESS] Attempt 1 converged: accuracy={acc_30k:.4f} >= {CONVERGENCE_THRESHOLD}")
        print(f"  Checkpoint: {path_30k}")
        print(f"  Use this checkpoint in the analysis.")
        _print_summary(converged=True, attempt=1, model_seed=2, data_seed=42,
                       n_steps=TRAIN_STEPS, acc=acc_30k, path=path_30k)
        return

    print(f"\n[WARN] Attempt 1 still stuck: accuracy={acc_30k:.4f} < {CONVERGENCE_THRESHOLD}")
    print("  Trying fallback: model_seed=2, data_seed=43...")

    # -----------------------------------------------------------------------
    # Attempt 2 (fallback): 30k steps, data_seed=43
    # -----------------------------------------------------------------------
    print("\n=== Attempt 2 (fallback): 30k steps, data_seed=43 ===")
    ckpt_dir_ds43 = run_dir / "step_030000_ds43"
    acc_ds43, path_ds43 = train_attempt(
        model_seed=2, data_seed=43, n_steps=TRAIN_STEPS, ckpt_dir=ckpt_dir_ds43
    )

    if acc_ds43 >= CONVERGENCE_THRESHOLD:
        print(f"\n[SUCCESS] Attempt 2 (data_seed=43) converged: accuracy={acc_ds43:.4f}")
        print(f"  Checkpoint: {path_ds43}")
        _print_summary(converged=True, attempt=2, model_seed=2, data_seed=43,
                       n_steps=TRAIN_STEPS, acc=acc_ds43, path=path_ds43)
    else:
        print(f"\n[FAIL] Both attempts failed to converge.")
        print(f"  Attempt 1 (data_seed=42): {acc_30k:.4f}")
        print(f"  Attempt 2 (data_seed=43): {acc_ds43:.4f}")
        print()
        print("  => Seed 2 will be treated as non-converged in the analysis.")
        print("  => The v2 analysis will use N=19 converged seeds.")
        _print_summary(converged=False, attempt=None, model_seed=2, data_seed=None,
                       n_steps=TRAIN_STEPS, acc=max(acc_30k, acc_ds43), path=None)


def _print_summary(converged: bool, attempt, model_seed: int, data_seed, n_steps: int,
                   acc: float, path):
    print()
    print("=" * 70)
    print("RETRAINING SUMMARY")
    print("=" * 70)
    print(f"  Original seed 2 (20k steps):  baseline_accuracy=0.193  [FAILED]")
    if converged:
        print(f"  Retrained (attempt {attempt}):      baseline_accuracy={acc:.4f}  [CONVERGED]")
        print(f"  model_seed={model_seed}, data_seed={data_seed}, steps={n_steps}")
        print(f"  Checkpoint: {path}")
        print(f"  => Use step_030000 checkpoint in content_matching_20_seeds_v2.py")
    else:
        print(f"  Retrained (best):             baseline_accuracy={acc:.4f}  [STILL FAILED]")
        print(f"  => Seed 2 excluded; analysis uses N=19 converged seeds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
