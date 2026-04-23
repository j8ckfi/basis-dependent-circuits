"""
Train a single content-matching induction model for a given seed.
Called as a subprocess by content_matching_20_seeds.py.

Usage:
    python analysis/train_single_seed.py --seed 4
"""

import sys
import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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

TRAIN_STEPS = 20000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
DATA_SEED = 42

BASE_DIR = Path(__file__).parent.parent
CKPT_BASE = BASE_DIR / "checkpoints" / "induction_content_match"


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_seed(seed: int) -> Path:
    run_name = f"induction_seed{seed}"
    run_dir = CKPT_BASE / run_name
    final_ckpt = run_dir / "step_020000" / "model.safetensors"

    if final_ckpt.exists():
        print(f"[seed {seed}] Already exists at {final_ckpt}, skipping.", flush=True)
        return final_ckpt

    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seed {seed}] Creating model (~{MODEL_CONFIG.param_count():,} params)...", flush=True)
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

    print(f"[seed {seed}] Training for {TRAIN_STEPS} steps...", flush=True)
    t0 = time.time()

    for step, (inputs, targets, mask) in enumerate(
        dataset.iter_batches(BATCH_SIZE, TRAIN_STEPS)
    ):
        lr = get_lr(step)
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

        if step % 1000 == 0:
            acc = compute_task_accuracy(model, inputs, targets, mask)
            elapsed = time.time() - t0
            rate = (step + 1) / max(elapsed, 1e-6)
            eta = (TRAIN_STEPS - step) / max(rate, 1e-6)
            print(
                f"  [seed {seed}] step {step:5d}/{TRAIN_STEPS} | "
                f"loss={loss.item():.4f} | acc={acc:.3f} | "
                f"{rate:.0f} steps/s | ETA {eta/60:.1f}min",
                flush=True,
            )

    # Save final checkpoint
    ckpt_dir = run_dir / "step_020000"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(ckpt_dir / "model.safetensors"))

    elapsed = time.time() - t0
    final_acc = compute_task_accuracy(model, inputs, targets, mask)
    print(
        f"[seed {seed}] DONE in {elapsed:.1f}s | "
        f"final_acc={final_acc:.4f} | "
        f"saved to {ckpt_dir / 'model.safetensors'}",
        flush=True,
    )
    return ckpt_dir / "model.safetensors"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_seed(args.seed)
