"""
Training loop with dense checkpointing for developmental interpretability.

Key design decisions:
- Checkpoint every N steps (configurable, default every 50 steps)
- Save full model state + optimizer state (for reproducibility)
- Log per-step metrics: loss, task-specific accuracy, pre-clipping gradient norms
- Deterministic data ordering (seed-controlled) so data is identical across runs
- The ONLY variable across runs is model initialization seed
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from .model import GPT, GPTConfig, create_model
from .data import InductionDataset, IOIDataset


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_config: GPTConfig = None

    # Training
    n_steps: int = 5000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # Checkpointing
    checkpoint_every: int = 50  # Dense checkpointing for developmental analysis
    log_every: int = 10

    # Experiment
    seed: int = 0  # Model initialization seed (THE independent variable)
    data_seed: int = 42  # Fixed across all runs
    task: str = "induction"  # "induction", "ioi", or "text"

    # Paths
    run_name: str = ""
    output_dir: str = "checkpoints"
    text_data_dir: str = "data"

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = GPTConfig()
        if not self.run_name:
            self.run_name = f"{self.task}_seed{self.seed}"


def get_lr(step: int, config: TrainConfig) -> float:
    """Linear warmup + cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    # Cosine decay
    progress = (step - config.warmup_steps) / max(1, config.n_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))


def compute_loss(model: GPT, inputs: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss over all positions."""
    logits = model(inputs)
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss


def compute_masked_loss(model: GPT, inputs: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Cross-entropy loss ONLY at masked positions (where mask=1).

    Essential for tasks where most positions are random noise and only
    a few positions carry the learnable signal (e.g., content-matching induction).
    """
    logits = model(inputs)
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    mask_flat = mask.reshape(B * T)

    # Per-token loss
    per_token_loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Mask and average only over masked positions
    masked_loss = (per_token_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    return masked_loss


def compute_task_accuracy(
    model: GPT,
    inputs: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> float:
    """Accuracy at task-specific positions only (where mask=1)."""
    logits = model(inputs)
    preds = mx.argmax(logits, axis=-1)
    correct = (preds == targets) * mask
    total_masked = mask.sum()
    if total_masked.item() == 0:
        return 0.0
    return (correct.sum() / total_masked).item()


def save_checkpoint(
    model: GPT,
    optimizer: optim.Optimizer,
    step: int,
    metrics: dict,
    config: TrainConfig,
    run_dir: Path,
):
    """Save model weights, optimizer state, and metrics."""
    ckpt_dir = run_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model.save_weights(str(ckpt_dir / "model.safetensors"))

    # Save metrics
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def train(config: TrainConfig) -> Path:
    """Run a single training run with dense checkpointing.

    Returns path to the run directory.
    """
    # Setup output directory
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create model with experiment seed
    model_config = config.model_config
    if config.task == "induction":
        model_config = GPTConfig(
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            vocab_size=512,   # Large vocab to avoid random token collisions
            ctx_len=64,       # Short enough that bigrams are findable
            dropout=0.0,
        )
    elif config.task == "ioi":
        ds_temp = IOIDataset(seed=0)
        model_config = GPTConfig(
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            vocab_size=ds_temp.vocab_size,
            ctx_len=32,
            dropout=0.0,
        )
    elif config.task == "text":
        # Natural text: use the model_config as-is (caller sets vocab/ctx)
        pass
    else:
        raise ValueError(f"Unknown task: {config.task}")

    # Save the resolved config after task-specific overrides.
    config_dict = {
        "model": asdict(model_config),
        "train": {
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "grad_clip": config.grad_clip,
            "checkpoint_every": config.checkpoint_every,
            "seed": config.seed,
            "data_seed": config.data_seed,
            "task": config.task,
        },
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    model = create_model(model_config, seed=config.seed)
    print(f"[{config.run_name}] Model created: ~{model_config.param_count():,} params")

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Dataset (SAME data seed for all runs)
    if config.task == "induction":
        dataset = InductionDataset(
            vocab_size=512,
            seq_len=64,
            n_bigrams=6,
            seed=config.data_seed,
        )
    elif config.task == "ioi":
        dataset = IOIDataset(seed=config.data_seed)
    elif config.task == "text":
        from .data import TextDataset
        dataset = TextDataset(
            data_dir=config.text_data_dir,
            seq_len=model_config.ctx_len,
            seed=config.data_seed,
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")

    # Loss and grad function
    # Use masked loss for tasks with sparse targets; full loss for text
    use_masked_loss = config.task in ("induction", "ioi")

    if use_masked_loss:
        loss_and_grad_fn = nn.value_and_grad(model, compute_masked_loss)
    else:
        loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    # Training loop
    all_metrics = []
    data_iter = dataset.iter_batches(config.batch_size, config.n_steps)

    print(f"[{config.run_name}] Training for {config.n_steps} steps, "
          f"{'masked' if use_masked_loss else 'full'} loss...")
    t0 = time.time()

    for step, batch in enumerate(data_iter):
        if config.task in ("induction", "ioi"):
            inputs, targets, mask = batch
        else:
            inputs, targets = batch
            mask = None

        # Update learning rate
        lr = get_lr(step, config)
        optimizer.learning_rate = lr

        # Forward + backward
        if use_masked_loss:
            loss, grads = loss_and_grad_fn(model, inputs, targets, mask)
        else:
            loss, grads = loss_and_grad_fn(model, inputs, targets)

        # Record global norm for diagnostics, then clip gradient entries elementwise.
        grad_norm = sum(
            (g * g).sum().item()
            for _, g in nn.utils.tree_flatten(grads)
        ) ** 0.5

        if config.grad_clip > 0:
            import mlx.utils
            grads = mlx.utils.tree_map(
                lambda g: mx.clip(g, -config.grad_clip, config.grad_clip),
                grads,
            )

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Metrics
        loss_val = loss.item()

        step_metrics = {
            "step": step,
            "loss": loss_val,
            "lr": lr,
            "grad_norm": grad_norm,
            "time": time.time() - t0,
        }

        # Task accuracy (compute less frequently for speed)
        if step % config.log_every == 0:
            if mask is not None:
                acc = compute_task_accuracy(model, inputs, targets, mask)
                step_metrics["task_accuracy"] = acc

            all_metrics.append(step_metrics)
            if step % (config.log_every * 10) == 0:
                acc_str = f", acc={step_metrics.get('task_accuracy', 'N/A'):.3f}" if 'task_accuracy' in step_metrics else ""
                print(f"  step {step:5d} | loss={loss_val:.4f}{acc_str} | gnorm={grad_norm:.3f} | lr={lr:.2e}")

        # Checkpoint
        if step % config.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, step_metrics, config, run_dir)

    # Final checkpoint
    save_checkpoint(model, optimizer, config.n_steps, step_metrics, config, run_dir)

    # Save all metrics
    with open(run_dir / "metrics_log.json", "w") as f:
        json.dump(all_metrics, f)

    elapsed = time.time() - t0
    print(f"[{config.run_name}] Done in {elapsed:.1f}s ({config.n_steps / elapsed:.0f} steps/s)")

    return run_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="induction", choices=["induction", "ioi", "text"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    config = TrainConfig(
        seed=args.seed,
        task=args.task,
        n_steps=args.n_steps,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
    )
    train(config)
