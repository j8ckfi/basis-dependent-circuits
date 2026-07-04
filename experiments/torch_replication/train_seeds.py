"""Train fresh induction models in PyTorch, matching the paper's recipe exactly.

Hyperparameters mirror analysis/train_single_seed.py: 20k steps, batch 64,
AdamW(lr=1e-3, wd=0.01), 500-step warmup + cosine decay, element-wise gradient
value clipping at ±1.0 (the paper's actual clipping, despite being labeled
"grad clip"), data seed 42 fixed across init seeds.

Extends the paper's design with data-seed variation runs (same init seed,
different data seed) to test whether basis-dependence also arises under
data-order variation, not just init variation.

Usage: python -m experiments.torch_replication.train_seeds --runs 0:42 1:42 2:42 3:42 0:43 1:43
"""

import argparse
import json
import time

import numpy as np
import torch

from .common import (CKPT_DIR, SMALL_CONFIG, evaluate_fixed, fixed_eval_set,
                     masked_loss, train_stream)
from .torch_model import create_model

TRAIN_STEPS = 20000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
EVAL_EVERY = 2000


def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_run(init_seed, data_seed, eval_batches):
    name = f"is{init_seed}_ds{data_seed}"
    run_dir = CKPT_DIR / name
    final = run_dir / "final.pt"
    if final.exists():
        print(f"[{name}] exists, skipping", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(SMALL_CONFIG, seed=init_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    stream = train_stream(data_seed)
    curve = []
    t0 = time.time()
    for step in range(TRAIN_STEPS):
        for group in opt.param_groups:
            group["lr"] = get_lr(step)
        x, y, m = next(stream)
        model.train()
        loss = masked_loss(model(x), y, m)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.clamp_(-GRAD_CLIP, GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == TRAIN_STEPS - 1:
            acc, _ = evaluate_fixed(model, eval_batches)
            curve.append({"step": step, "loss": float(loss.item()), "eval_acc": acc})
            rate = (step + 1) / (time.time() - t0)
            eta = (TRAIN_STEPS - step - 1) / max(rate, 1e-9) / 60
            print(f"[{name}] step {step:5d} loss={loss.item():.4f} "
                  f"eval_acc={acc:.4f} ({rate:.1f} it/s, ETA {eta:.0f}m)", flush=True)

    torch.save(model.state_dict(), final)
    acc, _ = evaluate_fixed(model, eval_batches)
    meta = {"init_seed": init_seed, "data_seed": data_seed,
            "steps": TRAIN_STEPS, "final_eval_acc": acc, "curve": curve,
            "wall_minutes": (time.time() - t0) / 60}
    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] DONE final_eval_acc={acc:.4f} "
          f"({meta['wall_minutes']:.1f} min)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["0:42", "1:42", "2:42", "3:42", "0:43", "1:43"],
                        help="init_seed:data_seed pairs")
    args = parser.parse_args()
    torch.set_num_threads(4)
    eval_batches = fixed_eval_set()
    for spec in args.runs:
        init_seed, data_seed = (int(v) for v in spec.split(":"))
        train_run(init_seed, data_seed, eval_batches)


if __name__ == "__main__":
    main()
