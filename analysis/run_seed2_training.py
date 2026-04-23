"""Train content-matching induction model with seed=2 to 20k steps."""
import sys
sys.path.insert(0, '.')
from src.model import GPTConfig, create_model
from src.data import InductionDataset
from src.train import compute_loss
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
import time
from pathlib import Path

config = GPTConfig(n_layers=2, n_heads=4, d_model=128, d_ff=512, vocab_size=512, ctx_len=64)
model = create_model(config, seed=2)
print(f'Training seed=2: {config.param_count():,} params', flush=True)

optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)

loss_and_grad = nn.value_and_grad(model, compute_loss)

out_dir = Path('checkpoints/induction_content_match/induction_seed2')
out_dir.mkdir(parents=True, exist_ok=True)

N_STEPS = 20000
t0 = time.time()
for step in range(N_STEPS):
    batch = dataset.generate_batch(64)
    inputs, targets, mask = batch

    if step < 500:
        lr = 1e-3 * step / 500
    else:
        progress = (step - 500) / (N_STEPS - 500)
        lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * progress))
    optimizer.learning_rate = lr

    loss, grads = loss_and_grad(model, inputs, targets)
    grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    if step % 1000 == 0:
        print(f'  step {step}/20000 loss={loss.item():.4f} t={time.time()-t0:.1f}s', flush=True)

ckpt = out_dir / 'step_020000'
ckpt.mkdir(exist_ok=True)
model.save_weights(str(ckpt / 'model.safetensors'))
print(f'Done in {time.time()-t0:.1f}s', flush=True)
