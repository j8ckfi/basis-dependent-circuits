"""Train Shakespeare seeds 1, 2, 3 for Experiment D scale test.

Config matches checkpoints/shakespeare/text_seed0/config.json:
  6L/8H/d_model=512/d_ff=2048/vocab=65/ctx=256/10000 steps
  data_seed=42 (fixed), model_seed varies (1, 2, 3)
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.model import GPTConfig, create_model
from src.train import TrainConfig, train
import time

model_cfg = GPTConfig(
    n_layers=6,
    n_heads=8,
    d_model=512,
    d_ff=2048,
    vocab_size=65,
    ctx_len=256,
    dropout=0.0,
)

results = {}

for seed in [1, 2, 3]:
    final_ckpt = BASE_DIR / f'checkpoints/shakespeare/text_seed{seed}/step_010000/model.safetensors'

    if final_ckpt.exists():
        print(f'\n=== Seed {seed}: already exists at {final_ckpt}, skipping ===', flush=True)
        results[seed] = ('skipped', str(final_ckpt))
        continue

    print(f'\n=== Starting seed {seed} at {time.strftime("%H:%M:%S")} ===', flush=True)
    t0 = time.time()

    cfg = TrainConfig(
        model_config=model_cfg,
        n_steps=10000,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=200,
        grad_clip=1.0,
        checkpoint_every=2000,
        log_every=10,
        seed=seed,
        data_seed=42,
        task='text',
        run_name=f'text_seed{seed}',
        output_dir=str(BASE_DIR / 'checkpoints/shakespeare'),
        text_data_dir=str(BASE_DIR / 'data'),
    )

    run_dir = train(cfg)

    # Read final loss from metrics
    import json
    metrics_file = run_dir / 'metrics_log.json'
    final_loss = None
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        if metrics:
            final_loss = metrics[-1]['loss']

    elapsed = time.time() - t0
    print(f'\n=== Seed {seed} done in {elapsed/60:.1f} min, final_loss={final_loss} ===', flush=True)
    results[seed] = (final_loss, str(final_ckpt))

print('\n\n========== FINAL SUMMARY ==========', flush=True)
for seed in [1, 2, 3]:
    loss, path = results[seed]
    print(f'Seed {seed}: final loss {loss}, saved to {path}', flush=True)
