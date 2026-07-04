"""Shared data/eval utilities for the PyTorch replication.

Data comes from src.data.InductionDataset (NumPy-seeded), so the token stream is
bit-identical to the one the paper's MLX models saw: training uses data seed 42
with the same batch size and order; the fixed eval set uses seed 9999 with 1024
sequences in batches of 32 and macro-averaged per-sequence accuracy, exactly
mirroring experiments/extended_controls/mlx_core.fixed_eval_set / evaluate_fixed.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import InductionDataset  # noqa: E402  (imports mlx for array types only)

from .torch_model import GPTConfig  # noqa: E402

SMALL_CONFIG = GPTConfig(n_layers=2, n_heads=4, d_model=128, d_ff=512,
                         vocab_size=512, ctx_len=64)
CKPT_DIR = ROOT / "checkpoints" / "torch_replication"
RESULTS_DIR = ROOT / "results" / "torch_replication"


def np_batch(batch):
    """Convert an (mx, mx, mx) batch from src.data into numpy arrays."""
    return tuple(np.array(t) for t in batch)


def torch_batch(batch_np, device="cpu"):
    x, y, m = batch_np
    return (torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device),
            torch.from_numpy(m.astype(np.float32)).to(device))


def train_stream(data_seed, batch_size=64):
    ds = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=data_seed)
    while True:
        yield torch_batch(np_batch(ds.generate_batch(batch_size)))


def fixed_eval_set(n_sequences=1024, batch_size=32, data_seed=9999):
    ds = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=data_seed)
    return [np_batch(ds.generate_batch(batch_size))
            for _ in range(n_sequences // batch_size)]


@torch.no_grad()
def evaluate_fixed(model, eval_batches, head_value_masks=None):
    """Macro-averaged per-sequence accuracy at masked positions (argmax)."""
    model.eval()
    per_sequence = []
    for x_np, y_np, m_np in eval_batches:
        x, _, _ = torch_batch((x_np, y_np, m_np))
        logits = model(x, head_value_masks=head_value_masks)
        preds = logits.argmax(dim=-1).numpy()
        at_mask = m_np > 0.5
        correct = (preds == y_np) & at_mask
        for row in range(x_np.shape[0]):
            denom = int(at_mask[row].sum())
            per_sequence.append(float(correct[row].sum() / denom) if denom else 0.0)
    return float(np.mean(per_sequence)), per_sequence


def masked_loss(logits, targets, mask):
    B, T, V = logits.shape
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="none")
    flat = mask.reshape(B * T)
    return (losses * flat).sum() / (flat.sum() + 1e-8)


def paired_bootstrap(per_seq_a, per_seq_b, n_boot=10000, seed=0):
    """Bootstrap CI of mean(a - b) over paired per-sequence accuracies."""
    rng = np.random.default_rng(seed)
    d = np.asarray(per_seq_a) - np.asarray(per_seq_b)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    # sign-flip test on the mean difference
    flips = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = (flips * np.abs(d)).mean(axis=1)
    p = float((np.abs(null) >= abs(d.mean())).mean())
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p_signflip": p}
