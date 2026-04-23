# Reproducibility Notes

## What is committed

- Source code for model training and circuit analyses.
- LaTeX manuscript source, bibliography, figures, and compiled PDF.
- Result JSON/CSV files used by the manuscript and provenance manifest.
- TinyShakespeare text data.

## What is not committed

- `checkpoints/`: local model weights and dense training checkpoints, about 13 GB.
- `.venv/`: local Python environment.
- LaTeX and Python build artifacts.

Place checkpoint artifacts back under the paths listed in `paper/PROVENANCE.md` before rerunning analyses that load weights.

## Basic verification

```bash
python -m compileall -q src analysis experiments code train_shakespeare_seeds.py
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Lightweight smoke test

```bash
python - <<'PY'
import mlx.core as mx
from src.model import GPTConfig, create_model
from src.data import InductionDataset
from src.train import compute_masked_loss, compute_task_accuracy

cfg = GPTConfig(n_layers=2, n_heads=4, d_model=128, d_ff=512, vocab_size=512, ctx_len=64)
model = create_model(cfg, seed=0)
dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)
inputs, targets, mask = dataset.generate_batch(4)
loss = compute_masked_loss(model, inputs, targets, mask)
mx.eval(loss)
print(float(loss.item()))
print(compute_task_accuracy(model, inputs, targets, mask))
PY
```

## Rerun policy

The manuscript distinguishes committed result artifacts from local checkpoints. If a code change modifies training dynamics, rerun the affected experiment family and update both result files and `paper/PROVENANCE.md`. Documentation-only corrections, numerical-stability fixes for future analysis runs, and release hygiene changes do not invalidate the existing checkpoints.
