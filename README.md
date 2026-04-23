# Circuits Are Basis-Dependent Computations

This repository contains the code, analysis outputs, figures, and LaTeX source for:

**Circuits Are Basis-Dependent Computations: Transplantation Failure and Rescue Across Task-Matched Networks**

The paper tests whether mechanistic-interpretability circuits are portable as raw weights across independently trained task-matched transformers. The central result is that direct head transplantation fails in small induction models, while compatibility can be learned through host retraining or supplied through broad residual-stream adapters. Larger-scale controls on Shakespeare and Pythia sharpen the boundary conditions.

## Repository layout

- `src/`: MLX transformer, synthetic datasets, and shared training utilities.
- `analysis/`: main analysis scripts and committed result JSON files for small-model experiments.
- `experiments/`: scale, depth, entropy, and extended-control experiment scripts.
- `results/extended_controls/`: extended-control result tables and JSON summaries.
- `paper/`: LaTeX source, figures, provenance manifest, bibliography, and compiled PDF.
- `data/shakespeare.txt`: TinyShakespeare text used for character-level runs.

Large checkpoints are intentionally excluded from git. Scripts expect them under `checkpoints/` when rerunning weight-level analyses.

## Environment

The project uses MLX on Apple Silicon.

```bash
conda env create -f environment.yaml
conda activate mechinterp2-extended-control
```

For a quick Python syntax check:

```bash
python -m compileall -q src analysis experiments code train_shakespeare_seeds.py
```

## Paper build

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The compiled manuscript is tracked at `paper/main.pdf`.

## Reproducibility

`paper/PROVENANCE.md` maps every main quantitative claim to a source script, result file, and JSON key. `REPRODUCIBILITY.md` gives the practical rerun path and artifact expectations.

The committed result files are sufficient to audit the reported numbers. Full reruns that inspect or transplant weights require local checkpoints, which are excluded because they are approximately 13 GB.
