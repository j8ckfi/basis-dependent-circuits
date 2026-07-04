# PyTorch Cross-Framework Replication & Extension

Independent replication of the paper's small-model transplant experiments in
PyTorch (CPU), plus the control arms the original design lacked and a new
data-order-variation axis. Produced during the audit session that generated
`AUDIT.md`; all numbers below are in `results/torch_replication/`.

## Setup

- Faithful port of `src/model.py` (`torch_model.py`): same architecture,
  head layout, init scales, exact-erf GELU, additive causal mask, fp32.
- Bit-identical data: training and eval streams come from `src.data.
  InductionDataset` (NumPy-seeded), same data seed (42), batch size/order,
  and the same fixed eval protocol (seed 9999, 1024 sequences, macro
  per-sequence accuracy) as `mlx_core.fixed_eval_set`/`evaluate_fixed`.
- Same recipe as `analysis/train_single_seed.py`: 20k steps, AdamW(1e-3,
  wd 0.01), 500-step warmup + cosine, element-wise grad value clipping ±1.
- Six models: init seeds 0–3 on data seed 42 (the paper's regime), plus
  init 0/data 43 and init 1/data 43 (new: data-order variation).
- All six converge to eval accuracy 0.961–0.965 (paper: 0.964); all six have
  layer-0 critical heads (layer-level universality replicates).

## Findings

### 1. The headline transplant failure replicates (6/6 init-variation pairs)

| arm (mean over 6 pairs) | accuracy |
|---|---|
| recipient baseline | 0.963 |
| zero critical head | 0.040 |
| **donor transplant (critical slot)** | **0.031** |
| shuffled donor | 0.034 |
| random head | 0.040 |
| donor → non-critical slot | 0.958 |

Transplants from a different-init donor sit at/below the ablation floor,
indistinguishable from junk controls — cross-framework confirmation of the
paper's central negative result.

### 2. NEW: with shared init, transplants partially work (4/4 data-variation pairs)

Same init seed, different data order: transplant accuracy **0.371 / 0.596 /
0.626 / 0.579** vs floors of 0.03–0.08 (all junk controls stay at floor;
paired sign-flip p < 1e-4 vs baseline in all pairs). The critical head also
forms in the *same slot* (L0H2) as the same-init sibling in all four pairs.

Interpretation: the residual-stream basis (and even slot assignment) is
substantially pinned at initialization. Networks that share an init and
diverge only through data order remain partially basis-compatible; networks
with different inits are basis-orthogonal. This sharpens the paper's claim:
the private basis is primarily an **initialization artifact**, and the
paper's fixed-data design (identical data across seeds) was in fact the
*conservative* choice — data-order variation alone produces weaker
incompatibility.

### 3. Table 4 adjudicated: head-local adapters DO rescue — but for the wrong reason

The paper's printed Table 4 (learned adapters fail at 0.095/0.118)
contradicts the repo's committed `alignment_tournament_results.json` (0.96).
Fresh runs side with the committed JSON — and the new control arms show what
the rescue actually is:

| arm (mean, all 10 pairs) | accuracy |
|---|---|
| adapters + donor head | 0.961 |
| **adapters + random never-trained head** | **0.961** |
| adapters + zeroed head | 0.038 |
| adapters + donor, then donor zeroed post-training | 0.042 |

Head-interface adapters (d×d linear, with bias, identity-init, base model
frozen, 1000 steps) fully rescue **regardless of whether the head contains
donor weights or fresh random weights**. They fail only when the head is
zero (a per-token linear map cannot do induction), and zeroing the head
after training collapses the system — the adapter routes *through* the head
but does not use donor-specific structure. Adapter rescue therefore
demonstrates interface learnability, not circuit portability. (This also
retroactively explains why the original bias-free zero-donor control was
vacuous: see AUDIT.md S4.)

### 4. Retraining "rescue" is rerouting — except when bases are shared

Freezing the transplanted head and retraining the host
(`retrain_rescue.py`, arms: donor / random / zero / shuffled frozen head,
reliance probe = ablate the frozen head after retraining):

| pair | arm | k* (80% recovery) | reliance drop |
|---|---|---|---|
| is0_ds42→is1_ds42 (diff init) | frozen donor | 50 | +0.000 |
| | frozen random / zero / shuffled | 50 | ≈ 0.000 |
| is2_ds42→is3_ds42 (diff init) | frozen donor | 125 | +0.000 |
| | frozen random / zero / shuffled | 100–125 | ≈ 0.001 |
| is0_ds42→is0_ds43 (same init) | **frozen donor** | **25** | **+0.921** |
| | frozen random / zero / shuffled | 100 | ≈ 0.000 |

With a different-init donor, all arms recover equally fast and the host
never relies on the frozen head afterward — recovery is redundant-capacity
rerouting, and the paper's "the donor head carries useful structure"
interpretation is unsupported. With a same-init donor, recovery is 4×
faster and the host genuinely adopts the donor head (ablating it afterward
collapses accuracy). Basis compatibility determines not just whether a
transplant works statically, but whether retraining integrates it or routes
around it.

### 5. Corrected statistics (committed artifacts only)

`corrected_stats.py` (results in `corrected_stats.json`):
- Head-slot agreement 23.2% vs the fair conditional null (uniform over 4
  heads given L0 criticality, 25%): p = 0.61 two-sided — the paper's
  p = 0.002 was entirely the layer effect. Slot counts 5/3/7/5 are uniform
  (χ² p = 0.66).
- Layer agreement 20/20: Clopper–Pearson 95% CI [83.2%, 100%].
- Effect-vector cosine similarity 0.329 **survives** a within-seed
  head-relabeling permutation null (0.208 ± 0.019, p < 5e-4) — genuine
  shared ablation structure across seeds, properly supported (the paper's
  test against 0 was vacuous).

`corrected_bigram_baseline.py` (results in `corrected_bigram_baselines.json`):
rebuilds both Shakespeare eval corpora bit-exactly (implemented baselines
0.076 / 0.090 reproduce the committed JSONs) and shows the correctly
conditioned bigram baseline is **0.554 / 0.551** — every "induction
advantage" level in the Shakespeare/NL sections must be recomputed against
it (advantage drops survive as raw accuracy differences).

## What this means for the paper

1. Central negative result: **stands** (replicated cross-framework, 6/6).
2. "Basis-dependent computations": **stands, sharpened** — the basis is set
   primarily at initialization (new data-variation evidence).
3. "Compatibility can be learned by the host" (retraining rescue):
   **reinterpret** — it is rerouting around the transplant unless the bases
   are already shared.
4. "Compatibility can be supplied by adapters": **reinterpret** — head-local
   adapters rescue any non-degenerate head; rescue ≠ donor-function
   transfer. Table 4 and the abstract's tournament claim must be corrected
   to match the committed artifacts.
5. Shakespeare/NL advantage numbers: recompute vs corrected baseline.
6. Head-level agreement significance claim: drop; keep the (honest)
   layer-level and effect-vector-similarity claims.

## Reproducing

```bash
pip install torch numpy scipy "mlx[cpu]"   # mlx only for src.data types
python -m experiments.torch_replication.train_seeds --runs 0:42 1:42 2:42 3:42 0:43 1:43
python -m experiments.torch_replication.transplant_suite --pairs is0_ds42:is1_ds42 ...
python -m experiments.torch_replication.retrain_rescue --pairs is0_ds42:is1_ds42 ...
python -m experiments.torch_replication.corrected_stats
python -m experiments.torch_replication.corrected_bigram_baseline
```

Total compute: ~2 h training + ~2 h experiments on a 4-core CPU.
