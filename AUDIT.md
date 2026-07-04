# Independent Code & Claims Audit

Audit of the repository at commit `198d107` ("Prepare reproducible paper release").
The code and paper were produced autonomously by an AI model (Opus 4.6); this audit was
performed independently (Claude, session 2026-07-04) by reading every in-scope source file,
numerically re-deriving the weight-layout and alignment math, reproducing the deterministic
eval corpora bit-for-bit, and cross-checking all headline numbers in `paper/main.tex` against
the committed result artifacts.

## Verdict in one paragraph

The **core negative result is sound**: the head-transplant machinery is mechanically correct
(per-head Q/K/V row slices and out-proj column slices match the model's actual layout — verified
numerically; a full 15-tensor transplant reproduces the donor's baseline to all digits), the
controls in the central experiment are fair, frozen weights are genuinely frozen, and the
0.964 → 0.095 collapse to the ablation floor is real as committed. However, the **rescue-side
narrative has a results-integrity break**: the paper's Table 4 and abstract claim head-local
learned adapters fail (0.095/0.118), while the committed `alignment_tournament_results.json`
*and* the raw per-sequence arrays show they fully rescue (0.960). Two paragraphs of §4.5 cite
geometry numbers matching nothing in the committed geometry file. Separately, an off-by-one in
the bigram baseline invalidates every "induction advantage" *level* in the Shakespeare /
natural-language sections (correct trivial baseline ≈ 0.55, not 0.28), though advantage *drops*
survive as raw accuracy drops. Several statistical constructions are flawed but mostly in ways
that soften rather than reverse conclusions. Roughly 90% of quantitative claims are faithfully
backed by committed artifacts.

## Severity-ranked findings

### S1 — Paper contradicts its own committed results (alignment tournament, Table 4)

- `paper/main.tex` Table `tab:alignment_tournament` reports learned linear adapter = 0.0951
  (p = 1.00) and learned nonlinear adapter = 0.1180, with a narrative that the adapter
  "learned to ignore the donor head."
- Committed `analysis/alignment_tournament_results.json` reports `learned_linear = 0.9604`,
  `learned_nonlinear = 0.9601` (paired diff vs unaligned +0.865, p < 1e-4). The raw arrays
  `analysis/alignment_tournament_per_seq/learned_linear.npy` (mean 0.9604, n = 1024) and
  `learned_nonlinear.npy` (0.9601) side with the JSON.
- Rows 1–5 and the donor-zero controls of the same table match the same JSON to 4 decimals,
  so the file is unambiguously the intended source. `paper/PROVENANCE.md` (Exp C) cites this
  file for the 0.095/0.118 values, which it does not contain.
- Impact: the abstract's "seven-method head-interface alignment tournament does not rescue the
  transplant" and the central "head-local maps fail / only broad multi-interface adapters
  rescue" contrast are unsupported by the committed artifacts. Note `learned_nonlinear`
  (0.9601236979166667) is byte-identical to the stitching `all4_mlp` result, suggesting a
  post-hoc regeneration of the JSON that was never reconciled with the manuscript.
- Also: the paper's mechanistic explanation for the (stale) numbers is wrong on its face — a
  linear map can trivially zero the donor subspace (A_out = 0), contradicting "lacking the
  capacity to zero-out the donor subspace."

### S2 — Bigram-baseline off-by-one invalidates all "induction advantage" levels (Shakespeare/NL)

- `analysis/natural_language_induction_v2.py:213,293-297` and (copied verbatim)
  `experiments/shakespeare_portability.py:129,287-290` condition the bigram baseline on
  `window[t-1]` when predicting `window[t+1]`; the correct conditioning token is `window[t]`.
  `experiments/extended_controls/shakespeare_redundancy_control.py` imports this wholesale.
- Reproduced bit-for-bit against the committed eval corpora: implemented bigram baseline
  0.076/0.090 (matches JSONs) vs **correct bigram baseline 0.554/0.551**.
- Consequences: every advantage *level* is overstated by ~0.27; "advantage 0.708" is actually
  ~0.446; "preserves induction advantage above 0.50" (`min_advantage=0.50`) is infeasible for
  every recipient under the correct null; post-ablation accuracy 0.38 is *below* the true
  bigram baseline. What survives: advantage *drops* equal raw accuracy drops (baseline cancels),
  so the 19M transplant drop 0.445 ± 0.040 and the direction of the redundancy-control result
  stand as accuracy statements; models at ~1.0 accuracy still beat the corrected 0.55 baseline,
  so "real induction formed" also stands.

### S3 — §4.5 geometry paragraph cites numbers absent from the committed artifact

- Paper: "mean pairwise CKA 0.013 pre-L0, 0.006 post-L0", "r = −0.03, p = 0.67".
- `analysis/residual_stream_geometry_results.json`: mean CKA pre-L0 = **0.307**, post-L0 =
  0.039, post-L1 = 0.217; correlations r = +0.038/−0.020/−0.025. No interface matches the
  printed values. With pre-L0 CKA ≈ 0.31, "seeds occupy almost entirely disjoint linear
  subspaces" is substantially overstated. PROVENANCE quotes the tex numbers, not the file.

### S4 — Missing decisive controls in both rescue experiments

- **Vacuous zero-donor adapter control** (`analysis/alignment_tournament.py:679-690`): with the
  donor head zeroed *before* bias-free adapters, the adapter is disconnected (0·A = 0, no-bias
  MLP maps 0→0) — zero gradients, training is a no-op. The control necessarily equals the
  ablation floor (confirmed: linear and nonlinear controls byte-identical, 0.1178). It cannot
  test whether adapters bypass the donor head.
- **Adapter expressivity confound**: the full-rank d×d input adapter composed with donor Q/K/V
  can synthesize an essentially arbitrary rank-d_head head; rescue at 0.96 does not show the
  *donor's* function was used. Missing arm: same adapters around a random-weight or
  non-critical donor head.
- **Partial-retraining rescue** (`experiments/extended_controls/run_mlx_experiment.py:100-189`):
  no arm freezes a zero/random head in the same slot, and the frozen donor head is never
  ablated post-retraining to show it became load-bearing. Given the redundant capacity of the
  other three L0 heads, "host reroutes around a frozen junk head in ~100 steps" is not excluded,
  so "the donor head carries useful structure" is not established.

### S5 — Statistical methodology

- The headline "23.2% head agreement vs uniform null, p = 0.002" is significant *only* because
  of the layer-0 effect that is reported separately as "20/20". Conditioned on L0 criticality,
  the fair null is uniform over 4 heads = 25%, and observed 23.2% is *below* it (slot counts
  5/3/7/5, χ² p ≈ 0.66). The narrative ("agree on layer, not head") is consistent with the
  data; the p = 0.002 decoration double-counts the layer effect.
- Bootstrap CIs on pairwise statistics resample the 190 seed-pairs as i.i.d., ignoring that
  each seed appears in 19 pairs → CIs too narrow. "100% CI [100%, 100%]" should be
  Clopper–Pearson [83%, 100%] for 20/20.
- `permutation_test_agreement` is a Monte-Carlo parametric-null test, not a permutation test,
  and is one-sided while described as two-sided.
- main.tex "copy-before-prev-token holds in 65% of seeds (p < 0.001 vs 4.2% null)": the 4.2%
  (1/24) null is for a full 4-element ordering; the pairwise null is 50%, under which 13/20
  gives p ≈ 0.13 — not significant.
- Cosine-similarity tests against a null of 0 are vacuous for non-negative effect vectors.
- n = 4 inferences (ablation t-test, depth-scaling slope) are fragile; the paper partially hedges.

### S6 — Extended-controls design weaknesses

- **Pythia**: 14 of 15 Procrustes transplants are *cross-layer* (donor and recipient induction
  heads live at different depths, L3–L8) — undisclosed in the paper; post-rotation R² ≈ 0.08
  or negative, so "Procrustes fails" partly reflects incommensurate interfaces. Effect sizes
  are small relative to eval noise (baselines 0.066–0.143, N = 1024, SE ≈ 0.01) and the 15
  pairs share 5 recipients (not independent).
- **Redundancy control**: the ablation subset is selected by searching all 2⁷ subsets on the
  first 256 of the same 1024 eval samples, with a selection key that maximizes load-bearing
  drop (`shakespeare_redundancy_control.py:166-178,250-251`) — eval-coupled selection bias on
  the controlled-baseline numbers (the transplant drop itself is less affected).
- The v1/older analysis scripts (`circuit_transplant.py`, `basis_aligned_transplant.py`,
  `whole_layer_transplant.py`) evaluate on the literal training stream (data seed 42) and v1
  fits Procrustes on its own eval set; superseded by the clean V3 unified eval (eval seed 9999,
  alignment 1234, materialized fixed set) which the paper cites — but any figure still derived
  from v1 scripts should be regenerated.
- v1 Procrustes computed the inverse rotation (`basis_aligned_transplant.py:109-125`,
  `SVD(X1ᵀX0)` instead of `SVD(X0ᵀX1)`), and aligned an L0 head on post-L0 activations (wrong
  interface). Self-documented and fixed in v2; but note the headline "aligned transplant fails"
  conclusion was first obtained under inverted alignment.
- "Git Re-Basin" tournament row is mathematically a duplicate of the Procrustes row for a
  single-head transplant (head permutation cannot change any head's weights; confirmed
  byte-identical accuracies). The "seven-method" tournament is effectively six.
- The v2 write-side rotation `R_out` is fit on post-block (post-MLP) residuals while the head
  writes pre-MLP — an unacknowledged approximation.
- IOI control task has a positional shortcut (S/IO/verb at fixed positions 3/6/10/11/12,
  `src/data.py:184-194`) — fine as a cross-task donor control, invalid for any mechanistic
  interpretation.
- `shakespeare_portability_results.json` derived booleans are unreliable
  (`slot_vs_func_asymmetry_consistent` predicate inverted relative to its name; hardcoded
  "interpretation" strings mismatch the paper's narrative). Trust the raw numbers only.

### S7 — Reproducibility gaps

- `src/train.py:116-133` never saves optimizer state despite the module docstring claiming it;
  checkpoint step labels are off by one (`step_000000` contains one update; the "init"
  checkpoint is not the initialization).
- `grad_clip=1.0` is element-wise **value** clipping (`mx.clip(g, -1, 1)`), not norm clipping,
  while the logged `grad_norm` is a global L2 norm — applied uniformly across seeds, so
  comparisons survive, but reproducers using true norm clipping will get different baselines.
- `statistical_rigor_results.json` claims 1–2 read `checkpoints/seed_divergence/*.json` —
  result files that are gitignored along with the weights, so they are unverifiable; PROVENANCE
  lists the file as CURRENT anyway. The NL loss deltas (0.080 → 7.95; +5.08) appear in no
  committed artifact.
- Stale `paper/methods.tex` and `paper/appendices.tex` (not `\input` by `main.tex`) carry
  deprecated statistics (13.2% agreement, similarity 0.985) that contradict main.tex.
- `checkpoints/` (~13 GB) absent as documented; additionally there is no committed trainer for
  Shakespeare seed 0 (`train_shakespeare_seeds.py` trains seeds 1–3 only).
- 20/20 seeds share `DATA_SEED = 42` — identical data *and batch order*. The paper's isolation
  claim is accurate, but the supported reading is "basis-dependence arises under init-only
  variation"; data-order variation is untested.

## What was verified and checks out

- **Head-slicing convention** (the one bug that would have trivially produced the headline
  failure): MLX `nn.Linear` stores (out, in) and computes `x @ Wᵀ`; the reshape in
  `src/model.py:70-73` puts head h of Q at rows `[h·d_h, (h+1)·d_h)`, K/V offset by
  d_model/2·d_model, and the head-major concat makes out-proj *columns* the correct slice.
  Verified by numpy simulation of the exact reshape/transpose; every transplant path
  (`circuit_transplant.py`, `transplant_unified_eval.py`, `alignment_tournament.py`,
  `mlx_core.py`, both Pythia scripts) uses this convention consistently, including the
  GPTNeoX interleaved fused-QKV layout with rotary handled head-internally.
- Full-model transplant reproduces the donor baseline to all committed digits (0.8605143229…),
  end-to-end proof of splice/save/load mechanics.
- Causal mask, loss masking, target indexing (predict-next at second bigram occurrence, no
  off-by-one), argmax scoring, fp32 throughout, no dropout anywhere.
- The 0.964 baseline is quantitatively plausible given token-collision statistics (~10–14%
  ambiguous targets), i.e., not suspicious.
- Freezing is genuine: gradient masks cover exactly the donor-head rows/columns, `weight_decay=0`
  in the frozen-arm optimizers, Adam moments stay zero for masked entries.
- v2/unified Procrustes math correct (SVD of `X_sourceᵀX_target`), orthogonality error ~4e-6;
  alignment (seed 1234), adapter-training (7777), and eval (9999) data disjoint.
- 17 of ~20 spot-checked paper tables/values match committed JSONs exactly (transplant matrix,
  20-seed statistics, rescue K*, stitching adapters, shared-basis CKA, redundancy-control
  summary, Pythia tables, depth scaling, metric validation, minimum-unit sweep, per-example
  agreement). The exceptions are S1 and S3 above plus one trivial rounding slip (0.965→"0.966").
- Chance level is 1/512 ≈ 0.002; the paper correctly describes 0.095 as the ablation floor,
  not chance.

## Bottom line for revision

1. Rerun the learned-adapter tournament arms and reconcile Table 4 + abstract with the
   artifacts (committed data currently say head-local adapters rescue — which *supports*
   basis-dependence as a learnable-interface phenomenon, but changes the story).
2. Recompute all Shakespeare/NL advantage numbers against the corrected bigram baseline.
3. Rewrite the §4.5 geometry paragraph from the committed JSON.
4. Add the missing controls: biased adapters around zero/random donor heads; frozen-random-head
   retraining arm; post-retraining ablation of the frozen donor.
5. Fix the statistics per S5 (conditional null for head agreement, hierarchical/jackknife CIs,
   Clopper–Pearson, correct pairwise-ordering null).
6. Delete or update stale `methods.tex`/`appendices.tex`; fix PROVENANCE entries written from
   the manuscript rather than the artifacts.
