# PROVENANCE.md — Claim-to-Source Manifest (V6, scale-control update)

**Purpose:** Every substantive quantitative claim in `paper/main.tex` is mapped
to its source script, result file, and exact JSON key/value. A reader should
be able to verify any number in under 30 seconds.

**Last updated:** 2026-04-23. Supersedes V5.

## V6 Additions (scale controls requested after informal review)

The following real-backend runs were added to address the scale-story critique.

| Experiment | Main-text claim | Result file | Key numbers |
|-----------|-----------------|-------------|-------------|
| **L Shakespeare redundancy control** | At 19M, slot-matched survival is explained by non-load-bearing slots; when other L0 heads are ablated while preserving the recipient's critical slot, transplant into that slot still fails | `results/extended_controls/shakespeare_redundancy_control_results.json` | selected controls ablate mean 4.0 L0 heads; controlled baseline advantage=0.5202; critical-slot zero drop=0.4489; unaligned transplant drop=0.4453; Procrustes drop=0.2871 |
| **M Pythia-160M Procrustes + adapters** | At 160M, unaligned and Procrustes selected-head transplants remain below baseline; broad adapters solve the synthetic probe but the zero-head adapter control solves it equally well, so this is not donor-specific rescue | `results/extended_controls/pythia_adapter_rescue_results.json` | unaligned-minus-baseline=-0.0589; Procrustes-minus-baseline=-0.0454; adapter accuracy=0.8188; zero-head adapter accuracy=0.8337 |

### V6 paper integration

- Abstract now foregrounds the 0.5--2.5% partial-retraining compatibility-map
  cost.
- Introduction replaces the French/English analogy with a compiled
  instruction/register-state analogy.
- Section 4 clarifies that the greedy 9/15 tensor peel-back is a local greedy
  solution, not an exhaustive lower bound.
- Section 4 resolves the fixed-recipient training-time probe vs. shared-fork
  tension by distinguishing donor-checkpoint portability from branch
  co-divergence.
- Section 4 reframes shared-basis training around the real partial rescue at
  lambda=0.1 and the non-monotonic lambda=1.0 result.
- Section 6 adds the 19M redundancy-controlled transplant and Pythia-160M
  Procrustes/adapter-control results.
- Limitations now state that the Pythia adapter result is not donor-specific
  because zero-head adapter controls solve the synthetic probe.

## V5 Additions (extended-control compute suite)

The following real-backend runs were added after the extended-control compute
plan. They change the paper framing from absolute non-portability to
basis-dependent portability with learned compatibility maps.

| Experiment | Main-text claim | Result file | Key numbers |
|-----------|-----------------|-------------|-------------|
| **G Partial-retraining rescue** | Frozen donor head can be used by a retrained host; all five pairs recover >=80% of recipient baseline by 100--500 steps | `results/extended_controls/partial_retrain_rescue_results.json` | K*: 0->1=100, 0->3=100, 1->4=100, 3->5=500, 4->6=100 |
| **H Multi-interface adapters** | Head-local adapters fail, but four residual-stream adapters recover near baseline with recipient frozen | `results/extended_controls/stitching_adapters_results.json` | baseline=0.9640, bare=0.0951, post_l0_linear=0.4791, all4_linear=0.9561, all4_mlp=0.9601 |
| **I Shared-trajectory forks** | Fork compatibility is non-monotonic; fork 500 is highest, fork >=2000 mostly collapses | `results/extended_controls/shared_trajectory_results.json` | fork 500 unaligned=0.6608; fork 2000 unaligned=0.0947; fork 19500 unaligned=0.0700 |
| **J Shared-basis joint training** | Residual CKA can be forced above 0.99 while raw head transplant still fails | `results/extended_controls/shared_basis_training_results.json` | lambda=0.1 CKA post-L0/L1=0.9917/0.9982, unaligned=0.0259, Procrustes=0.5049 |
| **K Pythia-160M unaligned probe** | All 15 160M unaligned cross-seed transplants drop relative to recipient baseline | `results/extended_controls/pythia_scale_portability_results.json` | baseline range=0.0664--0.1426, selected-head score range=0.4978--0.6920, mean transplant-minus-baseline=-0.0589 |

### V5 paper integration

- Title and abstract now use "basis-dependent computations" rather than
  "system-level emergent properties" as the primary claim.
- Section 4 adds host-retraining and multi-interface adapter rescue results.
- Section 4 adds shared-trajectory and shared-basis tests.
- Section 6 adds the Pythia-160M unaligned scale probe and explicitly declines
  any 1.4B cross-seed claim because public seed-pair checkpoints were
  unavailable under the configured repository IDs.
- Discussion and limitations now distinguish raw/head-local non-portability
  from learned compatibility-map rescue.

## V4 Additions (Round 1 + Round 2 robustness experiments)

The following six experiments were added after the anonymous paper review
requested them:

| Experiment | Claim | Result file | Key numbers |
|-----------|-------|-------------|-------------|
| **A Per-example agreement** | Two 96%-accuracy networks only agree 4pp above marginal-product chance | `analysis/per_example_agreement_results.json` | mean_pairwise_agreement=0.958, mean_chance_baseline=0.918, correlation_agreement_vs_shared_head=0.02 |
| **B Training-time portability** | Non-portability is constant across training (same at step 0 as at step 20k) | `analysis/training_time_portability_results.json` | unaligned_acc at step 0 = 0.118, at step 20000 = 0.095; donor_acc at step 20000 = 0.861 |
| **C Alignment tournament** | Seven head-interface methods tested (Procrustes, affine, CKA, Git Re-Basin, learned linear, learned nonlinear); none rescues the head-local transplant. Nonlinear adapter's 0.118 matches donor-zero control 0.118 | `analysis/alignment_tournament_results.json` | unaligned=0.095, two-iface=0.084, cka_linear=0.098, git_rebasin=0.084, learned_linear=0.095, learned_nonlinear=0.118, donor_zero_control=0.118 |
| **D 19M Shakespeare scale test** | Slot-matched transplant at 19M works (drop 0.09); functionally-matched cross-slot transplant still fails (drop 0.57) | `experiments/shakespeare_portability_results.json` | slot-matched drop=0.094, func-matched drop=0.569, procrustes drop=0.136, whole-L0-attn drop=0.601, full-L0 drop=0.926 |
| **E Minimum unit sweep** | Single attention projection transplant alone breaks the model (drop to 0.002); MLP/LN projections transplant freely; greedy rescue minimum is 9/15 tensors at 72% params | `analysis/minimum_unit_sweep_results.json` | per-tensor accs: L0_qkv_proj=0.006, L0_out_proj=0.002, L0_ln1=0.965, L1_mlp=0.962. Greedy: 9 tensors, 72% params, acc=0.593 |
| **F Residual stream geometry** | 20 seeds' residual streams have mean pairwise CKA=0.013; shared-critical-head pairs have same CKA as different-head pairs | `analysis/residual_stream_geometry_results.json` | mean_cka_pre_l0=0.013, correlation with shared-head=-0.031, p=0.67 |

### Paper integration

- Section 4.5 extended with alignment tournament (Table 2 + extended text)
- Section 4.6 softened: "minimum unit is large but not whole network"
- New Section 4.7 "Minimum functional unit: finer-grained characterization" (Exp E)
- New Section 4.8 "Training-time portability: when does non-portability emerge?" (Exp B + figure)
- New Section 5 paragraph: "Per-example agreement" (Exp A)
- New subsection in 4.5: "Residual stream geometry: distinct manifolds" (Exp F)
- New Section 6 "Scale Replication: 19M Shakespeare Model" (Exp D)
- Abstract + Limitations updated with scale and alignment results

### Text-only fixes

- Added analogy in Introduction; V6 replaces the earlier French/English
  analogy with a compiled instruction/register-state analogy
- Softened "minimum functional unit is the entire network" → refined to quantitative result
- Softened "non-portability is not a basis problem" → "not correctable by any orthogonal or learned alignment we tested"
- Strengthened Limitations section with explicit scale caveat

---

---

## RECONCILIATION: What Changed Across Versions

### Version history

| Version | Key event |
|---------|-----------|
| **V1** | Original submission. Used Model A (4L positional-shortcut). Wrong similarity metric (layer-level causal vectors). Wrong baseline (uniform 1.5% for NLP). |
| **V2** | Post-review rewrite. Switched to Model B (2L/4H content-matching). Correct metrics. Seed 2 was non-converged (baseline 0.193 at 20k steps); excluded from converged-only stats but noted. Source: `content_matching_20_seeds_results.json`. |
| **V2-post-retrain** | Seed 2 retrained to 30k steps; now converged (baseline 0.965, critical-head drop 0.940). All 20 seeds pass convergence threshold. Source: `content_matching_20_seeds_results_v2.json`. |
| **V2-unified** | Unified eval protocol introduced (`transplant_unified_eval.py`). All transplant/Procrustes/whole-layer conditions re-evaluated on a single fixed 1024-sequence batch (data_seed=9999). Source: `transplant_unified_results.json`. Numbers differ slightly from per-script predecessors due to larger/different eval batch. |

### Numbers that changed and why

| Claim | Old PROVENANCE (V2) value | Current value | Source of change |
|-------|--------------------------|---------------|-----------------|
| Head agreement rate | 24.7% CI [18.4%, 31.1%] p<0.001 | **23.2% CI [17.4%, 29.5%] p=0.002** | Seed 2 retrained; slot counts changed (L0H1: 2→3, L0H3: 6→5); agreement recomputed in v2.json |
| Similarity mean | 0.350 CI [0.30, 0.40] | **0.329 CI [0.28, 0.38]** | Same: seed 2 retrain shifted its effect vector |
| Transplant unaligned | 0.090 (circuit_transplant) / 0.092 (basis_aligned_v2) | **0.095** | Unified eval batch (1024 seqs, seed=9999) |
| Procrustes two-interface | 0.079 | **0.084** | Unified eval batch |
| Random rotation control | 0.080 (abstract) / 0.096 (table) | **0.091** | Unified eval batch resolves old batch discrepancy |
| Whole L0 attn block | 0.005 | **0.004** | Unified eval batch |
| Whole L0 block | 0.003 | **0.005** | Unified eval batch |
| All except L1 | 0.005 | **0.007** | Unified eval batch |
| Full transplant | 0.849 | **0.861** | Unified eval batch; seed0 baseline in unified = 0.861 |
| Slot counts | L0H0:5, L0H1:2, L0H2:7, L0H3:6 | **L0H0:5, L0H1:3, L0H2:7, L0H3:5** | Seed 2 retrained: critical head changed from L0H3 (old non-converged) to L0H1 (converged) |
| Seed 2 status | Non-converged (0.193 baseline) | **Converged (0.965 baseline), retrained 30k steps** | Explicit retrain; checkpoint at step_030000 |
| False positives | 38 (draft Appendix), 39 (main text) | **39 (consistent across manuscript)** | Historical: earlier draft Appendix had a ±1 transcription error; fixed during the consistency pass. JSON = 39, main text = 39, current Appendix = 39. |

### Old source files superseded

| Old file | Superseded by | Reason |
|----------|--------------|--------|
| `analysis/content_matching_20_seeds_results.json` | `analysis/content_matching_20_seeds_results_v2.json` | V2 includes seed 2 retrain at 30k steps |
| `analysis/circuit_transplant_results.json` | `analysis/transplant_unified_results.json` | Unified eval batch; paired bootstrap CIs |
| `analysis/basis_aligned_transplant_v2_results.json` | `analysis/transplant_unified_results.json` | Unified eval batch |
| `analysis/whole_layer_transplant_results.json` | `analysis/transplant_unified_results.json` | Unified eval batch |
| `analysis/basis_aligned_transplant_results.json` | — | Superseded in V2 (single-interface, wrong) |

### Seed 2 retraining disclosure

Seed 2 was initially trained for 20k steps like all other seeds. At 20k steps
its baseline accuracy was **0.193** (on grokking plateau; not converged). The
original PROVENANCE (V2) excluded it from converged-only statistics and noted it.

For V3 (`content_matching_20_seeds_results_v2.json`), seed 2 was **retrained to
30k steps** using the same data_seed=42 and architecture. At 30k steps:
- Baseline accuracy: **0.965**
- Critical head: **L0H1** (drop 0.940)
- Checkpoint: `checkpoints/induction_content_match/induction_seed2/step_030000/model.safetensors`

All 20 seeds now pass the convergence threshold (>0.5). The `non_converged_seeds`
list in v2.json is empty. This retrain changes the slot count distribution and
the agreement/similarity statistics.

---

## Model Disambiguation Table

| Label | Architecture | Task | Checkpoint root |
|-------|-------------|------|-----------------|
| **Model B** (transplant / 20-seed) | 2L, 4H, d=128, d_ff=512, vocab=512, ctx=64 (~460K params) | Content-matching induction (n_bigrams=6, random positions) | `checkpoints/induction_content_match/induction_seed{0-19}/` |
| **Model C** (Shakespeare, depth-6) | 6L, 8H, d=512, d_ff=2048, vocab=65, ctx=256 (~19M params) | TinyShakespeare char-level | `checkpoints/shakespeare/text_seed0/step_010000/` |
| **Model D** (depth-scaling) | 2/4/6/8 L, 8H, d=512, vocab=65 | TinyShakespeare char-level | `checkpoints/depth_scaling/depth_{2,4,6,8}/` |
| **Model E** (entropy continuum) | 4L, 4H, d=128, vocab=64, ctx=64 | Markov chain H=0.0–1.0 | Trained inline; no persistent checkpoints |
| **[Deprecated] Model A** | 4L, 4H, d=128, vocab=50 | Positional-shortcut induction | `checkpoints/seed_divergence/induction_seed{0-19}/` |

Model A is no longer cited as the source of any main-text claim.

Transplantation experiments use seeds 0 and 1 of Model B (the **donor** is seed 0
L0H3; the **recipient** is seed 1, whose critical head is L0H1).

---

## Section 3 (Methods) — Methodological Correction Claims

### Claim M-1: "pos_dep classifies 47/48 heads as positional in Shakespeare"
- **Paper location**: Section 3.4 (permutation-sensitivity metric)
- **Source script**: `analysis/metric_validation.py`
- **Model**: Model C (Shakespeare, 6L/8H)
- **Checkpoint**: `checkpoints/shakespeare/text_seed0/step_010000/model.safetensors`
- **Result file**: `analysis/metric_validation_results.json`
- **JSON path**: `shakespeare.n_heads_total` = 48; pos_dep classifies 47 as positional (L1H6 is the single content-classified head)
- **Value**: 47/48 positional by pos_dep

### Claim M-2: "permutation sensitivity resolves 39/48 false positives"
- **Paper location**: Section 3.4; Abstract; Section 9; Appendix B
- **Source script**: `analysis/metric_validation.py`
- **Result file**: `analysis/metric_validation_results.json`
- **JSON path**: `shakespeare.n_disagree` = **39**
- **Exact value**: 39 false positives (consistent across main text and Appendix B in the current manuscript; earlier drafts had a ±1 transcription error in the Appendix which was fixed during the consistency pass)

### Claim M-3: "entropy=0 model: both metrics agree (16/16 positional)"
- **Result file**: `analysis/metric_validation_results.json`
- **JSON path**: `entropy0_markov.n_heads_total` = 16; `entropy0_markov.n_agree` = 16; `entropy0_markov.n_disagree` = 0

---

## Section 4 (Transplantation Failure) — All Claims

**Primary source for all transplant conditions (V3):**
- **Result file**: `analysis/transplant_unified_results.json`
- **Script**: `analysis/transplant_unified_eval.py`
- **Eval protocol**: 1024 fixed sequences, data_seed=9999, vocab=512, n_bigrams=6, batch_size=32 × 32 batches
- **Critical property**: Eval set materialized once as list; RNG state does not advance during evaluation. All conditions see identical sequences.

### Claim T-1: "Seed 1 baseline accuracy 0.964"
- **Paper location**: Abstract; Section 4.1; Table 1
- **JSON key**: `baselines.seed1`
- **Value**: `0.9640299479166667` (reported as 0.964)

### Claim T-2: "Seed 0 baseline accuracy 0.861 (full transplant sanity check)"
- **Paper location**: Table 3 (full transplant row); text "reconstructed the donor network"
- **JSON key**: `baselines.seed0`
- **Value**: `0.8605143229166667` (reported as 0.861 in Table 3 / 0.860 in some references)
- **Note**: This is the unified-eval seed 0 baseline. The 20-seed file gives seed 0 baseline as 0.857 (different eval batch). Main text uses 0.861 from unified eval.

### Claim T-3: "Non-critical slot transplant: accuracy 0.966"
- **Paper location**: Table 1
- **JSON path**: `experiment_1.basic_transplants[0]` (condition "Seed0 L0H3 -> Seed1 L0H3 (non-critical slot)")
- **Value**: `0.9658203125000001` (reported as 0.966)
- **Bootstrap**: p=0.5935 vs seed1 baseline (not significantly different — correct)

### Claim T-4: "Functional head transplant (critical slot, unaligned): accuracy 0.095"
- **Paper location**: Abstract; Section 4.2; Table 1 bold row; Fig. 1
- **JSON path**: `experiment_1.basic_transplants[1]` (condition "Seed0 L0H3 -> Seed1 L0H1 (critical slot, unaligned)")
- **Value**: `0.09505208333333333` (reported as 0.095)
- **Delta vs seed1 baseline**: `−0.869`
- **Bootstrap**: p=0.0 vs seed1 baseline

### Claim T-5: "Control A (non-critical donor → critical slot): accuracy 0.100"
- **Paper location**: Table 1
- **JSON path**: `experiment_1.basic_transplants[2]` (condition "Control A: Seed0 non-critical L0H0 -> Seed1 L0H1")
- **Value**: `0.10026041666666666` (reported as 0.100)

### Claim T-6: "Control B (shuffled weights → critical slot): accuracy 0.098"
- **Paper location**: Abstract (cited as ~0.098); Table 1; Fig. 1
- **JSON path**: `experiment_1.basic_transplants[3]` (condition "Control B: Seed0 L0H3 shuffled weights -> Seed1 L0H1")
- **Value**: `0.09798177083333334` (reported as 0.098)

### Claim T-7: "Ablate seed1 L0H1 (no replacement): accuracy 0.118"
- **Paper location**: Table 1
- **JSON path**: `experiment_1.basic_transplants[4]` (condition "Ablate Seed1 L0H1 (no replacement)")
- **Value**: `0.11783854166666666` (reported as 0.118)

---

## Section 4.4 (Procrustes Alignment) — Basis Alignment Claims

**Source**: `analysis/transplant_unified_results.json`, `experiment_2.procrustes_alignment`.
Script: `analysis/transplant_unified_eval.py`.
Method: two-interface Procrustes — R_in aligns Q/K/V read interface (pre-L0 LN-normalized residual); R_out aligns output-projection write interface (post-L0 residual).

### Claim P-1: "Unaligned transplant: accuracy 0.095 (Table 2)"
- **JSON path**: `experiment_2.procrustes_alignment[0]`
- **Value**: `0.09505208333333333` (reported as 0.095)
- **Note**: Same condition as T-4; identical eval batch confirms consistency.

### Claim P-2: "One-interface aligned: accuracy 0.094 (Table 2)"
- **JSON path**: `experiment_2.procrustes_alignment[1]` (condition "One-interface aligned (post-L0 only)")
- **Value**: `0.09375` (reported as 0.094)

### Claim P-3: "Two-interface aligned: accuracy 0.084 (Table 2, bold)"
- **Paper location**: Abstract; Table 2 bold row; Section 4.4; Fig. 1
- **JSON path**: `experiment_2.procrustes_alignment[2]` (condition "Two-interface aligned (pre-L0 Q/K/V + post-L0 out_proj)")
- **Value**: `0.08382161458333334` (reported as 0.084)
- **Delta vs seed1 baseline**: `−0.880`
- **Bootstrap vs unaligned**: p=0.024 (two-interface is significantly *worse* than unaligned)

### Claim P-4: "Random orthogonal rotation control: accuracy 0.091 (Table 2)"
- **Paper location**: Table 2; Fig. 1
- **JSON path**: `experiment_2.procrustes_alignment[3]` (condition "Random orthogonal rotation control")
- **Value**: `0.09065755208333334` (reported as 0.091)
- **Bootstrap vs two-interface aligned**: p=0.179 (not significantly different — Procrustes provides no advantage over random rotation)

### Claim P-5: "Aligned non-critical donor: accuracy 0.107 (Table 2)"
- **JSON path**: `experiment_2.procrustes_alignment[4]` (condition "Aligned non-critical donor (Seed0 L0H0, two-interface)")
- **Value**: `0.107421875` (reported as 0.107)

### Claim P-6: "Procrustes R² improves from −1.03 to −0.04"
- **Paper location**: Section 4.4; Fig. 1 caption
- **Source**: `analysis/basis_aligned_transplant_v2_results.json` (alignment diagnostics retained from pre-unified run; alignment matrices are the same)
- **JSON key**: `alignment_diagnostics.pre_l0_ln.r2_unrotated` = `−1.033`; `r2_procrustes` = `−0.043`
- **Reported values**: −1.03 unrotated → −0.04 aligned

### Claim P-7: "Post-L0 write interface R² improves from −1.36 to −0.93"
- **Source**: `analysis/basis_aligned_transplant_v2_results.json`
- **JSON key**: `alignment_diagnostics.post_l0.r2_unrotated` = `−1.357`; `r2_procrustes` = `−0.935`

---

## Section 4 static partial transplants — Progressive Transplant Claims

**Source**: `analysis/transplant_unified_results.json`, `experiment_3.whole_layer`.
Seed 1 baseline (unified eval): **0.9640299479166667**.

### Claim W-1: "Single-head reference: accuracy 0.095"
- **JSON path**: `experiment_3.whole_layer[0]` ("Single-head L0H3 -> L0H1 (reference)")
- **Value**: `0.09505208333333333` (reported as 0.095)

### Claim W-2: "Whole L0 attention block (all 4 heads): accuracy 0.004"
- **Paper location**: static partial-transplant table
- **JSON path**: `experiment_3.whole_layer[1]` ("Whole L0 attention block (all 4 heads)")
- **Value**: `0.0037434895833333335` (reported as 0.004)
- **Keys transplanted**: `blocks.0.attn.qkv_proj.weight`, `blocks.0.attn.out_proj.weight`

### Claim W-3: "Whole L0 block (attention + MLP + LayerNorms): accuracy 0.005"
- **Paper location**: static partial-transplant table
- **JSON path**: `experiment_3.whole_layer[2]` ("Whole L0 block (attn + MLP + LayerNorms)")
- **Value**: `0.004557291666666666` (reported as 0.005)
- **Keys transplanted**: 6 keys (ln1, qkv, out, ln2, up_proj, down_proj)

### Claim W-4: "Whole L0 block + final LayerNorm: accuracy 0.006"
- **JSON path**: `experiment_3.whole_layer[3]` ("Whole L0 block + final LayerNorm")
- **Value**: `0.005859375` (reported as 0.006 in Table 3; 0.005 referenced elsewhere — see note)
- **Note**: current main text lists this row in the static partial-transplant table.

### Claim W-5: "Embeddings only: accuracy 0.012"
- **JSON path**: `experiment_3.whole_layer[4]` ("Embeddings only (wte + wpe)")
- **Value**: `0.01220703125` (reported as 0.012)

### Claim W-6: "Embeddings + L0 attention: accuracy 0.004"
- **Paper location**: static partial-transplant table
- **JSON path**: `experiment_3.whole_layer[5]` ("Embeddings + L0 attention")
- **Value**: `0.003580729166666666` (reported as 0.004)

### Claim W-7: "Embeddings + L0 full block: accuracy 0.007"
- **Paper location**: static partial-transplant table
- **JSON path**: `experiment_3.whole_layer[6]` ("Embeddings + L0 full block")
- **Value**: `0.00732421875` (reported as 0.007)

### Claim W-8: "All weights except L1 block: accuracy 0.007"
- **Paper location**: static partial-transplant table; static partial-transplant discussion
- **JSON path**: `experiment_3.whole_layer[7]` ("All except L1 block (embed + L0 + ln_f)")
- **Value**: `0.006998697916666667` (reported as 0.007)
- **Keys transplanted**: 9 keys (embed + full L0 + ln_f); only L1 block remains from seed 1

### Claim W-9: "Full transplant (all 15 weights): accuracy 0.861"
- **Paper location**: Table 3 bold row
- **JSON path**: `experiment_3.whole_layer[8]` ("Full transplant (all weights, sanity check)")
- **Value**: `0.8605143229166667` (reported as 0.861)
- **Sanity check**: This equals seed0 baseline (same JSON value) — confirms correct weight copy

---

## Section 5 (Functional Universality) — 20-Seed Claims

**Primary source**: `analysis/content_matching_20_seeds_results_v2.json`, block `converged_only`.
Script: `analysis/content_matching_20_seeds_v2.py`.
Model: 2L/4H, d=128, vocab=512, ctx=64. Eval: eval_seed=9999, batch_size=256.
Seeds: 0–19 (all 20 converged; seed 2 retrained to 30k steps).

### Claim U-1: "20/20 seeds place critical head in Layer 0 (100% layer agreement)"
- **Paper location**: Abstract; Section 5; Fig. 5 caption; Appendix bootstrap table
- **JSON path**: `converged_only.layer_distribution.L0_count` = 20; `L0_fraction` = 1.0
- **Value**: 100% (20/20)

### Claim U-2: "Head-level agreement 23.2% (95% CI [17.4%, 29.5%], p=0.002)"
- **Paper location**: Abstract; Section 5; Appendix bootstrap table; multiple in-text references
- **JSON key**: `converged_only.critical_head_stats.head_agreement_rate` = `0.23157894736842105`
- **CI**: `head_agreement_ci_95_lo` = `0.1736842105263158`; `head_agreement_ci_95_hi` = `0.29473684210526313`
- **Null model**: `converged_only.null_model.expected_agreement_uniform` = 0.125 (uniform over 8 heads)
- **p-value**: `converged_only.null_model.p_value` = **0.002** (10,000-permutation test)
- **Reported**: 23.2% CI [17.4%, 29.5%] p=0.002 — all match JSON exactly

### Claim U-3: "Head-level similarity 0.329 (95% CI [0.28, 0.38])"
- **Paper location**: Section 5; Fig. 5 caption; Appendix bootstrap table
- **JSON key**: `converged_only.similarity_stats.mean` = `0.3289896565443348`
- **CI**: `ci_95_lo` = `0.28108354214237713`; `ci_95_hi` = `0.3766155395840662`
- **n_pairs**: 190 (off-diagonal upper triangle of 20×20 matrix)
- **Metric**: cosine similarity of 10-dimensional ablation effect vectors (8 heads + 2 MLPs per seed)
- **Reported**: 0.329 CI [0.28, 0.38] — matches JSON (rounded)

### Claim U-4: "Head distribution: L0H0:5, L0H1:3, L0H2:7, L0H3:5"
- **Paper location**: Fig. 5B
- **JSON key**: `converged_only.critical_head_stats.slot_counts`
- **Values**: `{"L0H0": 5, "L0H1": 3, "L0H2": 7, "L0H3": 5, "L1H*": 0}`
- **Note**: L0H1 count is **3** (not 2 as in old PROVENANCE). L0H3 is **5** (not 6). Change due to seed 2 retrain (critical head shifted from non-converged L0H3 to converged L0H1).

### Claim U-5: "Mean ablation drop 0.775 ± 0.162 (95% CI [0.59, 0.91], p=0.004)"
- **Paper location**: Section 5; Appendix bootstrap table
- **Source**: `analysis/ablation_robustness_results.json` → `summary`
- **JSON key**: `summary.critical_head_drop_mean` = `0.7752641593268148`
- **Std**: `summary.critical_head_drop_std` = `0.1623246072259708`
- **CI**: `analysis/statistical_rigor_results.json` → `claim4.ci_95` = `[0.5904, 0.9060]`
- **p-value**: `claim4.p_value` = `0.0037` (reported as 0.004)
- **Note**: This 4-seed value (seeds 0–3) comes from `ablation_robustness_results.json`, not the v2 20-seed file.

---

## Section 4.1 (Ablation Identification) — Critical Head Claims

**Source**: `analysis/ablation_robustness_results.json`.

### Claim A-1: "Seed 0 critical head L0H3; baseline 0.854, post-ablation 0.041, drop 0.813"
- **JSON path**: `seeds[0].critical_head` key="L0H3"; `accuracy_after_ablation`=`0.04076`; `accuracy_drop`=`0.8131`
- **Baseline**: `seeds[0].baseline` = `0.8539`

### Claim A-2: "Seed 1 critical head L0H1; baseline 0.964, post-ablation 0.118, drop 0.846"
- **JSON path**: `seeds[1].critical_head` key="L0H1"; `accuracy_after_ablation`=`0.1180`; `accuracy_drop`=`0.8457`
- **Baseline**: `seeds[1].baseline` = `0.9637`

### Claim A-3: "Seed 2 critical head L0H1; baseline 0.965, post-ablation 0.028, drop 0.937"
- **Paper location**: Appendix C ablation table
- **Source for paper Appendix**: `analysis/ablation_robustness_results.json` seeds[2]
- **20-seed v2 file value**: seed 2, `baseline_accuracy`=`0.9655`, `critical_head_drop`=`0.9401`, head=L0H1
- **Note**: Appendix C uses `ablation_robustness_results.json` seed 2 (drop=0.937). The v2 20-seed file gives 0.940 for the retrained seed 2. These are numerically consistent (same seed, same critical head); minor difference due to eval batch size.

### Claim A-4: "Seed 3 critical head L0H3; baseline 0.968, post-ablation 0.463, drop 0.505"
- **JSON path**: `seeds[3].critical_head` key="L0H3"; `accuracy_after_ablation`=`0.4628`; `accuracy_drop`=`0.5053`

---

## Section 8 (Depth Scaling) — Suppression Cascade Claims

**Source**: `analysis/depth_scaling_rigorous_results.json`. Script: `analysis/depth_scaling_rigorous.py`.
Fixed eval seed: 9999. Attribution samples: 256. Models: Model D (depth 2/4/6/8).

### Claim DS-1: "Depth 2: eval loss 1.05, 8/16 suppression (50%)"
- **JSON**: `per_depth["2"].eval_loss_fixed_seed` = `1.05653`; `suppression_fraction` = `0.5`

### Claim DS-2: "Depth 4: eval loss 0.82, 23/32 suppression (72%)"
- **JSON**: `per_depth["4"].eval_loss_fixed_seed` = `0.81862`; `suppression_fraction` = `0.71875`

### Claim DS-3: "Depth 6: eval loss 0.66, 39/48 suppression (81%)"
- **JSON**: `per_depth["6"].eval_loss_fixed_seed` = `0.65898`; `suppression_fraction` = `0.8125`

### Claim DS-4: "Depth 8: eval loss 0.57, 55/64 suppression (86%)"
- **JSON**: `per_depth["8"].eval_loss_fixed_seed` = `0.56774`; `suppression_fraction` = `0.859375`

### Claim DS-5: "OLS slope 0.059/layer, R²=0.90, 95% CI [0.023, 0.109], p=0.052"
- **JSON key**: `regression_suppression_vs_depth.slope` = `0.05859375`; `r2` = `0.8993`; `ci_95_low` = `0.0234375`; `ci_95_high` = `0.109375`; `p_value` = `0.0517`
- **Note**: p=0.052 marginally above 0.05 with n=4. Paper correctly describes as "scaling trend not a law."

### Claim DS-6: "MLP dominates attention at every depth (8.56×, 9.35×, 9.02×, 9.08×)"
- **JSON**: `per_depth["2"].ratio_mlp_to_attn_abs` = `8.564`; depth 4: `9.345`; depth 6: `9.020`; depth 8: `9.082`

### Claim DS-7 (RETRACTED): "Old L2-norm crossover at depth 8"
- `per_depth["8"].comparison_with_original_l2.old_ratio_mlp_to_attn` = `0.976` (< 1, spurious crossover)
- `crossover_survived` = false. Retracted in Section 8.

---

## Section 9 (Natural Language Bridge) — Shakespeare Claims

**Source**: `analysis/natural_language_induction_v2_results.json`.
Script: `analysis/natural_language_induction_v2.py`.
Model: Model C. Eval: 500 induction positions.

### Claim NL-1: "Model achieves 1.000 accuracy on induction positions"
- **JSON key**: `model_accuracy_on_induction_positions` = `1.0`

### Claim NL-2: "Char-mode baseline (predict space): 0.292"
- **JSON key**: `corpus_baselines.char_mode.accuracy` = `0.292`; character space, corpus frequency 15.2%
- **JSON key**: `best_baseline_accuracy` = `0.292`

### Claim NL-3: "Bigram baseline: 0.076"
- **JSON key**: `corpus_baselines.bigram` = `0.076`

### Claim NL-4: "Induction advantage 1.000 − 0.292 = 0.708"
- **JSON key**: `induction_advantage` = `0.708`

### Claim NL-5: "Ablating L0H7+L1H6+L2H4 drops advantage by 0.586"
- **JSON key**: `ablation_results["All candidates"].induction_advantage_drop` = `0.586`

### Claim NL-6: "L0H7 ablation alone drops advantage by 0.620 (accuracy 0.380)"
- **JSON key**: `ablation_results["L0H7"].model_accuracy` = `0.38`; `induction_advantage_drop` = `0.620`

### Claim NL-7: "L1H6 ablation drops advantage by 0.038"
- **JSON key**: `ablation_results["L1H6"].induction_advantage_drop` = `0.038`

### Claim NL-8 (KNOWN GAP): "L0H7 ablation raises eval loss from 0.080 to 7.95 (Δ=+7.87)"
- **Source**: Not present in `natural_language_induction_v2_results.json`. Derives from
  `analysis/natural_language_induction_results.json` (V1 file) or NLP analysis script output.
- **Status**: Value not verifiable from v2 JSON alone; retain V1 file for this specific claim.

### Claim NL-9 (KNOWN GAP): "Ablating layers 2–5 attention raises loss by 5.08"
- **Source**: Not in v2 JSON. From NLP suppression cascade ablation scripts.
- **Status**: Verify against `analysis/natural_language_induction_results.json`.

---

## Appendix Claims

### Claim App-1: "Appendix C — 4-seed critical head drops"
Source: `analysis/ablation_robustness_results.json`

| Seed | Baseline | Critical head | Post-ablation | Drop |
|------|----------|--------------|---------------|------|
| 0 | 0.854 | L0H3 | 0.041 | **0.813** |
| 1 | 0.964 | L0H1 | 0.118 | **0.846** |
| 2 | 0.965 | L0H1 | 0.028 | **0.937** |
| 3 | 0.968 | L0H3 | 0.463 | **0.505** |

Mean drop: `0.7753` ± `0.1623`.

### Claim App-2: "Bootstrap CI table (Appendix)"
Source: `analysis/statistical_rigor_results.json` and `analysis/content_matching_20_seeds_results_v2.json`

| Claim | Point estimate | 95% CI | Null | p-value | Source |
|-------|--------------|--------|------|---------|--------|
| Layer-level agreement (20 seeds) | 100% | [100%, 100%] | 50% | <0.001 | v2.json |
| Head-level agreement (20 seeds) | **23.2%** | **[17.4%, 29.5%]** | 12.5% | **0.002** | v2.json |
| Head-level similarity (20 seeds) | **0.329** | **[0.28, 0.38]** | 0 | <0.001 | v2.json |
| Ablation drop (4 seeds) | 0.775 | [0.59, 0.91] | 0 | 0.004 | statistical_rigor |
| Suppression slope (n=4 depths) | 0.059/layer | [0.023, 0.109] | 0 | 0.052 | depth_scaling |

---

## Known Discrepancies and Gaps

### Active discrepancies between main.tex and JSON

| Location | main.tex value | JSON value | Status |
|----------|---------------|------------|--------|
| Table 1, "Shuffled weights" | 0.098 | `0.09798` in transplant_unified | Match (rounded) |
| Fig. 1 caption "shuffled random weights (0.098)" | 0.098 | `0.09798` | Match |
| Table 3, "Full transplant" | **0.861** | `0.8605` | Match (rounded) |
| Appendix B false positives | 39 | JSON=39 | **Consistent across manuscript; historical ±1 transcription error in earlier draft Appendix fixed** |
| NL-8 loss ablation values | 0.080→7.95 | Not in v2 JSON | **Known gap; source is V1 NLP file** |
| NL-9 cascade loss increase | 5.08 | Not in v2 JSON | **Known gap; source is V1 NLP file** |

### Values verified against main.tex (all match)

The following main.tex values are confirmed against `transplant_unified_results.json` and `content_matching_20_seeds_results_v2.json`:

- Abstract: 23.2%, [17.4%, 29.5%], p=0.002 — **confirmed**
- Abstract: 0.095 unaligned and 0.084 two-interface — **confirmed**
- Abstract: partial-retraining K*=100--500 (0.5--2.5% of 20k), all-four adapters 0.956--0.960, shared-basis CKA >0.99, 19M redundancy-control drop 0.445, Pythia-160M unaligned/Procrustes below baseline with non-donor-specific adapter controls — **confirmed against V6 extended-control files**
- Section 5: 0.329 CI [0.28, 0.38] — **confirmed**
- Section 5: slot counts L0H0:5, L0H1:3, L0H2:7, L0H3:5 — **confirmed**
- Table 3: 0.861 full transplant — **confirmed** (0.8605 rounds to 0.861)

---

## File Index: Result Files Used in V3 Paper

| File | Section(s) | Status |
|------|-----------|--------|
| `analysis/content_matching_20_seeds_results_v2.json` | §5 (universality), Abstract, Appendix CI table | **CURRENT** — primary source for all 20-seed claims |
| `analysis/transplant_unified_results.json` | §4.2, §4.4, §4.5 (all transplant/Procrustes/whole-layer) | **CURRENT** — unified eval, supersedes per-script files |
| `results/extended_controls/partial_retrain_rescue_results.json` | §4.5 rescue | **CURRENT** — partial-retraining rescue |
| `results/extended_controls/stitching_adapters_results.json` | §4.5 rescue | **CURRENT** — multi-interface adapter rescue |
| `results/extended_controls/shared_trajectory_results.json` | §4.8 shared-trajectory forks | **CURRENT** — fork compatibility sweep |
| `results/extended_controls/shared_basis_training_results.json` | §4.9 shared-basis training | **CURRENT** — residual CKA joint-training sweep |
| `results/extended_controls/shakespeare_redundancy_control_results.json` | §6 scale probes | **CURRENT** — 19M redundancy-controlled critical-slot transplant |
| `results/extended_controls/pythia_adapter_rescue_results.json` | §6 scale probes | **CURRENT** — Pythia-160M Procrustes and adapter-control probe |
| `results/extended_controls/pythia_scale_portability_results.json` | §6 scale probes | **CURRENT** — Pythia-160M unaligned cross-seed source probe |
| `results/extended_controls/pythia_model_manifest.json` | §6 scale probes, limitations | **CURRENT** — Pythia checkpoint availability |
| `analysis/depth_scaling_rigorous_results.json` | §8 (depth scaling) | **CURRENT** — DLA measurement |
| `analysis/natural_language_induction_v2_results.json` | §9 (NLP bridge) | **CURRENT** — char-mode baseline |
| `analysis/ablation_robustness_results.json` | §4.1, §5, Appendix C | **CURRENT** — 4-seed ablation |
| `analysis/statistical_rigor_results.json` | Appendix CI table, §5, §7 | **CURRENT** — bootstrap CIs |
| `analysis/metric_validation_results.json` | §3.4, Appendix B | **CURRENT** — permutation sensitivity |
| `analysis/basis_aligned_transplant_v2_results.json` | §4.4 (R² diagnostics only) | **RETAINED** — alignment quality diagnostics not re-run in unified eval |
| `analysis/natural_language_induction_results.json` | §9 (NL-8, NL-9 loss values) | **RETAINED (V1)** — loss ablation values not in v2 file |
| `analysis/content_matching_20_seeds_results.json` | — | **SUPERSEDED** by v2 (missing seed 2 retrain) |
| `analysis/circuit_transplant_results.json` | — | **SUPERSEDED** by transplant_unified |
| `analysis/whole_layer_transplant_results.json` | — | **SUPERSEDED** by transplant_unified |
| `analysis/basis_aligned_transplant_results.json` | — | **SUPERSEDED** in V2 (single-interface) |
| `checkpoints/induction_content_match/induction_seed{0,1}/` | §4 | Model B seeds 0,1 (transplant donor/recipient) |
| `checkpoints/induction_content_match/induction_seed2/step_030000/` | §5 | **Seed 2 retrained checkpoint (30k steps)** |
| `checkpoints/induction_content_match/induction_seed{0-19}/` | §5 | Model B seeds 0–19 (20-seed universality) |
| `checkpoints/shakespeare/text_seed0/step_010000/` | §3, §6, §9 | Model C checkpoint |
| `checkpoints/depth_scaling/depth_{2,4,6,8}/` | §8 | Model D checkpoints |
| `checkpoints/seed_divergence/induction_seed{0-19}/` | [DEPRECATED] | Model A — not cited in V3 |
