"""
Cross-seed analysis for 20 content-matching induction models (v2).

Changes from v1:
- Filters to CONVERGED seeds (baseline_accuracy > 0.5) for primary stats
- Seed 2 was non-converged at 20k steps (baseline_accuracy=0.193).
  Retraining attempt: seed 2 retrained to 30k steps.
  If step_030000 checkpoint exists AND converged, seed 2 is included.
  Otherwise seed 2 is excluded and N=19.
- Reports BOTH converged-only stats AND all-seeds stats for transparency
- Saves to content_matching_20_seeds_results_v2.json with clear labeling

Pipeline:
1. Load all 20 models (using retrained seed2 if available and converged)
2. Run ablation with a FIXED eval batch (seed=9999)
3. Compute stats separately for converged seeds vs all seeds
4. Save JSON + generate plot

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/content_matching_20_seeds_v2.py
"""

import sys
import json
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import InductionDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CONFIG = GPTConfig(
    n_layers=2,
    n_heads=4,
    d_model=128,
    d_ff=512,
    vocab_size=512,
    ctx_len=64,
    dropout=0.0,
)

N_SEEDS = 20
ALL_SEEDS = list(range(N_SEEDS))
DATA_SEED = 42
EVAL_SEED = 9999
EVAL_BATCH_SIZE = 256

CONVERGENCE_THRESHOLD = 0.5  # baseline_accuracy must exceed this to be "converged"

BASE_DIR = Path(__file__).parent.parent
CKPT_BASE = BASE_DIR / "checkpoints" / "induction_content_match"
RESULTS_V1_PATH = Path(__file__).parent / "content_matching_20_seeds_results.json"
RESULTS_PATH = Path(__file__).parent / "content_matching_20_seeds_results_v2.json"
PLOT_PATH = Path(__file__).parent / "plots" / "content_matching_20_seeds_v2.png"

N_BOOTSTRAP = 2000
N_PERMUTATIONS = 10000

# ---------------------------------------------------------------------------
# Ablation forward pass  (mirrors analysis/ablation.py patterns exactly)
# ---------------------------------------------------------------------------

def _attn_with_head_ablation(attn_module, x: mx.array, ablate_heads_set: set, layer_idx: int) -> mx.array:
    B, T, C = x.shape
    n_heads = attn_module.n_heads
    d_head = attn_module.d_head

    qkv = attn_module.qkv_proj(x)
    qkv = qkv.reshape(B, T, 3, n_heads, d_head)
    qkv = qkv.transpose(0, 3, 2, 1, 4)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    mask = mx.triu(mx.full((T, T), -1e9), k=1)
    attn = attn + mask
    attn = mx.softmax(attn, axis=-1)
    out = attn @ v  # (B, n_heads, T, d_head)

    head_mask_list = [0.0 if (layer_idx, h) in ablate_heads_set else 1.0 for h in range(n_heads)]
    head_mask = mx.array(head_mask_list).reshape(1, n_heads, 1, 1)
    out = out * head_mask

    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = attn_module.out_proj(out)
    return out


def forward_with_ablations(
    model: GPT,
    idx: mx.array,
    ablate_heads: Optional[list] = None,
    ablate_mlp_layers: Optional[list] = None,
) -> mx.array:
    ablate_heads_set = set(ablate_heads or [])
    ablate_mlp_set = set(ablate_mlp_layers or [])

    B, T = idx.shape
    pos = mx.arange(T)
    x = model.wte(idx) + model.wpe(pos)

    for layer_idx, block in enumerate(model.blocks):
        x_norm = block.ln1(x)
        if any(l == layer_idx for l, _ in ablate_heads_set):
            attn_out = _attn_with_head_ablation(block.attn, x_norm, ablate_heads_set, layer_idx)
        else:
            attn_out = block.attn(x_norm)
        x = x + attn_out

        x_norm2 = block.ln2(x)
        if layer_idx in ablate_mlp_set:
            mlp_out = mx.zeros_like(x_norm2)
        else:
            mlp_out = block.mlp(x_norm2)
        x = x + mlp_out

    x = model.ln_f(x)
    logits = x @ model.wte.weight.T
    return logits


def eval_accuracy(model: GPT, inputs: mx.array, targets: mx.array, mask: mx.array,
                  ablate_heads=None, ablate_mlp_layers=None) -> float:
    logits = forward_with_ablations(model, inputs,
                                     ablate_heads=ablate_heads,
                                     ablate_mlp_layers=ablate_mlp_layers)
    mx.eval(logits)
    logits_np = np.array(logits)
    targets_np = np.array(targets)
    mask_np = np.array(mask)

    preds = logits_np.argmax(axis=-1)
    at_mask = mask_np > 0.5
    total = int(at_mask.sum())
    if total == 0:
        return 0.0
    correct = int((preds == targets_np)[at_mask].sum())
    return correct / total


# ---------------------------------------------------------------------------
# Model loading — handles retrained seed 2
# ---------------------------------------------------------------------------

def find_seed2_checkpoint() -> tuple[Path, str]:
    """
    Returns (path, description) for the best available seed 2 checkpoint.
    Priority: step_030000 (retrained) > step_030000_ds43 (fallback) > step_020000 (original).
    """
    run_dir = CKPT_BASE / "induction_seed2"
    candidates = [
        (run_dir / "step_030000" / "model.safetensors", "retrained 30k steps, data_seed=42"),
        (run_dir / "step_030000_ds43" / "model.safetensors", "retrained 30k steps, data_seed=43"),
        (run_dir / "step_020000" / "model.safetensors", "original 20k steps (non-converged)"),
    ]
    for path, desc in candidates:
        if path.exists():
            return path, desc
    raise FileNotFoundError(f"No checkpoint found for seed 2 under {run_dir}")


def load_model(seed: int) -> tuple[GPT, str]:
    """Load model for given seed. Returns (model, checkpoint_description)."""
    if seed == 2:
        weights_path, ckpt_desc = find_seed2_checkpoint()
    else:
        weights_path = CKPT_BASE / f"induction_seed{seed}" / "step_020000" / "model.safetensors"
        ckpt_desc = "20k steps"

    model = GPT(MODEL_CONFIG)
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())
    return model, ckpt_desc


# ---------------------------------------------------------------------------
# Component effect vector
# ---------------------------------------------------------------------------

def compute_effect_vector(model: GPT, inputs: mx.array, targets: mx.array, mask: mx.array) -> np.ndarray:
    """
    10-D component effect vector.
    Layout: L0H0,L0H1,L0H2,L0H3,L0_MLP, L1H0,L1H1,L1H2,L1H3,L1_MLP
    Effect = baseline_acc - ablated_acc  (positive = component matters)
    """
    baseline = eval_accuracy(model, inputs, targets, mask)
    vector = []

    for layer in range(MODEL_CONFIG.n_layers):
        for head in range(MODEL_CONFIG.n_heads):
            ablated = eval_accuracy(model, inputs, targets, mask, ablate_heads=[(layer, head)])
            vector.append(baseline - ablated)
        ablated_mlp = eval_accuracy(model, inputs, targets, mask, ablate_mlp_layers=[layer])
        vector.append(baseline - ablated_mlp)

    return np.array(vector, dtype=np.float64)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    alpha = 1 - ci
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def permutation_test_agreement(observed_agreement: float, critical_heads: list[tuple[int, int]],
                                n_permutations: int = 10000) -> float:
    rng = np.random.default_rng(42)
    n = len(critical_heads)
    n_slots = MODEL_CONFIG.n_layers * MODEL_CONFIG.n_heads  # 8 slots
    null_agreements = []

    for _ in range(n_permutations):
        null_slots = rng.integers(0, n_slots, size=n)
        agreements = [
            1.0 if null_slots[i] == null_slots[j] else 0.0
            for i in range(n) for j in range(i + 1, n)
        ]
        null_agreements.append(np.mean(agreements))

    null_agreements = np.array(null_agreements)
    return float((null_agreements >= observed_agreement).mean())


def compute_stats_for_subset(
    seed_results: list[dict],
    effect_vectors_map: dict[int, np.ndarray],
    label: str,
) -> dict:
    """Compute all stats for a given subset of seeds."""
    n = len(seed_results)
    if n == 0:
        return {"n_seeds": 0, "label": label}

    seeds = [r["seed"] for r in seed_results]
    vecs = [effect_vectors_map[s] for s in seeds]

    # Pairwise cosine similarity
    sim_matrix = np.zeros((n, n))
    pairwise_sims = []
    for i in range(n):
        for j in range(n):
            s = cosine_similarity(vecs[i], vecs[j])
            sim_matrix[i, j] = s
            if i < j:
                pairwise_sims.append(s)

    pairwise_sims = np.array(pairwise_sims)
    mean_sim = float(pairwise_sims.mean()) if len(pairwise_sims) > 0 else 0.0
    sd_sim = float(pairwise_sims.std()) if len(pairwise_sims) > 0 else 0.0
    ci_lo, ci_hi = bootstrap_ci(pairwise_sims, N_BOOTSTRAP) if len(pairwise_sims) > 1 else (0.0, 0.0)

    # Critical heads
    critical_heads = [(r["critical_head_layer"], r["critical_head_index"]) for r in seed_results]
    per_seed_drops = [r["critical_head_drop"] for r in seed_results]

    unique_slots = sorted(set(critical_heads))
    slot_counts = {
        f"L{l}H{h}": critical_heads.count((l, h))
        for l in range(MODEL_CONFIG.n_layers)
        for h in range(MODEL_CONFIG.n_heads)
    }

    # Head agreement
    n_pairs = n * (n - 1) // 2
    pair_agreements = np.array([
        1.0 if critical_heads[i] == critical_heads[j] else 0.0
        for i in range(n) for j in range(i + 1, n)
    ])
    head_agreement_rate = float(pair_agreements.mean()) if n_pairs > 0 else 0.0
    agree_ci_lo, agree_ci_hi = bootstrap_ci(pair_agreements, N_BOOTSTRAP) if n_pairs > 1 else (0.0, 0.0)

    # Permutation test
    p_value = permutation_test_agreement(head_agreement_rate, critical_heads, N_PERMUTATIONS)

    # Layer distribution
    l0_count = sum(1 for layer, _ in critical_heads if layer == 0)
    l1_count = sum(1 for layer, _ in critical_heads if layer == 1)

    return {
        "label": label,
        "n_seeds": n,
        "seed_list": seeds,
        "n_pairs": n_pairs,
        "similarity_stats": {
            "mean": mean_sim,
            "sd": sd_sim,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "n_pairs": len(pairwise_sims),
        },
        "pairwise_similarity_matrix": sim_matrix.tolist(),
        "critical_head_stats": {
            "per_seed_critical_heads": [{"layer": l, "head": h} for l, h in critical_heads],
            "distinct_slots_used": len(unique_slots),
            "slot_counts": slot_counts,
            "head_agreement_rate": head_agreement_rate,
            "head_agreement_ci_95_lo": agree_ci_lo,
            "head_agreement_ci_95_hi": agree_ci_hi,
        },
        "null_model": {
            "expected_agreement_uniform": 1 / 8,
            "n_permutations": N_PERMUTATIONS,
            "p_value": p_value,
        },
        "layer_distribution": {
            "L0_count": l0_count,
            "L1_count": l1_count,
            "L0_fraction": l0_count / n,
            "L1_fraction": l1_count / n,
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(converged_stats: dict, all_stats: dict, per_seed_results: list[dict],
              converged_seeds: set[int], save_path: Path):
    """3-panel plot for converged seeds, with non-converged seeds marked."""
    n_conv = converged_stats["n_seeds"]
    n_all = all_stats["n_seeds"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Content-Matching Induction Models: Cross-Seed Analysis (v2)\n"
        f"Primary: {n_conv} converged seeds | All: {n_all} seeds",
        fontsize=13,
    )

    # Use converged-only similarity matrix for panel A
    sim_matrix = np.array(converged_stats["pairwise_similarity_matrix"])
    conv_seeds = converged_stats["seed_list"]

    # Panel A: similarity matrix heatmap (converged seeds only)
    ax = axes[0]
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title(f"A: Pairwise Cosine Similarity\n({n_conv} converged seeds)", fontsize=11)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Seed")
    ax.set_xticks(range(n_conv))
    ax.set_yticks(range(n_conv))
    ax.set_xticklabels(conv_seeds, fontsize=7)
    ax.set_yticklabels(conv_seeds, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B: critical head distribution (converged seeds)
    ax = axes[1]
    slot_labels = [f"L{l}H{h}" for l in range(2) for h in range(4)]
    slot_counts_conv = converged_stats["critical_head_stats"]["slot_counts"]
    counts_conv = [slot_counts_conv.get(lbl, 0) for lbl in slot_labels]
    max_c = max(counts_conv) if counts_conv else 1
    colors = ["#d62728" if c == max_c else "#1f77b4" for c in counts_conv]
    ax.bar(slot_labels, counts_conv, color=colors)
    ax.axhline(n_conv / 8, color="gray", linestyle="--", linewidth=1, label=f"Uniform (N/{8}={n_conv/8:.1f})")
    ax.set_title(f"B: Critical Head Distribution\n({n_conv} converged seeds)", fontsize=11)
    ax.set_xlabel("Head Slot")
    ax.set_ylabel(f"Count (out of {n_conv} seeds)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, n_conv + 1)
    for i, (lbl, c) in enumerate(zip(slot_labels, counts_conv)):
        if c > 0:
            ax.text(i, c + 0.2, str(c), ha="center", va="bottom", fontsize=9)

    # Panel C: per-seed ablation drops (all 20 seeds, marking non-converged)
    ax = axes[2]
    all_seed_list = [r["seed"] for r in per_seed_results]
    all_drops = [r["critical_head_drop"] for r in per_seed_results]
    all_crit = [(r["critical_head_layer"], r["critical_head_index"]) for r in per_seed_results]

    bar_colors = []
    for r in per_seed_results:
        if r["seed"] not in converged_seeds:
            bar_colors.append("#aaaaaa")  # gray for non-converged
        elif r["critical_head_layer"] == 1:
            bar_colors.append("#d62728")  # red for L1 critical
        else:
            bar_colors.append("#1f77b4")  # blue for L0 critical

    x_pos = list(range(len(all_seed_list)))
    ax.bar(x_pos, all_drops, color=bar_colors)
    ax.axhline(CONVERGENCE_THRESHOLD, color="orange", linestyle=":", linewidth=1.5,
               label=f"Convergence threshold ({CONVERGENCE_THRESHOLD})", alpha=0.7)
    ax.set_title("C: Per-Seed Max Ablation Drop\n(gray=non-converged, blue=L0, red=L1 critical)", fontsize=11)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Accuracy Drop (baseline - ablated)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_seed_list, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    for i, (r, layer, head) in enumerate(zip(per_seed_results, [c[0] for c in all_crit], [c[1] for c in all_crit])):
        if r["seed"] in converged_seeds:
            ax.text(i, all_drops[i] + 0.005, f"L{layer}H{head}", ha="center", va="bottom",
                    fontsize=6, rotation=90)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Load v1 results for comparison
# ---------------------------------------------------------------------------

def load_v1_stats() -> dict:
    """Load v1 results for old-vs-new comparison table."""
    if not RESULTS_V1_PATH.exists():
        return {}
    with open(RESULTS_V1_PATH) as f:
        v1 = json.load(f)
    return {
        "n_seeds": v1.get("n_seeds", 20),
        "mean_sim": v1["similarity_stats"]["mean"],
        "sd_sim": v1["similarity_stats"]["sd"],
        "ci_lo": v1["similarity_stats"]["ci_95_lo"],
        "ci_hi": v1["similarity_stats"]["ci_95_hi"],
        "head_agreement_rate": v1["critical_head_stats"]["head_agreement_rate"],
        "agree_ci_lo": v1["critical_head_stats"]["head_agreement_ci_95_lo"],
        "agree_ci_hi": v1["critical_head_stats"]["head_agreement_ci_95_hi"],
        "p_value": v1["null_model"]["p_value"],
        "l0_count": v1["layer_distribution"]["L0_count"],
        "l1_count": v1["layer_distribution"]["L1_count"],
        "slot_counts": v1["critical_head_stats"]["slot_counts"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CONTENT-MATCHING INDUCTION MODELS: 20-SEED ANALYSIS v2")
    print("(with convergence filtering and seed 2 retraining)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Check seed 2 retraining status
    # -----------------------------------------------------------------------
    print("\n=== SEED 2 STATUS ===")
    seed2_ckpt, seed2_desc = find_seed2_checkpoint()
    print(f"  Using checkpoint: {seed2_ckpt}")
    print(f"  Description: {seed2_desc}")
    is_retrained_seed2 = "030000" in str(seed2_ckpt)
    print(f"  Is retrained (30k steps): {is_retrained_seed2}")

    # Verify all 20 checkpoints exist
    print("\n=== CHECKPOINT VERIFICATION ===")
    for seed in ALL_SEEDS:
        if seed == 2:
            ckpt = seed2_ckpt
        else:
            ckpt = CKPT_BASE / f"induction_seed{seed}" / "step_020000" / "model.safetensors"
        if ckpt.exists():
            print(f"  seed {seed:2d}: OK  ({ckpt.parent.name})")
        else:
            raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    # -----------------------------------------------------------------------
    # Step 2: Generate fixed eval batch
    # -----------------------------------------------------------------------
    print(f"\nGenerating fixed eval batch (eval_seed={EVAL_SEED}, batch_size={EVAL_BATCH_SIZE})...")
    eval_dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=EVAL_SEED)
    eval_inputs, eval_targets, eval_mask = eval_dataset.generate_batch(EVAL_BATCH_SIZE)
    mx.eval(eval_inputs, eval_targets, eval_mask)
    print(f"  Eval batch shape: {eval_inputs.shape}")

    # -----------------------------------------------------------------------
    # Step 3: Per-seed ablation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-SEED ABLATION")
    print("=" * 70)

    per_seed_results = []
    effect_vectors_map: dict[int, np.ndarray] = {}

    for seed in ALL_SEEDS:
        print(f"\n[seed {seed}] Loading model...", flush=True)
        model, ckpt_desc = load_model(seed)

        baseline_acc = eval_accuracy(model, eval_inputs, eval_targets, eval_mask)
        converged = baseline_acc > CONVERGENCE_THRESHOLD
        status = "CONVERGED" if converged else "NON-CONVERGED"
        print(f"  [seed {seed}] baseline_accuracy={baseline_acc:.4f}  [{status}]  ({ckpt_desc})")

        print(f"  [seed {seed}] Computing effect vector (10 ablations)...", flush=True)
        vec = compute_effect_vector(model, eval_inputs, eval_targets, eval_mask)
        effect_vectors_map[seed] = vec

        head_drops = {}
        for layer in range(MODEL_CONFIG.n_layers):
            for head in range(MODEL_CONFIG.n_heads):
                idx = layer * (MODEL_CONFIG.n_heads + 1) + head
                key = f"L{layer}H{head}"
                head_drops[key] = float(vec[idx])
                print(f"    ablate {key}: drop={vec[idx]:+.4f}")
            mlp_idx = layer * (MODEL_CONFIG.n_heads + 1) + MODEL_CONFIG.n_heads
            print(f"    ablate L{layer}_MLP: drop={vec[mlp_idx]:+.4f}")

        best_key = max(head_drops, key=lambda k: head_drops[k])
        best_layer = int(best_key[1])
        best_head = int(best_key[3])
        best_drop = head_drops[best_key]
        print(f"  [seed {seed}] CRITICAL HEAD: {best_key} (drop={best_drop:.4f})")

        per_seed_results.append({
            "seed": seed,
            "baseline_accuracy": float(baseline_acc),
            "converged": converged,
            "checkpoint_description": ckpt_desc,
            "critical_head_layer": best_layer,
            "critical_head_index": best_head,
            "critical_head_drop": float(best_drop),
            "head_drops": head_drops,
            "effect_vector": vec.tolist(),
        })

    # -----------------------------------------------------------------------
    # Step 4: Split into converged vs all
    # -----------------------------------------------------------------------
    converged_results = [r for r in per_seed_results if r["converged"]]
    converged_seeds = {r["seed"] for r in converged_results}
    non_converged = [r for r in per_seed_results if not r["converged"]]

    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total seeds:        {len(per_seed_results)}")
    print(f"  Converged (>={CONVERGENCE_THRESHOLD}): {len(converged_results)}")
    print(f"  Non-converged:      {len(non_converged)}")
    for r in non_converged:
        print(f"    seed {r['seed']}: baseline_accuracy={r['baseline_accuracy']:.4f}  ({r['checkpoint_description']})")

    # -----------------------------------------------------------------------
    # Step 5: Compute stats for both subsets
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STATS: CONVERGED SEEDS ONLY (PRIMARY — for paper)")
    print(f"{'='*70}")
    converged_stats = compute_stats_for_subset(converged_results, effect_vectors_map, "converged_only")
    _print_stats(converged_stats)

    print(f"\n{'='*70}")
    print("STATS: ALL SEEDS (for transparency)")
    print(f"{'='*70}")
    all_stats = compute_stats_for_subset(per_seed_results, effect_vectors_map, "all_seeds")
    _print_stats(all_stats)

    # -----------------------------------------------------------------------
    # Step 6: Old vs new comparison table
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("OLD (v1, all 20 seeds) vs NEW (v2) COMPARISON")
    print(f"{'='*70}")
    v1 = load_v1_stats()
    _print_comparison_table(v1, converged_stats, all_stats)

    # -----------------------------------------------------------------------
    # Step 7: Save results
    # -----------------------------------------------------------------------
    results = {
        "experiment": "content_matching_20_seeds_v2",
        "description": (
            "v2: Filters to converged seeds (baseline_accuracy > 0.5). "
            "Seed 2 was non-converged at 20k steps; retrained to 30k steps. "
            "converged_only = primary numbers for paper. "
            "all_seeds = transparency / supplement."
        ),
        "model_config": {
            "n_layers": MODEL_CONFIG.n_layers,
            "n_heads": MODEL_CONFIG.n_heads,
            "d_model": MODEL_CONFIG.d_model,
            "d_ff": MODEL_CONFIG.d_ff,
            "vocab_size": MODEL_CONFIG.vocab_size,
            "ctx_len": MODEL_CONFIG.ctx_len,
        },
        "eval_seed": EVAL_SEED,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "seed2_retraining": {
            "original_20k_baseline_accuracy": 0.193,
            "original_20k_critical_head_drop": 0.064,
            "checkpoint_used": str(seed2_ckpt),
            "checkpoint_description": seed2_desc,
            "is_retrained": is_retrained_seed2,
        },
        "per_seed": per_seed_results,
        "converged_seeds": sorted(converged_seeds),
        "non_converged_seeds": [r["seed"] for r in non_converged],
        # Primary numbers for the paper
        "converged_only": converged_stats,
        # All-seeds numbers for transparency
        "all_seeds": all_stats,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Step 8: Plot
    # -----------------------------------------------------------------------
    make_plot(converged_stats, all_stats, per_seed_results, converged_seeds, PLOT_PATH)

    # -----------------------------------------------------------------------
    # Summary for paper
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PAPER NUMBERS (use converged_only)")
    print(f"{'='*70}")
    cs = converged_stats
    sim = cs["similarity_stats"]
    crit = cs["critical_head_stats"]
    null = cs["null_model"]
    layer = cs["layer_distribution"]
    print(f"  N converged seeds:             {cs['n_seeds']}")
    print(f"  N pairs:                       {cs['n_pairs']}")
    print(f"  Mean pairwise cosine sim:      {sim['mean']:.4f} (SD={sim['sd']:.4f})")
    print(f"  95% CI:                        [{sim['ci_95_lo']:.4f}, {sim['ci_95_hi']:.4f}]")
    print(f"  Head agreement rate:           {crit['head_agreement_rate']:.4f}")
    print(f"  Head agreement 95% CI:         [{crit['head_agreement_ci_95_lo']:.4f}, {crit['head_agreement_ci_95_hi']:.4f}]")
    print(f"  Permutation test p-value:      {null['p_value']:.6f}")
    print(f"  Distinct critical head slots:  {crit['distinct_slots_used']}")
    print(f"  L0 critical:                   {layer['L0_count']}/{cs['n_seeds']} ({layer['L0_fraction']:.1%})")
    print(f"  L1 critical:                   {layer['L1_count']}/{cs['n_seeds']} ({layer['L1_fraction']:.1%})")
    print(f"  Slot counts (converged):       {crit['slot_counts']}")
    print(f"  Results JSON:                  {RESULTS_PATH}")
    print(f"  Plot:                          {PLOT_PATH}")
    print("=" * 70)
    print("DONE.")


def _print_stats(stats: dict):
    if stats["n_seeds"] == 0:
        print("  No seeds in this subset.")
        return
    sim = stats["similarity_stats"]
    crit = stats["critical_head_stats"]
    null = stats["null_model"]
    layer = stats["layer_distribution"]
    print(f"  N seeds: {stats['n_seeds']}  |  N pairs: {stats['n_pairs']}")
    print(f"  Mean pairwise cosine similarity: {sim['mean']:.4f} (SD={sim['sd']:.4f})")
    print(f"  95% CI (bootstrap): [{sim['ci_95_lo']:.4f}, {sim['ci_95_hi']:.4f}]")
    print(f"  Head agreement rate: {crit['head_agreement_rate']:.4f}")
    print(f"  95% CI (bootstrap): [{crit['head_agreement_ci_95_lo']:.4f}, {crit['head_agreement_ci_95_hi']:.4f}]")
    print(f"  Expected under uniform (1/8): {null['expected_agreement_uniform']:.4f}")
    print(f"  Permutation test p-value (n={null['n_permutations']}): {null['p_value']:.6f}")
    if null['p_value'] < 0.001:
        print(f"  *** p < 0.001 ***")
    elif null['p_value'] < 0.05:
        print(f"  ** p < 0.05 **")
    print(f"  Distinct slots: {crit['distinct_slots_used']}  |  Slot counts: {crit['slot_counts']}")
    print(f"  Layer distribution: L0={layer['L0_count']}/{stats['n_seeds']}  L1={layer['L1_count']}/{stats['n_seeds']}")


def _print_comparison_table(v1: dict, converged: dict, all_s: dict):
    if not v1:
        print("  (v1 results not found, skipping comparison)")
        return

    c_sim = converged["similarity_stats"]
    c_crit = converged["critical_head_stats"]
    c_null = converged["null_model"]
    c_layer = converged["layer_distribution"]

    a_sim = all_s["similarity_stats"]
    a_crit = all_s["critical_head_stats"]
    a_null = all_s["null_model"]

    def _fmt(val, fmt=".4f"):
        return format(val, fmt) if val is not None else "N/A"

    print(f"  {'Metric':<40} {'v1 (all 20)':>15} {'v2 converged':>15} {'v2 all 20':>15}")
    print(f"  {'-'*85}")
    print(f"  {'N seeds':<40} {v1.get('n_seeds',20):>15} {converged['n_seeds']:>15} {all_s['n_seeds']:>15}")
    print(f"  {'N pairs':<40} {'190':>15} {converged['n_pairs']:>15} {all_s['n_pairs']:>15}")
    print(f"  {'Mean pairwise cosine sim':<40} {_fmt(v1['mean_sim']):>15} {_fmt(c_sim['mean']):>15} {_fmt(a_sim['mean']):>15}")
    print(f"  {'SD cosine sim':<40} {_fmt(v1['sd_sim']):>15} {_fmt(c_sim['sd']):>15} {_fmt(a_sim['sd']):>15}")
    print(f"  {'Head agreement rate':<40} {_fmt(v1['head_agreement_rate']):>15} {_fmt(c_crit['head_agreement_rate']):>15} {_fmt(a_crit['head_agreement_rate']):>15}")
    print(f"  {'p-value (permutation)':<40} {_fmt(v1['p_value'],'.6f'):>15} {_fmt(c_null['p_value'],'.6f'):>15} {_fmt(a_null['p_value'],'.6f'):>15}")
    print(f"  {'L0 critical count':<40} {v1.get('l0_count','?'):>15} {c_layer['L0_count']:>15} {all_s['layer_distribution']['L0_count']:>15}")
    print(f"  {'L1 critical count':<40} {v1.get('l1_count','?'):>15} {c_layer['L1_count']:>15} {all_s['layer_distribution']['L1_count']:>15}")
    print(f"  {'L0H3 slot count':<40} {v1.get('slot_counts',{}).get('L0H3','?'):>15} {converged['critical_head_stats']['slot_counts'].get('L0H3',0):>15} {all_s['critical_head_stats']['slot_counts'].get('L0H3',0):>15}")
    print()
    print("  => PRIMARY PAPER NUMBERS: use 'v2 converged' column")
    print("  => FOR TRANSPARENCY: report 'v2 all 20' in supplement")


if __name__ == "__main__":
    main()
