"""
Cross-seed analysis for 20 content-matching induction models.

Pipeline:
1. Train seeds 4-19 (seeds 0-3 already exist) using parallel subprocesses.
2. Load all 20 models and run ablation with a FIXED eval batch (seed=9999).
3. Compute:
   - Component effect vectors (10-D per seed: 2L x 4H ablation drops + 2 MLP drops)
   - Pairwise cosine similarity matrix (20x20)
   - Critical head per seed (largest accuracy drop on head ablation)
   - Head-agreement rate across 190 pairs + bootstrap 95% CI
   - Null model permutation test (p-value vs uniform 1/8 expected agreement)
   - Layer-level distribution (L0 vs L1 critical heads)
4. Save JSON results and generate 3-panel plot.

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/content_matching_20_seeds.py
"""

import sys
import json
import time
import subprocess
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
EVAL_SEED = 9999      # Fixed eval batch seed — same for ALL measurements
EVAL_BATCH_SIZE = 256  # Large fixed batch for stable estimates

BASE_DIR = Path(__file__).parent.parent
CKPT_BASE = BASE_DIR / "checkpoints" / "induction_content_match"
RESULTS_PATH = Path(__file__).parent / "content_matching_20_seeds_results.json"
PLOT_PATH = Path(__file__).parent / "plots" / "content_matching_20_seeds.png"

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

    # Zero out ablated heads via masked multiply
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
    """Accuracy at induction-masked positions using the given fixed eval batch."""
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
# Model loading
# ---------------------------------------------------------------------------

def load_model(seed: int) -> GPT:
    weights_path = CKPT_BASE / f"induction_seed{seed}" / "step_020000" / "model.safetensors"
    model = GPT(MODEL_CONFIG)
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Parallel training
# ---------------------------------------------------------------------------

def train_missing_seeds(seeds_to_train: list[int], max_concurrent: int = 2):
    """Launch training subprocesses for missing seeds, max_concurrent at a time."""
    if not seeds_to_train:
        print("All seeds already trained.")
        return

    venv_python = str(BASE_DIR / ".venv" / "bin" / "python")
    script_path = str(Path(__file__).parent / "train_single_seed.py")

    print(f"\nTraining {len(seeds_to_train)} seeds with up to {max_concurrent} concurrent jobs...")
    print(f"Seeds to train: {seeds_to_train}\n")

    queue = list(seeds_to_train)
    running: dict[int, subprocess.Popen] = {}  # seed -> process

    while queue or running:
        # Launch new jobs up to concurrency limit
        while queue and len(running) < max_concurrent:
            seed = queue.pop(0)
            print(f"[launcher] Starting seed {seed}...", flush=True)
            proc = subprocess.Popen(
                [venv_python, script_path, "--seed", str(seed)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            running[seed] = proc

        # Poll all running processes
        finished = []
        for seed, proc in running.items():
            # Non-blocking read of available output
            if proc.stdout:
                line = proc.stdout.readline()
                if line:
                    print(line, end="", flush=True)
            ret = proc.poll()
            if ret is not None:
                # Drain remaining output
                if proc.stdout:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                if ret == 0:
                    print(f"[launcher] Seed {seed} COMPLETED successfully.", flush=True)
                else:
                    print(f"[launcher] Seed {seed} FAILED with return code {ret}.", flush=True)
                    raise RuntimeError(f"Training seed {seed} failed (exit code {ret}).")
                finished.append(seed)

        for seed in finished:
            del running[seed]

        if running:
            time.sleep(0.5)

    print("\nAll seeds trained.\n")


# ---------------------------------------------------------------------------
# Component effect vector computation
# ---------------------------------------------------------------------------

def compute_effect_vector(model: GPT, inputs: mx.array, targets: mx.array, mask: mx.array) -> np.ndarray:
    """
    Compute the 10-D component effect vector for one model.

    Components (in order):
      L0H0, L0H1, L0H2, L0H3  (ablation drop for each head in layer 0)
      L0_MLP                   (ablation drop for MLP in layer 0)
      L1H0, L1H1, L1H2, L1H3  (ablation drop for each head in layer 1)
      L1_MLP                   (ablation drop for MLP in layer 1)

    Effect = baseline_acc - ablated_acc  (positive = component matters)
    """
    baseline = eval_accuracy(model, inputs, targets, mask)
    vector = []

    for layer in range(MODEL_CONFIG.n_layers):
        for head in range(MODEL_CONFIG.n_heads):
            ablated = eval_accuracy(model, inputs, targets, mask, ablate_heads=[(layer, head)])
            drop = baseline - ablated
            vector.append(drop)
        # MLP ablation for this layer
        ablated_mlp = eval_accuracy(model, inputs, targets, mask, ablate_mlp_layers=[layer])
        drop_mlp = baseline - ablated_mlp
        vector.append(drop_mlp)

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
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    alpha = 1 - ci
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def permutation_test_agreement(observed_agreement: float, critical_heads: list[tuple[int, int]],
                                n_permutations: int = 10000) -> float:
    """
    Permutation test against uniform null: randomly assign each seed a head slot
    drawn uniformly from all 8 possible slots (2 layers x 4 heads), then compute
    pairwise agreement. p-value = fraction of null draws >= observed agreement.

    This tests whether the observed concentration of critical heads in specific
    slots is greater than expected under pure chance (1/8 per slot).
    """
    rng = np.random.default_rng(42)
    n = len(critical_heads)
    n_slots = MODEL_CONFIG.n_layers * MODEL_CONFIG.n_heads  # 8 slots
    null_agreements = []

    for _ in range(n_permutations):
        # Draw uniformly from 8 slots for each seed
        null_slots = rng.integers(0, n_slots, size=n)
        agreements = []
        for i in range(n):
            for j in range(i + 1, n):
                agreements.append(1.0 if null_slots[i] == null_slots[j] else 0.0)
        null_agreements.append(np.mean(agreements))

    null_agreements = np.array(null_agreements)
    p_value = float((null_agreements >= observed_agreement).mean())
    return p_value


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(sim_matrix: np.ndarray, critical_heads: list, per_seed_drops: list[float],
              per_seed_results: list[dict], save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Content-Matching Induction Models: 20-Seed Cross-Seed Analysis", fontsize=14)

    # Panel A: similarity matrix heatmap
    ax = axes[0]
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title("A: Pairwise Cosine Similarity\n(Component Effect Vectors)", fontsize=11)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Seed")
    ax.set_xticks(range(N_SEEDS))
    ax.set_yticks(range(N_SEEDS))
    ax.set_xticklabels(range(N_SEEDS), fontsize=7)
    ax.set_yticklabels(range(N_SEEDS), fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B: critical head distribution (count per slot)
    ax = axes[1]
    slot_labels = [f"L{l}H{h}" for l in range(2) for h in range(4)]
    slot_counts = {lbl: 0 for lbl in slot_labels}
    for layer, head in critical_heads:
        key = f"L{layer}H{head}"
        slot_counts[key] = slot_counts.get(key, 0) + 1
    counts = [slot_counts[lbl] for lbl in slot_labels]
    colors = ["#d62728" if c == max(counts) else "#1f77b4" for c in counts]
    ax.bar(slot_labels, counts, color=colors)
    ax.axhline(N_SEEDS / 8, color="gray", linestyle="--", linewidth=1, label="Uniform (1/8)")
    ax.set_title("B: Critical Head Distribution\n(Head w/ largest ablation drop)", fontsize=11)
    ax.set_xlabel("Head Slot")
    ax.set_ylabel("Count (out of 20 seeds)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, N_SEEDS + 1)
    for i, (lbl, c) in enumerate(zip(slot_labels, counts)):
        if c > 0:
            ax.text(i, c + 0.2, str(c), ha="center", va="bottom", fontsize=9)

    # Panel C: per-seed ablation drops
    ax = axes[2]
    seeds = list(range(N_SEEDS))
    layer_colors = []
    for layer, head in critical_heads:
        layer_colors.append("#d62728" if layer == 1 else "#1f77b4")
    bars = ax.bar(seeds, per_seed_drops, color=layer_colors)
    ax.set_title("C: Per-Seed Max Ablation Drop\n(red=L1 critical, blue=L0 critical)", fontsize=11)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Accuracy Drop (baseline - ablated)")
    ax.set_xticks(seeds)
    ax.set_xticklabels(seeds, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate critical head label
    for i, (layer, head) in enumerate(critical_heads):
        ax.text(i, per_seed_drops[i] + 0.005, f"L{layer}H{head}", ha="center", va="bottom",
                fontsize=6, rotation=90)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CONTENT-MATCHING INDUCTION MODELS: 20-SEED CROSS-SEED ANALYSIS")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Train missing seeds
    # -----------------------------------------------------------------------
    seeds_to_train = []
    for seed in ALL_SEEDS:
        ckpt = CKPT_BASE / f"induction_seed{seed}" / "step_020000" / "model.safetensors"
        if ckpt.exists():
            print(f"[check] seed {seed}: checkpoint exists")
        else:
            print(f"[check] seed {seed}: MISSING — will train")
            seeds_to_train.append(seed)

    if seeds_to_train:
        train_missing_seeds(seeds_to_train, max_concurrent=2)
    else:
        print("All 20 checkpoints present. Skipping training.")

    # Verify all checkpoints exist
    for seed in ALL_SEEDS:
        ckpt = CKPT_BASE / f"induction_seed{seed}" / "step_020000" / "model.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint still missing after training: {ckpt}")
    print("\nAll 20 checkpoints verified.")

    # -----------------------------------------------------------------------
    # Step 2: Generate ONE fixed eval batch (seed=9999) used for ALL measurements
    # -----------------------------------------------------------------------
    print(f"\nGenerating fixed eval batch (eval_seed={EVAL_SEED}, batch_size={EVAL_BATCH_SIZE})...")
    eval_dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=EVAL_SEED)
    eval_inputs, eval_targets, eval_mask = eval_dataset.generate_batch(EVAL_BATCH_SIZE)
    mx.eval(eval_inputs, eval_targets, eval_mask)
    print(f"  Eval batch shape: {eval_inputs.shape}, masked positions: {int(np.array(eval_mask).sum())}")

    # -----------------------------------------------------------------------
    # Step 3: Per-seed ablation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-SEED ABLATION")
    print("=" * 70)

    per_seed_results = []
    effect_vectors = []

    for seed in ALL_SEEDS:
        print(f"\n[seed {seed}] Loading model...", flush=True)
        model = load_model(seed)

        # Baseline
        baseline_acc = eval_accuracy(model, eval_inputs, eval_targets, eval_mask)
        print(f"  [seed {seed}] baseline accuracy = {baseline_acc:.4f}")

        # Compute full 10-D effect vector
        print(f"  [seed {seed}] Computing effect vector (10 ablations)...", flush=True)
        vec = compute_effect_vector(model, eval_inputs, eval_targets, eval_mask)
        effect_vectors.append(vec)

        # Per-head ablation drops (first 8 elements of vector, interleaved with MLP)
        # Vector layout: L0H0,L0H1,L0H2,L0H3,L0_MLP, L1H0,L1H1,L1H2,L1H3,L1_MLP
        head_drops = {}
        for layer in range(MODEL_CONFIG.n_layers):
            for head in range(MODEL_CONFIG.n_heads):
                idx = layer * (MODEL_CONFIG.n_heads + 1) + head
                key = f"L{layer}H{head}"
                head_drops[key] = float(vec[idx])
                print(f"    ablate {key}: drop={vec[idx]:+.4f}")
            mlp_idx = layer * (MODEL_CONFIG.n_heads + 1) + MODEL_CONFIG.n_heads
            print(f"    ablate L{layer}_MLP: drop={vec[mlp_idx]:+.4f}")

        # Critical head: largest drop among attention heads only
        best_key = max(head_drops, key=lambda k: head_drops[k])
        best_layer = int(best_key[1])
        best_head = int(best_key[3])
        best_drop = head_drops[best_key]
        print(f"  [seed {seed}] CRITICAL HEAD: {best_key} (drop={best_drop:.4f})")

        per_seed_results.append({
            "seed": seed,
            "baseline_accuracy": float(baseline_acc),
            "critical_head_layer": best_layer,
            "critical_head_index": best_head,
            "critical_head_drop": float(best_drop),
            "head_drops": head_drops,
            "effect_vector": vec.tolist(),
        })

    # -----------------------------------------------------------------------
    # Step 4: Pairwise cosine similarity matrix
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PAIRWISE COSINE SIMILARITY")
    print("=" * 70)

    sim_matrix = np.zeros((N_SEEDS, N_SEEDS))
    pairwise_sims = []
    for i in range(N_SEEDS):
        for j in range(N_SEEDS):
            s = cosine_similarity(effect_vectors[i], effect_vectors[j])
            sim_matrix[i, j] = s
            if i < j:
                pairwise_sims.append(s)

    pairwise_sims = np.array(pairwise_sims)
    mean_sim = float(pairwise_sims.mean())
    sd_sim = float(pairwise_sims.std())
    ci_lo, ci_hi = bootstrap_ci(pairwise_sims, n_bootstrap=N_BOOTSTRAP)

    print(f"  Mean pairwise cosine similarity: {mean_sim:.4f}")
    print(f"  SD: {sd_sim:.4f}")
    print(f"  95% CI (bootstrap): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  N pairs: {len(pairwise_sims)}")

    # -----------------------------------------------------------------------
    # Step 5: Critical head identification and agreement
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CRITICAL HEAD IDENTIFICATION")
    print("=" * 70)

    critical_heads = [
        (r["critical_head_layer"], r["critical_head_index"]) for r in per_seed_results
    ]
    per_seed_drops = [r["critical_head_drop"] for r in per_seed_results]

    unique_slots = sorted(set(critical_heads))
    print(f"  Distinct (layer, head) slots used: {len(unique_slots)}")
    for slot in unique_slots:
        count = critical_heads.count(slot)
        print(f"    L{slot[0]}H{slot[1]}: {count}/20 seeds")

    # Pairwise head agreement
    pair_agreements = []
    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            pair_agreements.append(1.0 if critical_heads[i] == critical_heads[j] else 0.0)
    pair_agreements = np.array(pair_agreements)
    head_agreement_rate = float(pair_agreements.mean())
    agree_ci_lo, agree_ci_hi = bootstrap_ci(pair_agreements, n_bootstrap=N_BOOTSTRAP)

    print(f"\n  Pairwise head-agreement rate: {head_agreement_rate:.4f}")
    print(f"  95% CI (bootstrap): [{agree_ci_lo:.4f}, {agree_ci_hi:.4f}]")
    print(f"  Expected under uniform (1/8): {1/8:.4f}")

    # -----------------------------------------------------------------------
    # Step 6: Permutation test (null model)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERMUTATION TEST (NULL MODEL)")
    print("=" * 70)

    p_value = permutation_test_agreement(head_agreement_rate, critical_heads, N_PERMUTATIONS)
    print(f"  Observed agreement: {head_agreement_rate:.4f}")
    print(f"  Null (uniform 1/8): {1/8:.4f}")
    print(f"  p-value (permutation, n={N_PERMUTATIONS}): {p_value:.6f}")
    if p_value < 0.001:
        print(f"  *** p < 0.001: observed agreement is highly significant ***")
    elif p_value < 0.05:
        print(f"  ** p < 0.05: observed agreement is significant **")
    else:
        print(f"  p >= 0.05: not significant vs null")

    # -----------------------------------------------------------------------
    # Step 7: Layer-level distribution
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LAYER-LEVEL DISTRIBUTION")
    print("=" * 70)

    l0_count = sum(1 for layer, _ in critical_heads if layer == 0)
    l1_count = sum(1 for layer, _ in critical_heads if layer == 1)
    print(f"  Critical head in L0: {l0_count}/20 ({l0_count/N_SEEDS:.1%})")
    print(f"  Critical head in L1: {l1_count}/20 ({l1_count/N_SEEDS:.1%})")

    # -----------------------------------------------------------------------
    # Step 8: Save results
    # -----------------------------------------------------------------------
    results = {
        "experiment": "content_matching_20_seeds",
        "n_seeds": N_SEEDS,
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
        "per_seed": per_seed_results,
        "pairwise_similarity_matrix": sim_matrix.tolist(),
        "similarity_stats": {
            "mean": mean_sim,
            "sd": sd_sim,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "n_pairs": len(pairwise_sims),
        },
        "critical_head_stats": {
            "per_seed_critical_heads": [
                {"layer": l, "head": h} for l, h in critical_heads
            ],
            "distinct_slots_used": len(unique_slots),
            "slot_counts": {
                f"L{l}H{h}": critical_heads.count((l, h))
                for l in range(MODEL_CONFIG.n_layers)
                for h in range(MODEL_CONFIG.n_heads)
            },
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
            "L0_fraction": l0_count / N_SEEDS,
            "L1_fraction": l1_count / N_SEEDS,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Step 9: Plot
    # -----------------------------------------------------------------------
    make_plot(sim_matrix, critical_heads, per_seed_drops, per_seed_results, PLOT_PATH)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Models trained/loaded:         {N_SEEDS}")
    print(f"  Mean pairwise similarity:      {mean_sim:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  Head agreement rate:           {head_agreement_rate:.4f} (95% CI [{agree_ci_lo:.4f}, {agree_ci_hi:.4f}])")
    print(f"  Null (uniform 1/8):            {1/8:.4f}")
    print(f"  Permutation test p-value:      {p_value:.6f}")
    print(f"  Distinct critical head slots:  {len(unique_slots)}")
    print(f"  Layer distribution:            L0={l0_count}/20, L1={l1_count}/20")
    print(f"  Results JSON:                  {RESULTS_PATH}")
    print(f"  Plot:                          {PLOT_PATH}")
    print("=" * 70)
    print("DONE.")


if __name__ == "__main__":
    main()
