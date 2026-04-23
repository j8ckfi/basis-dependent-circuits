"""
Experiment: Entropy Continuum

RESEARCH QUESTION:
Does task entropy continuously determine circuit architecture?
Hypothesis: task entropy determines whether the model builds copying circuits,
suppression circuits, or mixed circuits — as a continuous relationship.

DESIGN:
Train 11 small models on Markov chain tasks with varying entropy levels (0.0 to 1.0).
For each, profile all attention heads and record circuit architecture metrics.

Entropy levels map to Markov chain structure:
- 0.0: deterministic (each token has exactly one successor)
- 0.5: ~8 successors per token
- 1.0: uniform (all 64 tokens equally likely)

OUTPUT:
- experiments/entropy_plots/entropy_vs_head_roles.png
- experiments/entropy_plots/entropy_vs_scores.png
- experiments/entropy_plots/entropy_vs_attribution.png
- experiments/entropy_continuum_results.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from src.model import GPT, GPTConfig, create_model
from src.circuit_mapping import profile_all_heads, decompose_residual_stream


# ============================================================
# Markov Dataset
# ============================================================

class MarkovDataset:
    """Sequences drawn from a first-order Markov chain with tunable entropy.

    The transition matrix is built so that each token has a controlled number
    of successors, yielding a specific expected Shannon entropy per step.

    entropy_level=0.0 -> each token has exactly 1 successor (deterministic)
    entropy_level=0.5 -> each token has ~8 successors (equal prob)
    entropy_level=1.0 -> all 64 tokens equally likely as successor (uniform)

    The actual measured Shannon entropy of the transition matrix is computed
    and stored as self.measured_entropy.
    """

    def __init__(
        self,
        entropy_level: float,
        vocab_size: int = 64,
        seq_len: int = 128,
        seed: int = 42,
    ):
        self.entropy_level = entropy_level
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

        # Build transition matrix with controlled entropy
        self.transition_matrix = self._build_transition_matrix(entropy_level)
        self.measured_entropy = self._compute_entropy()

        # Build CDF for fast sampling
        self.cdf = np.cumsum(self.transition_matrix, axis=1)

    def _build_transition_matrix(self, entropy_level: float) -> np.ndarray:
        """Build a row-stochastic transition matrix.

        At entropy_level=0: each row has one entry = 1, rest = 0.
        At entropy_level=1: each row is uniform (1/vocab_size each).
        In between: smooth interpolation of number of active successors.

        Strategy: for each token, assign non-zero probability to k successors
        where k is interpolated. We use a power-law distribution among those
        k successors to keep the within-support entropy smooth.
        """
        V = self.vocab_size
        T = np.zeros((V, V), dtype=np.float64)

        # Number of successors: from 1 (at 0.0) to V (at 1.0)
        # Use exponential interpolation for a perceptually linear entropy curve
        k_min = 1.0
        k_max = float(V)
        # k = k_min * (k_max / k_min) ** entropy_level
        k_float = k_min * ((k_max / k_min) ** entropy_level)
        k = max(1, min(V, int(round(k_float))))

        # Concentration of probability mass: at low entropy, most mass on one token
        # At high entropy, mass is spread equally across the k successors
        # Use a Dirichlet-like concentration: alpha = large -> uniform, small -> peaked
        # alpha interpolates from 0.1 (very peaked) to large (uniform)
        alpha = 0.1 + entropy_level * (10.0 - 0.1)

        for i in range(V):
            # Pick k successors (deterministic from seed for reproducibility)
            # Use a fixed permutation per row based on token index
            rng_row = np.random.default_rng(seed=12345 + i)
            successors = rng_row.choice(V, size=k, replace=False)

            if k == 1:
                # Fully deterministic
                probs = np.array([1.0])
            else:
                # Sample from symmetric Dirichlet(alpha) to get within-support probs
                rng_prob = np.random.default_rng(seed=99999 + i)
                probs = rng_prob.dirichlet(alpha=np.full(k, alpha))

            for j, tok in enumerate(successors):
                T[i, tok] = probs[j]

        # Normalize rows (should already sum to 1 but guard for float errors)
        row_sums = T.sum(axis=1, keepdims=True)
        T = T / (row_sums + 1e-12)
        return T

    def _compute_entropy(self) -> float:
        """Compute mean Shannon entropy (nats) of rows of the transition matrix."""
        T = self.transition_matrix
        # Only include nonzero entries
        mask = T > 1e-12
        log_T = np.where(mask, np.log(T + 1e-12), 0.0)
        row_entropy = -(T * log_T).sum(axis=1)
        return float(row_entropy.mean())

    def generate_batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
        """Generate a batch of Markov chain sequences.

        Returns:
            inputs:  (B, T) int32 token sequences
            targets: (B, T) int32 next-token targets
            mask:    (B, T) float32 all-ones (no special positions to mask)
        """
        V = self.vocab_size
        L = self.seq_len
        inputs_np = np.zeros((batch_size, L), dtype=np.int32)
        targets_np = np.zeros((batch_size, L), dtype=np.int32)

        for b in range(batch_size):
            # Random start token
            tok = int(self.rng.integers(0, V))
            seq = np.empty(L + 1, dtype=np.int32)
            seq[0] = tok

            for t in range(1, L + 1):
                # Sample next token from transition distribution
                u = self.rng.random()
                next_tok = int(np.searchsorted(self.cdf[tok], u))
                next_tok = min(next_tok, V - 1)
                seq[t] = next_tok
                tok = next_tok

            inputs_np[b] = seq[:L]
            targets_np[b] = seq[1:L + 1]

        # Dummy mask: all ones (all positions are targets)
        mask_np = np.ones((batch_size, L), dtype=np.float32)

        return mx.array(inputs_np), mx.array(targets_np), mx.array(mask_np)

    def iter_batches(self, batch_size: int, n_batches: int):
        for _ in range(n_batches):
            yield self.generate_batch(batch_size)


# ============================================================
# Training
# ============================================================

def get_lr(step: int, n_steps: int, lr: float, warmup: int = 200) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, n_steps - warmup)
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_model(dataset: MarkovDataset, entropy_level: float) -> GPT:
    """Train a small model on the given Markov dataset.

    Returns the trained model.
    """
    cfg = GPTConfig(
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_ff=512,
        vocab_size=64,
        ctx_len=128,
        dropout=0.0,
    )

    n_steps = 5000
    batch_size = 64
    lr = 1e-3

    model = create_model(cfg, seed=0)

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.1)

    def compute_loss(model, inputs, targets, mask):
        logits = model(inputs)
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        mask_flat = mask.reshape(B * T)
        per_tok = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (per_tok * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    t0 = time.time()
    data_iter = dataset.iter_batches(batch_size, n_steps)

    for step, (inputs, targets, mask) in enumerate(data_iter):
        current_lr = get_lr(step, n_steps, lr)
        optimizer.learning_rate = current_lr

        loss, grads = loss_and_grad(model, inputs, targets, mask)

        # Elementwise gradient clipping.
        import mlx.utils
        grads = mlx.utils.tree_map(
            lambda g: mx.clip(g, -1.0, 1.0),
            grads,
        )

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"    step {step:5d}/{n_steps} | loss={loss.item():.4f} | lr={current_lr:.2e} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s")
    return model


# ============================================================
# Circuit Measurement
# ============================================================

def measure_circuit(model: GPT, dataset: MarkovDataset) -> dict:
    """Profile all heads and compute logit attribution aggregates.

    Returns a dict with:
      - head_roles: counts of each role
      - head_fracs: fractions of copy/suppression/positional/other
      - mean_copy_score, mean_suppression_score, mean_pos_dep
      - attn_logit_sum, mlp_logit_sum: total attribution from attn vs MLP
    """
    profiles = profile_all_heads(model, dataset, n_samples=128)

    n_heads = len(profiles)
    role_counts = {}
    copy_scores = []
    suppression_scores = []
    pos_deps = []

    for p in profiles:
        role = p.role
        role_counts[role] = role_counts.get(role, 0) + 1
        copy_scores.append(p.copy_score)
        suppression_scores.append(p.suppression_score)
        pos_deps.append(p.positional_score)

    # Categorize into broad buckets: copy, suppression, positional, other
    copy_roles = {"copy", "induction"}
    suppress_roles = {"suppression"}
    positional_roles = {"positional", "previous_token", "bos_attention", "focused"}

    n_copy = sum(v for k, v in role_counts.items() if k in copy_roles)
    n_suppress = sum(v for k, v in role_counts.items() if k in suppress_roles)
    n_positional = sum(v for k, v in role_counts.items() if k in positional_roles)
    n_other = n_heads - n_copy - n_suppress - n_positional

    # Logit attribution: sum over attn components vs MLP components
    # Use a batch of data; average attribution across positions
    batch = dataset.generate_batch(32)
    inputs = batch[0]
    contributions = decompose_residual_stream(model, inputs)
    mx.eval(*list(contributions.values()))

    # Unembedding weight for logit projection
    unembed = np.array(model.wte.weight)  # (vocab, d_model)

    attn_logit_total = 0.0
    mlp_logit_total = 0.0

    for name, contrib in contributions.items():
        if name in ("embed", "final_resid"):
            continue
        contrib_np = np.array(contrib)  # (B, T, d_model)
        # Project each position onto its own max-logit token direction (mean)
        # Use mean over batch and positions of L2 norm as proxy for attribution magnitude
        # Simpler: compute mean absolute dot product with embedding matrix
        # contrib_np: (B, T, d_model), unembed: (V, d_model)
        # logit_effects: (B, T, V) = contrib_np @ unembed.T
        logit_effects = np.tensordot(contrib_np, unembed, axes=[[2], [1]])  # (B, T, V)
        mean_abs_logit = float(np.abs(logit_effects).mean())

        if "_attn" in name:
            attn_logit_total += mean_abs_logit
        elif "_mlp" in name:
            mlp_logit_total += mean_abs_logit

    return {
        "n_heads": n_heads,
        "role_counts": role_counts,
        "frac_copy": n_copy / n_heads,
        "frac_suppression": n_suppress / n_heads,
        "frac_positional": n_positional / n_heads,
        "frac_other": n_other / n_heads,
        "mean_copy_score": float(np.mean(copy_scores)),
        "mean_suppression_score": float(np.mean(suppression_scores)),
        "mean_pos_dep": float(np.mean(pos_deps)),
        "attn_logit_total": attn_logit_total,
        "mlp_logit_total": mlp_logit_total,
    }


# ============================================================
# Plotting
# ============================================================

def make_plots(results: list[dict], plot_dir: Path):
    """Generate the three plots from collected results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    entropies = [r["measured_entropy"] for r in results]
    entropy_levels = [r["entropy_level"] for r in results]

    # Sort by measured entropy for clean line plots
    order = np.argsort(entropies)
    entropies = [entropies[i] for i in order]
    results = [results[i] for i in order]

    frac_copy = [r["circuit"]["frac_copy"] for r in results]
    frac_supp = [r["circuit"]["frac_suppression"] for r in results]
    frac_pos = [r["circuit"]["frac_positional"] for r in results]
    frac_other = [r["circuit"]["frac_other"] for r in results]
    mean_copy = [r["circuit"]["mean_copy_score"] for r in results]
    mean_supp = [r["circuit"]["mean_suppression_score"] for r in results]
    mean_pos = [r["circuit"]["mean_pos_dep"] for r in results]
    attn_attr = [r["circuit"]["attn_logit_total"] for r in results]
    mlp_attr = [r["circuit"]["mlp_logit_total"] for r in results]

    # ----- Plot 1: Stacked area — head role fractions vs entropy -----
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = entropies

    ax.stackplot(
        xs,
        frac_copy, frac_supp, frac_pos, frac_other,
        labels=["Copy/Induction", "Suppression", "Positional", "Other"],
        colors=["#2196F3", "#F44336", "#4CAF50", "#9E9E9E"],
        alpha=0.85,
    )
    ax.set_xlabel("Task Entropy (nats)", fontsize=12)
    ax.set_ylabel("Fraction of Attention Heads", fontsize=12)
    ax.set_title("Head Role Distribution vs Task Entropy", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Annotate entropy level on top axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(entropies[::2])
    ax2.set_xticklabels([f"{r['entropy_level']:.1f}" for r in results[::2]], fontsize=8)
    ax2.set_xlabel("Entropy Level (parameter)", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_dir / "entropy_vs_head_roles.png", dpi=150)
    plt.close()
    print(f"  Saved {plot_dir / 'entropy_vs_head_roles.png'}")

    # ----- Plot 2: Lines — mean scores vs entropy -----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, mean_copy, "o-", color="#2196F3", label="Mean Copy Score", linewidth=2, markersize=6)
    ax.plot(xs, mean_supp, "s-", color="#F44336", label="Mean Suppression Score", linewidth=2, markersize=6)
    ax.plot(xs, mean_pos, "^-", color="#4CAF50", label="Mean Positional Dep", linewidth=2, markersize=6)
    ax.set_xlabel("Task Entropy (nats)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Head Scores vs Task Entropy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(min(xs), max(xs))
    plt.tight_layout()
    plt.savefig(plot_dir / "entropy_vs_scores.png", dpi=150)
    plt.close()
    print(f"  Saved {plot_dir / 'entropy_vs_scores.png'}")

    # ----- Plot 3: Attention vs MLP logit attribution -----
    fig, ax = plt.subplots(figsize=(9, 5))
    total = [a + m for a, m in zip(attn_attr, mlp_attr)]
    attn_frac = [a / (t + 1e-12) for a, t in zip(attn_attr, total)]
    mlp_frac = [m / (t + 1e-12) for m, t in zip(mlp_attr, total)]

    ax.stackplot(
        xs,
        attn_frac, mlp_frac,
        labels=["Attention Attribution", "MLP Attribution"],
        colors=["#FF9800", "#9C27B0"],
        alpha=0.85,
    )
    ax.set_xlabel("Task Entropy (nats)", fontsize=12)
    ax.set_ylabel("Fraction of Total Logit Attribution", fontsize=12)
    ax.set_title("Attention vs MLP Logit Attribution vs Entropy", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "entropy_vs_attribution.png", dpi=150)
    plt.close()
    print(f"  Saved {plot_dir / 'entropy_vs_attribution.png'}")


# ============================================================
# Main Experiment
# ============================================================

ENTROPY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

VOCAB_SIZE = 64
SEQ_LEN = 128


def run_experiment():
    """Run the full entropy continuum experiment."""
    print("=" * 60)
    print("ENTROPY CONTINUUM EXPERIMENT")
    print("=" * 60)
    print(f"Entropy levels: {ENTROPY_LEVELS}")
    print(f"Model: 4L4H d_model=128, vocab={VOCAB_SIZE}, seq_len={SEQ_LEN}")
    print(f"Training: 5000 steps each")
    print()

    results = []
    plot_dir = Path(__file__).parent / "entropy_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for idx, entropy_level in enumerate(ENTROPY_LEVELS):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(ENTROPY_LEVELS)}] Entropy level = {entropy_level:.1f}")
        print(f"{'='*60}")

        # Build dataset
        dataset = MarkovDataset(
            entropy_level=entropy_level,
            vocab_size=VOCAB_SIZE,
            seq_len=SEQ_LEN,
            seed=42,
        )
        print(f"  Dataset: measured_entropy = {dataset.measured_entropy:.4f} nats")
        print(f"           (max possible = {np.log(VOCAB_SIZE):.4f} nats)")

        # Train model
        print(f"  Training...")
        t0 = time.time()
        model = train_model(dataset, entropy_level)
        train_time = time.time() - t0

        # Profile circuits
        print(f"  Profiling circuits...")
        circuit_metrics = measure_circuit(model, dataset)

        print(f"  Results:")
        print(f"    frac_copy={circuit_metrics['frac_copy']:.3f}  "
              f"frac_suppress={circuit_metrics['frac_suppression']:.3f}  "
              f"frac_positional={circuit_metrics['frac_positional']:.3f}  "
              f"frac_other={circuit_metrics['frac_other']:.3f}")
        print(f"    mean_copy_score={circuit_metrics['mean_copy_score']:.4f}  "
              f"mean_suppression_score={circuit_metrics['mean_suppression_score']:.4f}  "
              f"mean_pos_dep={circuit_metrics['mean_pos_dep']:.4f}")
        print(f"    attn_attr={circuit_metrics['attn_logit_total']:.4f}  "
              f"mlp_attr={circuit_metrics['mlp_logit_total']:.4f}")

        results.append({
            "entropy_level": entropy_level,
            "measured_entropy": dataset.measured_entropy,
            "train_time_s": train_time,
            "circuit": circuit_metrics,
        })

    # Save results
    output_path = Path(__file__).parent / "entropy_continuum_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate plots
    print("\nGenerating plots...")
    make_plots(results, plot_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Entropy':>10} {'Measured':>10} {'Copy%':>8} {'Supp%':>8} {'Pos%':>8} {'Other%':>8}")
    print("-" * 60)
    for r in results:
        c = r["circuit"]
        print(f"  {r['entropy_level']:>6.1f}   {r['measured_entropy']:>8.3f}  "
              f"{c['frac_copy']*100:>6.1f}%  "
              f"{c['frac_suppression']*100:>6.1f}%  "
              f"{c['frac_positional']*100:>6.1f}%  "
              f"{c['frac_other']*100:>6.1f}%")

    print("\nDone.")
    return results


if __name__ == "__main__":
    run_experiment()
