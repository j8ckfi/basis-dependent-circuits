"""
Visualization for seed divergence experiment.

Produces:
1. Circuit effect heatmaps over training (per seed)
2. Cross-seed similarity matrix at convergence
3. Crystallization timeline (when does each seed's circuit snap into place?)
4. Attention head specialization over training
5. Component assignment comparison (do seeds use same layers?)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path


def load_experiment(experiment_dir: str) -> tuple[dict, dict, dict]:
    """Load experiment config, trajectories, and analysis."""
    exp_dir = Path(experiment_dir)

    with open(exp_dir / "experiment_config.json") as f:
        config = json.load(f)

    trajectories = {}
    traj_path = exp_dir / "trajectories.json"
    if traj_path.exists():
        with open(traj_path) as f:
            trajectories = json.load(f)

    analysis = {}
    analysis_path = exp_dir / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)

    return config, trajectories, analysis


def plot_circuit_development(trajectories: dict, save_path: str = None):
    """Plot how circuit effects evolve over training for each seed.

    Each subplot shows one seed: x-axis = training step, y-axis = component,
    color = causal effect magnitude.
    """
    seeds = list(trajectories.keys())
    n_seeds = len(seeds)

    fig, axes = plt.subplots(
        min(n_seeds, 5), max(1, (n_seeds + 4) // 5),
        figsize=(4 * max(1, (n_seeds + 4) // 5), 3 * min(n_seeds, 5)),
        squeeze=False,
    )

    for idx, seed_key in enumerate(seeds[:20]):
        row, col = idx % 5, idx // 5
        ax = axes[row, col]

        trajectory = trajectories[seed_key]
        steps = [t["step"] for t in trajectory]

        # Build effect matrix: (n_steps, n_components)
        effects = np.array([np.array(t["effects"]).flatten() for t in trajectory])

        # Plot as heatmap
        im = ax.imshow(
            effects.T,
            aspect="auto",
            cmap="viridis",
            extent=[steps[0], steps[-1], 0, effects.shape[1]],
            origin="lower",
        )

        # Label components
        n_layers = len(trajectory[0]["effects"])
        labels = []
        for l in range(n_layers):
            labels.extend([f"L{l}_attn", f"L{l}_mlp"])

        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(seed_key, fontsize=8)
        ax.set_xlabel("Step", fontsize=7)

    plt.suptitle("Circuit Development Over Training (Causal Effect Magnitude)", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_similarity_matrix(analysis: dict, save_path: str = None):
    """Plot cross-seed circuit similarity at convergence."""
    sim_matrix = np.array(analysis["similarity_matrix"])
    seed_keys = analysis["seed_keys"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(seed_keys)))
    ax.set_yticks(range(len(seed_keys)))
    ax.set_xticklabels(seed_keys, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(seed_keys, fontsize=7)

    plt.colorbar(im, label="Cosine Similarity")
    ax.set_title(f"Cross-Seed Circuit Similarity at Convergence\n(Mean pairwise = {analysis['mean_pairwise_similarity']:.4f})")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_crystallization_timeline(trajectories: dict, save_path: str = None):
    """Plot when circuits crystallize across seeds.

    Shows: for each seed, the correlation between current circuit and final circuit
    over training. The "crystallization point" is where this crosses a threshold.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for seed_key, trajectory in trajectories.items():
        if len(trajectory) < 3:
            continue

        steps = [t["step"] for t in trajectory]
        effects_over_time = [np.array(t["effects"]).flatten() for t in trajectory]
        final = effects_over_time[-1]

        # Compute similarity to final at each step
        similarities = []
        for eff in effects_over_time:
            norm_prod = np.linalg.norm(eff) * np.linalg.norm(final)
            if norm_prod < 1e-10:
                similarities.append(0.0)
            else:
                similarities.append(np.dot(eff, final) / norm_prod)

        ax.plot(steps, similarities, alpha=0.6, linewidth=1.5, label=seed_key)

    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="Crystallization threshold (0.9)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Similarity to Final Circuit")
    ax.set_title("Circuit Crystallization Over Training")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_attention_entropy(trajectories: dict, save_path: str = None):
    """Plot attention head entropy over training.

    Decreasing entropy = heads are specializing (attending to specific positions).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # One plot per layer
    seeds = list(trajectories.keys())
    first_traj = trajectories[seeds[0]]
    n_layers = len(first_traj[0]["head_entropies"])

    for layer_idx in range(min(n_layers, 4)):
        ax = axes[layer_idx // 2, layer_idx % 2]

        for seed_key, trajectory in trajectories.items():
            steps = [t["step"] for t in trajectory]
            # Average entropy across heads in this layer
            entropies = [np.mean(t["head_entropies"][layer_idx]) for t in trajectory]
            ax.plot(steps, entropies, alpha=0.5, linewidth=1)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Attention Entropy")
        ax.set_title(f"Layer {layer_idx} - Head Specialization")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Attention Head Entropy Over Training\n(Lower = more specialized)", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_dominant_component_histogram(analysis: dict, save_path: str = None):
    """Show which layer/component is dominant across seeds."""
    dominant = analysis.get("dominant_components", {})

    # Count occurrences of each (layer, component) pair
    from collections import Counter
    assignments = Counter()
    for seed, info in dominant.items():
        key = f"L{info['layer']}_{info['component']}"
        assignments[key] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = sorted(assignments.keys())
    counts = [assignments[k] for k in keys]

    ax.bar(keys, counts, color="steelblue")
    ax.set_xlabel("Component")
    ax.set_ylabel("Number of Seeds")
    ax.set_title("Dominant Circuit Component Across Seeds\n(Where is the strongest causal effect?)")
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def generate_all_plots(experiment_dir: str):
    """Generate all visualization plots for an experiment."""
    config, trajectories, analysis = load_experiment(experiment_dir)

    plots_dir = Path(experiment_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)

    if trajectories:
        print("Generating circuit development plot...")
        plot_circuit_development(trajectories, str(plots_dir / "circuit_development.png"))

        print("Generating crystallization timeline...")
        plot_crystallization_timeline(trajectories, str(plots_dir / "crystallization.png"))

        print("Generating attention entropy plot...")
        plot_attention_entropy(trajectories, str(plots_dir / "attention_entropy.png"))

    if analysis and "similarity_matrix" in analysis:
        print("Generating similarity matrix...")
        plot_similarity_matrix(analysis, str(plots_dir / "similarity_matrix.png"))

        print("Generating dominant component histogram...")
        plot_dominant_component_histogram(analysis, str(plots_dir / "dominant_components.png"))

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    import sys
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/seed_divergence"
    generate_all_plots(exp_dir)
