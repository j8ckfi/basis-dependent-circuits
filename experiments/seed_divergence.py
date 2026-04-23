"""
Experiment 1: Seed Divergence

RESEARCH QUESTION:
Given identical data and architecture, do different random initializations
converge to the SAME circuits (functional universality) or DIFFERENT circuits
(contingent development)?

DESIGN:
- N independent training runs (default: 20 seeds)
- Same data, same architecture, same hyperparameters
- Only model initialization seed varies
- Dense checkpointing (every 50 steps)
- At each checkpoint: extract circuit via activation patching

MEASUREMENTS:
1. Circuit effect vectors at each checkpoint (which components are causally important)
2. When circuits "crystallize" (sharp increase in causal importance of specific components)
3. Cross-seed correlation of circuit structure at convergence
4. Whether different seeds produce the same circuit in different components (same function, different location)

HYPOTHESES:
H1 (Strong universality): All seeds converge to essentially the same circuit
H2 (Functional universality): Same computation but different component assignment
H3 (Weak universality): Same gross structure (e.g., "induction in early layers") but different details
H4 (Contingent): Qualitatively different circuits depending on initialization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.model import GPTConfig
from src.train import TrainConfig, train
from src.circuits import extract_circuit_trajectory
from src.data import InductionDataset


# ============================================================
# Experiment Configuration
# ============================================================

EXPERIMENT_CONFIG = {
    "name": "seed_divergence_induction",
    "description": "Test circuit universality across random initializations on induction task",

    # Seeds to test
    "n_seeds": 20,
    "seeds": list(range(20)),

    # Training
    "task": "induction",
    "n_steps": 2000,
    "checkpoint_every": 25,  # Dense checkpointing — phase transition is steps 100-200
    "batch_size": 64,
    "learning_rate": 3e-4,

    # Model (small enough for many runs, big enough for circuits)
    "model": {
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 128,
        "d_ff": 512,
    },

    # Data (FIXED across all runs)
    "data_seed": 42,

    # Analysis
    "circuit_extraction_samples": 32,

    # Output
    "output_dir": "checkpoints/seed_divergence",
}


def run_single_seed(seed: int, config: dict) -> str:
    """Train one model from one seed. Returns run directory path."""
    model_config = GPTConfig(
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_model=config["model"]["d_model"],
        d_ff=config["model"]["d_ff"],
        vocab_size=50,
        ctx_len=64,
    )

    train_config = TrainConfig(
        model_config=model_config,
        seed=seed,
        data_seed=config["data_seed"],
        task=config["task"],
        n_steps=config["n_steps"],
        checkpoint_every=config["checkpoint_every"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        output_dir=config["output_dir"],
    )

    run_dir = train(train_config)
    return str(run_dir)


def extract_all_circuits(config: dict) -> dict:
    """Extract circuit trajectories from all completed runs."""
    output_dir = Path(config["output_dir"])
    dataset = InductionDataset(vocab_size=50, seq_len=64, seed=config["data_seed"])

    model_config = GPTConfig(
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_model=config["model"]["d_model"],
        d_ff=config["model"]["d_ff"],
        vocab_size=50,
        ctx_len=64,
    )

    all_trajectories = {}
    for seed in config["seeds"]:
        run_dir = output_dir / f"induction_seed{seed}"
        if not run_dir.exists():
            print(f"  Skipping seed {seed} (not found)")
            continue

        print(f"\nExtracting circuits for seed {seed}...")
        trajectory = extract_circuit_trajectory(
            run_dir, dataset, model_config,
            n_samples=config["circuit_extraction_samples"],
        )
        all_trajectories[f"seed_{seed}"] = trajectory

    return all_trajectories


def analyze_divergence(trajectories: dict) -> dict:
    """Analyze cross-seed circuit divergence.

    Key metrics:
    1. Crystallization time: when does each seed's circuit "snap into place"?
    2. Final circuit similarity: how similar are converged circuits across seeds?
    3. Component assignment: do seeds use the same layers/heads?
    """
    seeds = list(trajectories.keys())
    if len(seeds) < 2:
        return {"error": "Need at least 2 seeds to analyze divergence"}

    # Extract final circuit effect vectors
    final_effects = {}
    for seed_key, trajectory in trajectories.items():
        if trajectory:
            final = trajectory[-1]
            # Flatten effect matrix to vector
            effects_flat = np.array(final["effects"]).flatten()
            final_effects[seed_key] = effects_flat

    # Pairwise cosine similarity of final circuits
    seed_keys = list(final_effects.keys())
    n = len(seed_keys)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v1 = final_effects[seed_keys[i]]
            v2 = final_effects[seed_keys[j]]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            similarity_matrix[i, j] = cos_sim

    # Crystallization detection: when does the circuit effect vector stabilize?
    crystallization_steps = {}
    for seed_key, trajectory in trajectories.items():
        if len(trajectory) < 3:
            continue
        effects_over_time = [np.array(t["effects"]).flatten() for t in trajectory]
        steps = [t["step"] for t in trajectory]

        # Find first step where correlation with final > 0.9
        final = effects_over_time[-1]
        for idx, (eff, step) in enumerate(zip(effects_over_time, steps)):
            cos_sim = np.dot(eff, final) / (np.linalg.norm(eff) * np.linalg.norm(final) + 1e-10)
            if cos_sim > 0.9:
                crystallization_steps[seed_key] = step
                break

    # Component dominance: which layer/component is strongest per seed?
    dominant_components = {}
    for seed_key, effects_vec in final_effects.items():
        # Reshape back to (n_layers, 2)
        n_layers = len(effects_vec) // 2
        effects_2d = effects_vec.reshape(n_layers, 2)
        max_idx = np.unravel_index(np.argmax(effects_2d), effects_2d.shape)
        dominant_components[seed_key] = {
            "layer": int(max_idx[0]),
            "component": "attn" if max_idx[1] == 0 else "mlp",
            "effect": float(effects_2d[max_idx]),
        }

    return {
        "similarity_matrix": similarity_matrix.tolist(),
        "seed_keys": seed_keys,
        "mean_pairwise_similarity": float(
            similarity_matrix[np.triu_indices(n, k=1)].mean()
        ),
        "crystallization_steps": crystallization_steps,
        "dominant_components": dominant_components,
    }


def run_experiment(n_seeds: int = None, quick: bool = False):
    """Run the full seed divergence experiment."""
    config = EXPERIMENT_CONFIG.copy()

    if n_seeds is not None:
        config["n_seeds"] = n_seeds
        config["seeds"] = list(range(n_seeds))

    if quick:
        config["n_steps"] = 1000
        config["checkpoint_every"] = 100
        config["circuit_extraction_samples"] = 8

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("SEED DIVERGENCE EXPERIMENT")
    print("=" * 60)
    print(f"Seeds: {config['seeds']}")
    print(f"Steps per run: {config['n_steps']}")
    print(f"Checkpoints: every {config['checkpoint_every']} steps")
    print(f"Model: {config['model']}")
    print()

    # Phase 1: Train all seeds
    print("PHASE 1: Training models...")
    print("-" * 40)
    for seed in config["seeds"]:
        print(f"\n>>> Seed {seed}/{config['seeds'][-1]}")
        run_single_seed(seed, config)

    # Phase 2: Extract circuits
    print("\n\nPHASE 2: Extracting circuits...")
    print("-" * 40)
    trajectories = extract_all_circuits(config)

    # Save trajectories
    with open(output_dir / "trajectories.json", "w") as f:
        json.dump(trajectories, f)

    # Phase 3: Analyze divergence
    print("\n\nPHASE 3: Analyzing divergence...")
    print("-" * 40)
    analysis = analyze_divergence(trajectories)

    # Save analysis
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean pairwise circuit similarity: {analysis.get('mean_pairwise_similarity', 'N/A'):.4f}")
    print(f"  (1.0 = identical circuits, 0.0 = orthogonal)")
    print(f"\nCrystallization steps:")
    for seed, step in analysis.get("crystallization_steps", {}).items():
        print(f"  {seed}: step {step}")
    print(f"\nDominant components:")
    for seed, comp in analysis.get("dominant_components", {}).items():
        print(f"  {seed}: layer {comp['layer']} {comp['component']} (effect={comp['effect']:.4f})")

    # Interpretation
    mean_sim = analysis.get("mean_pairwise_similarity", 0)
    if mean_sim > 0.95:
        print("\n>> INTERPRETATION: Strong universality (H1) — circuits converge regardless of seed")
    elif mean_sim > 0.8:
        print("\n>> INTERPRETATION: Functional universality (H2/H3) — similar but not identical circuits")
    elif mean_sim > 0.5:
        print("\n>> INTERPRETATION: Weak universality — gross structure shared, details diverge")
    else:
        print("\n>> INTERPRETATION: Contingent development (H4) — circuits depend on initialization")

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed Divergence Experiment")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds (default 5 for quick test)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer steps)")
    args = parser.parse_args()

    run_experiment(n_seeds=args.n_seeds, quick=args.quick)
