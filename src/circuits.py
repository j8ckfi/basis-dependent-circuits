"""
Circuit extraction via activation patching.

Core idea: run the model on a clean input and a corrupted input.
Patch activations from one run into the other at specific (layer, position) sites.
Measure how much the output changes -> causal importance of that site.

For developmental interpretability, we run this at every checkpoint to
observe which components become causally important and WHEN.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .model import GPT, GPTConfig, create_model


@dataclass
class PatchResult:
    """Result of activation patching at one site."""
    layer: int
    component: str  # "attn" or "mlp" or "resid"
    effect: float  # Change in logit/loss when this site is patched


def get_activations_with_hooks(
    model: GPT,
    inputs: mx.array,
) -> dict[str, mx.array]:
    """Run forward pass and capture all intermediate activations.

    Returns dict mapping "layer_N_attn", "layer_N_mlp", "layer_N_resid" -> activations
    """
    activations = {}
    B, T = inputs.shape
    pos = mx.arange(T)
    x = model.wte(inputs) + model.wpe(pos)
    activations["embed"] = x

    for i, block in enumerate(model.blocks):
        # Pre-attention residual
        attn_input = block.ln1(x)
        attn_out = block.attn(attn_input)
        activations[f"layer_{i}_attn"] = attn_out
        x = x + attn_out

        # Pre-MLP residual
        mlp_input = block.ln2(x)
        mlp_out = block.mlp(mlp_input)
        activations[f"layer_{i}_mlp"] = mlp_out
        x = x + mlp_out

        activations[f"layer_{i}_resid"] = x

    return activations


def run_with_patch(
    model: GPT,
    inputs: mx.array,
    patch_activations: dict[str, mx.array],
    patch_site: str,
) -> mx.array:
    """Run forward pass but substitute activation at patch_site with patch_activations.

    This implements activation patching: run on clean input but inject
    the corrupted activation at one specific site.
    """
    B, T = inputs.shape
    pos = mx.arange(T)
    x = model.wte(inputs) + model.wpe(pos)

    if patch_site == "embed":
        x = patch_activations["embed"]

    for i, block in enumerate(model.blocks):
        attn_input = block.ln1(x)
        attn_out = block.attn(attn_input)

        if patch_site == f"layer_{i}_attn":
            attn_out = patch_activations[patch_site]

        x = x + attn_out

        mlp_input = block.ln2(x)
        mlp_out = block.mlp(mlp_input)

        if patch_site == f"layer_{i}_mlp":
            mlp_out = patch_activations[patch_site]

        x = x + mlp_out

    x = model.ln_f(x)
    logits = x @ model.wte.weight.T
    return logits


def activation_patch_all_sites(
    model: GPT,
    clean_input: mx.array,
    corrupted_input: mx.array,
    target_pos: int,
    target_token: int,
) -> list[PatchResult]:
    """Run activation patching across all sites.

    Measures: how much does patching corrupted activations into the clean run
    change the logit of the target token at target_pos?

    This gives us the causal importance of each component.
    """
    # Get clean logits (baseline)
    clean_logits = model(clean_input)
    clean_logit_val = clean_logits[0, target_pos, target_token].item()

    # Get corrupted activations
    corrupted_acts = get_activations_with_hooks(model, corrupted_input)

    results = []
    n_layers = len(model.blocks)

    for layer in range(n_layers):
        for component in ["attn", "mlp"]:
            site = f"layer_{layer}_{component}"
            patched_logits = run_with_patch(
                model, clean_input, corrupted_acts, site
            )
            patched_val = patched_logits[0, target_pos, target_token].item()
            effect = clean_logit_val - patched_val  # Positive = this site helps

            results.append(PatchResult(
                layer=layer,
                component=component,
                effect=effect,
            ))

    return results


def compute_circuit_snapshot(
    model: GPT,
    dataset,
    n_samples: int = 32,
) -> dict:
    """Compute a circuit snapshot: average activation patching effects across samples.

    This is what we compute at each checkpoint to track circuit development.

    Returns dict with:
    - "effects": 2D array (n_layers, 2) where [:,0]=attn, [:,1]=mlp
    - "attention_entropy": per-head entropy (are heads specializing?)
    - "task_accuracy": accuracy on induction/IOI positions
    """
    n_layers = len(model.blocks)
    effects = np.zeros((n_layers, 2))  # attn, mlp

    for sample_idx in range(n_samples):
        # Generate clean and corrupted pairs
        clean_batch = dataset.generate_batch(1)
        corrupted_batch = dataset.generate_batch(1)

        inputs, targets, mask = clean_batch
        corrupted_inputs = corrupted_batch[0]

        # Find a masked position (where the task matters)
        mask_np = np.array(mask[0])
        masked_positions = np.where(mask_np > 0)[0]
        if len(masked_positions) == 0:
            continue

        target_pos = int(masked_positions[0])
        target_token = int(targets[0, target_pos].item())

        # Patch all sites
        results = activation_patch_all_sites(
            model, inputs, corrupted_inputs, target_pos, target_token
        )

        for r in results:
            col = 0 if r.component == "attn" else 1
            effects[r.layer, col] += abs(r.effect)

    effects /= max(n_samples, 1)

    # Attention entropy (are heads becoming specialized?)
    # Run a forward pass and check attention patterns
    dummy_batch = dataset.generate_batch(8)
    _ = model(dummy_batch[0])
    patterns = model.get_attention_patterns()

    head_entropies = []
    for layer_patterns in patterns:
        # Shape: (B, n_heads, T, T)
        # Compute entropy of attention distribution per head
        p = np.array(layer_patterns)
        # Avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -(p * np.log(p)).sum(axis=-1).mean(axis=(0, 2))  # (n_heads,)
        head_entropies.append(entropy.tolist())

    return {
        "effects": effects.tolist(),
        "head_entropies": head_entropies,
    }


def extract_circuit_trajectory(
    run_dir: Path,
    dataset,
    model_config: GPTConfig,
    n_samples: int = 16,
) -> list[dict]:
    """Load checkpoints from a run and compute circuit snapshots for each.

    This is the core analysis: how does the circuit evolve over training?
    """
    checkpoints = sorted(run_dir.glob("step_*"))
    trajectory = []

    for ckpt_dir in checkpoints:
        step = int(ckpt_dir.name.split("_")[1])
        weights_path = ckpt_dir / "model.safetensors"
        if not weights_path.exists():
            continue

        # Load model at this checkpoint
        model = GPT(model_config)
        model.load_weights(str(weights_path))
        mx.eval(model.parameters())

        # Compute circuit snapshot
        snapshot = compute_circuit_snapshot(model, dataset, n_samples=n_samples)
        snapshot["step"] = step

        # Load training metrics if available
        metrics_path = ckpt_dir / "metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                snapshot["train_metrics"] = json.load(f)

        trajectory.append(snapshot)
        print(f"  Extracted circuit at step {step}")

    return trajectory


if __name__ == "__main__":
    from .data import InductionDataset

    config = GPTConfig(vocab_size=50, ctx_len=64)
    model = create_model(config, seed=42)

    dataset = InductionDataset(vocab_size=50, seq_len=64, seed=0)

    print("Computing circuit snapshot...")
    snapshot = compute_circuit_snapshot(model, dataset, n_samples=8)
    print(f"Effects shape: {len(snapshot['effects'])} layers x 2 components")
    print(f"Head entropies: {len(snapshot['head_entropies'])} layers")
    for i, effects in enumerate(snapshot['effects']):
        print(f"  Layer {i}: attn={effects[0]:.4f}, mlp={effects[1]:.4f}")
