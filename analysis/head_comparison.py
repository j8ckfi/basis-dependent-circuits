"""
Deeper analysis: per-head circuit comparison across seeds.

The key question at the attention-head level:
- Do the SAME heads become induction heads across seeds?
- Or does the model find different heads to serve the same function?

This distinguishes H1 (strong universality) from H2 (functional universality):
- H1: Same heads, same roles, every time
- H2: Same computation, different heads assigned to it
"""

import json
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import InductionDataset


def measure_induction_score_per_head(
    model: GPT,
    dataset: InductionDataset,
    n_samples: int = 64,
) -> np.ndarray:
    """Measure how much each attention head exhibits induction behavior.

    An induction head attends to the token that followed the previous occurrence
    of the current token. In our repeated-sequence setup:
    - Position i in the second half should attend to position i in the first half
    - Specifically, it should attend to position (i - half_len) which has the same token

    Returns: (n_layers, n_heads) array of induction scores (0-1)
    """
    half_len = dataset.half_len
    batch = dataset.generate_batch(n_samples)
    inputs = batch[0]

    # Forward pass to get attention patterns
    _ = model(inputs)
    patterns = model.get_attention_patterns()

    n_layers = len(patterns)
    n_heads = patterns[0].shape[1]
    induction_scores = np.zeros((n_layers, n_heads))

    for layer_idx, layer_patterns in enumerate(patterns):
        # layer_patterns shape: (B, n_heads, T, T)
        attn = np.array(layer_patterns)

        for head_idx in range(n_heads):
            # For positions in the second half (half_len to seq_len-1),
            # check how much attention goes to the "induction position":
            # position i in second half should attend to position (i - half_len + 1)
            # because in a repeated sequence ABCABC, when at second A (pos 3),
            # to predict B, the induction head attends to first A (pos 0) and copies what came after

            # Actually for a previous-token + induction head composition:
            # At position i (second half), induction head attends to position (i - half_len)
            # which is the SAME token in the first half

            induction_attention = 0.0
            count = 0
            for pos in range(half_len, dataset.seq_len):
                source_pos = pos - half_len  # Where induction should attend
                # Average over batch
                induction_attention += attn[:, head_idx, pos, source_pos].mean()
                count += 1

            induction_scores[layer_idx, head_idx] = induction_attention / max(count, 1)

    return induction_scores


def measure_previous_token_score_per_head(
    model: GPT,
    dataset: InductionDataset,
    n_samples: int = 64,
) -> np.ndarray:
    """Measure how much each head acts as a previous-token head.

    A previous-token head attends to position (i-1) regardless of content.

    Returns: (n_layers, n_heads) array of prev-token scores (0-1)
    """
    batch = dataset.generate_batch(n_samples)
    inputs = batch[0]

    _ = model(inputs)
    patterns = model.get_attention_patterns()

    n_layers = len(patterns)
    n_heads = patterns[0].shape[1]
    prev_token_scores = np.zeros((n_layers, n_heads))

    for layer_idx, layer_patterns in enumerate(patterns):
        attn = np.array(layer_patterns)

        for head_idx in range(n_heads):
            # Check attention to position (i-1) for each position i
            prev_attn = 0.0
            count = 0
            for pos in range(1, dataset.seq_len):
                prev_attn += attn[:, head_idx, pos, pos - 1].mean()
                count += 1

            prev_token_scores[layer_idx, head_idx] = prev_attn / max(count, 1)

    return prev_token_scores


def compare_head_roles_across_seeds(
    experiment_dir: str,
    step: Optional[int] = None,
) -> dict:
    """Load final checkpoints for all seeds and compare head roles.

    Returns analysis of which heads take on which roles across seeds.
    """
    exp_dir = Path(experiment_dir)

    with open(exp_dir / "experiment_config.json") as f:
        config = json.load(f)

    model_config = GPTConfig(
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_model=config["model"]["d_model"],
        d_ff=config["model"]["d_ff"],
        vocab_size=50,
        ctx_len=64,
    )

    dataset = InductionDataset(vocab_size=50, seq_len=64, seed=config["data_seed"])

    results = {}

    for seed in config["seeds"]:
        run_dir = exp_dir / f"induction_seed{seed}"
        if not run_dir.exists():
            continue

        # Find the checkpoint to analyze
        if step is not None:
            ckpt_dir = run_dir / f"step_{step:06d}"
        else:
            # Use latest checkpoint
            checkpoints = sorted(run_dir.glob("step_*"))
            if not checkpoints:
                continue
            ckpt_dir = checkpoints[-1]

        weights_path = ckpt_dir / "model.safetensors"
        if not weights_path.exists():
            continue

        model = GPT(model_config)
        model.load_weights(str(weights_path))
        mx.eval(model.parameters())

        induction_scores = measure_induction_score_per_head(model, dataset)
        prev_token_scores = measure_previous_token_score_per_head(model, dataset)

        results[f"seed_{seed}"] = {
            "induction_scores": induction_scores.tolist(),
            "prev_token_scores": prev_token_scores.tolist(),
            "best_induction_head": {
                "layer": int(np.unravel_index(np.argmax(induction_scores), induction_scores.shape)[0]),
                "head": int(np.unravel_index(np.argmax(induction_scores), induction_scores.shape)[1]),
                "score": float(np.max(induction_scores)),
            },
            "best_prev_token_head": {
                "layer": int(np.unravel_index(np.argmax(prev_token_scores), prev_token_scores.shape)[0]),
                "head": int(np.unravel_index(np.argmax(prev_token_scores), prev_token_scores.shape)[1]),
                "score": float(np.max(prev_token_scores)),
            },
        }

    # Cross-seed analysis
    if len(results) >= 2:
        # Do all seeds assign the induction role to the same head?
        induction_assignments = [
            (r["best_induction_head"]["layer"], r["best_induction_head"]["head"])
            for r in results.values()
        ]
        prev_token_assignments = [
            (r["best_prev_token_head"]["layer"], r["best_prev_token_head"]["head"])
            for r in results.values()
        ]

        from collections import Counter
        induction_counter = Counter(induction_assignments)
        prev_token_counter = Counter(prev_token_assignments)

        results["_summary"] = {
            "induction_head_assignments": {
                str(k): v for k, v in induction_counter.most_common()
            },
            "prev_token_head_assignments": {
                str(k): v for k, v in prev_token_counter.most_common()
            },
            "induction_head_agreement": induction_counter.most_common(1)[0][1] / len(results) if induction_counter else 0,
            "prev_token_head_agreement": prev_token_counter.most_common(1)[0][1] / len(results) if prev_token_counter else 0,
        }

        print("\n=== HEAD ROLE COMPARISON ===")
        print(f"\nInduction head assignments across {len(results)} seeds:")
        for (layer, head), count in induction_counter.most_common():
            print(f"  Layer {layer}, Head {head}: {count} seeds ({100*count/len(results):.0f}%)")

        print(f"\nPrevious-token head assignments:")
        for (layer, head), count in prev_token_counter.most_common():
            print(f"  Layer {layer}, Head {head}: {count} seeds ({100*count/len(results):.0f}%)")

        agreement = results["_summary"]["induction_head_agreement"]
        if agreement > 0.8:
            print(f"\n>> Strong agreement ({agreement:.0%}): Same heads become induction heads (H1)")
        elif agreement > 0.5:
            print(f"\n>> Moderate agreement ({agreement:.0%}): Preferred heads but some variation (H2/H3)")
        else:
            print(f"\n>> Low agreement ({agreement:.0%}): Different heads take on the role (H2 — functional universality)")

    return results


if __name__ == "__main__":
    import sys
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/seed_divergence"
    results = compare_head_roles_across_seeds(exp_dir)

    # Save
    output_path = Path(exp_dir) / "head_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
