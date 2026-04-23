"""
Circuit mapping: understanding WHAT each component computes.

This moves beyond activation patching ("does this matter?") to
characterizing the actual computation:

1. QK analysis: What does each head attend TO? (attention pattern characterization)
2. OV analysis: What does each head DO with what it attends to? (output transformation)
3. Information flow: How do components compose? (residual stream decomposition)
4. Feature probing: What information is represented at each layer?

The goal is to produce a human-readable description of each head's function.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .model import GPT, GPTConfig


# ============================================================
# Attention Pattern Characterization
# ============================================================

@dataclass
class HeadProfile:
    """Functional profile of a single attention head."""
    layer: int
    head: int

    # Attention pattern metrics
    prev_token_score: float    # How much does it attend to position i-1?
    same_token_score: float    # How much does it attend to the same token?
    first_token_score: float   # How much does it attend to position 0?
    positional_score: float    # How position-dependent is attention (vs content-dependent)?
    entropy: float             # How concentrated is attention?

    # OV circuit metrics
    copy_score: float          # Does the OV matrix copy the input identity?
    suppression_score: float   # Does the OV matrix suppress the input identity?

    # Inferred role
    role: str = ""

    def describe(self) -> str:
        parts = [f"L{self.layer}H{self.head}"]
        if self.role:
            parts.append(f"[{self.role}]")
        parts.append(f"prev_tok={self.prev_token_score:.3f}")
        parts.append(f"same_tok={self.same_token_score:.3f}")
        parts.append(f"pos_dep={self.positional_score:.3f}")
        parts.append(f"entropy={self.entropy:.3f}")
        parts.append(f"copy={self.copy_score:.3f}")
        parts.append(f"suppress={self.suppression_score:.3f}")
        return " | ".join(parts)


def classify_head(profile: HeadProfile) -> str:
    """Infer a head's functional role from its profile."""
    # Previous token head: strongly attends to i-1
    if profile.prev_token_score > 0.3:
        return "previous_token"

    # Induction head: content-dependent attention + copies input
    if profile.positional_score < 0.3 and profile.copy_score > 0.2:
        return "induction"

    # Copy/suppression heads
    if profile.copy_score > 0.3:
        return "copy"
    if profile.suppression_score > 0.3:
        return "suppression"

    # BOS/first-token head
    if profile.first_token_score > 0.4:
        return "bos_attention"

    # Positional head (attends based on position, not content)
    if profile.positional_score > 0.6:
        return "positional"

    # Low entropy = focused on specific position
    if profile.entropy < 1.5:
        return "focused"

    return "unclassified"


def profile_all_heads(
    model: GPT,
    dataset,
    n_samples: int = 128,
) -> list[HeadProfile]:
    """Profile every attention head in the model.

    Runs multiple forward passes and computes statistics about
    what each head attends to and what it does with the information.
    """
    # Generate data
    batch = dataset.generate_batch(n_samples)
    inputs = batch[0]
    B, T = inputs.shape

    # Forward pass to get attention patterns
    _ = model(inputs)
    patterns = model.get_attention_patterns()

    n_layers = len(patterns)
    n_heads = patterns[0].shape[1]

    profiles = []

    for layer_idx in range(n_layers):
        attn = np.array(patterns[layer_idx])  # (B, n_heads, T, T)

        # Get OV matrix for this layer
        block = model.blocks[layer_idx]
        qkv_weight = np.array(block.attn.qkv_proj.weight)  # (3*d_model, d_model)
        out_weight = np.array(block.attn.out_proj.weight)   # (d_model, d_model)
        d_model = model.config.d_model
        d_head = model.config.d_head

        # Extract V and O projections per head
        # QKV weight layout: [Q_all | K_all | V_all] where each is (d_model, d_model)
        v_weight = qkv_weight[2 * d_model:, :]  # (d_model, d_model) — all V heads

        for head_idx in range(n_heads):
            head_attn = attn[:, head_idx, :, :]  # (B, T, T)

            # === Attention pattern metrics ===

            # Previous token score: mean attention to position i-1
            prev_tok = 0.0
            for pos in range(1, T):
                prev_tok += head_attn[:, pos, pos - 1].mean()
            prev_tok /= (T - 1)

            # Same token score: mean attention to same position
            same_tok = np.mean([head_attn[:, pos, pos].mean() for pos in range(T)])

            # First token score: mean attention to position 0
            first_tok = head_attn[:, :, 0].mean()

            # Positional vs content dependence:
            # If attention is purely positional, the pattern should be the same
            # across all inputs. Measure variance across batch.
            pattern_variance = head_attn.var(axis=0).mean()  # Low = positional
            pattern_mean = head_attn.mean(axis=0).mean()
            positional_score = 1.0 - min(1.0, pattern_variance / (pattern_mean + 1e-10))

            # Entropy
            p = np.clip(head_attn, 1e-10, 1.0)
            entropy = -(p * np.log(p)).sum(axis=-1).mean()

            # === OV circuit metrics ===
            # Extract this head's V and O projections
            v_head = v_weight[head_idx * d_head:(head_idx + 1) * d_head, :]  # (d_head, d_model)
            o_head = out_weight[:, head_idx * d_head:(head_idx + 1) * d_head]  # (d_model, d_head)

            # OV matrix = O @ V (maps input residual stream to output residual stream)
            ov_matrix = o_head @ v_head  # (d_model, d_model)

            # Through the embedding: E^T @ OV @ E maps tokens to tokens
            embed_weight = np.array(model.wte.weight)  # (vocab, d_model)
            token_ov = embed_weight @ ov_matrix @ embed_weight.T  # (vocab, vocab)

            # Copy score: how much does the diagonal dominate? (OV copies token identity)
            diag = np.diag(token_ov)
            off_diag_mean = (token_ov.sum() - diag.sum()) / (token_ov.size - len(diag))
            copy_score = float((diag.mean() - off_diag_mean) / (np.abs(token_ov).mean() + 1e-10))

            # Suppression score: negative diagonal means it suppresses the input token
            suppression_score = float(max(0, -diag.mean()) / (np.abs(token_ov).mean() + 1e-10))

            profile = HeadProfile(
                layer=layer_idx,
                head=head_idx,
                prev_token_score=float(prev_tok),
                same_token_score=float(same_tok),
                first_token_score=float(first_tok),
                positional_score=float(positional_score),
                entropy=float(entropy),
                copy_score=float(copy_score),
                suppression_score=float(suppression_score),
            )
            profile.role = classify_head(profile)
            profiles.append(profile)

    return profiles


# ============================================================
# Residual Stream Decomposition
# ============================================================

def decompose_residual_stream(
    model: GPT,
    inputs: mx.array,
) -> dict[str, mx.array]:
    """Decompose the residual stream into per-component contributions.

    At any point in the network, the residual stream is a sum of:
    - The embedding
    - Each attention layer's output
    - Each MLP layer's output

    This function computes each component's contribution separately,
    which lets us see what information each component is writing.
    """
    B, T = inputs.shape
    pos = mx.arange(T)
    embed = model.wte(inputs) + model.wpe(pos)

    contributions = {"embed": embed}
    x = embed

    for i, block in enumerate(model.blocks):
        # Attention contribution
        attn_input = block.ln1(x)
        attn_out = block.attn(attn_input)
        contributions[f"L{i}_attn"] = attn_out
        x = x + attn_out

        # MLP contribution
        mlp_input = block.ln2(x)
        mlp_out = block.mlp(mlp_input)
        contributions[f"L{i}_mlp"] = mlp_out
        x = x + mlp_out

    contributions["final_resid"] = x

    return contributions


def logit_attribution(
    model: GPT,
    inputs: mx.array,
    target_pos: int,
    target_token: int,
) -> dict[str, float]:
    """Attribute the logit of a specific token to each component.

    Uses the logit lens: project each component's contribution through
    the unembedding to see how much it pushes toward the target token.

    logit(target) = sum over components of: component @ unembed[target]
    """
    contributions = decompose_residual_stream(model, inputs)
    # Unembedding direction for target token (weight-tied)
    unembed_dir = model.wte.weight[target_token]  # (d_model,)
    # Need to account for final layer norm
    # Approximate: apply LN to final residual, then project
    # More precise: project each component through LN (but LN is nonlinear)
    # We use the direct logit attribution (DLA) approximation

    attribution = {}
    for name, contrib in contributions.items():
        if name == "final_resid":
            continue
        # Project contribution at target_pos onto unembedding direction
        component_at_pos = contrib[0, target_pos]  # (d_model,)
        logit_effect = (component_at_pos * unembed_dir).sum().item()
        attribution[name] = logit_effect

    return attribution


# ============================================================
# Information Flow Between Components
# ============================================================

def trace_information_flow(
    model: GPT,
    dataset,
    n_samples: int = 64,
) -> dict:
    """Trace how information flows between layers.

    Measures: if we zero out layer i's output, how much does layer j's
    behavior change? This gives us a directed graph of information flow.

    Returns: (n_components, n_components) matrix of dependency strengths.
    """
    batch = dataset.generate_batch(n_samples)
    inputs, targets, mask = batch

    n_layers = len(model.blocks)
    components = []
    for i in range(n_layers):
        components.extend([f"L{i}_attn", f"L{i}_mlp"])

    n_comp = len(components)

    # Baseline activations
    baseline_acts = get_all_component_outputs(model, inputs)

    # For each source component, zero it and measure effect on all downstream
    flow_matrix = np.zeros((n_comp, n_comp))

    for src_idx, src_name in enumerate(components):
        # Run model with source component zeroed
        perturbed_acts = get_all_component_outputs_with_ablation(
            model, inputs, ablate_component=src_name
        )

        for dst_idx, dst_name in enumerate(components):
            if dst_idx <= src_idx:
                continue  # Only downstream
            # How much did dst change?
            baseline_val = np.array(baseline_acts[dst_name])
            perturbed_val = np.array(perturbed_acts[dst_name])
            change = np.abs(baseline_val - perturbed_val).mean()
            flow_matrix[src_idx, dst_idx] = change

    return {
        "flow_matrix": flow_matrix.tolist(),
        "components": components,
    }


def get_all_component_outputs(model: GPT, inputs: mx.array) -> dict[str, mx.array]:
    """Get the output of every component (attn, mlp) during a forward pass."""
    B, T = inputs.shape
    pos = mx.arange(T)
    x = model.wte(inputs) + model.wpe(pos)

    outputs = {}
    for i, block in enumerate(model.blocks):
        attn_input = block.ln1(x)
        attn_out = block.attn(attn_input)
        outputs[f"L{i}_attn"] = attn_out
        x = x + attn_out

        mlp_input = block.ln2(x)
        mlp_out = block.mlp(mlp_input)
        outputs[f"L{i}_mlp"] = mlp_out
        x = x + mlp_out

    return outputs


def get_all_component_outputs_with_ablation(
    model: GPT,
    inputs: mx.array,
    ablate_component: str,
) -> dict[str, mx.array]:
    """Forward pass with one component zeroed out."""
    B, T = inputs.shape
    pos = mx.arange(T)
    x = model.wte(inputs) + model.wpe(pos)

    outputs = {}
    for i, block in enumerate(model.blocks):
        attn_input = block.ln1(x)
        attn_out = block.attn(attn_input)

        if f"L{i}_attn" == ablate_component:
            attn_out = mx.zeros_like(attn_out)

        outputs[f"L{i}_attn"] = attn_out
        x = x + attn_out

        mlp_input = block.ln2(x)
        mlp_out = block.mlp(mlp_input)

        if f"L{i}_mlp" == ablate_component:
            mlp_out = mx.zeros_like(mlp_out)

        outputs[f"L{i}_mlp"] = mlp_out
        x = x + mlp_out

    return outputs


# ============================================================
# Full Circuit Map
# ============================================================

def map_circuit(
    model: GPT,
    dataset,
    n_samples: int = 128,
    verbose: bool = True,
) -> dict:
    """Produce a complete circuit map of the model.

    Combines:
    - Head profiling (what each head does)
    - Information flow (how components connect)
    - Logit attribution (what drives the output)

    Returns a structured description of the model's circuit.
    """
    if verbose:
        print("Profiling attention heads...")
    profiles = profile_all_heads(model, dataset, n_samples)

    if verbose:
        print("Tracing information flow...")
    flow = trace_information_flow(model, dataset, n_samples=min(n_samples, 64))

    # Logit attribution on sample inputs
    if verbose:
        print("Computing logit attribution...")
    batch = dataset.generate_batch(min(n_samples, 32))
    inputs, targets, mask = batch

    mask_np = np.array(mask)
    attr_samples = []

    for b in range(min(8, inputs.shape[0])):
        masked_positions = np.where(mask_np[b] > 0)[0]
        if len(masked_positions) == 0:
            continue
        pos = int(masked_positions[0])
        token = int(targets[b, pos].item())
        single_input = inputs[b:b+1]
        attr = logit_attribution(model, single_input, pos, token)
        attr_samples.append(attr)

    # Average attributions
    avg_attr = {}
    if attr_samples:
        for key in attr_samples[0]:
            avg_attr[key] = np.mean([a[key] for a in attr_samples])

    circuit_map = {
        "heads": [
            {
                "layer": p.layer,
                "head": p.head,
                "role": p.role,
                "prev_token_score": p.prev_token_score,
                "same_token_score": p.same_token_score,
                "first_token_score": p.first_token_score,
                "positional_score": p.positional_score,
                "entropy": p.entropy,
                "copy_score": p.copy_score,
                "suppression_score": p.suppression_score,
            }
            for p in profiles
        ],
        "information_flow": flow,
        "logit_attribution": {k: float(v) for k, v in avg_attr.items()},
    }

    if verbose:
        print("\n" + "=" * 60)
        print("CIRCUIT MAP")
        print("=" * 60)

        print("\nHead Profiles:")
        for p in profiles:
            print(f"  {p.describe()}")

        print("\nLogit Attribution (average):")
        sorted_attr = sorted(avg_attr.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_attr:
            bar = "+" * int(abs(val) * 5) if val > 0 else "-" * int(abs(val) * 5)
            print(f"  {name:12s}: {val:+.4f} {bar}")

        print("\nStrongest Information Flows:")
        flow_matrix = np.array(flow["flow_matrix"])
        components = flow["components"]
        # Top 5 flows
        flat_indices = np.argsort(flow_matrix.flatten())[::-1][:10]
        for flat_idx in flat_indices:
            src, dst = np.unravel_index(flat_idx, flow_matrix.shape)
            val = flow_matrix[src, dst]
            if val > 0:
                print(f"  {components[src]} -> {components[dst]}: {val:.4f}")

    return circuit_map


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from .model import create_model
    from .data import InductionDataset

    # Map circuits of a trained induction model
    ckpt_dir = Path("checkpoints/seed_divergence/induction_seed0")
    if ckpt_dir.exists():
        # Load latest checkpoint
        checkpoints = sorted(ckpt_dir.glob("step_*"))
        if checkpoints:
            config = GPTConfig(n_layers=4, n_heads=4, d_model=128, d_ff=512, vocab_size=50, ctx_len=64)
            model = GPT(config)
            model.load_weights(str(checkpoints[-1] / "model.safetensors"))
            mx.eval(model.parameters())

            dataset = InductionDataset(vocab_size=50, seq_len=64, seed=42)
            circuit_map = map_circuit(model, dataset, n_samples=128)
    else:
        print("No trained model found. Run training first.")
