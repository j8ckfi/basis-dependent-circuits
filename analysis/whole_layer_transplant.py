"""
Whole-layer transplant experiment: test whether transplanting an entire attention
layer (all heads) or full block preserves function across seeds.

Background:
- Single-head transplants (circuit_transplant.py, basis_aligned_transplant.py) failed
  catastrophically: transplanting seed0 L0H3 into seed1 L0H1 gave accuracy ~0.09 vs
  baseline 0.964, and even Procrustes alignment did not help (R² negative).
- Question: does the layer-level or block-level transplant work?

Experiments:
1. Whole L0 attention transplant (all 4 heads: qkv_proj + out_proj)
2. Whole L0 block (attention + MLP + layer norms)
3. Whole L0 + ln_f (final layer norm)
4. Progressive transplants: embed, embed+L0attn, embed+L0block, all except L1
5. Full transplant (sanity check — should reproduce seed0 accuracy ~0.85)

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/whole_layer_transplant.py
"""

import sys
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

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

SEED0_CKPT = (
    "checkpoints/induction_content_match/induction_seed0/step_020000/model.safetensors"
)
SEED1_CKPT = (
    "checkpoints/induction_content_match/induction_seed1/step_020000/model.safetensors"
)

BASE_DIR = Path(__file__).parent.parent
RESULTS_PATH = Path(__file__).parent / "whole_layer_transplant_results.json"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------

def get_flat_weights(model: GPT) -> dict:
    """Return all model weights as a flat dict of numpy arrays."""
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    return {k: np.array(v) for k, v in flat.items()}


def build_model_from_weights(weights: dict, config: GPTConfig) -> GPT:
    """Instantiate a fresh GPT and load the given weight dict."""
    new_model = GPT(config)
    mlx_weights = [(k, mx.array(v)) for k, v in weights.items()]
    new_model.load_weights(mlx_weights)
    mx.eval(new_model.parameters())
    return new_model


def splice_keys(recipient_weights: dict, donor_weights: dict, keys: list) -> dict:
    """Return a copy of recipient_weights with the listed keys replaced by donor values."""
    result = dict(recipient_weights)
    for k in keys:
        if k not in donor_weights:
            raise KeyError(f"Key '{k}' not found in donor weights")
        result[k] = donor_weights[k].copy()
    return result


# ---------------------------------------------------------------------------
# Key sets for each transplant scope
# ---------------------------------------------------------------------------

def keys_l0_attention(layer: int = 0) -> list:
    """Keys for one attention block (qkv + out proj, no layer norm)."""
    return [
        f"blocks.{layer}.attn.qkv_proj.weight",
        f"blocks.{layer}.attn.out_proj.weight",
    ]


def keys_l0_full_block(layer: int = 0) -> list:
    """Keys for entire transformer block (ln1 + attn + ln2 + mlp)."""
    return [
        f"blocks.{layer}.ln1.weight",
        f"blocks.{layer}.attn.qkv_proj.weight",
        f"blocks.{layer}.attn.out_proj.weight",
        f"blocks.{layer}.ln2.weight",
        f"blocks.{layer}.mlp.up_proj.weight",
        f"blocks.{layer}.mlp.down_proj.weight",
    ]


def keys_ln_f() -> list:
    return ["ln_f.weight"]


def keys_embeddings() -> list:
    return ["wte.weight", "wpe.weight"]


def keys_l1_full_block(layer: int = 1) -> list:
    return keys_l0_full_block(layer)


# ---------------------------------------------------------------------------
# Verification: confirm actual weight change occurred
# ---------------------------------------------------------------------------

def verify_keys_changed(original: dict, modified: dict, keys: list) -> bool:
    """Return True if at least one key's weights differ."""
    for k in keys:
        if not np.allclose(original[k], modified[k]):
            return True
    return False


def verify_keys_unchanged(original: dict, modified: dict, keys: list) -> bool:
    """Return True if ALL listed keys are unchanged (sanity for non-transplanted parts)."""
    return all(np.allclose(original[k], modified[k]) for k in keys)


# ---------------------------------------------------------------------------
# Accuracy measurement (mirrors circuit_transplant.py)
# ---------------------------------------------------------------------------

def compute_induction_accuracy(
    model: GPT,
    dataset: InductionDataset,
    n_batches: int = 20,
    batch_size: int = 32,
) -> float:
    """Top-1 accuracy at induction-masked positions."""
    correct = 0
    total = 0

    for _ in range(n_batches):
        inputs, targets, mask = dataset.generate_batch(batch_size)
        logits = model(inputs)
        mx.eval(logits)

        logits_np = np.array(logits)
        targets_np = np.array(targets)
        mask_np = np.array(mask)

        preds = logits_np.argmax(axis=-1)
        at_mask = mask_np > 0.5

        correct += int((preds == targets_np)[at_mask].sum())
        total += int(at_mask.sum())

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Helper: run one transplant experiment
# ---------------------------------------------------------------------------

def run_transplant(
    label: str,
    description: str,
    hypothesis: str,
    recipient_weights: dict,
    donor_weights: dict,
    transplant_keys: list,
    config: GPTConfig,
    dataset: InductionDataset,
    acc_seed1_baseline: float,
    acc_seed0_baseline: float,
) -> dict:
    """Splice donor keys into recipient, build model, measure accuracy, return result dict."""
    print(f"\n--- {label} ---")
    print(f"  Transplanting {len(transplant_keys)} weight tensors:")
    for k in transplant_keys:
        print(f"    {k}")

    modified = splice_keys(recipient_weights, donor_weights, transplant_keys)
    changed = verify_keys_changed(recipient_weights, modified, transplant_keys)
    print(f"  Weight change verified: {changed}")

    model = build_model_from_weights(modified, config)
    acc = compute_induction_accuracy(model, dataset)
    delta_vs_seed1 = acc - acc_seed1_baseline
    delta_vs_seed0 = acc - acc_seed0_baseline

    print(f"  Accuracy: {acc:.4f}  "
          f"(delta vs seed1: {delta_vs_seed1:+.4f}, "
          f"delta vs seed0: {delta_vs_seed0:+.4f})")

    return {
        "label": label,
        "description": description,
        "hypothesis": hypothesis,
        "transplant_keys": transplant_keys,
        "n_keys_transplanted": len(transplant_keys),
        "accuracy": acc,
        "delta_vs_seed1_baseline": delta_vs_seed1,
        "delta_vs_seed0_baseline": delta_vs_seed0,
        "weight_change_verified": changed,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("WHOLE-LAYER TRANSPLANT EXPERIMENT")
    print("Testing whether layer-level transplants preserve function across seeds")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print(f"\nLoading seed 0 model from: {SEED0_CKPT}")
    model0 = load_model(str(BASE_DIR / SEED0_CKPT), MODEL_CONFIG)

    print(f"Loading seed 1 model from: {SEED1_CKPT}")
    model1 = load_model(str(BASE_DIR / SEED1_CKPT), MODEL_CONFIG)

    # Fixed-seed dataset for reproducible evaluation (same as prior experiments)
    dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)

    # Flatten both models' weights into numpy dicts
    w0 = get_flat_weights(model0)  # donor: seed 0
    w1 = get_flat_weights(model1)  # recipient: seed 1

    print(f"\nWeight keys: {sorted(w0.keys())}")

    results = {}

    # -----------------------------------------------------------------------
    # 1. Baselines
    # -----------------------------------------------------------------------
    print("\n--- Baselines ---")
    acc0 = compute_induction_accuracy(model0, dataset)
    acc1 = compute_induction_accuracy(model1, dataset)
    print(f"  Seed 0 baseline accuracy: {acc0:.4f}")
    print(f"  Seed 1 baseline accuracy: {acc1:.4f}")

    results["baselines"] = {"seed0": acc0, "seed1": acc1}

    # -----------------------------------------------------------------------
    # Reference: single-head transplant (known result from prior work)
    # -----------------------------------------------------------------------
    results["reference_single_head"] = {
        "label": "Single-head transplant (L0H3 -> L0H1)",
        "description": "seed0 L0H3 -> seed1 L0H1 (rows/cols of qkv/out only)",
        "accuracy": 0.082,
        "delta_vs_seed1_baseline": 0.082 - acc1,
        "source": "circuit_transplant.py and basis_aligned_transplant.py",
        "note": "Catastrophic failure; Procrustes alignment also failed (R2 < 0)",
    }

    experiment_results = []

    # -----------------------------------------------------------------------
    # Exp 1: Whole L0 attention (qkv_proj + out_proj, all heads at once)
    # -----------------------------------------------------------------------
    r = run_transplant(
        label="Whole L0 attention block (all 4 heads)",
        description="Replace seed1's L0 qkv_proj and out_proj with seed0's",
        hypothesis=(
            "If circuits are self-contained at the attention-layer level, "
            "accuracy should be preserved (near seed1 baseline)."
        ),
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=keys_l0_attention(layer=0),
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    experiment_results.append(r)

    # -----------------------------------------------------------------------
    # Exp 2: Whole L0 block (attention + MLP + layer norms)
    # -----------------------------------------------------------------------
    r = run_transplant(
        label="Whole L0 block (attn + MLP + LayerNorms)",
        description="Replace seed1's entire L0 transformer block with seed0's",
        hypothesis=(
            "If circuits require attention+MLP together, this should work "
            "but Exp1 might fail."
        ),
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=keys_l0_full_block(layer=0),
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    experiment_results.append(r)

    # -----------------------------------------------------------------------
    # Exp 3: Whole L0 block + ln_f
    # -----------------------------------------------------------------------
    r = run_transplant(
        label="Whole L0 block + final LayerNorm",
        description="Replace seed1's L0 block and ln_f with seed0's",
        hypothesis="Tests whether final LayerNorm mismatch matters.",
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=keys_l0_full_block(layer=0) + keys_ln_f(),
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    experiment_results.append(r)

    # -----------------------------------------------------------------------
    # Exp 4a: Progressive — embeddings only
    # -----------------------------------------------------------------------
    r = run_transplant(
        label="Embeddings only (wte + wpe)",
        description="Replace seed1's token and position embeddings with seed0's",
        hypothesis=(
            "Embeddings alone shouldn't change induction accuracy much if "
            "the downstream weights adapt to the embedding basis."
        ),
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=keys_embeddings(),
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    experiment_results.append(r)

    # -----------------------------------------------------------------------
    # Exp 4b: Progressive — embeddings + L0 attention
    # -----------------------------------------------------------------------
    r = run_transplant(
        label="Embeddings + L0 attention",
        description="Replace seed1's embeddings and L0 qkv/out with seed0's",
        hypothesis=(
            "Embeddings define the input basis for L0; transplanting both "
            "may allow L0 attention to work correctly in seed1's context."
        ),
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=keys_embeddings() + keys_l0_attention(layer=0),
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    experiment_results.append(r)

    # -----------------------------------------------------------------------
    # Exp 4c: Progressive — embeddings + L0 full block
    # -----------------------------------------------------------------------
    r = run_transplant(
        label="Embeddings + L0 full block",
        description="Replace seed1's embeddings and entire L0 block with seed0's",
        hypothesis=(
            "Adding MLP and LayerNorms may help if L0 block needs its own basis."
        ),
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=keys_embeddings() + keys_l0_full_block(layer=0),
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    experiment_results.append(r)

    # -----------------------------------------------------------------------
    # Exp 4d: Everything except L1 block
    # -----------------------------------------------------------------------
    all_except_l1 = (
        keys_embeddings()
        + keys_l0_full_block(layer=0)
        + keys_ln_f()
    )
    r = run_transplant(
        label="All except L1 block (embed + L0 + ln_f)",
        description="Replace everything from seed1 except the L1 transformer block",
        hypothesis=(
            "Tests whether L1 is the bottleneck: if this works, L1 alone "
            "can adapt; if not, L1 is incompatible with seed0's L0 output."
        ),
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=all_except_l1,
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    experiment_results.append(r)

    # -----------------------------------------------------------------------
    # Exp 5: Full transplant (sanity check — should reproduce seed0 accuracy)
    # -----------------------------------------------------------------------
    all_keys = list(w0.keys())
    r = run_transplant(
        label="Full transplant (all weights, sanity check)",
        description="Replace ALL seed1 weights with seed0's — should reproduce seed0 accuracy",
        hypothesis=(
            "This must reproduce seed0 baseline (~0.85). If it doesn't, "
            "the transplant machinery is broken."
        ),
        recipient_weights=w1,
        donor_weights=w0,
        transplant_keys=all_keys,
        config=MODEL_CONFIG,
        dataset=dataset,
        acc_seed1_baseline=acc1,
        acc_seed0_baseline=acc0,
    )
    # Extra check: verify sanity
    sanity_ok = abs(r["accuracy"] - acc0) < 0.05
    r["sanity_check_passed"] = sanity_ok
    print(f"  Sanity check (accuracy ≈ seed0 baseline): {'PASS' if sanity_ok else 'FAIL'}")
    experiment_results.append(r)

    results["experiments"] = experiment_results

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(f"  Seed 0 baseline: {acc0:.4f}")
    print(f"  Seed 1 baseline: {acc1:.4f}")
    print()
    header = f"  {'Transplant Scope':<45} {'Accuracy':>10} {'vs Seed1':>10}"
    print(header)
    print("  " + "-" * 65)

    # Include the known single-head result first
    ref = results["reference_single_head"]
    print(f"  {'Single-head L0H3->L0H1 (known)':<45} {ref['accuracy']:>10.4f} {ref['delta_vs_seed1_baseline']:>+10.4f}")

    for r in experiment_results:
        print(f"  {r['label']:<45} {r['accuracy']:>10.4f} {r['delta_vs_seed1_baseline']:>+10.4f}")

    # -----------------------------------------------------------------------
    # Analysis: find minimum transplant scope for preserved function
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    preservation_threshold = acc1 * 0.90  # within 10% of seed1 baseline

    # Determine which experiments "work" (preserve function)
    working = [r for r in experiment_results if r["accuracy"] >= preservation_threshold]
    failing = [r for r in experiment_results if r["accuracy"] < preservation_threshold]

    print(f"\n  Preservation threshold (90% of seed1 baseline): {preservation_threshold:.4f}")
    print(f"\n  Failing transplants ({len(failing)}):")
    for r in failing:
        print(f"    - {r['label']}: {r['accuracy']:.4f}")
    print(f"\n  Succeeding transplants ({len(working)}):")
    for r in working:
        print(f"    + {r['label']}: {r['accuracy']:.4f}")

    # Determine interpretation
    whole_l0_attn_acc = next(
        (r["accuracy"] for r in experiment_results if "all 4 heads" in r["label"]), None
    )
    whole_l0_block_acc = next(
        (r["accuracy"] for r in experiment_results if "attn + MLP" in r["label"]), None
    )
    embed_l0_attn_acc = next(
        (r["accuracy"] for r in experiment_results if r["label"] == "Embeddings + L0 attention"), None
    )
    embed_l0_block_acc = next(
        (r["accuracy"] for r in experiment_results if r["label"] == "Embeddings + L0 full block"), None
    )
    all_except_l1_acc = next(
        (r["accuracy"] for r in experiment_results if "except L1" in r["label"]), None
    )
    full_acc = next(
        (r["accuracy"] for r in experiment_results if "sanity" in r["label"]), None
    )

    def preserved(acc):
        return acc is not None and acc >= preservation_threshold

    if preserved(whole_l0_attn_acc):
        minimal_unit = "attention layer (all heads)"
        interpretation = (
            "LAYER IS A MINIMAL FUNCTIONAL UNIT. "
            "Transplanting all heads of L0 attention together preserves function, "
            "even though single-head transplants fail catastrophically. "
            "The induction heads operate as a coherent multi-head system: "
            "each head contributes to a shared residual stream subspace that the "
            "full layer projects onto collectively. Individual heads are not "
            "independently portable because they are only meaningful in the context "
            "of their co-trained siblings, but the layer as a whole is self-contained."
        )
    elif preserved(whole_l0_block_acc):
        minimal_unit = "full block (attention + MLP)"
        interpretation = (
            "BLOCK IS A MINIMAL FUNCTIONAL UNIT (attention alone insufficient). "
            "Transplanting the full L0 block (attention + MLP + LayerNorms) works, "
            "but the attention layer alone fails. This means the MLP is load-bearing: "
            "it likely performs basis adaptation or feature cleanup that makes the "
            "attention heads' outputs interpretable to the downstream network. "
            "The circuit requires the paired attention+MLP computation, not just attention."
        )
    elif preserved(embed_l0_attn_acc):
        minimal_unit = "embeddings + L0 attention"
        interpretation = (
            "INPUT BASIS MATTERS. "
            "Transplanting attention alone fails, but transplanting embeddings + attention "
            "succeeds. This means L0's attention heads are only functional in seed0's "
            "embedding basis — they read from a specific representation of tokens/positions "
            "that they were co-trained with. The residual stream basis (set by embeddings) "
            "is a prerequisite for the attention weights to function correctly."
        )
    elif preserved(embed_l0_block_acc):
        minimal_unit = "embeddings + L0 full block"
        interpretation = (
            "EMBEDDING BASIS + FULL BLOCK REQUIRED. "
            "Embeddings + attention fails; adding the MLP makes it work. "
            "L0's MLP is necessary to re-encode the residual stream into a form "
            "that seed1's L1 block can read. The block as a unit, operating in its "
            "native embedding basis, is portable — but the MLP is essential glue."
        )
    elif preserved(all_except_l1_acc):
        minimal_unit = "everything except L1 (embed + L0 + ln_f)"
        interpretation = (
            "L1 IS THE KEY ADAPTOR. "
            "Transplanting everything except L1 works: seed1's L1 block can successfully "
            "process seed0's L0 output when given seed0's embeddings. This means L1 is "
            "flexible enough to read either seed's L0 output, or that L0's output space "
            "is sufficiently similar across seeds that L1 can adapt. The bottleneck is "
            "NOT L1 but rather the L0-L1 interface when L0 is partially replaced."
        )
    elif full_acc is not None and abs(full_acc - acc0) < 0.05:
        minimal_unit = "full network (everything)"
        interpretation = (
            "CIRCUITS ARE GENUINELY SYSTEM-LEVEL. "
            "Only transplanting the complete network reproduces seed0 accuracy. "
            "No proper subset of weights is independently portable. "
            "The induction computation is an emergent property of the full weight "
            "configuration — there is no 'module' that can be surgically extracted "
            "and re-implanted. The circuit is truly system-level, consistent with "
            "the Procrustes alignment failure (R² < 0): the two networks don't share "
            "any common representational subspace that would allow partial reuse."
        )
    else:
        minimal_unit = "unknown (even full transplant failed)"
        interpretation = (
            "UNEXPECTED: even the full transplant sanity check failed. "
            "This suggests a bug in the transplant machinery or dataset evaluation. "
            "All results should be treated as suspect."
        )

    print(f"\n  Minimal functional unit: {minimal_unit}")
    print(f"\n  Interpretation: {interpretation}")

    results["analysis"] = {
        "preservation_threshold": preservation_threshold,
        "seed1_baseline": acc1,
        "seed0_baseline": acc0,
        "n_experiments_working": len(working),
        "n_experiments_failing": len(failing),
        "working_experiments": [r["label"] for r in working],
        "failing_experiments": [r["label"] for r in failing],
        "minimal_functional_unit": minimal_unit,
        "interpretation": interpretation,
        "summary_table": [
            {
                "scope": "Single-head L0H3->L0H1 (prior work)",
                "accuracy": 0.082,
                "delta_vs_seed1": round(0.082 - acc1, 4),
            }
        ] + [
            {
                "scope": r["label"],
                "accuracy": round(r["accuracy"], 4),
                "delta_vs_seed1": round(r["delta_vs_seed1_baseline"], 4),
            }
            for r in experiment_results
        ],
    }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
