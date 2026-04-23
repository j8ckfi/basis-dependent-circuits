"""
Rigorous Depth Scaling Analysis

Fixes three problems in experiments/depth_scaling.py:

1. Fix 1: Proper logit attribution
   - Original used mean L2 norm of residual contributions as "attribution"
   - This is NOT logit attribution; it ignores the unembedding direction
   - We compute actual direct logit attribution (DLA):
     attr_c = <contrib_c, W_U[t]> where t is the target token
   - Applied with final LayerNorm approximation (project through LN of final residual)

2. Fix 2: Statistical honesty
   - n=4 (four depths) is too few to claim a "scaling law"
   - Report slope, 95% CI via bootstrap, OLS p-value, R^2
   - Recommend "scaling trend" rather than "scaling law" unless p<<0.01

3. Fix 3: Eval fairness
   - All models use SAME fixed eval batch (seed=9999)
   - Report baseline eval losses on this fixed batch

Outputs:
- analysis/plots/depth_scaling_rigorous.png
- analysis/depth_scaling_rigorous_results.json
"""

import sys
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import TextDataset
from src.circuit_mapping import decompose_residual_stream, profile_all_heads

# ============================================================
# Constants (must match training in experiments/depth_scaling.py)
# ============================================================

DEPTHS = [2, 4, 6, 8]
N_HEADS = 8
D_MODEL = 512
D_FF = 2048
VOCAB_SIZE = 65
CTX_LEN = 128
CHECKPOINT_BASE = Path("checkpoints/depth_scaling")
DATA_DIR = "data"
FIXED_EVAL_SEED = 9999
EVAL_BATCH_SIZE = 64
N_ATTR_SAMPLES = 256   # number of (batch, position) samples for attribution

PLOT_DIR = Path("analysis/plots")
RESULTS_FILE = Path("analysis/depth_scaling_rigorous_results.json")


# ============================================================
# Model loading
# ============================================================

def load_depth_model(n_layers: int) -> GPT:
    model_path = CHECKPOINT_BASE / f"depth_{n_layers}" / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint at {model_path}. Run experiments/depth_scaling.py first.")
    config = GPTConfig(
        n_layers=n_layers,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
        ctx_len=CTX_LEN,
        dropout=0.0,
    )
    model = GPT(config)
    model.load_weights(str(model_path))
    mx.eval(model.parameters())
    return model


# ============================================================
# Fixed eval batch (same across all models)
# ============================================================

def make_fixed_eval_dataset() -> TextDataset:
    return TextDataset(data_dir=DATA_DIR, seq_len=CTX_LEN, seed=FIXED_EVAL_SEED)


def compute_eval_loss(model: GPT, dataset: TextDataset, n_batches: int = 20) -> float:
    losses = []
    for batch in dataset.iter_batches(32, n_batches):
        inputs, targets = batch
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        losses.append(float(loss.item()))
    return float(np.mean(losses))


# ============================================================
# Direct Logit Attribution (DLA)
# ============================================================

def compute_logit_attribution(
    model: GPT,
    inputs: mx.array,
    targets: mx.array,
    n_samples: int = N_ATTR_SAMPLES,
) -> dict:
    """Compute actual direct logit attribution for each component.

    For each sampled (batch, position) pair:
      - Get the target token t at position+1 (i.e., targets[b, pos])
      - Decompose residual stream into per-component contributions
      - Apply final LayerNorm using the scale from the full forward pass
        (linear approximation: scale = LN(final_resid).std / final_resid.std)
      - Project each component's contribution onto unembedding direction W_U[t]:
        attr_c = <LN_scaled(contrib_c), W_U[t]>

    Returns dict mapping component name -> mean signed logit attribution
    and mean absolute logit attribution, plus per-layer breakdowns.

    The linear approximation for LN is:
      LN(x + delta) ≈ LN(x) + delta / std(x) * gamma
    We use the full residual's LN scale factor as a rough linearization.
    This is the standard DLA approximation used in interpretability work.
    """
    B, T = inputs.shape
    n_layers = model.config.n_layers

    # Decompose residual stream
    contributions = decompose_residual_stream(model, inputs)
    final_resid = np.array(contributions["final_resid"])  # (B, T, d_model)

    # Compute LN scale factor: per (B, T) normalization
    # LN(x) = (x - mean) / sqrt(var + eps) * gamma
    # We need per-token scale = gamma / sqrt(var + eps)
    # Using the final_residual's statistics as the linearization point
    ln_gamma = np.array(model.ln_f.weight)  # (d_model,)
    eps = 1e-5
    resid_mean = final_resid.mean(axis=-1, keepdims=True)  # (B, T, 1)
    resid_var = final_resid.var(axis=-1, keepdims=True)    # (B, T, 1)
    ln_scale = ln_gamma / np.sqrt(resid_var + eps)          # (B, T, d_model)

    # Unembedding matrix W_U = wte.weight  shape (vocab, d_model)
    W_U = np.array(model.wte.weight)  # (vocab, d_model)

    # Convert targets to numpy for indexing
    targets_np = np.array(targets)  # (B, T)

    # For each (b, pos) sample a position where we have a valid next-token target
    # Valid positions: 0 to T-2 (target at pos is targets[b, pos] = input[b, pos+1])
    rng = np.random.default_rng(42)
    sample_b = rng.integers(0, B, size=n_samples)
    sample_pos = rng.integers(0, T - 1, size=n_samples)  # positions 0..T-2

    # Component names (ordered: embed, then L0_attn, L0_mlp, L1_attn, ...)
    component_names = ["embed"]
    for i in range(n_layers):
        component_names.append(f"L{i}_attn")
        component_names.append(f"L{i}_mlp")

    # Accumulators: signed and absolute per component
    signed_attr = {name: [] for name in component_names}
    abs_attr = {name: [] for name in component_names}

    for b, pos in zip(sample_b, sample_pos):
        target_token = int(targets_np[b, pos])
        unembed_dir = W_U[target_token]           # (d_model,)
        scale_bp = ln_scale[b, pos]               # (d_model,) - per-dim LN scale

        for name in component_names:
            contrib = np.array(contributions[name])   # (B, T, d_model)
            c = contrib[b, pos]                        # (d_model,)
            # Apply LN linearization: scale element-wise
            c_scaled = c * scale_bp
            # Project onto unembedding direction
            attr = float(np.dot(c_scaled, unembed_dir))
            signed_attr[name].append(attr)
            abs_attr[name].append(abs(attr))

    # Aggregate: mean signed and mean absolute
    result = {}
    for name in component_names:
        result[name] = {
            "mean_signed": float(np.mean(signed_attr[name])),
            "mean_abs": float(np.mean(abs_attr[name])),
            "std": float(np.std(signed_attr[name])),
        }

    # Per-layer aggregates: attn and mlp separately
    layer_attn_dla = []
    layer_mlp_dla = []
    for i in range(n_layers):
        layer_attn_dla.append(result[f"L{i}_attn"]["mean_signed"])
        layer_mlp_dla.append(result[f"L{i}_mlp"]["mean_signed"])

    total_attn_dla = float(np.sum([abs(result[f"L{i}_attn"]["mean_signed"]) for i in range(n_layers)]))
    total_mlp_dla = float(np.sum([abs(result[f"L{i}_mlp"]["mean_signed"]) for i in range(n_layers)]))

    # Signed totals (preserve direction)
    total_attn_signed = float(np.sum([result[f"L{i}_attn"]["mean_signed"] for i in range(n_layers)]))
    total_mlp_signed = float(np.sum([result[f"L{i}_mlp"]["mean_signed"] for i in range(n_layers)]))

    return {
        "per_component": result,
        "layer_attn_dla": layer_attn_dla,
        "layer_mlp_dla": layer_mlp_dla,
        "total_attn_abs": total_attn_dla,
        "total_mlp_abs": total_mlp_dla,
        "total_attn_signed": total_attn_signed,
        "total_mlp_signed": total_mlp_signed,
        "ratio_mlp_to_attn_abs": total_mlp_dla / (total_attn_dla + 1e-10),
        "ratio_mlp_to_attn_signed": total_mlp_signed / (total_attn_signed + 1e-10) if total_attn_signed != 0 else float("nan"),
    }


# ============================================================
# Old L2-norm "attribution" (from experiments/depth_scaling.py)
# ============================================================

def compute_l2_attribution(model: GPT, inputs: mx.array) -> dict:
    """Reproduce the original (flawed) L2-norm attribution for comparison."""
    contributions = decompose_residual_stream(model, inputs)
    n_layers = model.config.n_layers

    layer_attn_l2 = []
    layer_mlp_l2 = []
    for i in range(n_layers):
        attn_c = np.array(contributions[f"L{i}_attn"])
        mlp_c = np.array(contributions[f"L{i}_mlp"])
        layer_attn_l2.append(float(np.mean(np.linalg.norm(attn_c, axis=-1))))
        layer_mlp_l2.append(float(np.mean(np.linalg.norm(mlp_c, axis=-1))))

    total_attn = float(np.sum(layer_attn_l2))
    total_mlp = float(np.sum(layer_mlp_l2))
    return {
        "layer_attn_l2": layer_attn_l2,
        "layer_mlp_l2": layer_mlp_l2,
        "total_attn": total_attn,
        "total_mlp": total_mlp,
        "ratio_mlp_to_attn": total_mlp / (total_attn + 1e-10),
    }


# ============================================================
# Suppression analysis
# ============================================================

def get_suppression_fraction(model: GPT, dataset: TextDataset) -> float:
    profiles = profile_all_heads(model, dataset, n_samples=64)
    suppression_heads = [p for p in profiles if p.suppression_score > 0.3]
    return len(suppression_heads) / len(profiles)


# ============================================================
# Statistics: OLS + bootstrap CI for suppression vs depth
# ============================================================

def regression_with_bootstrap(x: list, y: list, n_bootstrap: int = 10000, seed: int = 0) -> dict:
    """OLS regression with 95% CI via bootstrap (BCa or percentile)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    # OLS
    x_bar = x.mean()
    y_bar = y.mean()
    ss_xx = ((x - x_bar) ** 2).sum()
    ss_xy = ((x - x_bar) * (y - y_bar)).sum()
    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar

    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_bar) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # F-test p-value (1 predictor)
    from scipy import stats as scipy_stats
    if n > 2 and ss_tot > 0:
        f_stat = (ss_tot - ss_res) / (ss_res / (n - 2))
        p_value = float(1.0 - scipy_stats.f.cdf(f_stat, 1, n - 2))
    else:
        p_value = float("nan")

    # Bootstrap CI for slope
    rng = np.random.default_rng(seed)
    boot_slopes = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        xb_bar = xb.mean()
        ss_xb = ((xb - xb_bar) ** 2).sum()
        if ss_xb < 1e-10:
            continue
        sb = ((xb - xb_bar) * (yb - yb.mean())).sum() / ss_xb
        boot_slopes.append(sb)

    boot_slopes = np.array(boot_slopes)
    ci_low = float(np.percentile(boot_slopes, 2.5))
    ci_high = float(np.percentile(boot_slopes, 97.5))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "p_value": float(p_value),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n": n,
        "recommended_language": "scaling law" if p_value < 0.01 else "scaling trend (not a law; p≥0.01 or n too small)",
    }


# ============================================================
# Plotting
# ============================================================

def make_plots(all_depth_results: list, regression_stats: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    depths = [r["n_layers"] for r in all_depth_results]
    colors = ["steelblue", "tomato", "darkorange", "green"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---- Panel A: MLP vs Attn DLA per depth (grouped bars) ----
    ax = axes[0]
    x = np.arange(len(depths))
    width = 0.35
    attn_totals = [r["dla"]["total_attn_signed"] for r in all_depth_results]
    mlp_totals = [r["dla"]["total_mlp_signed"] for r in all_depth_results]

    bars_attn = ax.bar(x - width / 2, attn_totals, width, label="Attention (DLA)", color="steelblue", alpha=0.8)
    bars_mlp = ax.bar(x + width / 2, mlp_totals, width, label="MLP (DLA)", color="tomato", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Model Depth (n_layers)")
    ax.set_ylabel("Summed Direct Logit Attribution (signed)")
    ax.set_title("Panel A: MLP vs Attn\nLogit Attribution by Depth")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in depths])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate with ratio
    for i, r in enumerate(all_depth_results):
        ratio = r["dla"]["ratio_mlp_to_attn_signed"]
        if not np.isnan(ratio):
            ax.text(x[i], max(attn_totals[i], mlp_totals[i]) * 1.05,
                    f"r={ratio:.2f}", ha="center", fontsize=7, color="gray")

    # ---- Panel B: Per-layer DLA by depth (grouped bars per layer) ----
    ax = axes[1]
    max_layers = max(depths)

    # One line per depth showing per-layer attn and mlp attribution
    for i, r in enumerate(all_depth_results):
        n = r["n_layers"]
        layer_ids = np.arange(n)
        attn_vals = r["dla"]["layer_attn_dla"]
        mlp_vals = r["dla"]["layer_mlp_dla"]
        ax.plot(layer_ids, attn_vals, "o--", color=colors[i], alpha=0.7, linewidth=1.5, label=f"D={n} attn")
        ax.plot(layer_ids, mlp_vals, "s-", color=colors[i], alpha=0.9, linewidth=1.5, label=f"D={n} mlp")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Signed DLA")
    ax.set_title("Panel B: Per-layer DLA\n(dashed=attn, solid=MLP)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- Panel C: Suppression fraction vs depth with regression stats ----
    ax = axes[2]
    supp_fracs = [r["suppression_fraction"] for r in all_depth_results]
    ax.scatter(depths, supp_fracs, color="darkorange", s=80, zorder=5)

    # Regression line
    slope = regression_stats["slope"]
    intercept = regression_stats["intercept"]
    x_line = np.linspace(min(depths) - 0.5, max(depths) + 0.5, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "k--", linewidth=1.5, label=f"OLS slope={slope:.4f}")

    ax.set_xlabel("Model Depth (n_layers)")
    ax.set_ylabel("Suppression Head Fraction")
    ax.set_title(
        f"Panel C: Suppression vs Depth\n"
        f"R²={regression_stats['r2']:.3f}, p={regression_stats['p_value']:.3f}, n={regression_stats['n']}"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add 95% CI annotation
    ci_text = (f"slope 95% CI: [{regression_stats['ci_95_low']:.4f}, {regression_stats['ci_95_high']:.4f}]\n"
               f"({regression_stats['recommended_language']})")
    ax.text(0.02, 0.02, ci_text, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Depth Scaling Rigorous Analysis (Direct Logit Attribution)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = PLOT_DIR / "depth_scaling_rigorous.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Rigorous Depth Scaling Analysis")
    print("=" * 70)
    print(f"Fixed eval seed: {FIXED_EVAL_SEED}")
    print(f"Attribution samples: {N_ATTR_SAMPLES}")
    print()

    # Load fixed eval dataset — same for ALL models
    print("Loading fixed eval dataset (seed=9999)...")
    eval_dataset = make_fixed_eval_dataset()

    # Load original depth_scaling_results for comparison
    orig_results_path = Path("experiments/depth_scaling_results.json")
    orig_results = {}
    if orig_results_path.exists():
        with open(orig_results_path) as f:
            orig_list = json.load(f)
        for r in orig_list:
            orig_results[r["n_layers"]] = r

    all_depth_results = []

    for n_layers in DEPTHS:
        print(f"\n{'='*60}")
        print(f"Analyzing depth={n_layers}")
        print(f"{'='*60}")

        model = load_depth_model(n_layers)

        # Fixed-seed eval loss
        eval_loss = compute_eval_loss(model, eval_dataset)
        print(f"  Fixed-seed eval loss: {eval_loss:.4f}")

        # Attribution batch — same fixed seed
        batch_inputs, batch_targets = eval_dataset.generate_batch(EVAL_BATCH_SIZE)

        # --- DLA (new, rigorous) ---
        print("  Computing direct logit attribution (DLA)...")
        dla = compute_logit_attribution(model, batch_inputs, batch_targets, n_samples=N_ATTR_SAMPLES)
        print(f"  Total attn DLA (signed): {dla['total_attn_signed']:.4f}")
        print(f"  Total MLP  DLA (signed): {dla['total_mlp_signed']:.4f}")
        print(f"  MLP/Attn DLA ratio (abs): {dla['ratio_mlp_to_attn_abs']:.4f}")
        print(f"  Layer attn DLA: {[f'{v:.3f}' for v in dla['layer_attn_dla']]}")
        print(f"  Layer MLP  DLA: {[f'{v:.3f}' for v in dla['layer_mlp_dla']]}")

        # --- L2 norm (old, for comparison) ---
        print("  Computing L2-norm attribution (old method)...")
        l2 = compute_l2_attribution(model, batch_inputs)
        print(f"  Total attn L2: {l2['total_attn']:.4f}")
        print(f"  Total MLP  L2: {l2['total_mlp']:.4f}")
        print(f"  MLP/Attn L2 ratio: {l2['ratio_mlp_to_attn']:.4f}")

        # --- Suppression ---
        print("  Profiling suppression heads...")
        supp_frac = get_suppression_fraction(model, eval_dataset)
        print(f"  Suppression fraction: {supp_frac:.3f}")

        # Load original L2 values for comparison
        orig_attn = orig_results.get(n_layers, {}).get("analysis", {}).get("layer_attn_attr", [])
        orig_mlp = orig_results.get(n_layers, {}).get("analysis", {}).get("layer_mlp_attr", [])

        all_depth_results.append({
            "n_layers": n_layers,
            "eval_loss_fixed_seed": eval_loss,
            "dla": dla,
            "l2_attribution": l2,
            "suppression_fraction": supp_frac,
            "original_l2_attn": orig_attn,
            "original_l2_mlp": orig_mlp,
        })

    # ============================================================
    # Statistics: suppression vs depth regression
    # ============================================================
    print("\n" + "=" * 70)
    print("Regression: suppression fraction vs depth")
    print("=" * 70)

    x_depths = [r["n_layers"] for r in all_depth_results]
    y_supps = [r["suppression_fraction"] for r in all_depth_results]
    regression_stats = regression_with_bootstrap(x_depths, y_supps)
    print(f"  slope:        {regression_stats['slope']:.6f}")
    print(f"  intercept:    {regression_stats['intercept']:.6f}")
    print(f"  R²:           {regression_stats['r2']:.4f}")
    print(f"  OLS p-value:  {regression_stats['p_value']:.4f}")
    print(f"  95% CI slope: [{regression_stats['ci_95_low']:.6f}, {regression_stats['ci_95_high']:.6f}]")
    print(f"  n:            {regression_stats['n']}")
    print(f"  Recommended language: {regression_stats['recommended_language']}")

    # ============================================================
    # Key findings comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Old L2-norm vs New DLA Attribution")
    print("=" * 70)
    print(f"\n{'Depth':<8} {'OldMLP/Attn':>12} {'NewMLP/Attn(abs)':>18} {'NewMLP/Attn(sgn)':>18} {'MLP>Attn(old)':>14} {'MLP>Attn(new)':>14}")
    print("-" * 90)

    crossover_old = []
    crossover_new = []
    for r in all_depth_results:
        n = r["n_layers"]
        old_ratio = r["l2_attribution"]["ratio_mlp_to_attn"]
        new_abs = r["dla"]["ratio_mlp_to_attn_abs"]
        new_sgn = r["dla"]["ratio_mlp_to_attn_signed"]
        old_dom = "MLP" if old_ratio > 1.0 else "Attn"
        new_dom = "MLP" if new_abs > 1.0 else "Attn"
        crossover_old.append(old_dom)
        crossover_new.append(new_dom)
        sgn_str = f"{new_sgn:.4f}" if not np.isnan(new_sgn) else "nan"
        print(f"{n:<8} {old_ratio:>12.4f} {new_abs:>18.4f} {sgn_str:>18} {old_dom:>14} {new_dom:>14}")

    # Did crossover claim survive?
    print("\n--- Crossover Claim Analysis ---")
    print(f"Original L2-norm dominant component by depth: {list(zip(DEPTHS, crossover_old))}")
    print(f"New DLA (abs) dominant component by depth:    {list(zip(DEPTHS, crossover_new))}")

    old_crossovers = [i for i in range(1, len(crossover_old)) if crossover_old[i] != crossover_old[i-1]]
    new_crossovers = [i for i in range(1, len(crossover_new)) if crossover_new[i] != crossover_new[i-1]]

    if old_crossovers:
        cross_depth_old = DEPTHS[old_crossovers[0]]
        print(f"\nOriginal claimed crossover at depth {cross_depth_old}: {crossover_old[old_crossovers[0]-1]} -> {crossover_old[old_crossovers[0]]}")
    else:
        print("\nOriginal: no dominant-component crossover detected.")

    if new_crossovers:
        cross_depth_new = DEPTHS[new_crossovers[0]]
        print(f"New DLA crossover at depth {cross_depth_new}: {crossover_new[new_crossovers[0]-1]} -> {crossover_new[new_crossovers[0]]}")
        print("\nCROSSSOVER CLAIM: SURVIVES under DLA analysis.")
    else:
        print("\nCROSSOVER CLAIM: DOES NOT SURVIVE under DLA analysis.")
        print("  The MLP→Attn or Attn→MLP dominance shift seen with L2-norm is NOT")
        print("  replicated when using actual direct logit attribution.")
        print("  The two methods measure different quantities and yield different conclusions.")

    # ============================================================
    # Statistical honesty summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Statistical Honesty Summary")
    print("=" * 70)
    p = regression_stats["p_value"]
    r2 = regression_stats["r2"]
    ci_lo = regression_stats["ci_95_low"]
    ci_hi = regression_stats["ci_95_high"]
    slope = regression_stats["slope"]
    print(f"  Suppression fraction vs depth: n={regression_stats['n']} data points")
    print(f"  OLS: slope={slope:.4f}, R²={r2:.4f}, p={p:.4f}")
    print(f"  Bootstrap 95% CI for slope: [{ci_lo:.4f}, {ci_hi:.4f}]")
    if p > 0.05:
        print("  WARNING: p > 0.05. Trend is not statistically significant.")
        print("  The paper MUST NOT describe this as a 'scaling law'.")
        print("  Recommended: 'apparent scaling trend (p={:.3f}, n=4; not definitive)'".format(p))
    elif p > 0.01:
        print("  p < 0.05 but > 0.01. Marginal significance with n=4.")
        print("  Recommended: 'scaling trend' (not 'scaling law').")
    else:
        print("  p < 0.01. Statistically significant even at n=4.")
        print("  Recommended: 'scaling trend' still safer than 'law' given small n.")

    # ============================================================
    # Save results
    # ============================================================
    results_to_save = {
        "per_depth": {},
        "regression_suppression_vs_depth": regression_stats,
        "crossover_survived": bool(new_crossovers),
        "crossover_old_l2": list(zip(DEPTHS, crossover_old)),
        "crossover_new_dla": list(zip(DEPTHS, crossover_new)),
        "fixed_eval_seed": FIXED_EVAL_SEED,
        "n_attr_samples": N_ATTR_SAMPLES,
    }

    for r in all_depth_results:
        n = r["n_layers"]
        results_to_save["per_depth"][str(n)] = {
            "eval_loss_fixed_seed": r["eval_loss_fixed_seed"],
            "mlp_attribution_total_signed": r["dla"]["total_mlp_signed"],
            "attn_attribution_total_signed": r["dla"]["total_attn_signed"],
            "mlp_attribution_total_abs": r["dla"]["total_mlp_abs"],
            "attn_attribution_total_abs": r["dla"]["total_attn_abs"],
            "ratio_mlp_to_attn_abs": r["dla"]["ratio_mlp_to_attn_abs"],
            "ratio_mlp_to_attn_signed": r["dla"]["ratio_mlp_to_attn_signed"],
            "per_layer": {
                "attn_dla": r["dla"]["layer_attn_dla"],
                "mlp_dla": r["dla"]["layer_mlp_dla"],
            },
            "suppression_fraction": r["suppression_fraction"],
            "comparison_with_original_l2": {
                "original_layer_attn_l2": r["original_l2_attn"],
                "original_layer_mlp_l2": r["original_l2_mlp"],
                "new_layer_attn_dla": r["dla"]["layer_attn_dla"],
                "new_layer_mlp_dla": r["dla"]["layer_mlp_dla"],
                "old_ratio_mlp_to_attn": r["l2_attribution"]["ratio_mlp_to_attn"],
                "new_ratio_mlp_to_attn_abs": r["dla"]["ratio_mlp_to_attn_abs"],
            },
        }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Generate plots
    print("\nGenerating plots...")
    make_plots(all_depth_results, regression_stats)

    print("\nDone.")


if __name__ == "__main__":
    main()
