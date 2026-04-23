"""
statistical_rigor.py

Bootstrap confidence intervals and statistical tests for key paper claims.
"""

import json
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RNG = np.random.default_rng(42)
N_BOOT = 1000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_mean(data, n_boot=N_BOOT, rng=RNG):
    """Return (mean, se, ci_lo, ci_hi) via percentile bootstrap."""
    data = np.asarray(data, dtype=float)
    boot = np.array([rng.choice(data, size=len(data), replace=True).mean()
                     for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(data.mean()), float(boot.std()), float(ci_lo), float(ci_hi)


def binomial_bootstrap_fraction(n_success, n_total, n_boot=N_BOOT, rng=RNG):
    """Bootstrap CI for a fraction (Bernoulli trials)."""
    trials = np.array([1] * n_success + [0] * (n_total - n_success), dtype=float)
    mean, se, ci_lo, ci_hi = bootstrap_mean(trials, n_boot=n_boot, rng=rng)
    return mean, se, ci_lo, ci_hi


def one_sample_p_value_bootstrap(data, null_value, n_boot=N_BOOT, rng=RNG):
    """
    Bootstrap p-value: fraction of bootstrap means at least as extreme as
    the observed mean, under the null that the true mean equals null_value.
    Shifts the data so that the null is the centre, then checks how often
    the bootstrap mean exceeds the observed mean.
    """
    data = np.asarray(data, dtype=float)
    obs = data.mean()
    centred = data - obs + null_value          # shift so null_mean = null_value
    boot = np.array([rng.choice(centred, size=len(centred), replace=True).mean()
                     for _ in range(n_boot)])
    # two-sided: how often |boot - null| >= |obs - null|
    p = np.mean(np.abs(boot - null_value) >= np.abs(obs - null_value))
    return float(p)


# ---------------------------------------------------------------------------
# Claim 1 – Cross-seed circuit similarity
# ---------------------------------------------------------------------------

def claim1_cross_seed_similarity():
    path = "checkpoints/seed_divergence/analysis.json"
    with open(path) as f:
        data = json.load(f)

    mat = np.array(data["similarity_matrix"])   # 20×20
    n = mat.shape[0]
    # Extract upper-triangle (off-diagonal) pairs
    idx_i, idx_j = np.triu_indices(n, k=1)
    pairs = mat[idx_i, idx_j]                   # 190 values

    mean, se, ci_lo, ci_hi = bootstrap_mean(pairs)

    # p-value against null = 0 (random similarity)
    p = one_sample_p_value_bootstrap(pairs, null_value=0.0)

    return {
        "claim": "Cross-seed circuit similarity",
        "n": len(pairs),
        "point_estimate": mean,
        "se": se,
        "ci_95": [ci_lo, ci_hi],
        "null_hypothesis": 0.0,
        "p_value": p,
        "note": "190 unique pairs from 20-seed similarity matrix",
    }


# ---------------------------------------------------------------------------
# Claim 2 – Head-level agreement
# ---------------------------------------------------------------------------

def claim2_head_level_agreement():
    path = "checkpoints/seed_divergence/head_comparison.json"
    with open(path) as f:
        data = json.load(f)

    seeds = [k for k in data.keys() if k != "_summary"]
    # Critical induction head per seed: (layer, head) tuple
    critical_heads = []
    for s in seeds:
        bh = data[s]["best_induction_head"]
        critical_heads.append((bh["layer"], bh["head"]))

    n_seeds = len(critical_heads)
    n_heads_possible = 4 * 4   # 4 layers × 4 heads

    # Agreement = fraction of pairs sharing the same critical head
    def agreement_fraction(heads):
        heads = list(heads)
        n = len(heads)
        if n < 2:
            return 0.0
        pairs = n * (n - 1) / 2
        matches = sum(
            1 for a in range(n) for b in range(a + 1, n)
            if heads[a] == heads[b]
        )
        return matches / pairs

    obs_agreement = agreement_fraction(critical_heads)

    # Bootstrap over seeds (resample seeds with replacement)
    boot_agreements = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n_seeds, size=n_seeds)
        sample = [critical_heads[i] for i in idx]
        boot_agreements.append(agreement_fraction(sample))

    boot_agreements = np.array(boot_agreements)
    se = float(boot_agreements.std())
    ci_lo, ci_hi = np.percentile(boot_agreements, [2.5, 97.5])

    null_value = 1.0 / n_heads_possible   # 1/16 = 0.0625

    # One-sided p-value: probability of seeing >= obs under null
    # Use permutation approach: randomly assign heads, compute agreement
    perm_agreements = []
    for _ in range(N_BOOT):
        perm_heads = [
            (RNG.integers(0, 4), RNG.integers(0, 4))
            for _ in range(n_seeds)
        ]
        perm_agreements.append(agreement_fraction(perm_heads))
    p_value = float(np.mean(np.array(perm_agreements) >= obs_agreement))

    return {
        "claim": "Head-level agreement",
        "n_seeds": n_seeds,
        "n_heads_possible": n_heads_possible,
        "point_estimate": obs_agreement,
        "se": se,
        "ci_95": [ci_lo, ci_hi],
        "null_hypothesis": null_value,
        "p_value": p_value,
        "critical_heads": [{"layer": h[0], "head": h[1]} for h in critical_heads],
        "note": "Fraction of seed-pairs sharing the same best induction head",
    }


# ---------------------------------------------------------------------------
# Claim 3 – Developmental ordering
# ---------------------------------------------------------------------------

def claim3_developmental_ordering():
    path = "analysis/formation_film_results.json"
    with open(path) as f:
        data = json.load(f)

    ordering = data["ordering_analysis"]
    canonical = ordering["canonical_order"]
    # Use the authoritative counts pre-computed by formation_film.py
    n_seeds = ordering["n_seeds_total"]
    n_matching_reported = ordering["n_seeds_matching_canonical"]

    # Build binary array matching the reported counts for bootstrap
    per_seed_match = np.array(
        [1] * n_matching_reported + [0] * (n_seeds - n_matching_reported),
        dtype=float,
    )
    match_fraction = float(per_seed_match.mean())

    # Bootstrap CI
    _, se, ci_lo, ci_hi = bootstrap_mean(per_seed_match)

    # Null: 1/4! = 1/24 if orderings were uniformly random
    null_value = 1.0 / 24.0
    p_value = one_sample_p_value_bootstrap(per_seed_match, null_value=null_value)

    # Binomial test for additional check
    binom_result = stats.binomtest(int(per_seed_match.sum()), n=n_seeds, p=null_value,
                                   alternative="greater")

    return {
        "claim": "Developmental ordering (canonical)",
        "n_seeds": n_seeds,
        "n_matching": int(per_seed_match.sum()),
        "point_estimate": match_fraction,
        "se": se,
        "ci_95": [ci_lo, ci_hi],
        "null_hypothesis": null_value,
        "p_value_bootstrap": p_value,
        "p_value_binomial": float(binom_result.pvalue),
        "canonical_order": canonical,
        "note": "Fraction of seeds whose feature-crystallisation order matches canonical",
    }


# ---------------------------------------------------------------------------
# Claim 4 – Ablation drop CI
# ---------------------------------------------------------------------------

def claim4_ablation_drop():
    path = "analysis/ablation_robustness_results.json"
    with open(path) as f:
        data = json.load(f)

    drops = [s["critical_head"]["accuracy_drop"] for s in data["seeds"]]
    drops = np.array(drops, dtype=float)

    mean, se, ci_lo, ci_hi = bootstrap_mean(drops)

    # t-test against null = 0 (no drop)
    t_stat, p_value = stats.ttest_1samp(drops, popmean=0.0)

    return {
        "claim": "Ablation drop (content-matching induction head)",
        "n_seeds": len(drops),
        "point_estimate": float(drops.mean()),
        "std": float(drops.std(ddof=1)),
        "se": se,
        "ci_95": [ci_lo, ci_hi],
        "null_hypothesis": 0.0,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "per_seed_drops": drops.tolist(),
        "note": "Accuracy drop when ablating the critical induction head per seed",
    }


# ---------------------------------------------------------------------------
# Claim 5 – Suppression fraction vs depth
# ---------------------------------------------------------------------------

def claim5_suppression_depth():
    path = "experiments/depth_scaling_results.json"
    with open(path) as f:
        data = json.load(f)

    depths = np.array([item["n_layers"] for item in data], dtype=float)
    fracs = np.array([item["analysis"]["frac_suppression"] for item in data], dtype=float)

    # OLS regression
    slope, intercept, r_value, p_value, se_slope = stats.linregress(depths, fracs)

    # Bootstrap CI on slope
    n = len(depths)
    boot_slopes = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        d_b, f_b = depths[idx], fracs[idx]
        if len(np.unique(d_b)) < 2:
            continue
        s, *_ = stats.linregress(d_b, f_b)
        boot_slopes.append(s)
    boot_slopes = np.array(boot_slopes)
    slope_ci_lo, slope_ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

    return {
        "claim": "Suppression fraction scales with depth",
        "n_models": n,
        "depths": depths.tolist(),
        "suppression_fractions": fracs.tolist(),
        "regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "se_slope": float(se_slope),
            "slope_ci_95": [float(slope_ci_lo), float(slope_ci_hi)],
        },
        "note": "OLS: suppression_frac ~ n_layers; bootstrap CI on slope",
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results):
    header = f"{'Claim':<40} {'Point Est':>12} {'95% CI':>20} {'Null':>8} {'p-value':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    rows = [
        (
            results["claim1"]["claim"],
            f"{results['claim1']['point_estimate']:.3f}",
            f"[{results['claim1']['ci_95'][0]:.3f}, {results['claim1']['ci_95'][1]:.3f}]",
            f"{results['claim1']['null_hypothesis']:.3f}",
            f"<{max(results['claim1']['p_value'], 1/N_BOOT):.4f}",
        ),
        (
            results["claim2"]["claim"],
            f"{results['claim2']['point_estimate']:.3f}",
            f"[{results['claim2']['ci_95'][0]:.3f}, {results['claim2']['ci_95'][1]:.3f}]",
            f"{results['claim2']['null_hypothesis']:.4f}",
            f"{results['claim2']['p_value']:.4f}",
        ),
        (
            results["claim3"]["claim"],
            f"{results['claim3']['point_estimate']:.3f}",
            f"[{results['claim3']['ci_95'][0]:.3f}, {results['claim3']['ci_95'][1]:.3f}]",
            f"{results['claim3']['null_hypothesis']:.4f}",
            f"{results['claim3']['p_value_binomial']:.4f}",
        ),
        (
            results["claim4"]["claim"][:40],
            f"{results['claim4']['point_estimate']:.3f} ± {results['claim4']['std']:.3f}",
            f"[{results['claim4']['ci_95'][0]:.3f}, {results['claim4']['ci_95'][1]:.3f}]",
            f"{results['claim4']['null_hypothesis']:.3f}",
            f"{results['claim4']['p_value']:.4f}",
        ),
        (
            results["claim5"]["claim"][:40],
            f"slope={results['claim5']['regression']['slope']:.4f}",
            f"[{results['claim5']['regression']['slope_ci_95'][0]:.4f}, "
            f"{results['claim5']['regression']['slope_ci_95'][1]:.4f}]",
            "linear",
            f"{results['claim5']['regression']['p_value']:.4f}",
        ),
    ]

    for row in rows:
        print(f"{row[0]:<40} {row[1]:>12} {row[2]:>20} {row[3]:>8} {row[4]:>10}")
    print(sep)


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def make_forest_plot(results, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Bootstrap 95% Confidence Intervals — Key Claims", fontsize=13, pad=12)

    claims = [
        ("Cross-seed similarity\n(null=0)",
         results["claim1"]["point_estimate"],
         results["claim1"]["ci_95"],
         results["claim1"]["null_hypothesis"]),
        ("Head-level agreement\n(null=1/16)",
         results["claim2"]["point_estimate"],
         results["claim2"]["ci_95"],
         results["claim2"]["null_hypothesis"]),
        ("Canonical dev. order\n(null=1/24)",
         results["claim3"]["point_estimate"],
         results["claim3"]["ci_95"],
         results["claim3"]["null_hypothesis"]),
        ("Ablation drop\n(null=0)",
         results["claim4"]["point_estimate"],
         results["claim4"]["ci_95"],
         results["claim4"]["null_hypothesis"]),
        ("Suppression slope\n(null=0)",
         results["claim5"]["regression"]["slope"],
         results["claim5"]["regression"]["slope_ci_95"],
         0.0),
    ]

    y_positions = list(range(len(claims) - 1, -1, -1))
    colors = ["#2166ac", "#4dac26", "#d01c8b", "#f1a340", "#998ec3"]

    for i, (label, est, ci, null) in enumerate(claims):
        y = y_positions[i]
        lo, hi = ci
        ax.plot([lo, hi], [y, y], color=colors[i], linewidth=2.5, solid_capstyle="round")
        ax.plot(est, y, "o", color=colors[i], markersize=8, zorder=5)
        ax.axvline(null, color=colors[i], linestyle="--", alpha=0.4, linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in claims], fontsize=10)
    ax.set_xlabel("Estimate (normalised per claim — see axis labels)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for null lines
    null_patch = mpatches.Patch(color="grey", alpha=0.5, label="Null hypothesis (dashed)")
    ax.legend(handles=[null_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Forest plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Computing bootstrap CIs and statistical tests ...\n")

    r1 = claim1_cross_seed_similarity()
    print(f"[1] Cross-seed similarity:  mean={r1['point_estimate']:.4f}  "
          f"95% CI=[{r1['ci_95'][0]:.4f}, {r1['ci_95'][1]:.4f}]  "
          f"SE={r1['se']:.4f}  p={r1['p_value']:.4f}")

    r2 = claim2_head_level_agreement()
    print(f"[2] Head-level agreement:   mean={r2['point_estimate']:.4f}  "
          f"95% CI=[{r2['ci_95'][0]:.4f}, {r2['ci_95'][1]:.4f}]  "
          f"SE={r2['se']:.4f}  p={r2['p_value']:.4f}")

    r3 = claim3_developmental_ordering()
    print(f"[3] Canonical dev. order:   frac={r3['point_estimate']:.4f}  "
          f"95% CI=[{r3['ci_95'][0]:.4f}, {r3['ci_95'][1]:.4f}]  "
          f"p_binom={r3['p_value_binomial']:.4f}  p_boot={r3['p_value_bootstrap']:.4f}")

    r4 = claim4_ablation_drop()
    print(f"[4] Ablation drop:          mean={r4['point_estimate']:.4f} ± {r4['std']:.4f}  "
          f"95% CI=[{r4['ci_95'][0]:.4f}, {r4['ci_95'][1]:.4f}]  "
          f"p={r4['p_value']:.4f}")

    r5 = claim5_suppression_depth()
    reg = r5["regression"]
    print(f"[5] Suppression vs depth:   slope={reg['slope']:.4f}  "
          f"R²={reg['r_squared']:.4f}  "
          f"95% CI slope=[{reg['slope_ci_95'][0]:.4f}, {reg['slope_ci_95'][1]:.4f}]  "
          f"p={reg['p_value']:.4f}")

    print()
    results = {"claim1": r1, "claim2": r2, "claim3": r3, "claim4": r4, "claim5": r5}
    print_summary_table(results)

    # Save JSON
    out_json = "analysis/statistical_rigor_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Forest plot
    make_forest_plot(results, "analysis/plots/statistical_summary.png")


if __name__ == "__main__":
    main()
