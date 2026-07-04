"""Corrected statistics for the 20-seed critical-head analysis (AUDIT.md S5).

Recomputes, from the committed content_matching_20_seeds_results_v2.json:
1. Head agreement against the CONDITIONAL null (uniform over 4 heads given
   layer-0 criticality) — the fair null for a claim labeled "head-level",
   since the layer-0 effect is reported separately.
2. Seed-level jackknife CI for pairwise agreement (the original bootstrap
   resampled 190 dependent pair-indicators as i.i.d.).
3. Clopper–Pearson interval for the 20/20 layer-agreement count (the original
   percentile bootstrap of an all-ones vector gives a degenerate [100,100]).
4. Slot-count uniformity test within L0 (chi-square + exact Monte Carlo).
5. Effect-vector cosine similarity against a within-seed head-relabeling
   permutation null (the original tested against 0, vacuous for non-negative
   vectors).

Usage: python -m experiments.torch_replication.corrected_stats
"""

import json
from itertools import combinations

import numpy as np
from scipy import stats

from .common import RESULTS_DIR, ROOT


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def pairwise_agreement(slots):
    pairs = list(combinations(range(len(slots)), 2))
    return np.mean([slots[i] == slots[j] for i, j in pairs])


def main():
    rng = np.random.default_rng(0)
    data = json.loads((ROOT / "analysis" / "content_matching_20_seeds_results_v2.json").read_text())
    per_seed = data["per_seed"]
    n = len(per_seed)
    layers = np.array([s["critical_head_layer"] for s in per_seed])
    slots = np.array([s["critical_head_index"] for s in per_seed])
    n_heads = data["model_config"]["n_heads"]

    out = {"n_seeds": n}

    # 1. layer agreement with honest CI
    k_l0 = int((layers == 0).sum())
    out["layer_agreement"] = {
        "count": f"{k_l0}/{n}",
        "clopper_pearson_95": clopper_pearson(k_l0, n),
    }

    # 2. observed head agreement + jackknife CI (leave-one-seed-out)
    obs = pairwise_agreement(slots)
    jack = np.array([pairwise_agreement(np.delete(slots, i)) for i in range(n)])
    theta_dot = jack.mean()
    se = np.sqrt((n - 1) / n * ((jack - theta_dot) ** 2).sum())
    out["head_agreement"] = {
        "observed": float(obs),
        "jackknife_se": float(se),
        "jackknife_ci95": [float(obs - 1.96 * se), float(obs + 1.96 * se)],
    }

    # 3. conditional null: uniform over n_heads given L0 criticality
    n_mc = 100000
    null_agree = np.empty(n_mc)
    for i in range(n_mc):
        null_agree[i] = pairwise_agreement(rng.integers(0, n_heads, n))
    p_greater = float((null_agree >= obs).mean())
    p_two_sided = float((np.abs(null_agree - null_agree.mean())
                         >= abs(obs - null_agree.mean())).mean())
    out["head_agreement_conditional_null"] = {
        "null": f"uniform over {n_heads} heads given L0 criticality",
        "null_mean": float(null_agree.mean()),
        "p_one_sided_greater": p_greater,
        "p_two_sided": p_two_sided,
        "note": "observed agreement is at/below the conditional null mean; "
                "the original p=0.002 was driven entirely by the layer effect",
    }

    # 4. slot-count uniformity within L0
    counts = np.bincount(slots[layers == 0], minlength=n_heads)
    chi2, p_chi2 = stats.chisquare(counts)
    # exact Monte Carlo of the chi-square statistic under uniform
    null_chi = np.empty(n_mc // 10)
    for i in range(len(null_chi)):
        c = np.bincount(rng.integers(0, n_heads, k_l0), minlength=n_heads)
        null_chi[i] = stats.chisquare(c)[0]
    out["slot_uniformity_within_L0"] = {
        "counts": counts.tolist(),
        "chi2": float(chi2),
        "p_asymptotic": float(p_chi2),
        "p_monte_carlo": float((null_chi >= chi2).mean()),
    }

    # 5. effect-vector cosine similarity vs head-relabeling permutation null
    vecs = np.array([s["effect_vector"] for s in per_seed])
    head_idx = np.array([0, 1, 2, 3, 5, 6, 7, 8])  # positions of the 8 head drops
    def mean_pairwise_cos(v):
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        sim = vn @ vn.T
        iu = np.triu_indices(n, 1)
        return sim[iu].mean()
    obs_cos = mean_pairwise_cos(vecs)
    null_cos = np.empty(2000)
    for i in range(len(null_cos)):
        v = vecs.copy()
        for s in range(n):
            v[s, head_idx] = v[s, head_idx[rng.permutation(8)]]
        null_cos[i] = mean_pairwise_cos(v)
    out["effect_vector_similarity"] = {
        "observed_mean_cosine": float(obs_cos),
        "permutation_null_mean": float(null_cos.mean()),
        "permutation_null_sd": float(null_cos.std()),
        "p_greater": float((null_cos >= obs_cos).mean()),
        "note": "null = within-seed head-relabeling; tests shared structure "
                "beyond the magnitude profile (original null of 0 was vacuous)",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "corrected_stats.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
