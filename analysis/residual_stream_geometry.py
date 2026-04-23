"""
Experiment F: Residual Stream Geometry
=======================================
Question: Are 20 seeds rotations of the same manifold, or different manifolds?

Protocol:
1. Build fixed 1024-sequence eval batch (seed=9999)
2. For each of 20 seeds, collect residual stream activations at 3 interfaces:
   - pre-L0-LN: model.blocks[0].ln1(post_embedding)
   - post-L0: residuals[1]
   - post-L1: residuals[2]
3. PCA analysis: shared 50-D basis, visualize first 2 PCs colored by seed
4. CKA heatmap: linear CKA for all 190 pairs, 3 interfaces
5. Cross-reference: CKA vs shared critical heads (Pearson correlation)

Outputs:
  analysis/residual_stream_geometry_results.json
  analysis/plots/residual_pca_{pre_l0,post_l0,post_l1}.png
  analysis/plots/residual_cka_heatmap.png
  analysis/plots/cka_vs_shared_heads.png
"""

import sys
import json
import numpy as np
import mlx.core as mx
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from analysis.transplant_unified_eval import (
    build_fixed_eval_set,
    load_model,
    MODEL_CONFIG,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
RESULTS_PATH = Path(__file__).parent / "residual_stream_geometry_results.json"
PLOTS_DIR = Path(__file__).parent / "plots"

N_SEEDS = 20
SUBSAMPLE_N = 4096
SUBSAMPLE_SEED = 2024
PCA_N_COMPONENTS = 50
INTERFACE_NAMES = ["pre_l0", "post_l0", "post_l1"]


def get_checkpoint_path(seed: int) -> str:
    if seed == 2:
        step = "step_030000"
    else:
        step = "step_020000"
    return str(
        BASE_DIR
        / f"checkpoints/induction_content_match/induction_seed{seed}/{step}/model.safetensors"
    )


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def build_shared_subsample_idx(n_total: int) -> np.ndarray:
    """Build a single shared index array used for ALL seeds.

    CKA requires comparing activations at the SAME token positions across seeds.
    Using different indices per seed destroys the correspondence and gives near-zero CKA.
    """
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    idx = rng.choice(n_total, size=SUBSAMPLE_N, replace=False)
    idx.sort()
    return idx


def collect_activations_for_seed(
    model: GPT,
    fixed_eval_set,
    shared_idx: np.ndarray,
) -> dict:
    """Collect and subsample residual stream activations at 3 interfaces.

    shared_idx: the SAME index array used for all seeds, so CKA compares
                activations at identical (sequence, position) slots.

    Returns dict: interface_name -> np.ndarray shape (SUBSAMPLE_N, d_model)
    """
    all_pre_l0 = []
    all_post_l0 = []
    all_post_l1 = []

    for inputs_np, _targets_np, _mask_np in fixed_eval_set:
        inputs_mx = mx.array(inputs_np)
        residuals = model.get_residual_stream(inputs_mx)

        # post_embedding = residuals[0], shape (B, T, D)
        post_embed = residuals[0]
        # Apply LN manually: pre-L0-LN
        pre_l0_normed = model.blocks[0].ln1(post_embed)
        mx.eval(pre_l0_normed)
        mx.eval(residuals[1])
        mx.eval(residuals[2])

        B, T, D = inputs_np.shape[0], inputs_np.shape[1], MODEL_CONFIG.d_model

        pre_l0_np = np.array(pre_l0_normed).reshape(B * T, D)
        post_l0_np = np.array(residuals[1]).reshape(B * T, D)
        post_l1_np = np.array(residuals[2]).reshape(B * T, D)

        all_pre_l0.append(pre_l0_np)
        all_post_l0.append(post_l0_np)
        all_post_l1.append(post_l1_np)

    full_pre_l0 = np.concatenate(all_pre_l0, axis=0)   # (B*T_total, D)
    full_post_l0 = np.concatenate(all_post_l0, axis=0)
    full_post_l1 = np.concatenate(all_post_l1, axis=0)

    return {
        "pre_l0": full_pre_l0[shared_idx],
        "post_l0": full_post_l0[shared_idx],
        "post_l1": full_post_l1[shared_idx],
    }


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def center_matrix(X: np.ndarray) -> np.ndarray:
    """Row-center X (subtract column means)."""
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between matrices X (n x p) and Y (n x q).

    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    after centering X and Y.
    """
    Xc = center_matrix(X)
    Yc = center_matrix(Y)

    # Numerator: ||Xc^T Yc||_F^2
    XtY = Xc.T @ Yc
    num = float(np.linalg.norm(XtY, "fro") ** 2)

    # Denominator: ||Xc^T Xc||_F * ||Yc^T Yc||_F
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    den = float(np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro"))

    if den < 1e-12:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# PCA analysis
# ---------------------------------------------------------------------------

def run_pca_analysis(activations_per_seed: list, interface: str) -> dict:
    """Fit shared PCA on stacked activations, return results dict."""
    from sklearn.decomposition import PCA

    # Stack: (20 * SUBSAMPLE_N, D)
    stacked = np.concatenate([a[interface] for a in activations_per_seed], axis=0)
    print(f"  [{interface}] PCA on stacked shape {stacked.shape}")

    pca = PCA(n_components=PCA_N_COMPONENTS)
    pca.fit(stacked)

    explained_var = pca.explained_variance_ratio_.tolist()
    print(f"  [{interface}] Top-10 explained variance: {[f'{v:.4f}' for v in explained_var[:10]]}")

    # Project each seed separately
    projections = []
    for seed_idx, a in enumerate(activations_per_seed):
        proj = pca.transform(a[interface])  # (SUBSAMPLE_N, 50)
        projections.append(proj)

    return {
        "explained_variance_ratio_top10": explained_var[:10],
        "explained_variance_ratio_cumulative_top10": float(np.sum(explained_var[:10])),
        "projections": projections,  # list of 20 arrays (SUBSAMPLE_N, 50)
        "pca": pca,
    }


# ---------------------------------------------------------------------------
# CKA heatmap
# ---------------------------------------------------------------------------

def run_cka_analysis(activations_per_seed: list, interface: str) -> dict:
    """Compute 20x20 CKA matrix for the given interface."""
    print(f"  [{interface}] Computing {N_SEEDS}x{N_SEEDS} CKA matrix ({N_SEEDS*(N_SEEDS-1)//2} pairs)...")
    cka_matrix = np.zeros((N_SEEDS, N_SEEDS))

    for i in range(N_SEEDS):
        cka_matrix[i, i] = 1.0
        Xi = activations_per_seed[i][interface]
        for j in range(i + 1, N_SEEDS):
            Xj = activations_per_seed[j][interface]
            val = linear_cka(Xi, Xj)
            cka_matrix[i, j] = val
            cka_matrix[j, i] = val

    # Stats (off-diagonal)
    off_diag = cka_matrix[np.triu_indices(N_SEEDS, k=1)]
    print(f"  [{interface}] CKA off-diag: mean={off_diag.mean():.4f}, min={off_diag.min():.4f}, max={off_diag.max():.4f}")

    return {
        "cka_matrix": cka_matrix,
        "mean_cka": float(off_diag.mean()),
        "min_cka": float(off_diag.min()),
        "max_cka": float(off_diag.max()),
    }


# ---------------------------------------------------------------------------
# Cross-reference: CKA vs shared critical heads
# ---------------------------------------------------------------------------

def run_cross_reference(
    cka_results: dict,
    per_seed_critical_heads: list,
    interface: str = "post_l0",
) -> dict:
    """Scatter CKA(i,j) vs shared_critical_heads(i,j), compute Pearson r.

    Uses post_l0 CKA matrix as the primary interface for cross-reference.
    """
    cka_matrix = cka_results[interface]["cka_matrix"]

    cka_vals = []
    shared_vals = []

    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            head_i = per_seed_critical_heads[i]
            head_j = per_seed_critical_heads[j]
            shared = int(head_i == head_j)
            cka_vals.append(cka_matrix[i, j])
            shared_vals.append(shared)

    cka_arr = np.array(cka_vals)
    shared_arr = np.array(shared_vals)

    r, p = stats.pearsonr(shared_arr, cka_arr)
    print(f"  [cross-ref] Interface={interface}: Pearson r={r:.4f}, p={p:.4f}")
    print(f"  [cross-ref] shared_critical_head pairs: {shared_arr.sum()} / {len(shared_arr)}")

    # Mean CKA for shared vs non-shared
    mean_cka_shared = float(cka_arr[shared_arr == 1].mean()) if shared_arr.sum() > 0 else float("nan")
    mean_cka_nonshared = float(cka_arr[shared_arr == 0].mean()) if (shared_arr == 0).sum() > 0 else float("nan")
    print(f"  [cross-ref] mean CKA shared={mean_cka_shared:.4f}, non-shared={mean_cka_nonshared:.4f}")

    return {
        "interface_used": interface,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "n_pairs": len(cka_vals),
        "n_shared_critical_head_pairs": int(shared_arr.sum()),
        "mean_cka_shared_critical_heads": mean_cka_shared,
        "mean_cka_nonshared_critical_heads": mean_cka_nonshared,
        "cka_vals": cka_arr.tolist(),
        "shared_vals": shared_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pca_scatter(pca_results: dict, interface: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projections = pca_results["projections"]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8, 7))

    for seed_idx, proj in enumerate(projections):
        color = cmap(seed_idx / 20.0)
        # Subsample points for clarity in scatter (max 512 per seed)
        n_show = min(512, proj.shape[0])
        step = max(1, proj.shape[0] // n_show)
        ax.scatter(
            proj[::step, 0],
            proj[::step, 1],
            color=color,
            alpha=0.4,
            s=4,
            label=f"seed {seed_idx}",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ev = pca_results["explained_variance_ratio_top10"]
    ax.set_title(
        f"PCA scatter — {interface}\nPC1={ev[0]:.3f}, PC2={ev[1]:.3f} explained var"
    )
    # Legend outside
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=6,
        markerscale=3,
        ncol=1,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cka_heatmaps(cka_results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, interface in zip(axes, INTERFACE_NAMES):
        mat = cka_results[interface]["cka_matrix"]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(
            f"{interface}\nmean={cka_results[interface]['mean_cka']:.3f}"
        )
        ax.set_xlabel("Seed index")
        ax.set_ylabel("Seed index")
        ax.set_xticks(range(0, N_SEEDS, 5))
        ax.set_yticks(range(0, N_SEEDS, 5))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Linear CKA heatmaps (20 seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cka_vs_shared_heads(cross_ref: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    cka_vals = np.array(cross_ref["cka_vals"])
    shared_vals = np.array(cross_ref["shared_vals"])

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ["steelblue" if s == 0 else "tomato" for s in shared_vals]
    ax.scatter(shared_vals + np.random.default_rng(7).uniform(-0.05, 0.05, len(shared_vals)),
               cka_vals, c=colors, alpha=0.5, s=20)

    # Regression line
    slope, intercept, r, p, _ = scipy_stats.linregress(shared_vals, cka_vals)
    x_line = np.array([0, 1])
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=2,
            label=f"r={r:.3f}, p={p:.3f}")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Different critical head", "Shared critical head"])
    ax.set_ylabel(f"CKA ({cross_ref['interface_used']})")
    ax.set_title(
        f"CKA vs shared critical heads\nPearson r={cross_ref['pearson_r']:.3f}, p={cross_ref['pearson_p']:.4f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT F: Residual Stream Geometry")
    print("Question: Are 20 seeds rotations of the same manifold?")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Build fixed eval set
    # -----------------------------------------------------------------------
    print("\n[Step 1] Building fixed eval set (seed=9999, 1024 sequences)...")
    fixed_eval_set = build_fixed_eval_set()

    # -----------------------------------------------------------------------
    # Step 2: Load content matching results for critical heads
    # -----------------------------------------------------------------------
    print("\n[Step 2] Loading per-seed critical heads from content_matching_20_seeds_results_v2.json...")
    cm_path = Path(__file__).parent / "content_matching_20_seeds_results_v2.json"
    with open(cm_path) as f:
        cm_data = json.load(f)

    per_seed_critical_heads = []
    for entry in cm_data["per_seed"]:
        layer = entry["critical_head_layer"]
        head = entry["critical_head_index"]
        per_seed_critical_heads.append((layer, head))
        print(f"  seed {entry['seed']}: critical head = L{layer}H{head}")

    # -----------------------------------------------------------------------
    # Step 3: Collect activations for all 20 seeds
    # -----------------------------------------------------------------------
    print(f"\n[Step 3] Collecting residual stream activations for {N_SEEDS} seeds...")
    print(f"  Subsample: {SUBSAMPLE_N} rows per seed per interface (seed={SUBSAMPLE_SEED})")
    print(f"  NOTE: all seeds use the SAME shared index array (required for valid CKA)")

    # Build shared index ONCE — same positions compared across all seeds for valid CKA
    # Total rows: 1024 sequences * 64 tokens = 65536; subsample to 4096
    n_total = 1024 * 64  # EVAL_N_SEQUENCES * EVAL_SEQ_LEN
    shared_idx = build_shared_subsample_idx(n_total)
    print(f"  Shared index: {len(shared_idx)} rows sampled from {n_total} total")

    activations_per_seed = []

    for seed_idx in range(N_SEEDS):
        ckpt = get_checkpoint_path(seed_idx)
        print(f"  Loading seed {seed_idx}: {ckpt.split('/')[-3]}/{ckpt.split('/')[-2]}")
        model = load_model(ckpt, MODEL_CONFIG)

        acts = collect_activations_for_seed(model, fixed_eval_set, shared_idx)
        activations_per_seed.append(acts)

        for iface in INTERFACE_NAMES:
            print(f"    {iface}: shape={acts[iface].shape}")

        # Explicitly free model from memory
        del model

    print(f"\n  Done. Collected activations for {N_SEEDS} seeds.")

    # -----------------------------------------------------------------------
    # Step 4: PCA analysis (per interface)
    # -----------------------------------------------------------------------
    print("\n[Step 4] PCA analysis (shared 50-D basis per interface)...")

    pca_results = {}
    for interface in INTERFACE_NAMES:
        print(f"\n  --- Interface: {interface} ---")
        pca_results[interface] = run_pca_analysis(activations_per_seed, interface)

    # -----------------------------------------------------------------------
    # Step 5: CKA heatmap (per interface)
    # -----------------------------------------------------------------------
    print("\n[Step 5] CKA heatmap (linear CKA, 190 pairs per interface)...")

    cka_results = {}
    for interface in INTERFACE_NAMES:
        print(f"\n  --- Interface: {interface} ---")
        cka_results[interface] = run_cka_analysis(activations_per_seed, interface)

    # -----------------------------------------------------------------------
    # Step 6: Cross-reference CKA vs shared critical heads
    # -----------------------------------------------------------------------
    print("\n[Step 6] Cross-reference: CKA vs shared critical heads...")
    # Run cross-reference on all three interfaces, report all
    cross_ref_results = {}
    for interface in INTERFACE_NAMES:
        cross_ref_results[interface] = run_cross_reference(
            cka_results, per_seed_critical_heads, interface=interface
        )

    # Primary cross-reference is post_l0
    primary_cross_ref = cross_ref_results["post_l0"]

    # -----------------------------------------------------------------------
    # Step 7: Plotting
    # -----------------------------------------------------------------------
    print("\n[Step 7] Generating plots...")

    for interface in INTERFACE_NAMES:
        plot_pca_scatter(
            pca_results[interface],
            interface,
            PLOTS_DIR / f"residual_pca_{interface}.png",
        )

    plot_cka_heatmaps(cka_results, PLOTS_DIR / "residual_cka_heatmap.png")

    plot_cka_vs_shared_heads(
        primary_cross_ref,
        PLOTS_DIR / "cka_vs_shared_heads.png",
    )

    # -----------------------------------------------------------------------
    # Step 8: Save results JSON
    # -----------------------------------------------------------------------
    print("\n[Step 8] Saving results JSON...")

    results = {
        "experiment": "residual_stream_geometry",
        "question": "Are 20 seeds rotations of the same manifold, or different manifolds?",
        "protocol": {
            "n_seeds": N_SEEDS,
            "subsample_n": SUBSAMPLE_N,
            "subsample_seed": SUBSAMPLE_SEED,
            "pca_n_components": PCA_N_COMPONENTS,
            "eval_seed": 9999,
            "n_sequences": 1024,
        },
        "per_seed_critical_heads": [
            {"seed": i, "layer": h[0], "head": h[1]}
            for i, h in enumerate(per_seed_critical_heads)
        ],
        "interfaces": {},
        "cross_reference": {},
    }

    for interface in INTERFACE_NAMES:
        pca_r = pca_results[interface]
        cka_r = cka_results[interface]
        results["interfaces"][interface] = {
            "pca": {
                "explained_variance_ratio_top10": pca_r["explained_variance_ratio_top10"],
                "explained_variance_ratio_cumulative_top10": pca_r["explained_variance_ratio_cumulative_top10"],
            },
            "cka": {
                "matrix_20x20": cka_r["cka_matrix"].tolist(),
                "mean_cka": cka_r["mean_cka"],
                "min_cka": cka_r["min_cka"],
                "max_cka": cka_r["max_cka"],
            },
        }

    for interface in INTERFACE_NAMES:
        cr = cross_ref_results[interface]
        results["cross_reference"][interface] = {
            "pearson_r": cr["pearson_r"],
            "pearson_p": cr["pearson_p"],
            "n_pairs": cr["n_pairs"],
            "n_shared_critical_head_pairs": cr["n_shared_critical_head_pairs"],
            "mean_cka_shared_critical_heads": cr["mean_cka_shared_critical_heads"],
            "mean_cka_nonshared_critical_heads": cr["mean_cka_nonshared_critical_heads"],
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for interface in INTERFACE_NAMES:
        cka_r = cka_results[interface]
        pca_r = pca_results[interface]
        cr = cross_ref_results[interface]
        print(f"\n  Interface: {interface}")
        print(f"    CKA: mean={cka_r['mean_cka']:.4f}, min={cka_r['min_cka']:.4f}, max={cka_r['max_cka']:.4f}")
        print(f"    PCA top-10 cumulative explained var: {pca_r['explained_variance_ratio_cumulative_top10']:.4f}")
        print(f"    CKA vs shared heads: Pearson r={cr['pearson_r']:.4f}, p={cr['pearson_p']:.4f}")
        print(f"    mean CKA shared={cr['mean_cka_shared_critical_heads']:.4f}, non-shared={cr['mean_cka_nonshared_critical_heads']:.4f}")

    print("\n[Done] Experiment F complete.")


if __name__ == "__main__":
    main()
