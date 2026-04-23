"""
Mechanistic analysis of the induction grokking transition.

Loads checkpoints and tracks per-head circuit metrics through training
to understand the order in which components develop.
"""

import sys
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import GPT, GPTConfig
from src.data import InductionDataset


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG = GPTConfig(
    n_layers=2, n_heads=4, d_model=128, d_ff=512, vocab_size=512, ctx_len=64
)
DATASET = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=42)

CKPT_BASE = ROOT / "checkpoints" / "induction_content_match" / "induction_seed0"
OUT_DIR   = ROOT / "checkpoints" / "induction_content_match"
PLOT_DIR  = OUT_DIR / "plots"
RESULT_FILE = OUT_DIR / "transition_analysis.json"

N_SAMPLES = 256
TRANSITION_START = 11000
TRANSITION_END   = 13500


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir: Path) -> GPT:
    model = GPT(MODEL_CONFIG)
    model.load_weights(str(ckpt_dir / "model.safetensors"))
    mx.eval(model.parameters())
    return model


def list_checkpoints() -> list[tuple[int, Path]]:
    entries = []
    for d in sorted(CKPT_BASE.glob("step_*")):
        step = int(d.name.split("_")[1])
        entries.append((step, d))
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Metric 1: Task accuracy
# ──────────────────────────────────────────────────────────────────────────────

def compute_accuracy(model: GPT, n_samples: int = N_SAMPLES) -> float:
    """Induction accuracy on n_samples sequences."""
    inputs, targets, mask = DATASET.generate_batch(n_samples)
    logits = model(inputs)                          # (B, T, V)
    mx.eval(logits)
    preds   = np.array(logits).argmax(axis=-1)      # (B, T)
    tgts_np = np.array(targets)
    mask_np = np.array(mask)

    correct = ((preds == tgts_np) * mask_np).sum()
    total   = mask_np.sum()
    return float(correct) / float(total + 1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# Metric 2: Positional dependence per head
# ──────────────────────────────────────────────────────────────────────────────

def compute_pos_dep(attn_np: np.ndarray) -> float:
    """
    How positional (vs content-dependent) is this head?

    Positional attention: the pattern is the SAME across all inputs (low batch variance).
    Content attention: the pattern DIFFERS across inputs (high batch variance).

    Returns value in [0,1]: 1 = purely positional, 0 = purely content-based.
    """
    # attn_np: (B, T, T)
    mean_pattern = attn_np.mean(axis=0)          # (T, T)
    variance     = attn_np.var(axis=0).mean()    # scalar
    mean_val     = mean_pattern.mean()
    pos_dep      = 1.0 - min(1.0, variance / (mean_val + 1e-8))
    return float(pos_dep)


# ──────────────────────────────────────────────────────────────────────────────
# Metric 3: OV copy score (diagonal dominance in token space)
# ──────────────────────────────────────────────────────────────────────────────

def compute_copy_score(model: GPT, layer: int, head: int) -> float:
    """
    Does the OV circuit copy token identity?

    Compute token_ov = E @ (O_h @ V_h) @ E^T  (vocab x vocab)
    and measure diagonal dominance.
    """
    block   = model.blocks[layer]
    d_model = MODEL_CONFIG.d_model
    d_head  = MODEL_CONFIG.d_head

    qkv_w = np.array(block.attn.qkv_proj.weight)   # (3*d_model, d_model)
    out_w = np.array(block.attn.out_proj.weight)    # (d_model, d_model)
    embed = np.array(model.wte.weight)              # (vocab, d_model)

    # V weight for this head:  qkv layout rows = [Q0..Qh | K0..Kh | V0..Vh]
    v_all = qkv_w[2 * d_model:, :]                 # (d_model, d_model)
    v_head = v_all[head * d_head:(head + 1) * d_head, :]  # (d_head, d_model)

    # O weight for this head
    o_head = out_w[:, head * d_head:(head + 1) * d_head]  # (d_model, d_head)

    ov = o_head @ v_head                            # (d_model, d_model)
    token_ov = embed @ ov @ embed.T                 # (vocab, vocab)

    diag     = np.diag(token_ov)
    off_mean = (token_ov.sum() - diag.sum()) / (token_ov.size - len(diag))
    score    = float((diag.mean() - off_mean) / (np.abs(token_ov).mean() + 1e-10))
    return score


# ──────────────────────────────────────────────────────────────────────────────
# Metric 4: QK token-identity score
# ──────────────────────────────────────────────────────────────────────────────

def compute_qk_identity_score(model: GPT, layer: int, head: int) -> float:
    """
    Does QK produce high attention between IDENTICAL tokens?

    Compute qk_token = E @ Q_h^T @ K_h @ E^T  (vocab x vocab)
    and measure diagonal dominance.

    High diagonal = head learns to match tokens by identity (content matching).
    """
    block   = model.blocks[layer]
    d_model = MODEL_CONFIG.d_model
    d_head  = MODEL_CONFIG.d_head

    qkv_w  = np.array(block.attn.qkv_proj.weight)  # (3*d_model, d_model)
    embed  = np.array(model.wte.weight)             # (vocab, d_model)

    q_all  = qkv_w[:d_model, :]                    # (d_model, d_model) — all Q heads
    k_all  = qkv_w[d_model:2 * d_model, :]         # (d_model, d_model) — all K heads

    q_head = q_all[head * d_head:(head + 1) * d_head, :]  # (d_head, d_model)
    k_head = k_all[head * d_head:(head + 1) * d_head, :]  # (d_head, d_model)

    # QK score for each token pair: (E @ Q^T) @ (K @ E^T)
    # = vocab x d_head  @  d_head x vocab  = vocab x vocab
    qk_token = (embed @ q_head.T) @ (k_head @ embed.T)  # (vocab, vocab)

    scale = d_head ** -0.5
    qk_token = qk_token * scale

    diag     = np.diag(qk_token)
    off_mean = (qk_token.sum() - diag.sum()) / (qk_token.size - len(diag))
    score    = float((diag.mean() - off_mean) / (np.abs(qk_token).mean() + 1e-10))
    return score


# ──────────────────────────────────────────────────────────────────────────────
# Metric 5: Per-head induction score
# ──────────────────────────────────────────────────────────────────────────────

def compute_induction_score(attn_np: np.ndarray, inputs_np: np.ndarray) -> float:
    """
    For actual sequences, how much does each position attend to where the
    current token previously appeared?

    For position t with token x, the induction target is the position s < t
    where inputs[s] == x (the most recent prior occurrence).
    We measure the mean attention weight placed on that target.
    """
    B, T = inputs_np.shape
    # attn_np: (B, T, T) — attention from each query position to each key position

    scores = []
    for b in range(B):
        for t in range(2, T):
            tok = inputs_np[b, t]
            # Find most recent prior occurrence of tok
            prior_pos = None
            for s in range(t - 1, -1, -1):
                if inputs_np[b, s] == tok:
                    prior_pos = s
                    break
            if prior_pos is None:
                continue
            # Attention weight this position places on the prior occurrence
            scores.append(float(attn_np[b, t, prior_pos]))

    if not scores:
        return 0.0
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# Main per-checkpoint analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_checkpoint(step: int, ckpt_dir: Path) -> dict:
    model = load_checkpoint(ckpt_dir)

    # Generate shared batch
    inputs, targets, mask = DATASET.generate_batch(N_SAMPLES)
    logits = model(inputs)
    mx.eval(logits)
    patterns = model.get_attention_patterns()   # list of (B, n_heads, T, T)
    mx.eval(patterns)

    # Task accuracy
    preds   = np.array(logits).argmax(axis=-1)
    tgts_np = np.array(targets)
    mask_np = np.array(mask)
    correct = ((preds == tgts_np) * mask_np).sum()
    total   = mask_np.sum()
    accuracy = float(correct) / float(total + 1e-10)

    # Loss (cross-entropy on all positions)
    logits_np = np.array(logits)     # (B, T, V)
    # stable softmax cross-entropy
    B, T, V = logits_np.shape
    logits_flat = logits_np.reshape(B * T, V)
    tgts_flat   = tgts_np.reshape(B * T)
    log_softmax = logits_flat - np.log(np.exp(logits_flat - logits_flat.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)) - logits_flat.max(axis=-1, keepdims=True)
    loss_vals   = -log_softmax[np.arange(B * T), tgts_flat]
    loss        = float(loss_vals.mean())

    inputs_np = np.array(inputs)

    head_metrics = {}
    n_layers = MODEL_CONFIG.n_layers
    n_heads  = MODEL_CONFIG.n_heads

    for layer in range(n_layers):
        attn_all = np.array(patterns[layer])   # (B, n_heads, T, T)

        for head in range(n_heads):
            attn_h = attn_all[:, head, :, :]   # (B, T, T)
            key    = f"L{layer}H{head}"

            pos_dep    = compute_pos_dep(attn_h)
            copy_score = compute_copy_score(model, layer, head)
            qk_id_score = compute_qk_identity_score(model, layer, head)
            ind_score  = compute_induction_score(attn_h, inputs_np)

            head_metrics[key] = {
                "pos_dep":         pos_dep,
                "copy_score":      copy_score,
                "qk_identity_score": qk_id_score,
                "induction_score": ind_score,
            }

    return {
        "step":         step,
        "accuracy":     accuracy,
        "loss":         loss,
        "head_metrics": head_metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

HEAD_COLORS = {
    "L0H0": "#1f77b4",
    "L0H1": "#aec7e8",
    "L0H2": "#17becf",
    "L0H3": "#9edae5",
    "L1H0": "#d62728",
    "L1H1": "#ff9896",
    "L1H2": "#e377c2",
    "L1H3": "#f7b6d2",
}

METRIC_LABELS = {
    "pos_dep":           "Pos Dependence",
    "copy_score":        "OV Copy Score",
    "qk_identity_score": "QK Identity Score",
    "induction_score":   "Induction Score",
}


def shade_transition(ax, alpha=0.12):
    ax.axvspan(TRANSITION_START, TRANSITION_END, color="orange", alpha=alpha,
               label="Transition region")


def plot_transition_overview(results: list[dict], plot_dir: Path):
    steps    = [r["step"] for r in results]
    accs     = [r["accuracy"] for r in results]
    losses   = [r["loss"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_acc, ax_loss = axes

    shade_transition(ax_acc)
    shade_transition(ax_loss)

    ax_acc.plot(steps, accs, "o-", color="#2ca02c", markersize=4, lw=2)
    ax_acc.set_ylabel("Induction Accuracy", fontsize=12)
    ax_acc.set_ylim(-0.02, 1.05)
    ax_acc.set_title("Induction Grokking Transition", fontsize=14)
    ax_acc.axhline(0.85, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_acc.axhline(0.05, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_acc.legend(loc="upper left", fontsize=9)

    ax_loss.plot(steps, losses, "o-", color="#d62728", markersize=4, lw=2)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax_loss.set_xlabel("Training Step", fontsize=12)

    # Annotate transition
    ax_acc.annotate(
        f"Transition\n{TRANSITION_START}–{TRANSITION_END}",
        xy=((TRANSITION_START + TRANSITION_END) / 2, 0.5),
        fontsize=9, ha="center", color="darkorange",
    )

    plt.tight_layout()
    fig.savefig(plot_dir / "transition_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved transition_overview.png")


def plot_head_metrics_over_training(results: list[dict], plot_dir: Path):
    steps = [r["step"] for r in results]
    n_layers = MODEL_CONFIG.n_layers
    n_heads  = MODEL_CONFIG.n_heads
    metrics  = ["pos_dep", "copy_score", "qk_identity_score", "induction_score"]

    # 2 rows (layers) x 4 cols (metrics), one line per head per subplot
    # Actually task says: 2x4 grid one per head — so n_layers x n_heads = 8 subplots
    # Each subplot shows all 4 metrics for that head over training
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(16, 7), sharex=True)

    metric_colors = {
        "pos_dep":           "#1f77b4",
        "copy_score":        "#ff7f0e",
        "qk_identity_score": "#2ca02c",
        "induction_score":   "#d62728",
    }

    for layer in range(n_layers):
        for head in range(n_heads):
            ax  = axes[layer][head]
            key = f"L{layer}H{head}"

            shade_transition(ax, alpha=0.15)

            for metric in metrics:
                vals = [r["head_metrics"][key][metric] for r in results]
                ax.plot(steps, vals, lw=1.5, label=METRIC_LABELS[metric],
                        color=metric_colors[metric])

            ax.set_title(f"L{layer} H{head}", fontsize=10, fontweight="bold")
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            if head == 0:
                ax.set_ylabel(f"Layer {layer}", fontsize=9)
            if layer == n_layers - 1:
                ax.set_xlabel("Step", fontsize=8)

    # Single legend below the grid
    handles = [
        mpatches.Patch(color=metric_colors[m], label=METRIC_LABELS[m])
        for m in metrics
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Per-Head Circuit Metrics over Training\n(orange shading = transition region)",
                 fontsize=12)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(plot_dir / "head_metrics_over_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved head_metrics_over_training.png")


def find_crystallization_step(
    vals: np.ndarray,
    steps: np.ndarray,
    thresh: float,
    direction: str,
    require_recovery: bool = False,
) -> int:
    """
    Find the training step when a metric first crosses a threshold.

    If require_recovery=True, find the crossing that happens AFTER the metric
    has already gone to the other side of the threshold (i.e., the recovery
    crossing, not the initial state). This handles U-shaped trajectories where
    a metric starts above threshold, dips below during early training, then
    recovers during the phase transition.
    """
    if direction == "up":
        above = vals > thresh
    else:
        above = vals < thresh  # reuse variable; means "past threshold"

    if not require_recovery:
        # Simple: first time we cross
        crossed = np.where(above)[0]
        return int(steps[crossed[0]]) if len(crossed) > 0 else int(steps[-1]) + 1000

    # Recovery crossing: find first index where above is True,
    # but only after at least one index where above is False.
    found_below = False
    for i, a in enumerate(above):
        if not a:
            found_below = True
        elif found_below:
            return int(steps[i])
    return int(steps[-1]) + 1000   # never recovered


def plot_circuit_formation_order(results: list[dict], plot_dir: Path):
    """
    For each (head, metric), find when the metric crystallizes (crosses threshold).
    Plot crystallization order as a horizontal bar chart.

    Thresholds are calibrated to actual value ranges observed in this model:
    - copy_score:        1.0   (range -0.2 to 6.4; crosses monotonically early)
    - qk_identity_score: -3.5  (U-shaped: starts ~0, dips to -5, recovers during transition;
                                 use recovery crossing so we detect the grokking-related rise)
    - induction_score:   0.015 (range 0.005 to 0.037; baseline ~0.010, rises during transition)
    """
    steps = np.array([r["step"] for r in results])

    # (metric, threshold, direction, require_recovery)
    # copy_score: monotonically rises from 0; simple first-crossing
    # qk_identity_score: U-shaped (starts ~0, dips to -5, recovers); find recovery crossing
    # induction_score: U-shaped (starts ~0.027, dips to ~0.009, recovers); find recovery crossing
    #   threshold 0.020: well above the trough, clearly marks the post-transition rise
    thresholds = [
        ("copy_score",        1.0,   "up",  False),
        ("qk_identity_score", -3.5,  "up",  True),   # U-shaped; find recovery crossing
        ("induction_score",   0.020, "up",  True),    # U-shaped; find recovery crossing
    ]

    n_layers = MODEL_CONFIG.n_layers
    n_heads  = MODEL_CONFIG.n_heads

    crystallization = []

    for layer in range(n_layers):
        for head in range(n_heads):
            key = f"L{layer}H{head}"
            for metric, thresh, direction, recovery in thresholds:
                vals = np.array([r["head_metrics"][key][metric] for r in results])
                cryst_step = find_crystallization_step(vals, steps, thresh, direction, recovery)
                crystallization.append({
                    "label":  f"{key}.{metric}",
                    "head":   key,
                    "metric": metric,
                    "step":   cryst_step,
                })

    crystallization.sort(key=lambda x: x["step"])

    fig, ax = plt.subplots(figsize=(12, max(6, len(crystallization) * 0.35)))

    metric_colors = {
        "pos_dep":           "#1f77b4",
        "copy_score":        "#ff7f0e",
        "qk_identity_score": "#2ca02c",
        "induction_score":   "#d62728",
    }

    y_ticks = []
    y_labels = []
    for i, entry in enumerate(crystallization):
        color = metric_colors[entry["metric"]]
        never = entry["step"] > int(steps[-1])
        bar_step = int(steps[-1]) if never else entry["step"]
        ax.barh(i, bar_step, color=color, alpha=0.7 if not never else 0.2, height=0.7)
        y_ticks.append(i)
        marker = " (never)" if never else ""
        y_labels.append(f"{entry['head']} · {METRIC_LABELS[entry['metric']]}{marker}")

    ax.axvline(TRANSITION_START, color="darkorange", lw=1.5, ls="--", label=f"Transition start ({TRANSITION_START})")
    ax.axvline(TRANSITION_END,   color="darkorange", lw=1.5, ls=":",  label=f"Transition end ({TRANSITION_END})")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Training Step of First Crystallization", fontsize=11)
    ax.set_title("Circuit Formation Order\n(when each metric first crosses threshold)", fontsize=12)

    handles = [mpatches.Patch(color=metric_colors[m], label=METRIC_LABELS[m])
               for m in metric_colors]
    handles += [
        plt.Line2D([0], [0], color="darkorange", ls="--", lw=1.5, label=f"Transition start"),
        plt.Line2D([0], [0], color="darkorange", ls=":",  lw=1.5, label=f"Transition end"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(plot_dir / "circuit_formation_order.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved circuit_formation_order.png")


# ──────────────────────────────────────────────────────────────────────────────
# Summary analysis
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    steps = np.array([r["step"] for r in results])
    accs  = np.array([r["accuracy"] for r in results])

    # Find transition step (first step where accuracy > 50%)
    trans_idx = np.where(accs > 0.5)[0]
    trans_step = int(steps[trans_idx[0]]) if len(trans_idx) > 0 else None

    print("\n" + "=" * 65)
    print("SUMMARY: Mechanistic Transition Analysis")
    print("=" * 65)

    print(f"\nAccuracy jump: {'step ' + str(trans_step) if trans_step else 'not observed'}")
    print(f"Pre-transition acc  ({TRANSITION_START}):  "
          f"{accs[steps == TRANSITION_START][0]:.1%}" if TRANSITION_START in steps else "N/A")
    print(f"Post-transition acc ({TRANSITION_END}): "
          f"{accs[steps == TRANSITION_END][0]:.1%}" if TRANSITION_END in steps else "N/A")

    print("\n--- Crystallization order for key metrics ---")
    # Thresholds calibrated to actual value ranges:
    # copy_score: range -0.2 to 6.4  -> threshold 1.0 (crosses monotonically early)
    # qk_identity_score: U-shaped, range -5.1 to 0.38 -> track recovery above -3.5
    # induction_score: U-shaped, starts ~0.027, dips to ~0.009, recovers -> threshold 0.020
    thresholds = {
        "copy_score":        (1.0,   "up"),
        "qk_identity_score": (-3.5,  "up"),
        "induction_score":   (0.020, "up"),
    }
    events = []
    for layer in range(MODEL_CONFIG.n_layers):
        for head in range(MODEL_CONFIG.n_heads):
            key = f"L{layer}H{head}"
            for metric, (thresh, direction) in thresholds.items():
                vals = np.array([r["head_metrics"][key][metric] for r in results])
                # qk_identity_score and induction_score are U-shaped — find recovery crossing
                recovery = metric in ("qk_identity_score", "induction_score")
                cryst = find_crystallization_step(vals, steps, thresh, direction, recovery)
                events.append((cryst, key, metric))

    events.sort()
    for step_c, key, metric in events:
        marker = "(before transition)" if step_c < TRANSITION_START else \
                 "(during transition)" if step_c <= TRANSITION_END   else \
                 "(after transition)"  if step_c < 99999             else "(never)"
        print(f"  step {step_c:>6}  {key}.{metric:<22} {marker}")

    # Does L0 OV develop before L1 QK?
    # Note: qk_identity_score is always negative in this model (off-diagonal > diagonal in QK space).
    # We track the RISE toward 0 — less negative = more identity-selective.
    # Threshold: -3.5 (midpoint of observed movement from -5.0 at init to ~-2.5 post-transition).
    print("\n--- Key questions ---")
    l0_copy_steps = []
    l1_qk_steps   = []
    for h in range(MODEL_CONFIG.n_heads):
        vals = np.array([r["head_metrics"][f"L0H{h}"]["copy_score"] for r in results])
        cr   = np.where(vals > 1.0)[0]
        l0_copy_steps.append(int(steps[cr[0]]) if len(cr) > 0 else 99999)

        vals = np.array([r["head_metrics"][f"L1H{h}"]["qk_identity_score"] for r in results])
        # U-shaped: use recovery crossing (after dip) to find grokking-related rise
        cryst = find_crystallization_step(vals, steps, -3.5, "up", require_recovery=True)
        l1_qk_steps.append(cryst)

    l0_copy_first = min(l0_copy_steps)
    l1_qk_first   = min(l1_qk_steps)

    if l0_copy_first < l1_qk_first:
        print(f"  L0 OV copy develops FIRST (step {l0_copy_first}) before L1 QK identity (step {l1_qk_first})")
        print(f"  => L0 token-identity writing precedes L1 content matching.")
    elif l0_copy_first > l1_qk_first:
        print(f"  L1 QK identity develops FIRST (step {l1_qk_first}) before L0 OV copy (step {l0_copy_first})")
        print(f"  => L1 content matching precedes L0 token-identity writing.")
    else:
        print(f"  L0 OV copy and L1 QK identity develop SIMULTANEOUSLY (both step {l0_copy_first})")

    # Precursor detection: any metric shows meaningful activity before TRANSITION_START?
    print(f"\n  Checking for precursors before step {TRANSITION_START}:")
    # copy_score: look for values > 1.0 (already developed)
    # qk_identity_score: look for values rising above -4.0 (early identity selectivity)
    # induction_score: look for values above 0.012 (above baseline ~0.010)
    precursor_thresholds = {
        "copy_score":        (1.0,   "up"),
        "qk_identity_score": (-4.0,  "up"),
        "induction_score":   (0.012, "up"),
    }
    found_precursor = False
    for metric, (thresh, direction) in precursor_thresholds.items():
        for layer in range(MODEL_CONFIG.n_layers):
            for head in range(MODEL_CONFIG.n_heads):
                key = f"L{layer}H{head}"
                vals = np.array([r["head_metrics"][key][metric]
                                 for r in results if r["step"] < TRANSITION_START])
                if len(vals) == 0:
                    continue
                extreme = vals.max() if direction == "up" else vals.min()
                crossed = extreme > thresh if direction == "up" else extreme < thresh
                if crossed:
                    print(f"  PRECURSOR: {key}.{metric} reaches {extreme:.4f} "
                          f"(thresh {thresh}) before transition")
                    found_precursor = True
    if not found_precursor:
        print("  No precursors detected above thresholds before the transition.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute metrics even if cached results exist")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if RESULT_FILE.exists() and not args.recompute:
        print(f"Loading cached results from {RESULT_FILE}")
        with open(RESULT_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} checkpoint results")
    else:
        checkpoints = list_checkpoints()
        print(f"Found {len(checkpoints)} checkpoints")

        results = []
        t0 = time.time()

        for idx, (step, ckpt_dir) in enumerate(checkpoints):
            elapsed = time.time() - t0
            print(f"[{idx+1:02d}/{len(checkpoints)}] step={step:>6}  "
                  f"(elapsed {elapsed:.0f}s)", end="  ", flush=True)

            try:
                r = analyze_checkpoint(step, ckpt_dir)
                results.append(r)
                acc = r["accuracy"]
                loss = r["loss"]
                print(f"acc={acc:.3f}  loss={loss:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")

        print(f"\nAll checkpoints processed in {time.time() - t0:.1f}s")

        # Save results
        print(f"\nSaving results to {RESULT_FILE}")
        with open(RESULT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    plot_transition_overview(results, PLOT_DIR)
    plot_head_metrics_over_training(results, PLOT_DIR)
    plot_circuit_formation_order(results, PLOT_DIR)

    # Print summary
    print_summary(results)

    print(f"\nDone. Plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
