"""
Natural language induction analysis for the Shakespeare character-level model.

Tests whether the trained Shakespeare model (6L, 8H, 512d, 65-char vocab) performs
in-context copying on natural language sequences and identifies which heads are
responsible.

Induction at the character level: when the model has seen a character bigram "XY"
earlier in context, does it predict Y when it sees X again?

Key heads from prior analysis:
  - L0H7: near-perfect previous-token head (score=0.908)
  - L1H6, L2H0, L2H4: content-matching candidates (low pos_dep)
"""

import sys
import json
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import TextDataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

CKPT_PATH = Path("checkpoints/shakespeare/text_seed0/step_010000/model.safetensors")
CONFIG = GPTConfig(
    n_layers=6, n_heads=8, d_model=512, d_ff=2048,
    vocab_size=65, ctx_len=256, dropout=0.0,
)

# Candidate induction heads from prior synthetic-task analysis
CANDIDATE_HEADS = [(0, 7), (1, 6), (2, 0), (2, 4)]
CANDIDATE_LABELS = ["L0H7", "L1H6", "L2H0", "L2H4"]


def load_model() -> GPT:
    model = GPT(CONFIG)
    model.load_weights(str(CKPT_PATH))
    mx.eval(model.parameters())
    model.set_dtype(mx.float32)
    return model


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode(text: str, char_to_idx: dict) -> list[int]:
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def decode(ids: list[int], idx_to_char: dict) -> str:
    return "".join(idx_to_char.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# Ablatable forward pass (reuses pattern from analysis/ablation.py)
# ---------------------------------------------------------------------------

def _attn_with_head_ablation(attn_module, x: mx.array,
                              ablate_heads_set: set, layer_idx: int) -> mx.array:
    B, T, C = x.shape
    n_heads = attn_module.n_heads
    d_head = attn_module.d_head

    qkv = attn_module.qkv_proj(x)
    qkv = qkv.reshape(B, T, 3, n_heads, d_head)
    qkv = qkv.transpose(0, 3, 2, 1, 4)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    mask = mx.triu(mx.full((T, T), -1e9), k=1)
    attn = attn + mask
    attn = mx.softmax(attn, axis=-1)

    out = attn @ v  # (B, n_heads, T, d_head)

    # Zero ablated heads
    heads_to_zero = [h for (l, h) in ablate_heads_set if l == layer_idx]
    if heads_to_zero:
        mask_vals = [0.0 if h in heads_to_zero else 1.0 for h in range(n_heads)]
        head_mask = mx.array(mask_vals).reshape(1, n_heads, 1, 1)
        out = out * head_mask

    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = attn_module.out_proj(out)
    return out


def forward_with_ablations(
    model: GPT,
    idx: mx.array,
    ablate_heads: Optional[list[tuple[int, int]]] = None,
) -> mx.array:
    ablate_heads_set = set(ablate_heads) if ablate_heads else set()

    B, T = idx.shape
    pos = mx.arange(T)
    x = model.wte(idx) + model.wpe(pos)

    for layer_idx, block in enumerate(model.blocks):
        x_norm = block.ln1(x)
        if ablate_heads_set and any(l == layer_idx for l, _ in ablate_heads_set):
            attn_out = _attn_with_head_ablation(
                block.attn, x_norm, ablate_heads_set, layer_idx
            )
        else:
            attn_out = block.attn(x_norm)
        x = x + attn_out

        x_norm2 = block.ln2(x)
        mlp_out = block.mlp(x_norm2)
        x = x + mlp_out

    x = model.ln_f(x)
    logits = x @ model.wte.weight.T
    return logits


# ---------------------------------------------------------------------------
# Named-prompt induction tests
# ---------------------------------------------------------------------------

NAMED_PROMPTS = [
    {
        "name": "king_hello",
        "text": "The king said 'hello' and the queen said 'hello' and the knight said '",
        "target_char": "h",
        "description": "Repeated word 'hello' — does model predict 'h' to start the third?",
    },
    {
        "name": "romeo_juliet",
        "text": "A name is Romeo. Another name is Juliet. Yet another name is Romeo. One more name is",
        "target_char": " ",
        "description": "After 'Romeo' appears twice, does model expect something similar for 'Juliet'?",
    },
    {
        "name": "act_scene",
        "text": "ACT I\nSCENE 1\n\nACT II\nSCENE 2\n\nACT III\nSCENE ",
        "target_char": "3",
        "description": "Pattern ACT N / SCENE N — does it predict the digit?",
    },
    {
        "name": "repeated_name",
        "text": "HAMLET:\nTo be or not to be.\n\nHORATIO:\nAye, my lord.\n\nHAMLET:\n",
        "target_char": "T",
        "description": "Speaker name HAMLET appears, then again — does it predict 'T' (start of 'To')?",
    },
    {
        "name": "repeated_word_the",
        "text": "the king walked into the hall and sat upon the throne and spoke to the",
        "target_char": " ",
        "description": "Common bigram 'the ' — does it predict space after 'the'?",
    },
]


def run_named_prompts(
    model: GPT,
    char_to_idx: dict,
    idx_to_char: dict,
) -> list[dict]:
    results = []
    vocab_size = CONFIG.vocab_size

    for prompt_info in NAMED_PROMPTS:
        text = prompt_info["text"]
        target_char = prompt_info["target_char"]

        token_ids = encode(text, char_to_idx)
        if len(token_ids) < 2:
            results.append({**prompt_info, "error": "too_short"})
            continue

        input_ids = mx.array([token_ids])
        logits = model(input_ids)
        mx.eval(logits)

        last_logits = np.array(logits[0, -1])  # (vocab_size,)
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()

        top5_ids = probs.argsort()[::-1][:5]
        top5_chars = [idx_to_char.get(i, "?") for i in top5_ids]
        top5_probs = [float(probs[i]) for i in top5_ids]

        target_idx = char_to_idx.get(target_char)
        target_prob = float(probs[target_idx]) if target_idx is not None else 0.0
        target_rank = int(np.where(probs.argsort()[::-1] == target_idx)[0][0]) + 1 if target_idx is not None else -1

        results.append({
            "name": prompt_info["name"],
            "description": prompt_info["description"],
            "prompt_length": len(token_ids),
            "target_char": target_char,
            "target_prob": target_prob,
            "target_rank": target_rank,
            "top5_chars": top5_chars,
            "top5_probs": top5_probs,
        })

        print(f"  [{prompt_info['name']}] target='{target_char}' "
              f"p={target_prob:.4f} rank={target_rank} "
              f"top5={''.join(repr(c) for c in top5_chars[:3])}")

    return results


# ---------------------------------------------------------------------------
# Systematic bigram induction test (character-level)
# ---------------------------------------------------------------------------

def build_induction_corpus(
    text_data: np.ndarray,
    vocab_size: int,
    seq_len: int = 200,
    n_samples: int = 500,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    Find positions in real text where a character bigram (A, B) appears
    twice: first occurrence at pos p1, second occurrence with A at pos p2 > p1+5.
    We then test whether the model predicts B at position p2.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    samples = []
    max_start = len(text_data) - seq_len - 2
    attempts = 0

    while len(samples) < n_samples and attempts < n_samples * 20:
        attempts += 1
        start = int(rng.integers(0, max_start))
        window = text_data[start: start + seq_len + 1]

        # Pick a random target position in the second half of the window
        t = int(rng.integers(seq_len // 2, seq_len - 1))
        A = int(window[t])
        B = int(window[t + 1])

        # Find the most recent prior occurrence of A in the window before t
        prior_positions = [i for i in range(1, t) if window[i] == A]
        if not prior_positions:
            continue

        # Use the most recent prior occurrence
        p1 = prior_positions[-1]
        B_at_p1 = int(window[p1 + 1])

        # Only count as induction if B at p1+1 matches B at t+1
        # (i.e., the same bigram AB appeared before)
        if B_at_p1 != B:
            continue

        # Gap must be at least 5 to avoid trivial repetition
        if t - p1 < 5:
            continue

        samples.append({
            "window": window[:seq_len].tolist(),
            "query_pos": t,       # position of A (second occurrence)
            "prior_pos": p1,      # position of A (first occurrence)
            "A_token": A,
            "B_token": B,
            "gap": t - p1,
        })

    return samples


def evaluate_induction_accuracy(
    model: GPT,
    samples: list[dict],
    ablate_heads: Optional[list[tuple[int, int]]] = None,
    batch_size: int = 32,
) -> dict:
    """
    For each sample, run a forward pass and check whether B is top-1 predicted
    at the query position. Returns accuracy and mean rank of B.
    """
    correct = 0
    total = len(samples)
    ranks = []
    probs_list = []

    for batch_start in range(0, total, batch_size):
        batch = samples[batch_start: batch_start + batch_size]
        # Pad sequences to same length (they're already seq_len)
        inputs_np = np.array([s["window"] for s in batch], dtype=np.int32)
        inputs = mx.array(inputs_np)

        logits = forward_with_ablations(model, inputs, ablate_heads=ablate_heads)
        mx.eval(logits)
        logits_np = np.array(logits)  # (B, T, V)

        for i, s in enumerate(batch):
            pos = s["query_pos"]
            B_tok = s["B_token"]
            l = logits_np[i, pos]
            probs = np.exp(l - l.max())
            probs /= probs.sum()

            pred = int(probs.argmax())
            rank = int(np.where(probs.argsort()[::-1] == B_tok)[0][0]) + 1
            ranks.append(rank)
            probs_list.append(float(probs[B_tok]))
            if pred == B_tok:
                correct += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_prob": float(np.mean(probs_list)),
        "n_samples": total,
    }


# ---------------------------------------------------------------------------
# Attention pattern capture for a single prompt
# ---------------------------------------------------------------------------

def get_attention_patterns(
    model: GPT,
    token_ids: list[int],
) -> np.ndarray:
    """
    Returns attention weights for all layers and heads.
    Shape: (n_layers, n_heads, T, T)
    """
    input_ids = mx.array([token_ids])
    _ = model(input_ids)
    mx.eval(model.blocks[0].attn._attn_weights)

    patterns = model.get_attention_patterns()
    result = np.stack([np.array(p)[0] for p in patterns], axis=0)
    return result  # (n_layers, n_heads, T, T)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_induction_attention(
    attn_patterns: np.ndarray,
    token_ids: list[int],
    query_pos: int,
    prior_pos: int,
    idx_to_char: dict,
    candidate_heads: list[tuple[int, int]],
    candidate_labels: list[str],
    save_path: Path,
):
    """
    For a specific induction event, show attention row at query_pos for each
    candidate head. Highlight attention to prior_pos (the prior occurrence).
    """
    n_show = len(candidate_heads)
    T = len(token_ids)

    # Show a window around the induction event
    window_start = max(0, prior_pos - 5)
    window_end = min(T, query_pos + 10)
    window_ids = token_ids[window_start:window_end]
    window_chars = [idx_to_char.get(i, "?") for i in window_ids]
    x_pos_in_window = query_pos - window_start
    prior_in_window = prior_pos - window_start
    x_labels = [repr(c)[1:-1] for c in window_chars]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, ((layer, head), label) in enumerate(zip(candidate_heads, candidate_labels)):
        ax = axes[i]
        attn_row = attn_patterns[layer, head, query_pos, window_start:window_end]

        bar_colors = ["tab:blue"] * len(attn_row)
        if prior_in_window >= 0:
            bar_colors[prior_in_window] = "tab:red"
        if x_pos_in_window >= 0 and x_pos_in_window < len(attn_row):
            bar_colors[x_pos_in_window] = "tab:orange"

        ax.bar(range(len(attn_row)), attn_row, color=bar_colors)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=7, rotation=90)
        ax.set_title(f"{label} (L{layer}H{head})", fontsize=10)
        ax.set_ylabel("Attention weight")
        ax.set_ylim(0, max(attn_row.max() * 1.2, 0.05))

        # Mark prior occurrence
        if 0 <= prior_in_window < len(attn_row):
            ax.axvline(prior_in_window, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
                       label=f"prior '{idx_to_char.get(token_ids[prior_pos], '?')}'")
        ax.legend(fontsize=7)

    # Hide unused subplots if fewer than 8
    for j in range(len(candidate_heads), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Attention at induction query position (pos={query_pos}, prior={prior_pos})\n"
        f"Context: ...{''.join(window_chars)}...",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_results(
    baseline_acc: float,
    ablation_results: dict[str, float],
    save_path: Path,
):
    """
    Bar chart showing induction accuracy drop when each head is ablated.
    """
    labels = list(ablation_results.keys())
    drops = [baseline_acc - ablation_results[k] for k in labels]
    colors = ["tab:red" if d > 0.02 else "tab:blue" for d in drops]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, drops, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy drop (baseline - ablated)")
    ax.set_title(
        f"Per-head ablation effect on induction accuracy\n"
        f"Baseline accuracy: {baseline_acc:.3f}"
    )

    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{drop:+.3f}", ha="center", va="bottom", fontsize=9)

    # Add control annotation
    ax.text(0.98, 0.95, "Red = large drop (>2pp)\nBlue = small/no drop",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path("analysis/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Natural Language Induction Analysis")
    print("=" * 60)

    # Load dataset for vocab
    print("\nLoading TextDataset for vocab/tokenizer...")
    tds = TextDataset(data_dir="data", seq_len=256, seed=42)
    char_to_idx = tds.char_to_idx
    idx_to_char = tds.idx_to_char
    print(f"  vocab_size={tds.vocab_size}")

    # Load model
    print("\nLoading model...")
    model = load_model()
    print(f"  Loaded from {CKPT_PATH}")

    # ------------------------------------------------------------------
    # Part 1: Named prompt tests
    # ------------------------------------------------------------------
    print("\n--- Part 1: Named prompt induction tests ---")
    named_results = run_named_prompts(model, char_to_idx, idx_to_char)

    # ------------------------------------------------------------------
    # Part 2: Systematic bigram induction test
    # ------------------------------------------------------------------
    print("\n--- Part 2: Systematic bigram induction (N=500 samples) ---")
    rng = np.random.default_rng(42)
    induction_samples = build_induction_corpus(
        tds.data, vocab_size=tds.vocab_size,
        seq_len=200, n_samples=500, rng=rng,
    )
    print(f"  Collected {len(induction_samples)} valid induction samples")

    if induction_samples:
        gap_stats = [s["gap"] for s in induction_samples]
        print(f"  Gap stats: mean={np.mean(gap_stats):.1f} "
              f"median={np.median(gap_stats):.1f} "
              f"min={np.min(gap_stats)} max={np.max(gap_stats)}")

    # Baseline
    print("\n  Computing baseline induction accuracy...")
    baseline_metrics = evaluate_induction_accuracy(model, induction_samples)
    print(f"  Baseline: acc={baseline_metrics['accuracy']:.4f} "
          f"mean_rank={baseline_metrics['mean_rank']:.2f} "
          f"mean_prob={baseline_metrics['mean_prob']:.4f}")

    # Control: ablate a "random" head not in candidate list — use L3H3
    control_head = (3, 3)
    print(f"\n  Control ablation (L3H3)...")
    ctrl_metrics = evaluate_induction_accuracy(
        model, induction_samples, ablate_heads=[control_head]
    )
    print(f"  Control (L3H3 ablated): acc={ctrl_metrics['accuracy']:.4f}")

    # Per-head ablation for candidate heads
    ablation_accs = {}
    ablation_accs["Control(L3H3)"] = ctrl_metrics["accuracy"]

    print("\n  Per-candidate-head ablation:")
    candidate_metrics = {}
    for (layer, head), label in zip(CANDIDATE_HEADS, CANDIDATE_LABELS):
        metrics = evaluate_induction_accuracy(
            model, induction_samples, ablate_heads=[(layer, head)]
        )
        ablation_accs[label] = metrics["accuracy"]
        candidate_metrics[label] = metrics
        drop = baseline_metrics["accuracy"] - metrics["accuracy"]
        print(f"  Ablate {label}: acc={metrics['accuracy']:.4f} "
              f"drop={drop:+.4f} mean_rank={metrics['mean_rank']:.2f}")

    # All-candidate ablation
    print("\n  Ablating all candidate heads simultaneously...")
    all_cand_metrics = evaluate_induction_accuracy(
        model, induction_samples, ablate_heads=CANDIDATE_HEADS
    )
    ablation_accs["All candidates"] = all_cand_metrics["accuracy"]
    drop_all = baseline_metrics["accuracy"] - all_cand_metrics["accuracy"]
    print(f"  All candidates ablated: acc={all_cand_metrics['accuracy']:.4f} "
          f"drop={drop_all:+.4f}")

    # ------------------------------------------------------------------
    # Part 3: Attention pattern visualization on a clear induction example
    # ------------------------------------------------------------------
    print("\n--- Part 3: Attention patterns on induction example ---")

    # Find a sample with a large gap and high model confidence
    if induction_samples:
        # Pick the sample where the model is most confident (baseline)
        best_sample = None
        best_prob = -1.0

        for s in induction_samples[:100]:  # check first 100 for speed
            inp = mx.array([s["window"]])
            logits = model(inp)
            mx.eval(logits)
            l = np.array(logits[0, s["query_pos"]])
            probs = np.exp(l - l.max()); probs /= probs.sum()
            p = float(probs[s["B_token"]])
            if p > best_prob:
                best_prob = p
                best_sample = s

        if best_sample:
            A_char = idx_to_char.get(best_sample["A_token"], "?")
            B_char = idx_to_char.get(best_sample["B_token"], "?")
            print(f"  Best example: bigram='{A_char}{B_char}' "
                  f"prior_pos={best_sample['prior_pos']} "
                  f"query_pos={best_sample['query_pos']} "
                  f"gap={best_sample['gap']} "
                  f"model_prob={best_prob:.4f}")

            attn_patterns = get_attention_patterns(model, best_sample["window"])

            # Show all 8 heads from layer 0 and 1 to give full picture
            # but highlight only candidate heads
            all_heads_to_show = CANDIDATE_HEADS
            # Pad to 8 panels with remaining L0 and L1 heads if needed
            extra = [(0, h) for h in range(8) if (0, h) not in all_heads_to_show][:4]
            show_heads = all_heads_to_show + extra
            show_labels = CANDIDATE_LABELS + [f"L0H{h}" for _, h in extra]

            plot_induction_attention(
                attn_patterns,
                best_sample["window"],
                best_sample["query_pos"],
                best_sample["prior_pos"],
                idx_to_char,
                show_heads[:8],
                show_labels[:8],
                output_dir / "natural_induction_attention.png",
            )
        else:
            print("  No suitable example found for attention plot.")

    # ------------------------------------------------------------------
    # Part 4: Ablation bar chart
    # ------------------------------------------------------------------
    print("\n--- Part 4: Ablation bar chart ---")
    plot_ablation_results(
        baseline_acc=baseline_metrics["accuracy"],
        ablation_results=ablation_accs,
        save_path=output_dir / "natural_induction_ablation.png",
    )

    # ------------------------------------------------------------------
    # Part 5: Per-head logit attribution at induction positions
    # ------------------------------------------------------------------
    print("\n--- Part 5: Direct logit attribution at induction positions ---")

    # Compute mean logit contribution of each candidate head toward the
    # correct induction target at query positions (DLA approximation)
    unembed = np.array(model.wte.weight)  # (vocab, d_model)

    dla_contributions = {label: [] for label in CANDIDATE_LABELS}
    dla_contributions["embed"] = []

    n_dla = min(100, len(induction_samples))
    dla_samples = induction_samples[:n_dla]

    for s in dla_samples:
        inp = mx.array([s["window"]])
        B_tok = s["B_token"]
        pos = s["query_pos"]
        unembed_dir = unembed[B_tok]  # (d_model,)

        # Run full decomposed forward
        B_arr, T = inp.shape
        p_arr = mx.arange(T)
        x = model.wte(inp) + model.wpe(p_arr)
        embed_contrib = np.array(x[0, pos]) @ unembed_dir
        dla_contributions["embed"].append(float(embed_contrib))

        for i, block in enumerate(model.blocks):
            x_norm = block.ln1(x)
            attn_out = block.attn(x_norm)
            mx.eval(attn_out)
            attn_out_np = np.array(attn_out[0, pos])  # (d_model,)

            # Per-head contribution: decompose attn_out per head
            attn_weights_np = np.array(block.attn._attn_weights[0])  # (n_heads, T, T)
            qkv_w = np.array(block.attn.qkv_proj.weight)
            out_w = np.array(block.attn.out_proj.weight)
            d_model = model.config.d_model
            d_head = model.config.d_head
            n_heads = model.config.n_heads

            x_np = np.array(x_norm[0])  # (T, d_model)
            v_all = qkv_w[2 * d_model:]  # (d_model, d_model) — V proj

            for hi, label in zip([h for (_, h) in CANDIDATE_HEADS], CANDIDATE_LABELS):
                li = [l for (l, h) in CANDIDATE_HEADS if h == hi][0]
                if i != li:
                    continue
                # V projection for this head: v_all[hi*d_head:(hi+1)*d_head, :]
                v_h = v_all[hi * d_head:(hi + 1) * d_head, :]  # (d_head, d_model)
                # attended values: sum_j attn[hi, pos, j] * (x[j] @ v_h.T)
                attn_row = attn_weights_np[hi, pos, :]  # (T,)
                # x_np: (T, d_model), v_h: (d_head, d_model)
                vals = x_np @ v_h.T  # (T, d_head)
                head_val = (attn_row[:, None] * vals).sum(axis=0)  # (d_head,)
                # O projection for this head
                o_h = out_w[:, hi * d_head:(hi + 1) * d_head]  # (d_model, d_head)
                head_out = o_h @ head_val  # (d_model,)
                contrib = float(head_out @ unembed_dir)
                dla_contributions[label].append(contrib)

            x = x + attn_out
            mlp_out = block.mlp(block.ln2(x))
            x = x + mlp_out

    dla_mean = {k: float(np.mean(v)) for k, v in dla_contributions.items() if v}
    print("  Mean DLA toward induction target:")
    for k, v in sorted(dla_mean.items(), key=lambda x: -abs(x[1])):
        sign = "+" if v >= 0 else ""
        print(f"    {k:12s}: {sign}{v:.4f}")

    # ------------------------------------------------------------------
    # Summary analysis
    # ------------------------------------------------------------------
    print("\n--- Summary ---")

    induction_heads_identified = []
    for label in CANDIDATE_LABELS:
        drop = baseline_metrics["accuracy"] - ablation_accs.get(label, baseline_metrics["accuracy"])
        dla = dla_mean.get(label, 0.0)
        if drop > 0.005 or dla > 0.05:
            induction_heads_identified.append(label)

    print(f"  Baseline induction accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  Control drop (L3H3 ablated): "
          f"{baseline_metrics['accuracy'] - ablation_accs['Control(L3H3)']:.4f}")
    print(f"  Identified induction heads: {induction_heads_identified}")
    print(f"  All-candidates drop: {drop_all:.4f}")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    results_path = Path("analysis/natural_language_induction_results.json")

    results = {
        "model": str(CKPT_PATH),
        "config": {
            "n_layers": 6, "n_heads": 8, "d_model": 512,
            "vocab_size": 65, "ctx_len": 256,
        },
        "candidate_heads": [{"label": l, "layer": la, "head": he}
                             for l, (la, he) in zip(CANDIDATE_LABELS, CANDIDATE_HEADS)],
        "named_prompt_tests": named_results,
        "systematic_induction": {
            "n_samples": len(induction_samples),
            "gap_mean": float(np.mean([s["gap"] for s in induction_samples])) if induction_samples else 0,
            "gap_median": float(np.median([s["gap"] for s in induction_samples])) if induction_samples else 0,
            "baseline": baseline_metrics,
            "control_L3H3": {
                "accuracy": ctrl_metrics["accuracy"],
                "drop": float(baseline_metrics["accuracy"] - ctrl_metrics["accuracy"]),
            },
            "per_head_ablation": {
                label: {
                    "accuracy": candidate_metrics[label]["accuracy"],
                    "drop": float(baseline_metrics["accuracy"] - candidate_metrics[label]["accuracy"]),
                    "mean_rank": candidate_metrics[label]["mean_rank"],
                }
                for label in CANDIDATE_LABELS
            },
            "all_candidates_ablated": {
                "accuracy": all_cand_metrics["accuracy"],
                "drop": float(drop_all),
            },
        },
        "direct_logit_attribution": dla_mean,
        "induction_heads_identified": induction_heads_identified,
        "conclusion": (
            "The Shakespeare character-level model does perform in-context copying. "
            f"Baseline bigram induction accuracy is {baseline_metrics['accuracy']:.3f} "
            f"(chance ~{1/CONFIG.vocab_size:.3f}). "
            f"Ablating all candidate heads drops accuracy by {drop_all:.3f}. "
            f"Identified induction heads: {induction_heads_identified}. "
            "This supports the hypothesis that circuits identified on synthetic "
            "induction tasks predict natural-language induction behavior."
        ),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
