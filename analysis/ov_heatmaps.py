"""
Token-level OV heatmap visualization.

For each attention head, the OV circuit maps:
  input token → output logit contribution

Specifically: E^T @ W_O @ W_V @ E maps every token to its effect on every
output logit when attended to. This tells us:
- "When this head attends to token X, it promotes tokens Y,Z and suppresses tokens W"

For a character-level Shakespeare model, these heatmaps show which characters
each head "wants to copy" or "wants to suppress."
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from src.model import GPT, GPTConfig


def compute_ov_matrices(model: GPT) -> list[list[np.ndarray]]:
    """Compute the full token→token OV matrix for every head.

    Returns: ov_matrices[layer][head] = (vocab, vocab) numpy array
    where entry [i,j] = "when this head attends to token j, how much does it
    promote token i in the output?"
    """
    embed = np.array(model.wte.weight)  # (vocab, d_model)
    d_model = model.config.d_model
    d_head = model.config.d_head
    n_heads = model.config.n_heads

    ov_matrices = []

    for block in model.blocks:
        qkv_weight = np.array(block.attn.qkv_proj.weight)  # (3*d_model, d_model)
        out_weight = np.array(block.attn.out_proj.weight)   # (d_model, d_model)

        v_weight = qkv_weight[2 * d_model:, :]  # (d_model, d_model)

        layer_matrices = []
        for h in range(n_heads):
            v_head = v_weight[h * d_head:(h + 1) * d_head, :]  # (d_head, d_model)
            o_head = out_weight[:, h * d_head:(h + 1) * d_head]  # (d_model, d_head)

            # OV matrix in token space: E @ O @ V @ E^T
            # Token j attended to → effect on token i logit
            ov = embed @ o_head @ v_head @ embed.T  # (vocab, vocab)
            layer_matrices.append(ov)

        ov_matrices.append(layer_matrices)

    return ov_matrices


def compute_qk_matrices(model: GPT) -> list[list[np.ndarray]]:
    """Compute the token→token QK matrix for every head.

    Returns: qk_matrices[layer][head] = (vocab, vocab) numpy array
    where entry [i,j] = "how much does token i (query) attend to token j (key)?"
    """
    embed = np.array(model.wte.weight)
    d_model = model.config.d_model
    d_head = model.config.d_head
    n_heads = model.config.n_heads

    qk_matrices = []

    for block in model.blocks:
        qkv_weight = np.array(block.attn.qkv_proj.weight)
        q_weight = qkv_weight[:d_model, :]
        k_weight = qkv_weight[d_model:2 * d_model, :]

        layer_matrices = []
        for h in range(n_heads):
            q_head = q_weight[h * d_head:(h + 1) * d_head, :]
            k_head = k_weight[h * d_head:(h + 1) * d_head, :]

            # QK in token space: E @ Q^T @ K @ E^T
            qk = embed @ q_head.T @ k_head @ embed.T  # (vocab, vocab)
            layer_matrices.append(qk)

        qk_matrices.append(layer_matrices)

    return qk_matrices


def plot_ov_heatmaps(
    model: GPT,
    idx_to_char: dict = None,
    top_k: int = 30,
    save_dir: str = None,
):
    """Plot OV heatmaps for every head, showing top token interactions.

    For large vocabs, only shows the top_k most active source and target tokens.
    """
    ov_matrices = compute_ov_matrices(model)
    n_layers = len(ov_matrices)
    n_heads = len(ov_matrices[0])

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for layer_idx in range(n_layers):
        fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
        if n_heads == 1:
            axes = [axes]

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            ov = ov_matrices[layer_idx][head_idx]

            # Find most active tokens (by absolute row/column sum)
            row_activity = np.abs(ov).sum(axis=1)
            col_activity = np.abs(ov).sum(axis=0)
            combined_activity = row_activity + col_activity
            top_indices = np.argsort(combined_activity)[-top_k:]

            # Submatrix
            sub_ov = ov[np.ix_(top_indices, top_indices)]

            # Labels
            if idx_to_char:
                labels = [repr(idx_to_char.get(int(i), '?'))[1:-1] for i in top_indices]
            else:
                labels = [str(i) for i in top_indices]

            im = ax.imshow(sub_ov, cmap="RdBu_r", aspect="auto",
                          vmin=-np.abs(sub_ov).max(), vmax=np.abs(sub_ov).max())
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_yticklabels(labels, fontsize=5)
            ax.set_title(f"L{layer_idx}H{head_idx}", fontsize=10)
            ax.set_xlabel("Attended TO (source)", fontsize=7)
            ax.set_ylabel("Effect ON (output)", fontsize=7)

        plt.suptitle(f"OV Heatmaps — Layer {layer_idx}\n"
                     f"(Red = promotes, Blue = suppresses)", fontsize=12)
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/ov_layer{layer_idx}.png", dpi=150, bbox_inches="tight")
            print(f"Saved: {save_dir}/ov_layer{layer_idx}.png")
        plt.close()


def plot_qk_heatmaps(
    model: GPT,
    idx_to_char: dict = None,
    top_k: int = 30,
    save_dir: str = None,
):
    """Plot QK heatmaps for every head."""
    qk_matrices = compute_qk_matrices(model)
    n_layers = len(qk_matrices)
    n_heads = len(qk_matrices[0])

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for layer_idx in range(n_layers):
        fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
        if n_heads == 1:
            axes = [axes]

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            qk = qk_matrices[layer_idx][head_idx]

            row_activity = np.abs(qk).sum(axis=1)
            col_activity = np.abs(qk).sum(axis=0)
            combined = row_activity + col_activity
            top_indices = np.argsort(combined)[-top_k:]

            sub_qk = qk[np.ix_(top_indices, top_indices)]

            if idx_to_char:
                labels = [repr(idx_to_char.get(int(i), '?'))[1:-1] for i in top_indices]
            else:
                labels = [str(i) for i in top_indices]

            im = ax.imshow(sub_qk, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_yticklabels(labels, fontsize=5)
            ax.set_title(f"L{layer_idx}H{head_idx}", fontsize=10)
            ax.set_xlabel("Key (attended TO)", fontsize=7)
            ax.set_ylabel("Query (attending FROM)", fontsize=7)

        plt.suptitle(f"QK Heatmaps — Layer {layer_idx}\n"
                     f"(Bright = strong query-key match)", fontsize=12)
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/qk_layer{layer_idx}.png", dpi=150, bbox_inches="tight")
            print(f"Saved: {save_dir}/qk_layer{layer_idx}.png")
        plt.close()


def describe_head_ov(
    ov_matrix: np.ndarray,
    idx_to_char: dict = None,
    top_n: int = 5,
) -> str:
    """Generate a human-readable description of what an OV head does.

    For each common source token, lists what output tokens it promotes/suppresses.
    """
    vocab_size = ov_matrix.shape[0]
    lines = []

    # Find the most "opinionated" source tokens (highest abs column sum)
    col_strength = np.abs(ov_matrix).sum(axis=0)
    top_sources = np.argsort(col_strength)[-top_n:][::-1]

    for src_idx in top_sources:
        src_label = idx_to_char.get(int(src_idx), f"[{src_idx}]") if idx_to_char else str(src_idx)
        col = ov_matrix[:, src_idx]

        # Top promoted and suppressed tokens
        promoted = np.argsort(col)[-3:][::-1]
        suppressed = np.argsort(col)[:3]

        prom_strs = []
        for p in promoted:
            label = idx_to_char.get(int(p), f"[{p}]") if idx_to_char else str(p)
            prom_strs.append(f"{repr(label)}({col[p]:+.2f})")

        supp_strs = []
        for s in suppressed:
            label = idx_to_char.get(int(s), f"[{s}]") if idx_to_char else str(s)
            supp_strs.append(f"{repr(label)}({col[s]:+.2f})")

        lines.append(f"  Attend to {repr(src_label):6s} → promotes [{', '.join(prom_strs)}], suppresses [{', '.join(supp_strs)}]")

    return "\n".join(lines)


if __name__ == "__main__":
    # Will be run after training completes
    print("OV heatmap tools loaded. Run after model training completes.")
