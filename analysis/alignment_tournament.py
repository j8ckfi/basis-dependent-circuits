"""
Alignment Tournament (Experiment C) — Round 1

Question: Is transplant failure correctable by any alignment method?

Donor:     seed 0's L0H3
Recipient: seed 1
Target:    recipient's L0H1 (critical head)

Seven methods:
  1. Unaligned (baseline)
  2. Procrustes orthogonal two-interface
  3. Affine (Procrustes + per-axis scale)
  4. CKA-optimized linear map
  5. Git Re-Basin permutation matching + Procrustes
  6. Learned linear adapter
  7. Learned nonlinear adapter

Usage:
    cd /path/to/Mechinterp2
    source .venv/bin/activate
    python analysis/alignment_tournament.py
"""

import sys
import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT, GPTConfig
from src.data import InductionDataset
from analysis.transplant_unified_eval import (
    build_fixed_eval_set,
    evaluate_on_fixed_set,
    paired_sign_flip_pvalue,
    paired_bootstrap_ci,
    get_head_weights,
    transplant_head,
    collect_pre_l0_ln,
    collect_post_l0,
    procrustes_orthogonal,
    rotate_head_weights_two_interface,
    MODEL_CONFIG,
    SEED0_CKPT,
    SEED1_CKPT,
    BASE_DIR,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DONOR_LAYER, DONOR_HEAD = 0, 3       # seed0 L0H3
TARGET_LAYER, TARGET_HEAD = 0, 1     # seed1 L0H1 (critical slot)

D_MODEL = MODEL_CONFIG.d_model       # 128
D_HEAD = MODEL_CONFIG.d_head         # 32
D_FF = MODEL_CONFIG.d_ff             # 512
N_HEADS = MODEL_CONFIG.n_heads       # 4

ALIGN_SEED = 1234
ALIGN_BATCH = 256

TRAIN_SEED = 7777   # training data for adapters — DIFFERENT from eval
EVAL_SEED = 9999    # matches build_fixed_eval_set()
ADAPTER_STEPS = 1000
ADAPTER_BATCH = 32
ADAPTER_LR = 3e-4

CKA_STEPS = 500
CKA_LR = 1e-3

RESULTS_PATH = Path(__file__).parent / "alignment_tournament_results.json"
PLOTS_DIR = Path(__file__).parent / "plots"

SEED1_BASELINE_ACC = 0.964   # reference line for plots


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str) -> GPT:
    model = GPT(MODEL_CONFIG)
    model.load_weights(path)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Helper: collect alignment activations
# ---------------------------------------------------------------------------

def get_alignment_activations(model0: GPT, model1: GPT):
    """Return pre-LN and post-L0 activations for both models on 256-seq batch."""
    align_ds = InductionDataset(
        vocab_size=512, seq_len=64, n_bigrams=6, seed=ALIGN_SEED
    )
    align_inputs_mx, _, _ = align_ds.generate_batch(ALIGN_BATCH)
    mx.eval(align_inputs_mx)

    pre_ln_0 = collect_pre_l0_ln(model0, align_inputs_mx)   # (B*T, D)
    pre_ln_1 = collect_pre_l0_ln(model1, align_inputs_mx)
    post_l0_0 = collect_post_l0(model0, align_inputs_mx)
    post_l0_1 = collect_post_l0(model1, align_inputs_mx)

    return pre_ln_0, pre_ln_1, post_l0_0, post_l0_1, align_inputs_mx


# ---------------------------------------------------------------------------
# Method 1: Unaligned baseline
# ---------------------------------------------------------------------------

def method_unaligned(model1: GPT, donor_w: dict, fixed_eval):
    print("\n[Method 1] Unaligned baseline")
    m = transplant_head(model1, donor_w, TARGET_LAYER, TARGET_HEAD)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval)
    print(f"  Accuracy: {acc:.4f}")
    return acc, fracs, {"params_added": 0}


# ---------------------------------------------------------------------------
# Method 2: Procrustes two-interface
# ---------------------------------------------------------------------------

def method_procrustes(model1: GPT, donor_w: dict, fixed_eval,
                      pre_ln_0, pre_ln_1, post_l0_0, post_l0_1):
    print("\n[Method 2] Procrustes two-interface")
    R_in = procrustes_orthogonal(pre_ln_0, pre_ln_1)
    R_out = procrustes_orthogonal(post_l0_0, post_l0_1)
    aligned_w = rotate_head_weights_two_interface(donor_w, R_in, R_out)
    m = transplant_head(model1, aligned_w, TARGET_LAYER, TARGET_HEAD)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval)
    print(f"  Accuracy: {acc:.4f}")
    return acc, fracs, {"params_added": 2 * D_MODEL ** 2}


# ---------------------------------------------------------------------------
# Method 3: Affine alignment (Procrustes + per-axis diagonal scale)
# ---------------------------------------------------------------------------

def fit_affine(X_src: np.ndarray, X_tgt: np.ndarray):
    """Fit affine map: R (orthogonal) + per-axis scale s.

    Returns R (d,d) and s (d,) such that X_src @ (R * s) ~= X_tgt
    """
    R = procrustes_orthogonal(X_src, X_tgt)
    X_rot = X_src @ R  # (N, d)
    # Per-axis scale: ratio of stdevs in target vs rotated source
    std_rot = X_rot.std(axis=0)
    std_tgt = X_tgt.std(axis=0)
    s = std_tgt / (std_rot + 1e-8)
    return R, s


def apply_affine_to_head(weights: dict, R_in, s_in, R_out, s_out) -> dict:
    """Apply affine map (R, s) to head weights at both interfaces.

    QKV:     W_new = W @ (R * s)   [columns scaled by s after rotation]
    out_proj: new_out = (R_out * s_out).T @ out
              = diag(s_out) @ R_out.T @ out
    """
    # For QKV: the transform to apply is A_in = R_in * s_in (broadcast s over rows)
    # W_new[row, :] = W[row, :] @ A_in
    A_in = R_in * s_in[None, :]    # (d, d) — right-multiply
    # For out: we want the map A_out = R_out * s_out applied to the output direction
    # out contribution: v @ out_cols.T @ A_out = v @ (A_out.T @ out_cols).T
    A_out = R_out * s_out[None, :]  # (d, d)

    return {
        "q":   (weights["q"] @ A_in).astype(np.float32),
        "k":   (weights["k"] @ A_in).astype(np.float32),
        "v":   (weights["v"] @ A_in).astype(np.float32),
        "out": (A_out.T @ weights["out"]).astype(np.float32),
    }


def method_affine(model1: GPT, donor_w: dict, fixed_eval,
                  pre_ln_0, pre_ln_1, post_l0_0, post_l0_1):
    print("\n[Method 3] Affine alignment (Procrustes + per-axis scale)")
    R_in, s_in = fit_affine(pre_ln_0, pre_ln_1)
    R_out, s_out = fit_affine(post_l0_0, post_l0_1)
    aligned_w = apply_affine_to_head(donor_w, R_in, s_in, R_out, s_out)
    m = transplant_head(model1, aligned_w, TARGET_LAYER, TARGET_HEAD)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval)
    print(f"  Accuracy: {acc:.4f}  (s_in range [{s_in.min():.3f}, {s_in.max():.3f}])")
    return acc, fracs, {"params_added": 2 * D_MODEL ** 2}


# ---------------------------------------------------------------------------
# Method 4: CKA-optimized linear map
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between centered matrices X and Y."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    denom = (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def method_cka_linear(model1: GPT, donor_w: dict, fixed_eval,
                      pre_ln_0, pre_ln_1, post_l0_0, post_l0_1):
    """CKA-optimized unconstrained linear map A (d x d).

    Separate A_in (for QKV) and A_out (for out_proj), each initialized
    from Procrustes R and optimized via gradient ascent on linear CKA.
    """
    print("\n[Method 4] CKA-optimized linear map")

    # Initialize from Procrustes
    R_in = procrustes_orthogonal(pre_ln_0, pre_ln_1).astype(np.float32)
    R_out = procrustes_orthogonal(post_l0_0, post_l0_1).astype(np.float32)

    # Work in numpy/pure python — MLX gradient through matrix ops
    X_src = pre_ln_0.astype(np.float32)    # (N, d)
    X_tgt = pre_ln_1.astype(np.float32)
    Y_src = post_l0_0.astype(np.float32)
    Y_tgt = post_l0_1.astype(np.float32)

    # Center targets once
    X_tgt_c = (X_tgt - X_tgt.mean(0)).astype(np.float32)
    Y_tgt_c = (Y_tgt - Y_tgt.mean(0)).astype(np.float32)
    XtXt_norm = float(np.linalg.norm(X_tgt_c.T @ X_tgt_c, "fro"))
    YtYt_norm = float(np.linalg.norm(Y_tgt_c.T @ Y_tgt_c, "fro"))

    def cka_loss_and_grad(A, X_s, X_t_c, XtXt_n):
        """Compute -CKA(X_s @ A, X_t) and gradient w.r.t. A."""
        XA = X_s @ A
        XA_c = XA - XA.mean(0)
        cross = XA_c.T @ X_t_c                  # (d, d)
        num = np.sum(cross ** 2)
        XAA_norm = np.linalg.norm(XA_c.T @ XA_c, "fro")
        denom = XAA_norm * XtXt_n + 1e-12

        cka_val = num / denom

        # Gradient: d(-CKA)/dA
        # dcka/d(num) = 1/denom
        # dcka/d(denom) = -num/denom^2
        # num = ||XA_c.T @ Xt_c||^2 = tr(cross.T @ cross)
        # d(num)/d(XA_c) = 2 * Xt_c @ cross.T
        # denom involves ||XA_c.T @ XA_c||_F; we use chain rule
        # For simplicity: gradient of -CKA w.r.t. A via numerical finite diff
        # is too slow; use analytic (approximate) via:
        #   d(-CKA)/dA = d(-CKA)/d(XA) * d(XA)/dA = grad_XA.T ... @ X_s
        N = X_s.shape[0]
        # d(num)/d(XA_c): cross = XA_c.T @ Xt_c => d||cross||^2/d(XA_c)=2*Xt_c@cross.T
        grad_num_XAc = 2.0 * X_t_c @ cross.T  # (N, d)
        # d(denom)/d(XA_c): denom = ||XA_c.T @ XA_c||_F * XtXt_n
        #   let G = XA_c.T @ XA_c; d||G||_F/d(XA_c) via chain:
        #   d||G||_F/dG = G/||G||_F; dG/d(XA_c) contributes 2 * XA_c @ (G/||G||_F)
        G = XA_c.T @ XA_c
        G_norm = np.linalg.norm(G, "fro") + 1e-12
        grad_denom_XAc = 2.0 * XA_c @ (G / G_norm) * XtXt_n   # (N, d)

        # Gradient of -CKA w.r.t. XA_c:
        grad_neg_cka_XAc = -(grad_num_XAc * denom - num * grad_denom_XAc) / (denom ** 2)

        # XA_c = XA - mean(XA); d(XA_c)/d(XA) = I - 1/N * 11^T
        # grad w.r.t. XA:
        grad_XA = grad_neg_cka_XAc - grad_neg_cka_XAc.mean(0, keepdims=True)

        # grad w.r.t. A: X_s.T @ grad_XA
        grad_A = X_s.T @ grad_XA / N   # normalize by N for stability

        return -cka_val, grad_A

    A_in = R_in.copy()
    A_out = R_out.copy()
    lr = CKA_LR
    # Adam state
    m_in = np.zeros_like(A_in)
    v_in = np.zeros_like(A_in)
    m_out = np.zeros_like(A_out)
    v_out = np.zeros_like(A_out)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    cka_in_init = linear_cka(X_src @ A_in, X_tgt)
    cka_out_init = linear_cka(Y_src @ A_out, Y_tgt)
    print(f"  CKA init: in={cka_in_init:.4f}  out={cka_out_init:.4f}")

    for step in range(CKA_STEPS):
        t = step + 1
        loss_in, g_in = cka_loss_and_grad(A_in, X_src, X_tgt_c, XtXt_norm)
        loss_out, g_out = cka_loss_and_grad(A_out, Y_src, Y_tgt_c, YtYt_norm)

        # Adam update for A_in
        m_in = beta1 * m_in + (1 - beta1) * g_in
        v_in = beta2 * v_in + (1 - beta2) * (g_in ** 2)
        m_in_hat = m_in / (1 - beta1 ** t)
        v_in_hat = v_in / (1 - beta2 ** t)
        A_in -= lr * m_in_hat / (np.sqrt(v_in_hat) + eps)

        # Adam update for A_out
        m_out = beta1 * m_out + (1 - beta1) * g_out
        v_out = beta2 * v_out + (1 - beta2) * (g_out ** 2)
        m_out_hat = m_out / (1 - beta1 ** t)
        v_out_hat = v_out / (1 - beta2 ** t)
        A_out -= lr * m_out_hat / (np.sqrt(v_out_hat) + eps)

        if (step + 1) % 100 == 0:
            c_in = -loss_in
            c_out = -loss_out
            print(f"  Step {step+1:4d}  CKA_in={c_in:.4f}  CKA_out={c_out:.4f}")

    converged_cka_in = linear_cka(X_src @ A_in, X_tgt)
    converged_cka_out = linear_cka(Y_src @ A_out, Y_tgt)
    print(f"  CKA final: in={converged_cka_in:.4f}  out={converged_cka_out:.4f}")

    # Apply: QKV gets A_in, out_proj gets A_out
    aligned_w = {
        "q":   (donor_w["q"] @ A_in).astype(np.float32),
        "k":   (donor_w["k"] @ A_in).astype(np.float32),
        "v":   (donor_w["v"] @ A_in).astype(np.float32),
        "out": (A_out.T @ donor_w["out"]).astype(np.float32),
    }
    m = transplant_head(model1, aligned_w, TARGET_LAYER, TARGET_HEAD)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval)
    print(f"  Accuracy: {acc:.4f}")
    return acc, fracs, {
        "params_added": D_MODEL * D_MODEL,
        "converged_cka_in": converged_cka_in,
        "converged_cka_out": converged_cka_out,
    }


# ---------------------------------------------------------------------------
# Method 5: Git Re-Basin permutation matching + Procrustes
# ---------------------------------------------------------------------------

def collect_per_head_outputs(model: GPT, inputs: mx.array, layer: int) -> np.ndarray:
    """Collect per-head attention outputs at a given layer.

    Returns shape (N*T, n_heads, d_head).
    """
    # We need head-level outputs: attn_weights @ V for each head
    # Use the stored _attn_weights after a forward pass
    residuals = model.get_residual_stream(inputs)
    # Run forward to populate _attn_weights
    _ = model(inputs)
    mx.eval(model.blocks[layer].attn._attn_weights)

    attn_w = np.array(model.blocks[layer].attn._attn_weights)  # (B, H, T, T)
    B, H, T, _ = attn_w.shape

    # Get V: need the value projections
    # residuals[layer] is the pre-block input; apply ln1 then project
    pre_block = residuals[layer]  # (B, T, D)
    ln1 = model.blocks[layer].ln1
    x_ln = ln1(pre_block)
    mx.eval(x_ln)
    x_ln_np = np.array(x_ln)  # (B, T, D)

    # Get V weights
    qkv_w = np.array(model.blocks[layer].attn.qkv_proj.weight)  # (3D, D)
    D = MODEL_CONFIG.d_model
    dh = MODEL_CONFIG.d_head

    # V rows: [2*D : 3*D, :]
    V_w = qkv_w[2*D:3*D, :]  # (D, D)  -- (n_heads*d_head, D)

    # Compute V for all heads: (B, T, D) @ V_w.T = (B, T, D)
    V = x_ln_np @ V_w.T  # (B, T, D)
    # reshape to (B, T, H, dh)
    V = V.reshape(B, T, H, dh)
    V = V.transpose(0, 2, 1, 3)  # (B, H, T, dh)

    # Head outputs: attn_w @ V  (B, H, T, T) @ (B, H, T, dh) -> (B, H, T, dh)
    head_out = np.einsum("bhtj,bhjd->bhtd", attn_w, V)  # (B, H, T, dh)
    head_out = head_out.transpose(0, 2, 1, 3)  # (B, T, H, dh)
    return head_out.reshape(B * T, H, dh)


def collect_mlp_hidden(model: GPT, inputs: mx.array, layer: int) -> np.ndarray:
    """Collect MLP hidden activations (post-GELU) at a given layer.

    Returns shape (N*T, d_ff).
    """
    residuals = model.get_residual_stream(inputs)
    # After attention at this layer: residuals[layer+1] = residuals[layer] + attn_out + mlp_out
    # We need the input to MLP: residuals[layer] + attn_out, then apply ln2
    # But residuals[layer+1] includes mlp; we need intermediate.
    # Easier: collect pre-ln2 input to MLP by running attn manually.
    pre_block = residuals[layer]  # (B, T, D)
    ln1 = model.blocks[layer].ln1
    attn_out = model.blocks[layer].attn(ln1(pre_block))
    mx.eval(attn_out)
    pre_mlp = pre_block + attn_out  # (B, T, D)
    ln2 = model.blocks[layer].ln2
    mlp_input = ln2(pre_mlp)   # (B, T, D)
    # MLP: up_proj -> GELU -> down_proj
    up_w = np.array(model.blocks[layer].mlp.up_proj.weight)  # (d_ff, D)
    mlp_input_np = np.array(mlp_input)
    B, T, D = mlp_input_np.shape
    h = mlp_input_np.reshape(B*T, D) @ up_w.T   # (B*T, d_ff)
    # GELU
    h_gelu = h * 0.5 * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (h + 0.044715 * h**3)))
    return h_gelu


def permute_heads_in_donor(donor_model: GPT, recipient_model: GPT,
                           inputs: mx.array, layer: int) -> np.ndarray:
    """Find head permutation that best matches donor heads to recipient heads.

    Returns perm (n_heads,) such that donor head perm[i] should map to recipient head i.
    Cost: -cosine_similarity between head outputs.
    """
    donor_out = collect_per_head_outputs(donor_model, inputs, layer)    # (N, H, dh)
    recip_out = collect_per_head_outputs(recipient_model, inputs, layer)  # (N, H, dh)
    H = N_HEADS
    cost = np.zeros((H, H), dtype=np.float32)
    for i in range(H):
        for j in range(H):
            d = donor_out[:, i, :].flatten()
            r = recip_out[:, j, :].flatten()
            d_norm = np.linalg.norm(d)
            r_norm = np.linalg.norm(r)
            if d_norm < 1e-10 or r_norm < 1e-10:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(d, r) / (d_norm * r_norm))
            cost[i, j] = -cos_sim  # minimize negative cosine = maximize cosine
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = recipient slot that donor head i maps to
    # We want: for each recipient slot j, which donor head i? -> inverse permutation
    perm = np.argsort(col_ind)  # perm[j] = donor head that maps to recipient slot j
    return perm


def permute_mlp_in_donor(donor_model: GPT, recipient_model: GPT,
                         inputs: mx.array, layer: int) -> np.ndarray:
    """Find MLP hidden unit permutation via Pearson correlation cost matrix.

    Returns perm (d_ff,) such that donor unit perm[i] maps to recipient unit i.
    """
    donor_hid = collect_mlp_hidden(donor_model, inputs, layer)    # (N, d_ff)
    recip_hid = collect_mlp_hidden(recipient_model, inputs, layer)  # (N, d_ff)
    N, D = donor_hid.shape

    # Pearson correlation matrix: (d_ff, d_ff)
    # Normalize each column
    d_centered = donor_hid - donor_hid.mean(0)
    r_centered = recip_hid - recip_hid.mean(0)
    d_std = d_centered.std(0) + 1e-8
    r_std = r_centered.std(0) + 1e-8
    d_norm = d_centered / d_std
    r_norm = r_centered / r_std

    # Pearson corr matrix: C[i,j] = correlation between donor[:,i] and recip[:,j]
    # = (1/N) * d_norm[:,i] . r_norm[:,j]
    C = (d_norm.T @ r_norm) / N   # (d_ff, d_ff)
    cost = -C
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.argsort(col_ind)
    return perm


def apply_head_permutation_to_flat_weights(flat_weights: dict, perm: np.ndarray,
                                           layer: int) -> dict:
    """Permute attention heads in the flat weight dict for the given layer.

    perm[j] = donor head that maps to slot j.
    We reorder all H heads in qkv_proj and out_proj.
    """
    new_weights = dict(flat_weights)
    D = D_MODEL
    dh = D_HEAD
    H = N_HEADS
    qkv_key = f"blocks.{layer}.attn.qkv_proj.weight"
    out_key = f"blocks.{layer}.attn.out_proj.weight"

    qkv_w = flat_weights[qkv_key].copy()  # (3D, D)
    out_w = flat_weights[out_key].copy()   # (D, D)

    new_qkv = np.zeros_like(qkv_w)
    new_out = np.zeros_like(out_w)

    for j in range(H):
        src = perm[j]   # donor head src -> recipient slot j
        s_start = src * dh
        s_end = (src + 1) * dh
        j_start = j * dh
        j_end = (j + 1) * dh
        # Q block
        new_qkv[j_start:j_end, :] = qkv_w[s_start:s_end, :]
        # K block
        new_qkv[D + j_start:D + j_end, :] = qkv_w[D + s_start:D + s_end, :]
        # V block
        new_qkv[2*D + j_start:2*D + j_end, :] = qkv_w[2*D + s_start:2*D + s_end, :]
        # out_proj columns
        new_out[:, j_start:j_end] = out_w[:, s_start:s_end]

    new_weights[qkv_key] = new_qkv
    new_weights[out_key] = new_out
    return new_weights


def apply_mlp_permutation_to_flat_weights(flat_weights: dict, perm: np.ndarray,
                                          layer: int) -> dict:
    """Permute MLP hidden units: up_proj rows, down_proj columns."""
    new_weights = dict(flat_weights)
    up_key = f"blocks.{layer}.mlp.up_proj.weight"
    down_key = f"blocks.{layer}.mlp.down_proj.weight"

    up_w = flat_weights[up_key].copy()     # (d_ff, D)
    down_w = flat_weights[down_key].copy() # (D, d_ff)

    # perm[j] = donor unit that maps to recipient slot j
    new_weights[up_key] = up_w[perm, :]
    new_weights[down_key] = down_w[:, perm]
    return new_weights


def method_git_rebasin(model0: GPT, model1: GPT, donor_w: dict, fixed_eval,
                       pre_ln_0, pre_ln_1, post_l0_0, post_l0_1,
                       align_inputs_mx):
    """Git Re-Basin: permute donor's heads+MLP to match recipient, then Procrustes."""
    print("\n[Method 5] Git Re-Basin permutation matching + Procrustes")

    # Step 1: Get flat weights from donor (model0)
    flat0 = {k: np.array(v) for k, v in dict(nn.utils.tree_flatten(model0.parameters())).items()}

    # Step 2: Find head permutations for each layer and apply
    print("  Finding head permutations...")
    perm_info = {}
    for layer in range(MODEL_CONFIG.n_layers):
        perm = permute_heads_in_donor(model0, model1, align_inputs_mx, layer)
        flat0 = apply_head_permutation_to_flat_weights(flat0, perm, layer)
        perm_info[f"head_perm_L{layer}"] = perm.tolist()
        print(f"    L{layer} head perm: {perm.tolist()}")

    # Step 3: Find MLP permutations for each layer and apply
    print("  Finding MLP hidden unit permutations...")
    # Rebuild a temporary model from permuted weights to collect MLP activations
    donor_perm_model = GPT(MODEL_CONFIG)
    donor_perm_model.load_weights([(k, mx.array(v)) for k, v in flat0.items()])
    mx.eval(donor_perm_model.parameters())

    for layer in range(MODEL_CONFIG.n_layers):
        perm = permute_mlp_in_donor(donor_perm_model, model1, align_inputs_mx, layer)
        flat0 = apply_mlp_permutation_to_flat_weights(flat0, perm, layer)
        n_fixed = int((perm == np.arange(len(perm))).sum())
        perm_info[f"mlp_perm_L{layer}_fixed_pts"] = n_fixed
        print(f"    L{layer} MLP perm: {n_fixed}/{D_FF} fixed points")

    # Step 4: Rebuild permuted donor model and extract L0H3 weights
    donor_perm2 = GPT(MODEL_CONFIG)
    donor_perm2.load_weights([(k, mx.array(v)) for k, v in flat0.items()])
    mx.eval(donor_perm2.parameters())

    permuted_donor_w = get_head_weights(donor_perm2, DONOR_LAYER, DONOR_HEAD)

    # Step 5: Apply Procrustes two-interface on top of permuted donor
    R_in = procrustes_orthogonal(pre_ln_0, pre_ln_1)
    R_out = procrustes_orthogonal(post_l0_0, post_l0_1)
    aligned_w = rotate_head_weights_two_interface(permuted_donor_w, R_in, R_out)

    m = transplant_head(model1, aligned_w, TARGET_LAYER, TARGET_HEAD)
    acc, fracs = evaluate_on_fixed_set(m, fixed_eval)
    print(f"  Accuracy: {acc:.4f}")
    return acc, fracs, {"params_added": 0, "perm_info": perm_info}


# ---------------------------------------------------------------------------
# Methods 6 & 7: Learned linear/nonlinear adapter
# ---------------------------------------------------------------------------

class LinearAdapter(nn.Module):
    """Two d_model x d_model linear maps at donor head interface."""
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.A_in = mx.eye(d_model)
        self.A_out = mx.eye(d_model)

    def __call__(self, x_in, x_out_update):
        return x_in @ self.A_in, x_out_update @ self.A_out


class NonlinearAdapter(nn.Module):
    """Two 2-layer MLP adapters: d_model -> 4*d_model -> d_model with GELU."""
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        hidden = 4 * d_model
        self.in_up = nn.Linear(d_model, hidden, bias=False)
        self.in_down = nn.Linear(hidden, d_model, bias=False)
        self.out_up = nn.Linear(d_model, hidden, bias=False)
        self.out_down = nn.Linear(hidden, d_model, bias=False)

        # Initialize to near-identity: small weights for up, identity-like for down
        # Start with near-identity by setting weights to approximate identity
        eye = np.eye(d_model, dtype=np.float32)
        # up_proj: d -> 4d; we tile 4 copies of identity for first d rows
        up_init = np.zeros((hidden, d_model), dtype=np.float32)
        up_init[:d_model, :] = eye
        down_init = np.zeros((d_model, hidden), dtype=np.float32)
        down_init[:, :d_model] = eye / 4  # compensate for 4x expansion

        self.in_up.weight = mx.array(up_init)
        self.in_down.weight = mx.array(down_init)
        self.out_up.weight = mx.array(up_init)
        self.out_down.weight = mx.array(down_init)

    def __call__(self, x_in, x_out_update):
        h_in = self.in_down(nn.gelu(self.in_up(x_in)))
        h_out = self.out_down(nn.gelu(self.out_up(x_out_update)))
        return h_in, h_out


def adapted_forward_with_donor(
    recipient_model: GPT,
    donor_weights: dict,      # numpy dict {q, k, v, out}
    adapter,                   # LinearAdapter or NonlinearAdapter
    inputs: mx.array,
    use_donor: bool = True,
) -> mx.array:
    """Forward pass where head TARGET_HEAD in layer 0 is replaced by donor head.

    The adapter transforms the interface:
      - A_in applied to the LN-normed residual before Q/K/V
      - A_out applied to the head output before it's added to residual

    If use_donor=False, the donor head is zeroed (control experiment).
    """
    B, T = inputs.shape
    config = recipient_model.config
    D = config.d_model
    dh = config.d_head
    H = config.n_heads

    # Embed
    pos = mx.arange(T)
    x = recipient_model.wte(inputs) + recipient_model.wpe(pos)  # (B, T, D)

    # ----- Layer 0 (modified) -----
    ln1 = recipient_model.blocks[0].ln1
    x_ln = ln1(x)  # (B, T, D)

    # Adapter input transform
    x_ln_adapted, _ = adapter(x_ln, mx.zeros((B, T, D)))
    # Compute head output using donor weights with adapted input
    q_w = mx.array(donor_weights["q"])  # (dh, D)
    k_w = mx.array(donor_weights["k"])
    v_w = mx.array(donor_weights["v"])
    o_w = mx.array(donor_weights["out"])  # (D, dh)

    # Q, K, V for donor head: x_ln_adapted @ W.T
    Q = x_ln_adapted @ q_w.T  # (B, T, dh)
    K = x_ln_adapted @ k_w.T
    V = x_ln_adapted @ v_w.T

    # Scaled dot-product attention
    scale = dh ** -0.5
    attn_scores = (Q @ K.transpose(0, 2, 1)) * scale  # (B, T, T)
    mask = mx.triu(mx.full((T, T), -1e9), k=1)
    attn_scores = attn_scores + mask
    attn_p = mx.softmax(attn_scores, axis=-1)
    head_val = attn_p @ V  # (B, T, dh)

    # If zeroing donor, set head_val = 0
    if not use_donor:
        head_val = mx.zeros_like(head_val)

    # head contribution: head_val @ o_w.T — but o_w is (D, dh), so head_val @ o_w.T shape...
    # o_w shape: (D, dh), forward is: out = v @ out_cols.T -> (B, T, dh) @ (dh, D) = (B, T, D)
    donor_head_contrib = head_val @ o_w.T  # (B, T, D)

    # Apply adapter to the output update
    _, donor_head_contrib_adapted = adapter(mx.zeros_like(donor_head_contrib), donor_head_contrib)

    # Run recipient's full L0 attention (all heads)
    recip_attn_out = recipient_model.blocks[0].attn(x_ln)  # (B, T, D)

    # Subtract recipient's head TARGET_HEAD contribution and add adapted donor contribution
    # Recipient's head TARGET_HEAD contribution:
    recip_qkv_w = recipient_model.blocks[0].attn.qkv_proj.weight  # (3D, D)
    recip_out_w = recipient_model.blocks[0].attn.out_proj.weight   # (D, D)

    h = TARGET_HEAD
    q_start = h * dh
    q_end = (h + 1) * dh

    recip_Q_h = x_ln @ recip_qkv_w[q_start:q_end, :].T          # (B, T, dh)
    recip_K_h = x_ln @ recip_qkv_w[D + q_start:D + q_end, :].T
    recip_V_h = x_ln @ recip_qkv_w[2*D + q_start:2*D + q_end, :].T

    recip_attn_scores_h = (recip_Q_h @ recip_K_h.transpose(0, 2, 1)) * scale
    recip_attn_scores_h = recip_attn_scores_h + mask
    recip_attn_p_h = mx.softmax(recip_attn_scores_h, axis=-1)
    recip_head_val_h = recip_attn_p_h @ recip_V_h   # (B, T, dh)

    recip_head_contrib = recip_head_val_h @ recip_out_w[:, q_start:q_end].T  # (B, T, D)

    # Modified attn output: replace head TARGET_HEAD
    modified_attn_out = recip_attn_out - recip_head_contrib + donor_head_contrib_adapted

    # Residual add after attention
    x = x + modified_attn_out

    # MLP of layer 0
    x = x + recipient_model.blocks[0].mlp(recipient_model.blocks[0].ln2(x))

    # ----- Layer 1 onwards (unchanged) -----
    for block in recipient_model.blocks[1:]:
        x = block(x)

    x = recipient_model.ln_f(x)
    logits = x @ recipient_model.wte.weight.T
    return logits


def masked_ce_loss(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Masked cross-entropy loss at induction positions."""
    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)
    mask_1d = mask.reshape(B * T)
    log_probs = nn.log_softmax(logits_2d, axis=-1)
    nll = -log_probs[mx.arange(B * T), targets_1d]  # (B*T,)
    # Masked mean
    masked_sum = (nll * mask_1d).sum()
    n = mask_1d.sum()
    return masked_sum / (n + 1e-8)


def train_adapter(
    recipient_model: GPT,
    donor_weights: dict,
    adapter,
    use_donor: bool = True,
    label: str = "adapter",
) -> float:
    """Train the adapter for ADAPTER_STEPS steps. Returns final train loss.

    Uses the correct MLX nn.value_and_grad pattern: loss_fn does NOT take
    params as first arg; nn.value_and_grad calls model.update(params) internally
    before invoking loss_fn(*args, **kwargs).
    """
    train_ds = InductionDataset(
        vocab_size=512, seq_len=64, n_bigrams=6, seed=TRAIN_SEED
    )
    optimizer = optim.AdamW(learning_rate=ADAPTER_LR)
    mx.eval(adapter.parameters())

    # NOTE: loss_fn receives (inputs, targets, mask) directly — no params arg.
    # nn.value_and_grad(adapter, loss_fn) handles the update internally.
    def loss_fn(inputs, targets, mask):
        logits = adapted_forward_with_donor(
            recipient_model, donor_weights, adapter, inputs, use_donor=use_donor
        )
        return masked_ce_loss(logits, targets, mask)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    final_loss = 0.0
    for step in range(ADAPTER_STEPS):
        inputs_mx, targets_mx, mask_mx = train_ds.generate_batch(ADAPTER_BATCH)
        mx.eval(inputs_mx); mx.eval(targets_mx); mx.eval(mask_mx)

        loss, grads = loss_and_grad(inputs_mx, targets_mx, mask_mx)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), loss)

        final_loss = float(loss)
        if (step + 1) % 200 == 0:
            print(f"    Step {step+1:4d}  loss={final_loss:.4f}")

    return final_loss


def eval_adapter_on_fixed_set(
    recipient_model: GPT,
    donor_weights: dict,
    adapter,
    fixed_eval,
    use_donor: bool = True,
) -> Tuple[float, np.ndarray]:
    """Evaluate the adapted model on the fixed eval set."""
    per_seq_fracs = []
    for inputs_np, targets_np, mask_np in fixed_eval:
        inputs_mx = mx.array(inputs_np)
        logits = adapted_forward_with_donor(
            recipient_model, donor_weights, adapter, inputs_mx, use_donor=use_donor
        )
        mx.eval(logits)
        logits_np = np.array(logits)
        preds = logits_np.argmax(axis=-1)
        at_mask = mask_np > 0.5
        correct = (preds == targets_np) & at_mask
        for b in range(inputs_np.shape[0]):
            n_mask = at_mask[b].sum()
            if n_mask > 0:
                per_seq_fracs.append(correct[b].sum() / n_mask)
            else:
                per_seq_fracs.append(0.0)
    per_seq_arr = np.array(per_seq_fracs)
    accuracy = float(per_seq_arr.mean())
    return accuracy, per_seq_arr


def method_learned_linear(model1: GPT, donor_w: dict, fixed_eval):
    print("\n[Method 6] Learned linear adapter")
    adapter = LinearAdapter(D_MODEL)
    mx.eval(adapter.parameters())

    print(f"  Training linear adapter ({ADAPTER_STEPS} steps, lr={ADAPTER_LR}, seed={TRAIN_SEED})...")
    t0 = time.time()
    final_loss = train_adapter(model1, donor_w, adapter, use_donor=True, label="linear")
    print(f"  Training done in {time.time()-t0:.1f}s  final_loss={final_loss:.4f}")

    acc, fracs = eval_adapter_on_fixed_set(model1, donor_w, adapter, fixed_eval, use_donor=True)
    print(f"  Accuracy: {acc:.4f}")

    return acc, fracs, {
        "params_added": 2 * D_MODEL * D_MODEL,
        "final_train_loss": final_loss,
    }


def method_learned_linear_zero_control(model1: GPT, donor_w: dict, fixed_eval):
    print("\n[Method 6b] Learned linear adapter (donor zeroed — control)")
    adapter = LinearAdapter(D_MODEL)
    mx.eval(adapter.parameters())
    print(f"  Training linear adapter with zeroed donor ({ADAPTER_STEPS} steps)...")
    final_loss = train_adapter(model1, donor_w, adapter, use_donor=False, label="linear_zero")
    acc, fracs = eval_adapter_on_fixed_set(model1, donor_w, adapter, fixed_eval, use_donor=False)
    print(f"  Accuracy: {acc:.4f}  (should be near chance if adapter can't cheat)")
    return acc, fracs, {"final_train_loss": final_loss}


def method_learned_nonlinear(model1: GPT, donor_w: dict, fixed_eval):
    print("\n[Method 7] Learned nonlinear adapter")
    adapter = NonlinearAdapter(D_MODEL)
    mx.eval(adapter.parameters())

    print(f"  Training nonlinear adapter ({ADAPTER_STEPS} steps, lr={ADAPTER_LR}, seed={TRAIN_SEED})...")
    t0 = time.time()
    final_loss = train_adapter(model1, donor_w, adapter, use_donor=True, label="nonlinear")
    print(f"  Training done in {time.time()-t0:.1f}s  final_loss={final_loss:.4f}")

    acc, fracs = eval_adapter_on_fixed_set(model1, donor_w, adapter, fixed_eval, use_donor=True)
    print(f"  Accuracy: {acc:.4f}")

    hidden = 4 * D_MODEL
    return acc, fracs, {
        "params_added": 2 * (D_MODEL * hidden + hidden * D_MODEL),
        "final_train_loss": final_loss,
    }


def method_learned_nonlinear_zero_control(model1: GPT, donor_w: dict, fixed_eval):
    print("\n[Method 7b] Learned nonlinear adapter (donor zeroed — control)")
    adapter = NonlinearAdapter(D_MODEL)
    mx.eval(adapter.parameters())
    print(f"  Training nonlinear adapter with zeroed donor ({ADAPTER_STEPS} steps)...")
    final_loss = train_adapter(model1, donor_w, adapter, use_donor=False, label="nonlinear_zero")
    acc, fracs = eval_adapter_on_fixed_set(model1, donor_w, adapter, fixed_eval, use_donor=False)
    print(f"  Accuracy: {acc:.4f}  (should be near chance if adapter can't cheat)")
    return acc, fracs, {"final_train_loss": final_loss}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_bar_chart(results: dict, out_path: Path):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    method_labels = {
        "unaligned":                   "Unaligned",
        "procrustes":                  "Procrustes\n(2-iface)",
        "affine":                      "Affine",
        "cka_linear":                  "CKA\nLinear",
        "git_rebasin":                 "Git\nRe-Basin",
        "learned_linear":              "Learned\nLinear",
        "learned_nonlinear":           "Learned\nNonlinear",
        "donor_zero_control_linear":   "Zero-donor\n(linear)",
        "donor_zero_control_nonlinear":"Zero-donor\n(nonlinear)",
    }

    methods = list(method_labels.keys())
    methods_data = results.get("methods", {})

    accs = []
    labels = []
    for m in methods:
        if m in methods_data:
            accs.append(methods_data[m]["accuracy"])
            labels.append(method_labels[m])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(accs))
    bars = ax.bar(x, accs, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.7)

    # Baseline lines
    seed0_acc = results["baselines"]["seed0"]
    seed1_acc = results["baselines"]["seed1"]
    ax.axhline(seed1_acc, color="green", linestyle="--", linewidth=1.5, label=f"Seed1 baseline ({seed1_acc:.3f})")
    ax.axhline(seed0_acc, color="orange", linestyle="--", linewidth=1.5, label=f"Seed0 baseline ({seed0_acc:.3f})")
    ax.axhline(1.0 / 512, color="red", linestyle=":", linewidth=1.2, label=f"Chance (1/512)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Induction Accuracy")
    ax.set_title("Alignment Tournament: Is transplant failure correctable?")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(accs + [seed1_acc, seed0_acc]) * 1.15)

    # Annotate bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ALIGNMENT TOURNAMENT — Experiment C")
    print("Question: Is transplant failure correctable by any alignment method?")
    print(f"Donor: seed0 L{DONOR_LAYER}H{DONOR_HEAD}  ->  Recipient: seed1 L{TARGET_LAYER}H{TARGET_HEAD}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Fixed eval set (seed=9999, 1024 seqs)
    # -----------------------------------------------------------------------
    print("\n[Step 1] Building fixed eval set (seed=9999, 1024 seqs)...")
    fixed_eval = build_fixed_eval_set()

    # -----------------------------------------------------------------------
    # 2. Load models
    # -----------------------------------------------------------------------
    print("\n[Step 2] Loading models...")
    model0 = load_model(str(BASE_DIR / SEED0_CKPT))
    model1 = load_model(str(BASE_DIR / SEED1_CKPT))
    print("  Models loaded.")

    # -----------------------------------------------------------------------
    # 3. Baselines
    # -----------------------------------------------------------------------
    print("\n[Step 3] Baselines...")
    acc0, fracs0 = evaluate_on_fixed_set(model0, fixed_eval)
    acc1, fracs1 = evaluate_on_fixed_set(model1, fixed_eval)
    print(f"  Seed 0: {acc0:.4f}")
    print(f"  Seed 1: {acc1:.4f}")

    # -----------------------------------------------------------------------
    # 4. Alignment activations (shared across methods 2-5)
    # -----------------------------------------------------------------------
    print("\n[Step 4] Collecting alignment activations (seed=1234, 256 seqs)...")
    pre_ln_0, pre_ln_1, post_l0_0, post_l0_1, align_inputs_mx = get_alignment_activations(model0, model1)
    print(f"  Pre-LN shape: {pre_ln_0.shape}  Post-L0 shape: {post_l0_0.shape}")

    # Donor weights (original, for static methods)
    donor_w = get_head_weights(model0, DONOR_LAYER, DONOR_HEAD)

    # -----------------------------------------------------------------------
    # 5. Run all methods
    # -----------------------------------------------------------------------
    all_fracs = {}
    methods_results = {}

    # Method 1: Unaligned
    acc, fracs, info = method_unaligned(model1, donor_w, fixed_eval)
    methods_results["unaligned"] = {"accuracy": acc, **info}
    all_fracs["unaligned"] = fracs

    # Method 2: Procrustes
    acc, fracs, info = method_procrustes(model1, donor_w, fixed_eval,
                                         pre_ln_0, pre_ln_1, post_l0_0, post_l0_1)
    methods_results["procrustes"] = {"accuracy": acc, **info}
    all_fracs["procrustes"] = fracs

    # Method 3: Affine
    acc, fracs, info = method_affine(model1, donor_w, fixed_eval,
                                      pre_ln_0, pre_ln_1, post_l0_0, post_l0_1)
    methods_results["affine"] = {"accuracy": acc, **info}
    all_fracs["affine"] = fracs

    # Method 4: CKA linear
    acc, fracs, info = method_cka_linear(model1, donor_w, fixed_eval,
                                          pre_ln_0, pre_ln_1, post_l0_0, post_l0_1)
    methods_results["cka_linear"] = {"accuracy": acc, **info}
    all_fracs["cka_linear"] = fracs

    # Method 5: Git Re-Basin
    acc, fracs, info = method_git_rebasin(model0, model1, donor_w, fixed_eval,
                                           pre_ln_0, pre_ln_1, post_l0_0, post_l0_1,
                                           align_inputs_mx)
    methods_results["git_rebasin"] = {"accuracy": acc, **info}
    all_fracs["git_rebasin"] = fracs

    # Method 6: Learned linear adapter
    acc, fracs, info = method_learned_linear(model1, donor_w, fixed_eval)
    methods_results["learned_linear"] = {"accuracy": acc, **info}
    all_fracs["learned_linear"] = fracs

    # Method 6 control: zero donor
    acc_ctrl, fracs_ctrl, info_ctrl = method_learned_linear_zero_control(model1, donor_w, fixed_eval)
    methods_results["donor_zero_control_linear"] = {"accuracy": acc_ctrl, **info_ctrl}
    all_fracs["donor_zero_control_linear"] = fracs_ctrl

    # Method 7: Learned nonlinear adapter
    acc, fracs, info = method_learned_nonlinear(model1, donor_w, fixed_eval)
    methods_results["learned_nonlinear"] = {"accuracy": acc, **info}
    all_fracs["learned_nonlinear"] = fracs

    # Method 7 control: zero donor
    acc_ctrl, fracs_ctrl, info_ctrl = method_learned_nonlinear_zero_control(model1, donor_w, fixed_eval)
    methods_results["donor_zero_control_nonlinear"] = {"accuracy": acc_ctrl, **info_ctrl}
    all_fracs["donor_zero_control_nonlinear"] = fracs_ctrl

    # -----------------------------------------------------------------------
    # 6. Pairwise statistics vs unaligned baseline
    # -----------------------------------------------------------------------
    print("\n[Step 6] Computing pairwise statistics vs unaligned baseline...")
    pairwise = {}
    unaligned_fracs = all_fracs["unaligned"]
    comparison_methods = [k for k in all_fracs if k != "unaligned"]
    for method_key in comparison_methods:
        mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(unaligned_fracs, all_fracs[method_key])
        p, _ = paired_sign_flip_pvalue(unaligned_fracs, all_fracs[method_key])
        pairwise[f"{method_key}_vs_unaligned"] = {
            "mean_diff": mean_diff,
            "ci_95": [ci_lo, ci_hi],
            "p": p,
        }
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {method_key:35s}: diff={mean_diff:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}] p={p:.4f} {sig}")

    # -----------------------------------------------------------------------
    # 7. Results summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} | {'Accuracy':>8} | {'vs unaligned':>12} | {'p-value':>8}")
    print("-" * 70)
    unaligned_acc = methods_results["unaligned"]["accuracy"]
    for method_key, mdata in methods_results.items():
        acc = mdata["accuracy"]
        diff = acc - unaligned_acc
        pair_key = f"{method_key}_vs_unaligned"
        p_str = f"{pairwise[pair_key]['p']:.4f}" if pair_key in pairwise else "  —  "
        print(f"  {method_key:<33} | {acc:>8.4f} | {diff:>+12.4f} | {p_str:>8}")
    print(f"\n  Seed 0 baseline: {acc0:.4f}")
    print(f"  Seed 1 baseline: {acc1:.4f}")

    # -----------------------------------------------------------------------
    # 8. Save JSON
    # -----------------------------------------------------------------------
    # Convert numpy fracs to per_seq paths (save as npy)
    per_seq_dir = Path(__file__).parent / "alignment_tournament_per_seq"
    per_seq_dir.mkdir(parents=True, exist_ok=True)

    methods_json = {}
    for k, mdata in methods_results.items():
        entry = dict(mdata)
        if k in all_fracs:
            npy_path = per_seq_dir / f"{k}.npy"
            np.save(str(npy_path), all_fracs[k])
            entry["per_seq_path"] = str(npy_path)
        methods_json[k] = entry

    results = {
        "baselines": {"seed0": float(acc0), "seed1": float(acc1)},
        "methods": methods_json,
        "pairwise_pvalues_vs_unaligned": {
            k: {
                "mean_diff": float(v["mean_diff"]),
                "ci_95": [float(v["ci_95"][0]), float(v["ci_95"][1])],
                "p": float(v["p"]),
            }
            for k, v in pairwise.items()
        },
        "eval_protocol": {
            "n_sequences": 1024,
            "data_seed": 9999,
            "align_seed": ALIGN_SEED,
            "align_batch": ALIGN_BATCH,
            "train_seed": TRAIN_SEED,
            "adapter_steps": ADAPTER_STEPS,
            "adapter_batch": ADAPTER_BATCH,
            "adapter_lr": ADAPTER_LR,
        },
    }

    # Ensure all float values are JSON serializable
    def to_json_safe(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]
        return obj

    results = to_json_safe(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # 9. Plot
    # -----------------------------------------------------------------------
    plot_path = PLOTS_DIR / "alignment_tournament.png"
    make_bar_chart(results, plot_path)

    print("\n" + "=" * 70)
    print("DONE — Alignment Tournament complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
