"""Faithful PyTorch port of src/model.py for cross-framework replication.

Matches the MLX model exactly in architecture and layout:
- Pre-norm blocks, LayerNorm without bias, no linear biases anywhere.
- Fused QKV: rows [0,d) = Q, [d,2d) = K, [2d,3d) = V; head h occupies rows
  [h*d_head, (h+1)*d_head) within each section. out_proj input (weight columns)
  is head-major, so head h owns columns [h*d_head, (h+1)*d_head).
- Learned positional embeddings, weight-tied unembedding, exact (erf) GELU,
  additive -1e9 causal mask, fp32.
- Init mimics MLX defaults: Linear ~ U(±1/sqrt(fan_in)) (torch's default),
  Embedding ~ N(0, 1/sqrt(d_model)) (set explicitly; torch default is N(0,1)).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    vocab_size: int = 512
    ctx_len: int = 64

    @property
    def d_head(self):
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self._attn_weights = None

    def forward(self, x, head_value_mask=None, cache_attn=False):
        """head_value_mask: optional (n_heads,) 0/1 tensor; zeros a head's value
        output before out_proj (the ablation used to identify critical heads)."""
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if head_value_mask is None and not cache_attn:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
            mask = torch.triu(torch.full((T, T), -1e9, device=x.device), diagonal=1)
            attn = torch.softmax(attn + mask, dim=-1)
            if cache_attn:
                self._attn_weights = attn.detach()
            out = attn @ v
            if head_value_mask is not None:
                out = out * head_value_mask.view(1, -1, 1, 1)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, bias=False)
        self.mlp = MLP(config)

    def forward(self, x, head_value_mask=None, cache_attn=False):
        x = x + self.attn(self.ln1(x), head_value_mask=head_value_mask, cache_attn=cache_attn)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.ctx_len, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.d_model, bias=False)
        # Mimic MLX embedding init scale.
        with torch.no_grad():
            self.wte.weight.normal_(0.0, config.d_model ** -0.5)
            self.wpe.weight.normal_(0.0, config.d_model ** -0.5)

    def forward(self, idx, head_value_masks=None, cache_attn=False):
        """head_value_masks: optional dict {layer_index: (n_heads,) mask tensor}."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for i, block in enumerate(self.blocks):
            hm = None if head_value_masks is None else head_value_masks.get(i)
            x = block(x, head_value_mask=hm, cache_attn=cache_attn)
        x = self.ln_f(x)
        return x @ self.wte.weight.T


def create_model(config, seed=0):
    torch.manual_seed(seed)
    return GPT(config)


# ---------------------------------------------------------------------------
# Per-head weight surgery (mirrors mlx_core.get_head_weights conventions).
# ---------------------------------------------------------------------------

def head_slices(config, head):
    d, dh = config.d_model, config.d_head
    rows = slice(head * dh, (head + 1) * dh)
    return {
        "q": slice(0 * d + head * dh, 0 * d + (head + 1) * dh),
        "k": slice(1 * d + head * dh, 1 * d + (head + 1) * dh),
        "v": slice(2 * d + head * dh, 2 * d + (head + 1) * dh),
        "out_cols": rows,
    }


def get_head_weights(model, layer, head):
    sl = head_slices(model.config, head)
    attn = model.blocks[layer].attn
    return {
        "q": attn.qkv_proj.weight[sl["q"], :].detach().clone(),
        "k": attn.qkv_proj.weight[sl["k"], :].detach().clone(),
        "v": attn.qkv_proj.weight[sl["v"], :].detach().clone(),
        "out": attn.out_proj.weight[:, sl["out_cols"]].detach().clone(),
    }


def set_head_weights(model, layer, head, weights):
    sl = head_slices(model.config, head)
    attn = model.blocks[layer].attn
    with torch.no_grad():
        attn.qkv_proj.weight[sl["q"], :] = weights["q"]
        attn.qkv_proj.weight[sl["k"], :] = weights["k"]
        attn.qkv_proj.weight[sl["v"], :] = weights["v"]
        attn.out_proj.weight[:, sl["out_cols"]] = weights["out"]


def zero_head_weights(model, layer, head):
    z = get_head_weights(model, layer, head)
    set_head_weights(model, layer, head, {k: torch.zeros_like(v) for k, v in z.items()})


def shuffled_head_weights(weights, seed):
    """Element-shuffle each tensor (preserves weight statistics, destroys structure)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, w in weights.items():
        flat = w.flatten()
        out[k] = flat[torch.randperm(flat.numel(), generator=g)].reshape(w.shape)
    return out


def random_head_weights(config, seed):
    """Fresh init-scale random head (never trained)."""
    g = torch.Generator().manual_seed(seed)
    d, dh = config.d_model, config.d_head
    bound = 1.0 / math.sqrt(d)
    def u(*shape):
        return (torch.rand(*shape, generator=g) * 2 - 1) * bound
    return {"q": u(dh, d), "k": u(dh, d), "v": u(dh, d), "out": u(d, dh)}
