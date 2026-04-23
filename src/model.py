"""
Minimal GPT-2 style transformer for developmental interpretability research.

Architecture: standard decoder-only transformer with:
- Learned positional embeddings
- Pre-norm (LayerNorm before attention/MLP)
- No bias terms (cleaner for interpretability)
- Configurable depth/width for scaling experiments

Default config:
- 6 layers, 6 heads, d_model=384, d_ff=1536, vocab=50257, ctx=256
This generic default is not one of the reported paper configurations.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """Model configuration."""
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536
    vocab_size: int = 50257
    ctx_len: int = 256
    dropout: float = 0.0  # No dropout for interpretability (deterministic forward pass)

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    def param_count(self):
        """Estimate total parameters."""
        embed = self.vocab_size * self.d_model + self.ctx_len * self.d_model
        attn_per_layer = 4 * self.d_model * self.d_model  # QKV + out proj
        ff_per_layer = 2 * self.d_model * self.d_ff  # up + down
        ln_per_layer = 2 * self.d_model  # 2 layer norms per layer (gamma only, no bias)
        layer_total = (attn_per_layer + ff_per_layer + ln_per_layer) * self.n_layers
        final_ln = self.d_model
        # Weight tying: no separate unembedding
        return embed + layer_total + final_ln


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Stores attention weights for circuit extraction.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Cache for interpretability
        self._attn_weights = None

    def __call__(self, x: mx.array, cache: Optional[dict] = None) -> mx.array:
        B, T, C = x.shape

        # QKV projection and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.transpose(0, 3, 2, 1, 4)  # (B, n_heads, 3, T, d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Scaled dot-product attention with causal mask
        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        attn = attn + mask
        attn = mx.softmax(attn, axis=-1)

        # Store for interpretability hooks
        self._attn_weights = attn

        # Apply attention to values
        out = attn @ v  # (B, n_heads, T, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_proj(out)

        return out


class MLP(nn.Module):
    """Standard FFN with GELU activation."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, bias=False)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style model for developmental interpretability.

    Features:
    - Weight-tied embedding/unembedding
    - Residual stream access at every layer (for activation patching)
    - Attention weight caching (for circuit extraction)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.ctx_len, config.d_model)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.ln_f = nn.LayerNorm(config.d_model, bias=False)

    def __call__(self, idx: mx.array) -> mx.array:
        """Forward pass returning logits.

        Args:
            idx: Token indices, shape (B, T)
        Returns:
            logits: Shape (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.config.ctx_len, f"Sequence length {T} > context length {self.config.ctx_len}"

        pos = mx.arange(T)
        x = self.wte(idx) + self.wpe(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Weight-tied unembedding
        logits = x @ self.wte.weight.T
        return logits

    def get_residual_stream(self, idx: mx.array) -> list[mx.array]:
        """Forward pass returning residual stream at every layer.

        Returns list of (n_layers + 1) tensors: [after_embed, after_block_0, ..., after_block_n]
        """
        B, T = idx.shape
        pos = mx.arange(T)
        x = self.wte(idx) + self.wpe(pos)

        residuals = [x]
        for block in self.blocks:
            x = block(x)
            residuals.append(x)

        return residuals

    def get_attention_patterns(self) -> list[mx.array]:
        """Return cached attention weights from last forward pass.

        Returns list of n_layers tensors, each shape (B, n_heads, T, T).
        """
        return [block.attn._attn_weights for block in self.blocks]


def create_model(config: Optional[GPTConfig] = None, seed: int = 0) -> GPT:
    """Create and initialize a model with a specific seed.

    Different seeds = different initialization = our independent variable.
    """
    if config is None:
        config = GPTConfig()

    mx.random.seed(seed)
    model = GPT(config)
    mx.eval(model.parameters())

    return model


if __name__ == "__main__":
    config = GPTConfig()
    print(f"Model config: {config}")
    print(f"Estimated parameters: {config.param_count():,}")

    model = create_model(config, seed=42)

    # Test forward pass
    dummy_input = mx.array([[1, 2, 3, 4, 5]])
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")

    # Test residual stream
    residuals = model.get_residual_stream(dummy_input)
    print(f"Residual stream layers: {len(residuals)}")
    print(f"Each residual shape: {residuals[0].shape}")

    # Test attention patterns
    patterns = model.get_attention_patterns()
    print(f"Attention layers: {len(patterns)}")
    print(f"Each pattern shape: {patterns[0].shape}")
