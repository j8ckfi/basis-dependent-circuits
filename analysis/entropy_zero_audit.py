"""
Entropy-Zero Audit: Is the "positional" architecture at entropy=0 genuine or an artifact?

BACKGROUND:
Experiment 7 (entropy continuum) showed that at entropy=0 (deterministic Markov chain),
81% of heads are classified as "positional" and 19% as "suppression". This stands out
from every other entropy level.

At entropy=0, each token has exactly ONE successor, so the model could use either:
  Strategy A (truly positional): learn position -> token lookup
      -- IMPOSSIBLE because token identities are random per-sequence
  Strategy B (content-based via MLP): attend to self (diagonal), then use MLP as a
      token-to-successor lookup table. The attention pattern would be purely diagonal
      (same-position always = same token for causal attention), which our metric would
      misread as "positional" because patterns don't vary across inputs.

This script runs three diagnostics to distinguish these strategies:
  A. Attention pattern inspection: are patterns truly positional or just diagonal/self?
  B. MLP ablation: does zeroing any MLP layer destroy accuracy?
  C. Content vs position intervention: swap token content but keep positions, and vice versa.

Verdict: "positional", "content-based-through-MLP", or "hybrid"
"""

import sys
import json
import time
import numpy as np
from pathlib import Path


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars and arrays to plain Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from src.model import GPT, GPTConfig, create_model
from src.circuit_mapping import profile_all_heads

# Re-use MarkovDataset from entropy_continuum experiment
from experiments.entropy_continuum import MarkovDataset


# ============================================================
# Config
# ============================================================

VOCAB_SIZE = 64
SEQ_LEN = 128
N_LAYERS = 4
N_HEADS = 4
D_MODEL = 128
D_FF = 512
N_TRAIN_STEPS = 3000

CKPT_DIR = Path("checkpoints/entropy_zero_audit")
RESULTS_PATH = Path("analysis/entropy_zero_audit.json")


# ============================================================
# Training
# ============================================================

def get_lr(step: int, n_steps: int, lr: float, warmup: int = 200) -> float:
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, n_steps - warmup)
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_entropy_zero_model(dataset: MarkovDataset) -> GPT:
    """Train a fresh entropy=0 model for 3000 steps."""
    cfg = GPTConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
        ctx_len=SEQ_LEN,
        dropout=0.0,
    )

    n_steps = N_TRAIN_STEPS
    batch_size = 64
    lr = 1e-3

    model = create_model(cfg, seed=0)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.1)

    def compute_loss(model, inputs, targets, mask):
        logits = model(inputs)
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        mask_flat = mask.reshape(B * T)
        per_tok = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (per_tok * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    import mlx.utils

    t0 = time.time()
    data_iter = dataset.iter_batches(batch_size, n_steps)

    for step, (inputs, targets, mask) in enumerate(data_iter):
        current_lr = get_lr(step, n_steps, lr)
        optimizer.learning_rate = current_lr

        loss, grads = loss_and_grad(model, inputs, targets, mask)
        grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"    step {step:5d}/{n_steps} | loss={loss.item():.4f} | "
                  f"lr={current_lr:.2e} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"    Training done in {elapsed:.1f}s")
    return model


# ============================================================
# Accuracy helpers
# ============================================================

def compute_accuracy(model: GPT, dataset: MarkovDataset, n_batches: int = 20) -> float:
    """Measure next-token prediction accuracy on the deterministic Markov task."""
    total_correct = 0
    total_tokens = 0
    for _ in range(n_batches):
        inputs, targets, _ = dataset.generate_batch(32)
        logits = model(inputs)
        preds = mx.argmax(logits, axis=-1)
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_tokens += targets.size
    return total_correct / max(total_tokens, 1)


def compute_accuracy_with_forward(
    model: GPT,
    inputs: mx.array,
    targets: mx.array,
    forward_fn,
) -> float:
    """Accuracy using a custom forward function."""
    logits = forward_fn(inputs)
    preds = mx.argmax(logits, axis=-1)
    correct = (preds == targets).sum().item()
    return correct / targets.size


# ============================================================
# Diagnostic A: Attention pattern inspection
# ============================================================

def diagnostic_a_attention_patterns(
    model: GPT,
    dataset: MarkovDataset,
    n_samples: int = 20,
) -> dict:
    """
    Inspect actual attention patterns for 20 random inputs.

    Key question: are patterns positional (vary with position regardless of content)
    or diagonal/self-attending (token at pos i always attends to itself)?

    We measure:
    - mean_diag_weight: average weight on same-position (self) token
    - positional_score: variance-based metric from circuit_mapping (low = same across inputs = "positional")
    - cross_input_pattern_corr: correlation of attention patterns across different inputs at the same positions

    If patterns are truly positional (location-based), they should be consistent across
    inputs that share the same length. If they are content-based, rearranging input tokens
    should change the attended-to positions.
    """
    print("\n--- Diagnostic A: Attention pattern inspection ---")

    # Generate inputs
    inputs_list = []
    for _ in range(n_samples):
        inp, _, _ = dataset.generate_batch(1)
        inputs_list.append(inp)

    # Collect patterns across all samples
    all_patterns = []  # List of (n_layers, n_heads, T, T) arrays
    for inp in inputs_list:
        _ = model(inp)
        patterns = model.get_attention_patterns()
        # patterns is a list of n_layers tensors, each (1, n_heads, T, T)
        layer_patterns = [np.array(p[0]) for p in patterns]  # list of (n_heads, T, T)
        all_patterns.append(layer_patterns)

    # Per layer/head statistics
    n_layers = len(all_patterns[0])
    n_heads = all_patterns[0][0].shape[0]
    T = all_patterns[0][0].shape[1]

    layer_head_stats = []

    for l in range(n_layers):
        for h in range(n_heads):
            # Stack patterns: (n_samples, T, T)
            stack = np.stack([all_patterns[s][l][h] for s in range(n_samples)])

            # Diagonal (self-attention) weight
            diag_weights = np.array([np.diag(stack[s]) for s in range(n_samples)])
            mean_self_attn = float(diag_weights.mean())

            # Variance of pattern across inputs (low = positional-like)
            pattern_variance = stack.var(axis=0).mean()
            pattern_mean = stack.mean(axis=0).mean()
            positional_score = 1.0 - min(1.0, pattern_variance / (pattern_mean + 1e-10))

            # Previous token weight
            prev_tok_weights = []
            for pos in range(1, T):
                prev_tok_weights.append(stack[:, pos, pos - 1].mean())
            mean_prev_tok = float(np.mean(prev_tok_weights)) if prev_tok_weights else 0.0

            # Cross-input pattern correlation: do different inputs produce the same pattern?
            # Flatten T*T, compute pairwise correlations
            flat_patterns = stack.reshape(n_samples, T * T)
            # Correlation matrix
            corr_matrix = np.corrcoef(flat_patterns)
            # Mean off-diagonal correlation (how similar patterns are across inputs)
            mask_offdiag = ~np.eye(n_samples, dtype=bool)
            mean_cross_corr = float(corr_matrix[mask_offdiag].mean())

            layer_head_stats.append({
                "layer": l,
                "head": h,
                "mean_self_attn": mean_self_attn,
                "mean_prev_tok": mean_prev_tok,
                "positional_score": positional_score,
                "mean_cross_input_pattern_corr": mean_cross_corr,
            })

            print(f"  L{l}H{h}: self_attn={mean_self_attn:.3f}  "
                  f"prev_tok={mean_prev_tok:.3f}  "
                  f"pos_score={positional_score:.3f}  "
                  f"cross_corr={mean_cross_corr:.3f}")

    # Aggregate: are most heads attending to self (diagonal)?
    all_self = [s["mean_self_attn"] for s in layer_head_stats]
    all_cross_corr = [s["mean_cross_input_pattern_corr"] for s in layer_head_stats]
    mean_self_overall = float(np.mean(all_self))
    mean_cross_corr_overall = float(np.mean(all_cross_corr))

    print(f"\n  Overall mean self-attention weight: {mean_self_overall:.3f}")
    print(f"  Overall mean cross-input pattern correlation: {mean_cross_corr_overall:.3f}")
    print(f"  Interpretation: cross_corr near 1.0 means patterns are ~identical across inputs")
    print(f"  (which could be positional OR just diagonal — both look the same across inputs)")

    return {
        "layer_head_stats": layer_head_stats,
        "mean_self_attn_overall": mean_self_overall,
        "mean_cross_input_pattern_corr_overall": mean_cross_corr_overall,
    }


# ============================================================
# Diagnostic B: MLP and attention ablation
# ============================================================

def forward_with_mlp_ablated(model: GPT, inputs: mx.array, ablate_layer: int) -> mx.array:
    """Forward pass with one MLP layer's output zeroed."""
    B, T = inputs.shape
    pos = mx.arange(T)
    x = model.wte(inputs) + model.wpe(pos)

    for i, block in enumerate(model.blocks):
        attn_out = block.attn(block.ln1(x))
        x = x + attn_out

        mlp_out = block.mlp(block.ln2(x))
        if i == ablate_layer:
            mlp_out = mx.zeros_like(mlp_out)
        x = x + mlp_out

    x = model.ln_f(x)
    return x @ model.wte.weight.T


def forward_with_attn_ablated(model: GPT, inputs: mx.array, ablate_layer: int) -> mx.array:
    """Forward pass with one attention layer's output zeroed."""
    B, T = inputs.shape
    pos = mx.arange(T)
    x = model.wte(inputs) + model.wpe(pos)

    for i, block in enumerate(model.blocks):
        attn_out = block.attn(block.ln1(x))
        if i == ablate_layer:
            attn_out = mx.zeros_like(attn_out)
        x = x + attn_out

        mlp_out = block.mlp(block.ln2(x))
        x = x + mlp_out

    x = model.ln_f(x)
    return x @ model.wte.weight.T


def diagnostic_b_ablations(
    model: GPT,
    dataset: MarkovDataset,
    baseline_acc: float,
    n_batches: int = 20,
) -> dict:
    """
    Ablate each MLP and each attention layer one at a time.
    Measure accuracy drop.

    If ablating MLP_i causes a large drop: the computation is MLP-based (Strategy B).
    If ablating Attn_i causes a large drop but MLPs don't: truly positional (Strategy A).
    """
    print("\n--- Diagnostic B: Component ablation ---")
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    # Gather test batches
    test_inputs_list = []
    test_targets_list = []
    for _ in range(n_batches):
        inputs, targets, _ = dataset.generate_batch(32)
        test_inputs_list.append(inputs)
        test_targets_list.append(targets)

    def batch_accuracy(forward_fn) -> float:
        total_correct = 0
        total_tokens = 0
        for inputs, targets in zip(test_inputs_list, test_targets_list):
            logits = forward_fn(inputs)
            preds = mx.argmax(logits, axis=-1)
            correct = (preds == targets).sum().item()
            total_correct += correct
            total_tokens += targets.size
        return total_correct / max(total_tokens, 1)

    mlp_ablation_results = []
    attn_ablation_results = []

    for layer_idx in range(N_LAYERS):
        # MLP ablation
        acc_mlp = batch_accuracy(
            lambda inp, li=layer_idx: forward_with_mlp_ablated(model, inp, li)
        )
        drop_mlp = baseline_acc - acc_mlp
        mlp_ablation_results.append({
            "layer": layer_idx,
            "accuracy": float(acc_mlp),
            "accuracy_drop": float(drop_mlp),
        })
        print(f"  Ablate MLP L{layer_idx}: acc={acc_mlp:.4f} (drop={drop_mlp:+.4f})")

        # Attention ablation
        acc_attn = batch_accuracy(
            lambda inp, li=layer_idx: forward_with_attn_ablated(model, inp, li)
        )
        drop_attn = baseline_acc - acc_attn
        attn_ablation_results.append({
            "layer": layer_idx,
            "accuracy": float(acc_attn),
            "accuracy_drop": float(drop_attn),
        })
        print(f"  Ablate ATN L{layer_idx}: acc={acc_attn:.4f} (drop={drop_attn:+.4f})")

    max_mlp_drop = max(r["accuracy_drop"] for r in mlp_ablation_results)
    max_attn_drop = max(r["accuracy_drop"] for r in attn_ablation_results)

    print(f"\n  Max MLP ablation drop: {max_mlp_drop:.4f}")
    print(f"  Max Attn ablation drop: {max_attn_drop:.4f}")

    if max_mlp_drop > 0.3:
        print("  => MLP is essential: computation is MLP-based (Strategy B)")
    elif max_attn_drop > 0.3 and max_mlp_drop < 0.1:
        print("  => Attention is essential, MLP is not: truly positional (Strategy A)")
    else:
        print("  => Mixed or unclear from ablation alone")

    return {
        "baseline_accuracy": float(baseline_acc),
        "mlp_ablation": mlp_ablation_results,
        "attn_ablation": attn_ablation_results,
        "max_mlp_drop": float(max_mlp_drop),
        "max_attn_drop": float(max_attn_drop),
    }


# ============================================================
# Diagnostic C: Content vs Position intervention
# ============================================================

def forward_standard(model: GPT, inputs: mx.array) -> mx.array:
    return model(inputs)


def forward_with_shuffled_content(
    model: GPT,
    inputs: mx.array,
    rng: np.random.Generator,
) -> mx.array:
    """
    Keep same POSITIONS, replace token CONTENT with a random permutation.
    Each sequence gets a fresh random token permutation applied.

    If accuracy drops significantly: the model relies on token identity (content-based).
    If accuracy stays similar: the model doesn't care which token is at each position.
    """
    inputs_np = np.array(inputs)
    B, T = inputs_np.shape
    shuffled = np.zeros_like(inputs_np)
    for b in range(B):
        # Permute the token vocabulary randomly: token x -> perm[x]
        perm = rng.permutation(VOCAB_SIZE).astype(np.int32)
        shuffled[b] = perm[inputs_np[b]]
    return model(mx.array(shuffled))


def forward_with_shifted_positions(
    model: GPT,
    inputs: mx.array,
    shift: int = 4,
) -> mx.array:
    """
    Keep same token CONTENT, but shift positional embeddings by `shift`.
    We do this by passing pos_offset to the embedding.

    Implementation: reuse token embeddings but use offset positional embeddings.
    If accuracy drops: the model relies on absolute positions (positional).
    If accuracy stays similar: the model doesn't rely on absolute positions.
    """
    B, T = inputs.shape
    # Clamp positions to valid range
    pos = mx.arange(shift, shift + T) % model.config.ctx_len

    x = model.wte(inputs) + model.wpe(pos)
    for block in model.blocks:
        x = x + block.attn(block.ln1(x))
        x = x + block.mlp(block.ln2(x))
    x = model.ln_f(x)
    return x @ model.wte.weight.T


def diagnostic_c_content_vs_position(
    model: GPT,
    dataset: MarkovDataset,
    baseline_acc: float,
    n_batches: int = 20,
) -> dict:
    """
    Test whether accuracy depends on token content or absolute position.

    Intervention 1: Shuffle token identities (content) -> same positions, different content.
      Large drop => content matters => Strategy B
    Intervention 2: Shift positional embeddings -> same content, different positions.
      Large drop => positions matter => Strategy A
    """
    print("\n--- Diagnostic C: Content vs position intervention ---")
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    rng = np.random.default_rng(seed=777)

    test_inputs = []
    test_targets = []
    for _ in range(n_batches):
        inp, tgt, _ = dataset.generate_batch(32)
        test_inputs.append(inp)
        test_targets.append(tgt)

    # --- Intervention 1: Shuffle token content (vocabulary permutation) ---
    total_correct_shuffled = 0
    total_tokens = 0
    for inputs, targets in zip(test_inputs, test_targets):
        inputs_np = np.array(inputs)
        targets_np = np.array(targets)
        B, T = inputs_np.shape

        # Apply random vocab permutation to both inputs AND targets consistently
        perm = rng.permutation(VOCAB_SIZE).astype(np.int32)
        shuffled_inputs = mx.array(perm[inputs_np])
        shuffled_targets = mx.array(perm[targets_np])

        logits = model(shuffled_inputs)
        preds = mx.argmax(logits, axis=-1)
        correct = (preds == shuffled_targets).sum().item()
        total_correct_shuffled += correct
        total_tokens += shuffled_targets.size

    acc_shuffled_content = total_correct_shuffled / max(total_tokens, 1)
    drop_shuffled = baseline_acc - acc_shuffled_content
    print(f"  Shuffled token content (relabeled vocab): acc={acc_shuffled_content:.4f} (drop={drop_shuffled:+.4f})")

    # --- Intervention 2: Shift positional embeddings ---
    shift_amounts = [4, 8, 16, 32]
    shift_results = []

    for shift in shift_amounts:
        total_correct_shifted = 0
        total_tok = 0
        for inputs, targets in zip(test_inputs, test_targets):
            logits = forward_with_shifted_positions(model, inputs, shift=shift)
            preds = mx.argmax(logits, axis=-1)
            correct = (preds == targets).sum().item()
            total_correct_shifted += correct
            total_tok += targets.size

        acc_shifted = total_correct_shifted / max(total_tok, 1)
        drop_shifted = baseline_acc - acc_shifted
        shift_results.append({
            "shift": shift,
            "accuracy": float(acc_shifted),
            "accuracy_drop": float(drop_shifted),
        })
        print(f"  Shifted positions by {shift:2d}: acc={acc_shifted:.4f} (drop={drop_shifted:+.4f})")

    # --- Intervention 3: Scramble token positions (permute token order within sequence) ---
    # If positional: accuracy should stay because tokens land at valid positions
    # If content-based with sequence context: accuracy drops
    total_correct_permuted = 0
    total_tok_p = 0
    rng2 = np.random.default_rng(seed=888)
    for inputs, targets in zip(test_inputs, test_targets):
        inputs_np = np.array(inputs)
        targets_np = np.array(targets)
        B, T_len = inputs_np.shape

        perm_idx = rng2.permutation(T_len)
        permuted_inputs = mx.array(inputs_np[:, perm_idx])
        # Targets are also re-ordered (they follow inputs)
        permuted_targets = mx.array(targets_np[:, perm_idx])

        logits = model(permuted_inputs)
        preds = mx.argmax(logits, axis=-1)
        correct = (preds == permuted_targets).sum().item()
        total_correct_permuted += correct
        total_tok_p += permuted_targets.size

    acc_permuted_seq = total_correct_permuted / max(total_tok_p, 1)
    drop_permuted_seq = baseline_acc - acc_permuted_seq
    print(f"  Permuted sequence order: acc={acc_permuted_seq:.4f} (drop={drop_permuted_seq:+.4f})")
    print(f"  (If truly positional: permuting seq should collapse accuracy since position->token mapping breaks)")
    print(f"  (If content-based: permuting seq should NOT hurt since the Markov chain still holds at each position)")

    # Interpretation
    print(f"\n  Summary of drops:")
    print(f"    Vocab relabeling (shuffle content): {drop_shuffled:+.4f}")
    mean_pos_drop = float(np.mean([r["accuracy_drop"] for r in shift_results]))
    print(f"    Position shift (mean over shifts):  {mean_pos_drop:+.4f}")
    print(f"    Sequence permutation:               {drop_permuted_seq:+.4f}")

    if drop_shuffled < 0.05 and mean_pos_drop > 0.2:
        print("  => Position matters, content doesn't: TRULY POSITIONAL (Strategy A)")
    elif drop_shuffled > 0.3 and mean_pos_drop < 0.05:
        print("  => Content matters, position doesn't: CONTENT-BASED-THROUGH-MLP (Strategy B)")
    elif drop_shuffled < 0.05 and drop_permuted_seq > 0.3:
        print("  => Sequence order matters but vocab labels don't: POSITIONAL WITHIN SEQUENCE (consistent with either)")
    else:
        print("  => Mixed signals")

    return {
        "baseline_accuracy": float(baseline_acc),
        "acc_shuffled_vocab": float(acc_shuffled_content),
        "drop_shuffled_vocab": float(drop_shuffled),
        "position_shift_results": shift_results,
        "mean_position_shift_drop": mean_pos_drop,
        "acc_permuted_sequence": float(acc_permuted_seq),
        "drop_permuted_sequence": float(drop_permuted_seq),
    }


# ============================================================
# Bonus: positional_score metric validity check
# ============================================================

def check_positional_score_metric(
    model: GPT,
    dataset: MarkovDataset,
    n_samples: int = 64,
) -> dict:
    """
    Directly examine what drives high positional_score.
    The metric is: 1 - variance(pattern_across_batch) / mean(pattern)

    For entropy=0: each input token has one fixed successor. The model might learn
    to attend to position i (diagonal) because f(X_i) is fully determined by X_i.
    The attention pattern would be the identity matrix (pos i attends to pos i).

    This is NOT a position-based mechanism: it's a content-based mechanism that
    happens to produce an invariant attention pattern (diagonal = same across any input).

    We check: is the dominant attention pattern the diagonal?
    And: does the pattern vary at all across inputs sharing the same token at position i?
    """
    print("\n--- Positional score metric validity check ---")

    profiles = profile_all_heads(model, dataset, n_samples=n_samples)

    positional_heads = [(p.layer, p.head, p.positional_score, p.same_token_score)
                        for p in profiles if p.positional_score > 0.5]

    print(f"  Heads with positional_score > 0.5: {len(positional_heads)}")
    for l, h, ps, st in positional_heads:
        print(f"    L{l}H{h}: pos_score={ps:.3f}, self_attn={st:.3f}")

    # For each "positional" head, check if it's really just diagonal
    # Generate inputs and inspect
    inp, _, _ = dataset.generate_batch(n_samples)
    _ = model(inp)
    patterns = model.get_attention_patterns()

    diagonal_heads = []  # (layer, head, mean_diag_weight)
    for l, block_patterns in enumerate(patterns):
        attn_np = np.array(block_patterns)  # (B, n_heads, T, T)
        for h in range(N_HEADS):
            head_pat = attn_np[:, h, :, :]  # (B, T, T)
            mean_diag = np.mean([np.diag(head_pat[b]).mean() for b in range(n_samples)])
            if mean_diag > 0.3:
                diagonal_heads.append((l, h, float(mean_diag)))

    print(f"\n  Heads with mean diagonal weight > 0.3: {len(diagonal_heads)}")
    for l, h, d in diagonal_heads:
        print(f"    L{l}H{h}: mean_diag_weight={d:.3f}")

    print("\n  KEY INSIGHT CHECK:")
    print("  For entropy=0, if a head attends mostly to itself (diagonal),")
    print("  the pattern is CONSTANT across all inputs (because pos i always has pos i).")
    print("  This gives positional_score -> 1.0 artificially, even though the")
    print("  computation is content-based (token identity -> MLP lookup).")

    return {
        "n_positional_heads": len(positional_heads),
        "positional_head_details": [
            {"layer": l, "head": h, "pos_score": ps, "self_attn": st}
            for l, h, ps, st in positional_heads
        ],
        "n_diagonal_heads": len(diagonal_heads),
        "diagonal_head_details": [
            {"layer": l, "head": h, "mean_diag_weight": d}
            for l, h, d in diagonal_heads
        ],
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("ENTROPY-ZERO AUDIT")
    print("Is the 'positional' architecture at entropy=0 genuine or an artifact?")
    print("=" * 70)

    # Build dataset
    dataset = MarkovDataset(
        entropy_level=0.0,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        seed=42,
    )
    print(f"\nDataset: entropy_level=0.0, measured_entropy={dataset.measured_entropy:.6f} nats")
    print(f"  (should be ~0 for fully deterministic transitions)")

    # Train or load model
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = CKPT_DIR / "model.safetensors"

    cfg = GPTConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
        ctx_len=SEQ_LEN,
        dropout=0.0,
    )

    if weights_path.exists():
        print(f"\nLoading existing model from {weights_path}")
        model = GPT(cfg)
        model.load_weights(str(weights_path))
        mx.eval(model.parameters())
    else:
        print(f"\nTraining fresh entropy=0 model ({N_TRAIN_STEPS} steps)...")
        model = train_entropy_zero_model(dataset)
        model.save_weights(str(weights_path))
        print(f"Saved to {weights_path}")

    # Verify task accuracy
    print("\nVerifying task accuracy...")
    baseline_acc = compute_accuracy(model, dataset, n_batches=20)
    print(f"  Task accuracy: {baseline_acc:.4f} (target: ~1.0)")

    if baseline_acc < 0.90:
        print("  WARNING: accuracy below 0.90 — model may not have converged.")
        print("  Proceeding with diagnostics anyway.")

    # ---- Run diagnostics ----

    results = {
        "task_accuracy": float(baseline_acc),
        "measured_entropy": float(dataset.measured_entropy),
    }

    # A. Attention pattern inspection
    diag_a = diagnostic_a_attention_patterns(model, dataset, n_samples=20)
    results["attention_patterns"] = diag_a

    # B. MLP and attention ablation
    diag_b = diagnostic_b_ablations(model, dataset, baseline_acc, n_batches=20)
    results["mlp_ablation"] = diag_b["mlp_ablation"]
    results["attn_ablation"] = diag_b["attn_ablation"]
    results["ablation_summary"] = {
        "baseline_accuracy": float(baseline_acc),
        "max_mlp_drop": diag_b["max_mlp_drop"],
        "max_attn_drop": diag_b["max_attn_drop"],
    }

    # C. Content vs position intervention
    diag_c = diagnostic_c_content_vs_position(model, dataset, baseline_acc, n_batches=20)
    results["content_vs_position"] = diag_c

    # Metric validity check
    diag_metric = check_positional_score_metric(model, dataset, n_samples=64)
    results["metric_validity"] = diag_metric

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    max_mlp_drop = diag_b["max_mlp_drop"]
    max_attn_drop = diag_b["max_attn_drop"]
    drop_shuffled = diag_c["drop_shuffled_vocab"]
    mean_pos_drop = diag_c["mean_position_shift_drop"]
    drop_permuted = diag_c["drop_permuted_sequence"]
    n_diagonal = diag_metric["n_diagonal_heads"]
    n_positional = diag_metric["n_positional_heads"]

    # Decision logic
    evidence_content = 0
    evidence_positional = 0

    # MLP evidence
    if max_mlp_drop > 0.3:
        evidence_content += 2
        print(f"  [+content] MLP ablation causes {max_mlp_drop:.3f} accuracy drop")
    elif max_mlp_drop < 0.05:
        evidence_positional += 1
        print(f"  [+positional] MLP ablation causes only {max_mlp_drop:.3f} drop")

    # Attention evidence
    if max_attn_drop > 0.3:
        # Both strategies require attention (diagonal for content, non-diagonal for positional)
        print(f"  [neutral] Attention ablation causes {max_attn_drop:.3f} drop (expected for both strategies)")

    # Content intervention evidence
    if drop_shuffled < 0.05:
        evidence_positional += 2
        print(f"  [+positional] Vocab relabeling causes only {drop_shuffled:.3f} drop")
    elif drop_shuffled > 0.3:
        evidence_content += 2
        print(f"  [+content] Vocab relabeling causes {drop_shuffled:.3f} drop")

    # Position shift evidence
    if mean_pos_drop > 0.2:
        evidence_positional += 1
        print(f"  [+positional] Position shifting causes mean {mean_pos_drop:.3f} drop")
    elif mean_pos_drop < 0.05:
        evidence_content += 1
        print(f"  [+content] Position shifting causes only {mean_pos_drop:.3f} drop (model is position-invariant)")

    # Diagonal attention evidence
    if n_diagonal > N_LAYERS * N_HEADS // 2:
        evidence_content += 1
        print(f"  [+content] {n_diagonal}/{N_LAYERS*N_HEADS} heads attend to self (diagonal) — "
              f"suggests content lookup via self-attention + MLP")

    # Sequence permutation evidence
    if drop_permuted > 0.3:
        print(f"  [neutral] Permuting sequence order causes {drop_permuted:.3f} drop "
              f"(expected — Markov chain loses context when positions scrambled)")

    print(f"\n  Evidence for content-based (Strategy B): {evidence_content}")
    print(f"  Evidence for truly positional (Strategy A): {evidence_positional}")

    if evidence_content > evidence_positional + 1:
        verdict = "content-based-through-MLP"
        explanation = (
            "The entropy=0 'positional' classification is an ARTIFACT of our metric. "
            "The model learns a diagonal attention pattern (each position attends to itself), "
            "which is CONSTANT across all inputs and therefore registers as high positional_score. "
            "However, the actual computation is content-based: the token at each position is "
            "attended to by itself, and the MLP performs a lookup of token X -> successor f(X). "
            "This is Strategy B."
        )
    elif evidence_positional > evidence_content + 1:
        verdict = "positional"
        explanation = (
            "The entropy=0 architecture is GENUINELY positional. "
            "The model relies on absolute position information, not token identity, "
            "to predict successors. This is Strategy A."
        )
    else:
        verdict = "hybrid"
        explanation = (
            "The evidence is mixed. The model may use both positional and content-based "
            "computations, or the diagnostics are insufficient to cleanly distinguish the strategies."
        )

    print(f"\n  VERDICT: {verdict.upper()}")
    print(f"\n  {explanation}")

    results["verdict"] = verdict
    results["verdict_explanation"] = explanation
    results["evidence_scores"] = {
        "content_based": evidence_content,
        "positional": evidence_positional,
    }

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    main()
