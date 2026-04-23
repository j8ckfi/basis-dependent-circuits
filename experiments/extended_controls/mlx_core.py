"""Optimized MLX utilities for extended-control experiments."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from src.data import InductionDataset, IOIDataset
from src.model import GPT, GPTConfig, create_model

from .common import ROOT


SMALL_CONFIG = GPTConfig(
    n_layers=2,
    n_heads=4,
    d_model=128,
    d_ff=512,
    vocab_size=512,
    ctx_len=64,
    dropout=0.0,
)


def cosine_lr(step: int, total_steps: int, lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def masked_loss(model: GPT, inputs: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    logits = model(inputs)
    return masked_ce_from_logits(logits, targets, mask)


def masked_ce_from_logits(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    bsz, seq_len, vocab = logits.shape
    losses = nn.losses.cross_entropy(
        logits.reshape(bsz * seq_len, vocab),
        targets.reshape(bsz * seq_len),
        reduction="none",
    )
    flat_mask = mask.reshape(bsz * seq_len)
    return (losses * flat_mask).sum() / (flat_mask.sum() + 1e-8)


def logits_and_residuals(model: GPT, idx: mx.array) -> tuple[mx.array, list[mx.array]]:
    _, seq_len = idx.shape
    pos = mx.arange(seq_len)
    x = model.wte(idx) + model.wpe(pos)
    residuals = [x]
    for block in model.blocks:
        x = block(x)
        residuals.append(x)
    x = model.ln_f(x)
    return x @ model.wte.weight.T, residuals


def linear_cka_mx(source: mx.array, target: mx.array) -> mx.array:
    x = source.reshape(-1, source.shape[-1])
    y = target.reshape(-1, target.shape[-1])
    x = x - mx.mean(x, axis=0, keepdims=True)
    y = y - mx.mean(y, axis=0, keepdims=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    numerator = mx.sum(xty * xty)
    denominator = mx.sqrt(mx.sum(xtx * xtx) * mx.sum(yty * yty) + 1e-8)
    return numerator / denominator


def compute_task_accuracy(model: GPT, inputs: mx.array, targets: mx.array, mask: mx.array) -> float:
    logits = model(inputs)
    preds = mx.argmax(logits, axis=-1)
    correct = (preds == targets) * mask
    total = mask.sum()
    mx.eval(correct, total)
    total_value = total.item()
    if total_value == 0:
        return 0.0
    return float((correct.sum() / total).item())


def fixed_eval_set(
    n_sequences: int = 1024,
    batch_size: int = 32,
    data_seed: int = 9999,
    n_bigrams: int = 6,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=n_bigrams, seed=data_seed)
    batches = []
    for _ in range(n_sequences // batch_size):
        inputs, targets, mask = dataset.generate_batch(batch_size)
        mx.eval(inputs, targets, mask)
        batches.append((np.array(inputs), np.array(targets), np.array(mask)))
    return batches


def evaluate_fixed(model: GPT, eval_batches: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, list[float]]:
    per_sequence: list[float] = []
    for inputs_np, targets_np, mask_np in eval_batches:
        inputs = mx.array(inputs_np)
        logits = model(inputs)
        mx.eval(logits)
        preds = np.array(logits).argmax(axis=-1)
        at_mask = mask_np > 0.5
        correct = (preds == targets_np) & at_mask
        for row in range(inputs_np.shape[0]):
            denom = int(at_mask[row].sum())
            per_sequence.append(float(correct[row].sum() / denom) if denom else 0.0)
    return float(np.mean(per_sequence)), per_sequence


def load_seed_checkpoint(seed: int) -> Path:
    base = ROOT / "checkpoints" / "induction_content_match" / f"induction_seed{seed}"
    if seed == 2:
        for name in ("step_030000", "step_030000_ds43", "step_020000"):
            candidate = base / name / "model.safetensors"
            if candidate.exists():
                return candidate
    candidate = base / "step_020000" / "model.safetensors"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def load_small_model(path: Path | str, config: GPTConfig = SMALL_CONFIG) -> GPT:
    model = GPT(config)
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model


def clone_model(model: GPT) -> GPT:
    flat = [(k, mx.array(np.array(v))) for k, v in nn.utils.tree_flatten(model.parameters())]
    cloned = GPT(model.config)
    cloned.load_weights(flat)
    mx.eval(cloned.parameters())
    return cloned


def clone_optimizer(optimizer: optim.Optimizer) -> optim.Optimizer:
    cloned = optim.AdamW(
        learning_rate=float(np.array(optimizer.state["learning_rate"])),
        weight_decay=getattr(optimizer, "weight_decay", 0.0),
    )
    state_flat = [(k, mx.array(np.array(v))) for k, v in nn.utils.tree_flatten(optimizer.state)]
    cloned.state = nn.utils.tree_unflatten(state_flat)
    return cloned


def get_head_weights(model: GPT, layer: int, head: int) -> dict[str, np.ndarray]:
    d_model = model.config.d_model
    d_head = model.config.d_head
    start = head * d_head
    end = (head + 1) * d_head
    qkv = np.array(model.blocks[layer].attn.qkv_proj.weight)
    out = np.array(model.blocks[layer].attn.out_proj.weight)
    return {
        "q": qkv[start:end, :].copy(),
        "k": qkv[d_model + start : d_model + end, :].copy(),
        "v": qkv[2 * d_model + start : 2 * d_model + end, :].copy(),
        "out": out[:, start:end].copy(),
    }


def transplant_head_inplace(model: GPT, donor: dict[str, np.ndarray], layer: int, head: int) -> None:
    d_model = model.config.d_model
    d_head = model.config.d_head
    start = head * d_head
    end = (head + 1) * d_head
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    qkv_key = f"blocks.{layer}.attn.qkv_proj.weight"
    out_key = f"blocks.{layer}.attn.out_proj.weight"
    qkv = np.array(flat[qkv_key])
    out = np.array(flat[out_key])
    qkv[start:end, :] = donor["q"]
    qkv[d_model + start : d_model + end, :] = donor["k"]
    qkv[2 * d_model + start : 2 * d_model + end, :] = donor["v"]
    out[:, start:end] = donor["out"]
    flat[qkv_key] = mx.array(qkv)
    flat[out_key] = mx.array(out)
    model.load_weights(list(flat.items()))
    mx.eval(model.parameters())


def transplanted_model(recipient: GPT, donor: dict[str, np.ndarray], layer: int, head: int) -> GPT:
    model = clone_model(recipient)
    transplant_head_inplace(model, donor, layer, head)
    return model


def zero_head_weights(config: GPTConfig, dtype=np.float32) -> dict[str, np.ndarray]:
    return {
        "q": np.zeros((config.d_head, config.d_model), dtype=dtype),
        "k": np.zeros((config.d_head, config.d_model), dtype=dtype),
        "v": np.zeros((config.d_head, config.d_model), dtype=dtype),
        "out": np.zeros((config.d_model, config.d_head), dtype=dtype),
    }


def head_freeze_mask(model: GPT, layer: int, head: int) -> dict[str, Any]:
    d_model = model.config.d_model
    d_head = model.config.d_head
    start = head * d_head
    end = (head + 1) * d_head
    masks: list[tuple[str, mx.array]] = []
    for key, value in nn.utils.tree_flatten(model.parameters()):
        mask = np.ones_like(np.array(value), dtype=np.float32)
        if key == f"blocks.{layer}.attn.qkv_proj.weight":
            mask[start:end, :] = 0.0
            mask[d_model + start : d_model + end, :] = 0.0
            mask[2 * d_model + start : 2 * d_model + end, :] = 0.0
        elif key == f"blocks.{layer}.attn.out_proj.weight":
            mask[:, start:end] = 0.0
        masks.append((key, mx.array(mask)))
    return nn.utils.tree_unflatten(masks)


def apply_grad_mask(grads: dict[str, Any], mask: dict[str, Any]) -> dict[str, Any]:
    return mlx.utils.tree_map(lambda g, m: g * m, grads, mask)


def train_induction_steps(
    model: GPT,
    optimizer: optim.Optimizer,
    steps: int,
    total_steps: int,
    batch_size: int,
    data_seed: int,
    learning_rate: float,
    warmup_steps: int,
    grad_clip: float,
    start_step: int = 0,
    grad_mask: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=data_seed)
    loss_grad = nn.value_and_grad(model, masked_loss)
    t0 = time.perf_counter()
    last = {"loss": None, "accuracy": None}
    for local_step, (inputs, targets, mask) in enumerate(dataset.iter_batches(batch_size, steps)):
        global_step = start_step + local_step
        optimizer.learning_rate = cosine_lr(global_step, total_steps, learning_rate, warmup_steps)
        loss, grads = loss_grad(model, inputs, targets, mask)
        if grad_mask is not None:
            grads = apply_grad_mask(grads, grad_mask)
        if grad_clip > 0:
            grads = mlx.utils.tree_map(lambda g: mx.clip(g, -grad_clip, grad_clip), grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if local_step == steps - 1:
            last["loss"] = float(loss.item())
            last["accuracy"] = compute_task_accuracy(model, inputs, targets, mask)
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": float(steps / elapsed) if elapsed > 0 else None,
        "final_loss": last["loss"],
        "final_batch_accuracy": last["accuracy"],
    }


def collect_pre_l0_ln(model: GPT, inputs: mx.array) -> np.ndarray:
    residual = model.get_residual_stream(inputs)[0]
    arr = model.blocks[0].ln1(residual)
    mx.eval(arr)
    np_arr = np.array(arr)
    return np_arr.reshape(-1, np_arr.shape[-1])


def collect_post_l0(model: GPT, inputs: mx.array) -> np.ndarray:
    residual = model.get_residual_stream(inputs)[1]
    mx.eval(residual)
    np_arr = np.array(residual)
    return np_arr.reshape(-1, np_arr.shape[-1])


def procrustes_orthogonal(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(source.T @ target, full_matrices=True)
    return (u @ vt).astype(np.float32)


def rotate_head_two_interface(weights: dict[str, np.ndarray], r_in: np.ndarray, r_out: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q": (weights["q"] @ r_in).astype(np.float32),
        "k": (weights["k"] @ r_in).astype(np.float32),
        "v": (weights["v"] @ r_in).astype(np.float32),
        "out": (r_out.T @ weights["out"]).astype(np.float32),
    }


def fit_head_procrustes(donor_model: GPT, recipient_model: GPT, align_seed: int = 1234) -> tuple[np.ndarray, np.ndarray]:
    dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=align_seed)
    inputs, _, _ = dataset.generate_batch(256)
    mx.eval(inputs)
    r_in = procrustes_orthogonal(collect_pre_l0_ln(donor_model, inputs), collect_pre_l0_ln(recipient_model, inputs))
    r_out = procrustes_orthogonal(collect_post_l0(donor_model, inputs), collect_post_l0(recipient_model, inputs))
    return r_in, r_out


def load_critical_heads() -> dict[int, tuple[int, int]]:
    path = ROOT / "analysis" / "content_matching_20_seeds_results_v2.json"
    if not path.exists():
        return {0: (0, 3), 1: (0, 1)}
    import json

    with path.open() as f:
        data = json.load(f)
    result = {}
    for row in data["per_seed"]:
        if row.get("converged", True):
            result[int(row["seed"])] = (int(row["critical_head_layer"]), int(row["critical_head_index"]))
    return result


class PostL0LinearAdapter(nn.Module):
    def __init__(self, base: GPT):
        super().__init__()
        self.base = base
        self.adapter = nn.Linear(base.config.d_model, base.config.d_model, bias=False)

    def __call__(self, idx: mx.array) -> mx.array:
        _, seq_len = idx.shape
        pos = mx.arange(seq_len)
        x = self.base.wte(idx) + self.base.wpe(pos)
        x = self.base.blocks[0](x)
        x = x + self.adapter(x)
        x = self.base.blocks[1](x)
        x = self.base.ln_f(x)
        return x @ self.base.wte.weight.T


class ResidualLinearAdapter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.linear(x)


class ResidualMLPAdapter(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_mult * d_model, bias=False)
        self.down = nn.Linear(hidden_mult * d_model, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.down(nn.gelu(self.up(x)))


class StitchedAdapterModel(nn.Module):
    """Frozen transformer with optional residual-stream adapters."""

    def __init__(self, base: GPT, condition: str):
        super().__init__()
        self.base = base
        self.condition = condition
        d_model = base.config.d_model

        if condition == "none":
            self.pre_embed = None
            self.post_l0 = None
            self.post_l1 = None
            self.pre_unembed = None
        elif condition == "post_l0_linear":
            self.pre_embed = None
            self.post_l0 = ResidualLinearAdapter(d_model)
            self.post_l1 = None
            self.pre_unembed = None
        elif condition == "all4_linear":
            self.pre_embed = ResidualLinearAdapter(d_model)
            self.post_l0 = ResidualLinearAdapter(d_model)
            self.post_l1 = ResidualLinearAdapter(d_model)
            self.pre_unembed = ResidualLinearAdapter(d_model)
        elif condition == "all4_mlp":
            self.pre_embed = ResidualMLPAdapter(d_model)
            self.post_l0 = ResidualMLPAdapter(d_model)
            self.post_l1 = ResidualMLPAdapter(d_model)
            self.pre_unembed = ResidualMLPAdapter(d_model)
        else:
            raise ValueError(f"Unknown adapter condition: {condition}")

    def __call__(self, idx: mx.array) -> mx.array:
        _, seq_len = idx.shape
        pos = mx.arange(seq_len)
        x = self.base.wte(idx) + self.base.wpe(pos)
        if self.pre_embed is not None:
            x = self.pre_embed(x)
        x = self.base.blocks[0](x)
        if self.post_l0 is not None:
            x = self.post_l0(x)
        x = self.base.blocks[1](x)
        if self.post_l1 is not None:
            x = self.post_l1(x)
        x = self.base.ln_f(x)
        if self.pre_unembed is not None:
            x = self.pre_unembed(x)
        return x @ self.base.wte.weight.T


def adapter_only_grad_mask(model: nn.Module) -> dict[str, Any]:
    masks: list[tuple[str, mx.array]] = []
    for key, value in nn.utils.tree_flatten(model.parameters()):
        scale = 0.0 if key.startswith("base.") else 1.0
        masks.append((key, mx.array(np.full_like(np.array(value), scale, dtype=np.float32))))
    return nn.utils.tree_unflatten(masks)
