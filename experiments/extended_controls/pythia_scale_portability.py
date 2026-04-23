"""Pythia 160M single-head portability sweep.

This is a real-backend scale probe for the public 160M seed family. It
implements a conservative unaligned transplant condition plus zero-head
controls. It does not claim a 1.4B cross-seed result because matching public
1.4B seed checkpoints were unavailable under the configured repo IDs.

Run with Python 3.11 + Torch/Transformers:

    uv run --python /opt/homebrew/bin/python3.11 --with torch --with transformers --with safetensors --with huggingface_hub \
      python -m experiments.extended_controls.pythia_scale_portability
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .common import RESULTS_DIR, ensure_dirs, run_manifest, write_json


MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-seed1",
    "EleutherAI/pythia-160m-seed2",
    "EleutherAI/pythia-160m-seed3",
    "EleutherAI/pythia-160m-seed4",
    "EleutherAI/pythia-160m-seed5",
]


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dtype_for(dev: str):
    # MPS fp16 eager attention produced NaN attention weights in local probes.
    # Use fp32 for this metric script so baseline, scan, and transplant evals
    # share one numerically stable configuration.
    return torch.float32


def build_induction_batch(
    n_sequences: int,
    seq_len: int,
    pos1: int,
    pos2: int,
    vocab_size: int,
    seed: int,
    dev: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    low = 1000
    high = max(low + 1, vocab_size - 1000)
    inputs = rng.integers(low, high, size=(n_sequences, seq_len), dtype=np.int64)
    targets = np.zeros((n_sequences,), dtype=np.int64)
    for i in range(n_sequences):
        a = int(rng.integers(low, high))
        b = int(rng.integers(low, high))
        inputs[i, pos1] = a
        inputs[i, pos1 + 1] = b
        inputs[i, pos2] = a
        targets[i] = b
    return torch.tensor(inputs, device=dev), torch.tensor(targets, device=dev)


def load_model(repo_id: str, dev: str, *, require_attentions: bool = False):
    kwargs: dict[str, Any] = {"dtype": dtype_for(dev)}
    if require_attentions:
        # SDPA does not reliably return attention weights in current
        # Transformers/PyTorch builds. The scan phase needs weights; normal
        # forward-only evaluation keeps the faster default implementation.
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs).to(dev)
    model.eval()
    return model


@torch.no_grad()
def eval_accuracy(model, inputs: torch.Tensor, targets: torch.Tensor, pos2: int, batch_size: int) -> float:
    correct = 0
    total = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch = inputs[start : start + batch_size]
        target = targets[start : start + batch_size]
        logits = model(batch).logits[:, pos2, :]
        preds = logits.argmax(dim=-1)
        correct += int((preds == target).sum().item())
        total += int(target.numel())
        if batch.device.type == "mps":
            torch.mps.synchronize()
    return correct / total if total else 0.0


@torch.no_grad()
def scan_induction_head(model, inputs: torch.Tensor, pos1: int, pos2: int, batch_size: int) -> dict[str, Any]:
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    sums = torch.zeros((n_layers, n_heads), dtype=torch.float64)
    count = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch = inputs[start : start + batch_size]
        outputs = model(batch, output_attentions=True)
        for layer_idx, attn in enumerate(outputs.attentions):
            # Attention paid from second A to the B after first A.
            vals = attn[:, :, pos2, pos1 + 1].detach().to("cpu").to(torch.float64)
            sums[layer_idx] += vals.sum(dim=0)
        count += batch.shape[0]
        if batch.device.type == "mps":
            torch.mps.synchronize()
    scores = sums / max(count, 1)
    flat_idx = int(scores.argmax().item())
    layer = flat_idx // n_heads
    head = flat_idx % n_heads
    return {
        "layer": layer,
        "head": head,
        "score": float(scores[layer, head].item()),
    }


def head_slices(model, head: int) -> tuple[slice, slice, slice, slice]:
    d_head = model.config.hidden_size // model.config.num_attention_heads
    qkv_start = head * 3 * d_head
    q_slice = slice(qkv_start, qkv_start + d_head)
    k_slice = slice(qkv_start + d_head, qkv_start + 2 * d_head)
    v_slice = slice(qkv_start + 2 * d_head, qkv_start + 3 * d_head)
    out_slice = slice(head * d_head, (head + 1) * d_head)
    return q_slice, k_slice, v_slice, out_slice


def extract_head(model, layer: int, head: int) -> dict[str, torch.Tensor]:
    attn = model.gpt_neox.layers[layer].attention
    q_slice, k_slice, v_slice, out_slice = head_slices(model, head)
    return {
        "q_weight": attn.query_key_value.weight[q_slice, :].detach().to("cpu").clone(),
        "k_weight": attn.query_key_value.weight[k_slice, :].detach().to("cpu").clone(),
        "v_weight": attn.query_key_value.weight[v_slice, :].detach().to("cpu").clone(),
        "q_bias": attn.query_key_value.bias[q_slice].detach().to("cpu").clone(),
        "k_bias": attn.query_key_value.bias[k_slice].detach().to("cpu").clone(),
        "v_bias": attn.query_key_value.bias[v_slice].detach().to("cpu").clone(),
        "out_weight": attn.dense.weight[:, out_slice].detach().to("cpu").clone(),
    }


def insert_head(model, layer: int, head: int, weights: dict[str, torch.Tensor]) -> None:
    attn = model.gpt_neox.layers[layer].attention
    q_slice, k_slice, v_slice, out_slice = head_slices(model, head)

    def copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    with torch.no_grad():
        copy_into(attn.query_key_value.weight[q_slice, :], weights["q_weight"])
        copy_into(attn.query_key_value.weight[k_slice, :], weights["k_weight"])
        copy_into(attn.query_key_value.weight[v_slice, :], weights["v_weight"])
        copy_into(attn.query_key_value.bias[q_slice], weights["q_bias"])
        copy_into(attn.query_key_value.bias[k_slice], weights["k_bias"])
        copy_into(attn.query_key_value.bias[v_slice], weights["v_bias"])
        copy_into(attn.dense.weight[:, out_slice], weights["out_weight"])


def zero_head(model, layer: int, head: int) -> None:
    attn = model.gpt_neox.layers[layer].attention
    q_slice, k_slice, v_slice, out_slice = head_slices(model, head)
    with torch.no_grad():
        for s in (q_slice, k_slice, v_slice):
            attn.query_key_value.weight[s, :].zero_()
            attn.query_key_value.bias[s].zero_()
        attn.dense.weight[:, out_slice].zero_()


def main() -> None:
    ensure_dirs()
    dev = device()
    inputs, targets = build_induction_batch(
        n_sequences=1024,
        seq_len=128,
        pos1=20,
        pos2=80,
        vocab_size=50304,
        seed=12345,
        dev=dev,
    )

    started = time.perf_counter()
    model_summaries = {}
    head_weights_by_model = {}
    for repo_id in MODELS:
        model = load_model(repo_id, dev, require_attentions=True)
        baseline = eval_accuracy(model, inputs, targets, pos2=80, batch_size=8)
        head = scan_induction_head(model, inputs[:256], pos1=20, pos2=80, batch_size=4)
        head_weights_by_model[repo_id] = extract_head(model, head["layer"], head["head"])
        model_summaries[repo_id] = {
            "baseline_accuracy": baseline,
            "selected_head": head,
        }
        del model
        if dev == "mps":
            torch.mps.empty_cache()

    pair_rows = []
    for recipient_idx, recipient_id in enumerate(MODELS):
        donor_ids = MODELS[:recipient_idx]
        if not donor_ids:
            continue
        recipient = load_model(recipient_id, dev)
        recipient_head = model_summaries[recipient_id]["selected_head"]
        recipient_original = extract_head(recipient, recipient_head["layer"], recipient_head["head"])

        zero_head(recipient, recipient_head["layer"], recipient_head["head"])
        zero_acc = eval_accuracy(recipient, inputs, targets, pos2=80, batch_size=8)
        insert_head(recipient, recipient_head["layer"], recipient_head["head"], recipient_original)

        for donor_id in donor_ids:
            donor_head = model_summaries[donor_id]["selected_head"]
            insert_head(
                recipient,
                recipient_head["layer"],
                recipient_head["head"],
                head_weights_by_model[donor_id],
            )
            transplant_acc = eval_accuracy(recipient, inputs, targets, pos2=80, batch_size=8)
            insert_head(recipient, recipient_head["layer"], recipient_head["head"], recipient_original)

            pair_rows.append(
                {
                    "donor": donor_id,
                    "recipient": recipient_id,
                    "donor_head": donor_head,
                    "recipient_head": recipient_head,
                    "recipient_baseline_accuracy": model_summaries[recipient_id]["baseline_accuracy"],
                    "zero_head_accuracy": zero_acc,
                    "unaligned_transplant_accuracy": transplant_acc,
                }
            )
        del recipient
        if dev == "mps":
            torch.mps.empty_cache()

    result = {
        "manifest": run_manifest(
            "pythia_scale_portability",
            {
                "models": MODELS,
                "n_sequences": 1024,
                "seq_len": 128,
                "pos1": 20,
                "pos2": 80,
                "batch_size": 8,
                "head_scan_sequences": 256,
                "head_scan_attention_implementation": "eager",
            },
        ),
        "backend_status": "REAL_BACKEND",
        "scope": "PYTHIA_160M_UNALIGNED_ONLY",
        "device": dev,
        "elapsed_seconds": time.perf_counter() - started,
        "model_summaries": model_summaries,
        "pairs": pair_rows,
    }
    write_json(RESULTS_DIR / "pythia_scale_portability_results.json", result)
    print(f"Wrote {RESULTS_DIR / 'pythia_scale_portability_results.json'}")


if __name__ == "__main__":
    main()
