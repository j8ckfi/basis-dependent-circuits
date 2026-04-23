"""Pythia-160M Procrustes and adapter-rescue portability probe.

This extends ``pythia_scale_portability.py`` beyond the unaligned condition.
It uses the same six public Pythia-160M seed-family checkpoints and the same
synthetic induction probe, then evaluates:

1. unaligned selected-head transplant,
2. two-interface Procrustes selected-head transplant,
3. frozen-recipient all-four linear adapter rescue after transplant,
4. zero-head all-four linear adapter controls, one per recipient.

The adapter condition is intentionally narrow: all pretrained model weights are
frozen, and only four zero-initialized residual linear maps are trained.

Run with Python 3.11 + Torch/Transformers:

    uv run --python /opt/homebrew/bin/python3.11 --with torch --with transformers --with safetensors --with huggingface_hub \
      python -m experiments.extended_controls.pythia_adapter_rescue
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .common import RESULTS_DIR, ensure_dirs, run_manifest, write_json
from .pythia_scale_portability import (
    MODELS,
    build_induction_batch,
    device,
    dtype_for,
    eval_accuracy,
    extract_head,
    head_slices,
    insert_head,
    load_model,
    scan_induction_head,
    zero_head,
)


DEFAULT_RESULT_PATH = RESULTS_DIR / "pythia_adapter_rescue_results.json"
DEFAULT_CSV_PATH = RESULTS_DIR / "pythia_adapter_rescue.csv"
HEAD_SOURCE_PATH = RESULTS_DIR / "pythia_scale_portability_results.json"


@dataclass(frozen=True)
class PythiaProbeConfig:
    n_eval_sequences: int = 1024
    n_align_sequences: int = 256
    seq_len: int = 128
    pos1: int = 20
    pos2: int = 80
    vocab_size: int = 50304
    eval_seed: int = 12345
    align_seed: int = 22345
    train_seed: int = 32345


class ResidualLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class AllFourPythiaAdapters(nn.Module):
    """Four zero-initialized residual-stream maps around the selected layer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.post_embed = ResidualLinear(d_model)
        self.pre_target_layer = ResidualLinear(d_model)
        self.post_target_layer = ResidualLinear(d_model)
        self.pre_unembed = ResidualLinear(d_model)
        self._handles: list[Any] = []

    def attach(self, model, target_layer: int) -> None:
        self.remove()

        def embed_hook(_module, _inputs, output):
            return self.post_embed(output)

        def pre_layer_hook(_module, inputs):
            hidden = self.pre_target_layer(inputs[0])
            return (hidden,) + tuple(inputs[1:])

        def post_layer_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (self.post_target_layer(output[0]),) + tuple(output[1:])
            return self.post_target_layer(output)

        def final_ln_hook(_module, _inputs, output):
            return self.pre_unembed(output)

        self._handles = [
            model.gpt_neox.embed_in.register_forward_hook(embed_hook),
            model.gpt_neox.layers[target_layer].register_forward_pre_hook(pre_layer_hook),
            model.gpt_neox.layers[target_layer].register_forward_hook(post_layer_hook),
            model.gpt_neox.final_layer_norm.register_forward_hook(final_ln_hook),
        ]

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def load_head_source() -> dict[str, Any] | None:
    if not HEAD_SOURCE_PATH.exists():
        return None
    with HEAD_SOURCE_PATH.open() as f:
        data = json.load(f)
    if data.get("backend_status") != "REAL_BACKEND":
        return None
    return data


def r2_score(source: np.ndarray, target: np.ndarray, rotation: np.ndarray | None = None) -> float:
    mapped = source if rotation is None else source @ rotation
    denom = float(np.sum(target * target))
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.sum((mapped - target) ** 2)) / denom


def procrustes_orthogonal(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(source.T @ target, full_matrices=True)
    return (u @ vt).astype(np.float32)


@torch.no_grad()
def collect_layer_interfaces(
    model,
    layer: int,
    inputs: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pre_chunks: list[torch.Tensor] = []
    post_chunks: list[torch.Tensor] = []

    def pre_hook(_module, _inputs, output):
        pre_chunks.append(output.detach().cpu().float())

    def post_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        post_chunks.append(hidden.detach().cpu().float())

    h1 = model.gpt_neox.layers[layer].input_layernorm.register_forward_hook(pre_hook)
    h2 = model.gpt_neox.layers[layer].register_forward_hook(post_hook)
    try:
        for start in range(0, inputs.shape[0], batch_size):
            _ = model(inputs[start : start + batch_size])
            if inputs.device.type == "mps":
                torch.mps.synchronize()
    finally:
        h1.remove()
        h2.remove()

    pre = torch.cat(pre_chunks, dim=0).numpy()
    post = torch.cat(post_chunks, dim=0).numpy()
    return pre.reshape(-1, pre.shape[-1]), post.reshape(-1, post.shape[-1])


def rotate_pythia_head_two_interface(
    weights: dict[str, torch.Tensor],
    r_in: np.ndarray,
    r_out: np.ndarray,
) -> dict[str, torch.Tensor]:
    rotated = {
        "q_weight": torch.from_numpy(weights["q_weight"].numpy() @ r_in),
        "k_weight": torch.from_numpy(weights["k_weight"].numpy() @ r_in),
        "v_weight": torch.from_numpy(weights["v_weight"].numpy() @ r_in),
        "q_bias": weights["q_bias"].clone(),
        "k_bias": weights["k_bias"].clone(),
        "v_bias": weights["v_bias"].clone(),
        "out_weight": torch.from_numpy(r_out.T @ weights["out_weight"].numpy()),
    }
    return {k: v.to(dtype=weights[k].dtype if k in weights else torch.float32) for k, v in rotated.items()}


def set_requires_grad(model, value: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(value)


def train_all4_adapter(
    model,
    target_layer: int,
    cfg: PythiaProbeConfig,
    dev: str,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
) -> dict[str, Any]:
    set_requires_grad(model, False)
    adapters = AllFourPythiaAdapters(model.config.hidden_size).to(dev)
    adapters.attach(model, target_layer)
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=learning_rate, weight_decay=0.0)
    t0 = time.perf_counter()
    last_loss = None

    try:
        model.train()
        for step in range(steps):
            inputs, targets = build_induction_batch(
                n_sequences=batch_size,
                seq_len=cfg.seq_len,
                pos1=cfg.pos1,
                pos2=cfg.pos2,
                vocab_size=cfg.vocab_size,
                seed=seed + step,
                dev=dev,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs).logits[:, cfg.pos2, :]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            if dev == "mps":
                torch.mps.synchronize()

        elapsed = time.perf_counter() - t0
        model.eval()
        acc = eval_accuracy(model, eval_inputs, eval_targets, cfg.pos2, batch_size=max(1, batch_size))
    finally:
        adapters.remove()
        model.eval()

    return {
        "adapter_accuracy": acc,
        "train_stats": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / elapsed if elapsed > 0 else None,
            "final_loss": last_loss,
        },
        "adapter_params": sum(p.numel() for p in adapters.parameters()),
    }


def build_or_load_head_summaries(
    dev: str,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    cfg: PythiaProbeConfig,
    batch_size: int,
) -> dict[str, Any]:
    source = load_head_source()
    if source is not None and set(source.get("model_summaries", {}).keys()) >= set(MODELS):
        return source["model_summaries"]

    summaries = {}
    for repo_id in MODELS:
        model = load_model(repo_id, dev, require_attentions=True)
        baseline = eval_accuracy(model, eval_inputs, eval_targets, cfg.pos2, batch_size=batch_size)
        head = scan_induction_head(model, eval_inputs[:256], cfg.pos1, cfg.pos2, batch_size=max(1, batch_size // 2))
        summaries[repo_id] = {
            "baseline_accuracy": baseline,
            "selected_head": head,
        }
        del model
        if dev == "mps":
            torch.mps.empty_cache()
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_pair_rows(rows: list[dict[str, Any]], zero_controls: dict[str, Any]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    summary: dict[str, Any] = {
        "n_pairs": len(rows),
        "n_zero_controls": len(zero_controls),
    }
    for key in (
        "recipient_baseline_accuracy",
        "zero_head_accuracy",
        "unaligned_transplant_accuracy",
        "procrustes_transplant_accuracy",
        "adapter_accuracy",
    ):
        xs = vals(key)
        if xs:
            summary[key] = {
                "mean": float(np.mean(xs)),
                "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                "min": float(np.min(xs)),
                "max": float(np.max(xs)),
            }

    if rows:
        summary["mean_unaligned_minus_baseline"] = float(
            np.mean([row["unaligned_transplant_accuracy"] - row["recipient_baseline_accuracy"] for row in rows])
        )
        summary["mean_procrustes_minus_baseline"] = float(
            np.mean([row["procrustes_transplant_accuracy"] - row["recipient_baseline_accuracy"] for row in rows])
        )
        summary["mean_adapter_minus_baseline"] = float(
            np.mean([row["adapter_accuracy"] - row["recipient_baseline_accuracy"] for row in rows])
        )
        summary["mean_adapter_minus_zero_control"] = float(
            np.mean([row["adapter_accuracy"] - zero_controls[row["recipient"]]["zero_adapter_accuracy"] for row in rows])
        )
        summary["all_unaligned_below_baseline"] = all(
            row["unaligned_transplant_accuracy"] < row["recipient_baseline_accuracy"] for row in rows
        )
        summary["all_procrustes_below_baseline"] = all(
            row["procrustes_transplant_accuracy"] < row["recipient_baseline_accuracy"] for row in rows
        )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    cfg = PythiaProbeConfig(
        n_eval_sequences=args.eval_sequences,
        n_align_sequences=args.align_sequences,
    )
    dev = device()
    eval_inputs, eval_targets = build_induction_batch(
        cfg.n_eval_sequences, cfg.seq_len, cfg.pos1, cfg.pos2, cfg.vocab_size, cfg.eval_seed, dev
    )
    align_inputs, _ = build_induction_batch(
        cfg.n_align_sequences, cfg.seq_len, cfg.pos1, cfg.pos2, cfg.vocab_size, cfg.align_seed, dev
    )

    started = time.perf_counter()
    head_summaries = build_or_load_head_summaries(dev, eval_inputs, eval_targets, cfg, args.eval_batch_size)

    donor_head_weights: dict[str, dict[str, torch.Tensor]] = {}
    donor_interfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for repo_id in MODELS:
        print(f"[pythia] loading donor/interface source {repo_id}", flush=True)
        model = load_model(repo_id, dev)
        head = head_summaries[repo_id]["selected_head"]
        donor_head_weights[repo_id] = extract_head(model, int(head["layer"]), int(head["head"]))
        donor_interfaces[repo_id] = collect_layer_interfaces(
            model, int(head["layer"]), align_inputs, args.eval_batch_size
        )
        del model
        if dev == "mps":
            torch.mps.empty_cache()

    pair_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    zero_controls: dict[str, Any] = {}
    pair_count = 0

    for recipient_idx, recipient_id in enumerate(MODELS):
        donor_ids = MODELS[:recipient_idx]
        if not donor_ids:
            continue
        print(f"[pythia] recipient {recipient_id}: {len(donor_ids)} donor pairs", flush=True)
        recipient = load_model(recipient_id, dev)
        recipient_head = head_summaries[recipient_id]["selected_head"]
        rec_layer = int(recipient_head["layer"])
        rec_head = int(recipient_head["head"])
        recipient_original = extract_head(recipient, rec_layer, rec_head)
        rec_pre, rec_post = collect_layer_interfaces(recipient, rec_layer, align_inputs, args.eval_batch_size)

        zero_head(recipient, rec_layer, rec_head)
        zero_acc = eval_accuracy(recipient, eval_inputs, eval_targets, cfg.pos2, batch_size=args.eval_batch_size)
        print(f"[pythia] recipient {recipient_id}: training zero-head adapter control", flush=True)
        zero_control = train_all4_adapter(
            recipient,
            rec_layer,
            cfg,
            dev,
            steps=args.adapter_steps,
            batch_size=args.train_batch_size,
            learning_rate=args.adapter_lr,
            seed=cfg.train_seed + 100000 * recipient_idx,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
        )
        zero_controls[recipient_id] = {
            "recipient": recipient_id,
            "recipient_head": recipient_head,
            "zero_head_accuracy": zero_acc,
            "zero_adapter_accuracy": zero_control["adapter_accuracy"],
            "train_stats": zero_control["train_stats"],
            "adapter_params": zero_control["adapter_params"],
        }
        insert_head(recipient, rec_layer, rec_head, recipient_original)

        for donor_position, donor_id in enumerate(donor_ids):
            if args.limit_pairs is not None and pair_count >= args.limit_pairs:
                break
            pair_count += 1
            print(f"[pythia] pair {pair_count}: {donor_id} -> {recipient_id}", flush=True)
            donor_head = head_summaries[donor_id]["selected_head"]

            insert_head(recipient, rec_layer, rec_head, donor_head_weights[donor_id])
            unaligned_acc = eval_accuracy(recipient, eval_inputs, eval_targets, cfg.pos2, batch_size=args.eval_batch_size)
            insert_head(recipient, rec_layer, rec_head, recipient_original)

            donor_pre, donor_post = donor_interfaces[donor_id]
            r_in = procrustes_orthogonal(donor_pre, rec_pre)
            r_out = procrustes_orthogonal(donor_post, rec_post)
            aligned_weights = rotate_pythia_head_two_interface(donor_head_weights[donor_id], r_in, r_out)
            insert_head(recipient, rec_layer, rec_head, aligned_weights)
            procrustes_acc = eval_accuracy(recipient, eval_inputs, eval_targets, cfg.pos2, batch_size=args.eval_batch_size)
            insert_head(recipient, rec_layer, rec_head, recipient_original)

            insert_head(recipient, rec_layer, rec_head, donor_head_weights[donor_id])
            adapter = train_all4_adapter(
                recipient,
                rec_layer,
                cfg,
                dev,
                steps=args.adapter_steps,
                batch_size=args.train_batch_size,
                learning_rate=args.adapter_lr,
                seed=cfg.train_seed + 100000 * recipient_idx + 1000 * donor_position,
                eval_inputs=eval_inputs,
                eval_targets=eval_targets,
            )
            insert_head(recipient, rec_layer, rec_head, recipient_original)

            row = {
                "donor": donor_id,
                "recipient": recipient_id,
                "donor_head": donor_head,
                "recipient_head": recipient_head,
                "recipient_baseline_accuracy": float(head_summaries[recipient_id]["baseline_accuracy"]),
                "zero_head_accuracy": zero_acc,
                "unaligned_transplant_accuracy": unaligned_acc,
                "procrustes_transplant_accuracy": procrustes_acc,
                "adapter_accuracy": adapter["adapter_accuracy"],
                "adapter_train_stats": adapter["train_stats"],
                "adapter_params": adapter["adapter_params"],
                "pre_l0_r2_unrotated": r2_score(donor_pre, rec_pre),
                "pre_l0_r2_procrustes": r2_score(donor_pre, rec_pre, r_in),
                "post_layer_r2_unrotated": r2_score(donor_post, rec_post),
                "post_layer_r2_procrustes": r2_score(donor_post, rec_post, r_out),
                "zero_adapter_accuracy_for_recipient": zero_controls[recipient_id]["zero_adapter_accuracy"],
            }
            pair_rows.append(row)
            csv_rows.append(
                {
                    "donor": donor_id,
                    "recipient": recipient_id,
                    "recipient_baseline_accuracy": row["recipient_baseline_accuracy"],
                    "zero_head_accuracy": row["zero_head_accuracy"],
                    "unaligned_transplant_accuracy": row["unaligned_transplant_accuracy"],
                    "procrustes_transplant_accuracy": row["procrustes_transplant_accuracy"],
                    "adapter_accuracy": row["adapter_accuracy"],
                    "zero_adapter_accuracy_for_recipient": row["zero_adapter_accuracy_for_recipient"],
                    "pre_l0_r2_unrotated": row["pre_l0_r2_unrotated"],
                    "pre_l0_r2_procrustes": row["pre_l0_r2_procrustes"],
                    "post_layer_r2_unrotated": row["post_layer_r2_unrotated"],
                    "post_layer_r2_procrustes": row["post_layer_r2_procrustes"],
                }
            )

        del recipient
        if dev == "mps":
            torch.mps.empty_cache()
        if args.limit_pairs is not None and pair_count >= args.limit_pairs:
            break

    result = {
        "manifest": run_manifest(
            "pythia_adapter_rescue",
            vars(args),
            {
                "head_source_path": str(HEAD_SOURCE_PATH),
                "torch_version": torch.__version__,
                "transformers_model_class": "AutoModelForCausalLM",
            },
        ),
        "backend_status": "REAL_BACKEND",
        "scope": "PYTHIA_160M_PROCRUSTES_AND_ADAPTERS",
        "device": dev,
        "dtype": str(dtype_for(dev)),
        "elapsed_seconds": time.perf_counter() - started,
        "probe_config": cfg.__dict__,
        "model_summaries": head_summaries,
        "zero_controls": zero_controls,
        "pairs": pair_rows,
        "summary": summarize_pair_rows(pair_rows, zero_controls),
    }
    write_json(DEFAULT_RESULT_PATH, result)
    write_csv(DEFAULT_CSV_PATH, csv_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-steps", type=int, default=500)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-sequences", type=int, default=1024)
    parser.add_argument("--align-sequences", type=int, default=256)
    parser.add_argument("--limit-pairs", type=int)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Wrote {DEFAULT_RESULT_PATH}")
    print(f"Wrote {DEFAULT_CSV_PATH}")


if __name__ == "__main__":
    main()
