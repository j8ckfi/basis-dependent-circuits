"""Run local MLX extended-control experiments.

Use:
    .venv/bin/python -m experiments.extended_controls.run_mlx_experiment smoke
    .venv/bin/python -m experiments.extended_controls.run_mlx_experiment partial-retrain --max-steps 20000
    .venv/bin/python -m experiments.extended_controls.run_mlx_experiment shared-trajectory --fork-points 15000 18000 19500 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from src.data import InductionDataset, IOIDataset
from src.model import GPT, create_model

from .common import RESULTS_DIR, ensure_dirs, load_config, run_manifest, write_json
from .mlx_core import (
    SMALL_CONFIG,
    StitchedAdapterModel,
    adapter_only_grad_mask,
    clone_model,
    clone_optimizer,
    cosine_lr,
    compute_task_accuracy,
    evaluate_fixed,
    fit_head_procrustes,
    fixed_eval_set,
    get_head_weights,
    head_freeze_mask,
    load_critical_heads,
    load_seed_checkpoint,
    load_small_model,
    logits_and_residuals,
    linear_cka_mx,
    masked_ce_from_logits,
    masked_loss,
    rotate_head_two_interface,
    train_induction_steps,
    transplanted_model,
    transplant_head_inplace,
    zero_head_weights,
)


class PairedGPT(nn.Module):
    def __init__(self, seed_a: int, seed_b: int):
        super().__init__()
        self.a = create_model(SMALL_CONFIG, seed=seed_a)
        self.b = create_model(SMALL_CONFIG, seed=seed_b)


def _eval_config(config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    e = config["evaluation"]
    return fixed_eval_set(
        n_sequences=e["n_sequences"],
        batch_size=e["batch_size"],
        data_seed=e["data_seed"],
        n_bigrams=e["n_bigrams"],
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    eval_batches = fixed_eval_set(n_sequences=128, batch_size=32)
    donor = create_model(SMALL_CONFIG, seed=0)
    recipient = create_model(SMALL_CONFIG, seed=1)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    train_stats = train_induction_steps(
        recipient,
        optimizer,
        steps=args.steps,
        total_steps=args.steps,
        batch_size=32,
        data_seed=42,
        learning_rate=1e-3,
        warmup_steps=5,
        grad_clip=1.0,
    )
    donor_head = get_head_weights(donor, 0, 0)
    transplanted = transplanted_model(recipient, donor_head, 0, 0)
    acc, _ = evaluate_fixed(transplanted, eval_batches)
    result = {
        "manifest": run_manifest("smoke", vars(args), {"config": config}),
        "label": "DEBUG_ONLY",
        "train_stats": train_stats,
        "transplanted_accuracy": acc,
    }
    write_json(RESULTS_DIR / "smoke_results.json", result)
    return result


def run_partial_retrain(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    train_cfg = config["training"]
    exp_cfg = config["experiments"]["partial_retrain_rescue"]
    eval_batches = _eval_config(config)
    critical_heads = load_critical_heads()

    rows = []
    for donor_seed, recipient_seed in exp_cfg["pairs"]:
        donor = load_small_model(load_seed_checkpoint(donor_seed))
        recipient = load_small_model(load_seed_checkpoint(recipient_seed))
        donor_head = critical_heads.get(donor_seed, (0, 3))
        recipient_head = critical_heads.get(recipient_seed, (0, 1))
        donor_weights = get_head_weights(donor, *donor_head)

        baseline_acc, baseline_fracs = evaluate_fixed(recipient, eval_batches)
        zeroed = clone_model(recipient)
        transplant_head_inplace(zeroed, zero_head_weights(SMALL_CONFIG), *recipient_head)
        zero_acc, _ = evaluate_fixed(zeroed, eval_batches)

        model = transplanted_model(recipient, donor_weights, *recipient_head)
        freeze_mask = head_freeze_mask(model, *recipient_head)
        optimizer = optim.AdamW(learning_rate=train_cfg["learning_rate"], weight_decay=0.0)

        checkpoints = [k for k in exp_cfg["checkpoints"] if k <= args.max_steps]
        if 0 not in checkpoints:
            checkpoints = [0] + checkpoints
        checkpoints = sorted(set(checkpoints))
        last_step = 0
        pair_rows = []
        for checkpoint in checkpoints:
            delta = checkpoint - last_step
            train_stats = None
            if delta > 0:
                train_stats = train_induction_steps(
                    model,
                    optimizer,
                    steps=delta,
                    total_steps=train_cfg["total_steps"],
                    batch_size=train_cfg["batch_size"],
                    data_seed=train_cfg["data_seed"] + 1000 * recipient_seed + donor_seed,
                    learning_rate=train_cfg["learning_rate"],
                    warmup_steps=train_cfg["warmup_steps"],
                    grad_clip=train_cfg["grad_clip"],
                    start_step=last_step,
                    grad_mask=freeze_mask,
                )
                last_step = checkpoint
            acc, fracs = evaluate_fixed(model, eval_batches)
            pair_rows.append(
                {
                    "k": checkpoint,
                    "accuracy": acc,
                    "baseline_accuracy": baseline_acc,
                    "zero_head_accuracy": zero_acc,
                    "recovery_fraction": acc / baseline_acc if baseline_acc else None,
                    "train_stats": train_stats,
                    "n_eval_sequences": len(fracs),
                }
            )

        target = exp_cfg["recovery_fraction"] * baseline_acc
        k_star = None
        for row in pair_rows:
            if row["accuracy"] >= target:
                k_star = row["k"]
                break
        rows.append(
            {
                "donor_seed": donor_seed,
                "recipient_seed": recipient_seed,
                "donor_head": donor_head,
                "recipient_head": recipient_head,
                "baseline_accuracy": baseline_acc,
                "zero_head_accuracy": zero_acc,
                "target_accuracy": target,
                "k_star": k_star,
                "trajectory": pair_rows,
            }
        )

    result = {
        "manifest": run_manifest("partial_retrain_rescue", vars(args), {"config": config}),
        "backend_status": "REAL_BACKEND",
        "scope": "FULL" if args.max_steps >= train_cfg["total_steps"] else "DRY_RUN_REDUCED_STEPS",
        "results": rows,
    }
    out_name = "partial_retrain_rescue_results.json" if result["scope"] == "FULL" else f"partial_retrain_rescue_results_max{args.max_steps}.json"
    write_json(RESULTS_DIR / out_name, result)
    return result


def _train_prefix(fork_point: int, config: dict[str, Any]) -> tuple[GPT, optim.Optimizer, dict[str, Any] | None]:
    train_cfg = config["training"]
    exp_cfg = config["experiments"]["shared_trajectory"]
    model = create_model(SMALL_CONFIG, seed=exp_cfg["init_seed"])
    optimizer = optim.AdamW(learning_rate=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    if fork_point == 0:
        return model, optimizer, None
    stats = train_induction_steps(
        model,
        optimizer,
        steps=fork_point,
        total_steps=train_cfg["total_steps"],
        batch_size=train_cfg["batch_size"],
        data_seed=train_cfg["data_seed"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        grad_clip=train_cfg["grad_clip"],
    )
    return model, optimizer, stats


def run_shared_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    train_cfg = config["training"]
    exp_cfg = config["experiments"]["shared_trajectory"]
    eval_batches = _eval_config(config)
    target_layer, target_head = config["heads"]["seed1_critical"]
    donor_layer, donor_head = config["heads"]["seed0_critical"]

    fork_points = args.fork_points or exp_cfg["fork_points"]
    rows = []
    for fork_point in fork_points:
        prefix_model, prefix_opt, prefix_stats = _train_prefix(fork_point, config)
        branch_steps = train_cfg["total_steps"] - fork_point
        for pair_idx in range(args.pairs):
            branches = []
            for side in (0, 1):
                model = clone_model(prefix_model)
                optimizer = clone_optimizer(prefix_opt)
                branch_seed = exp_cfg["branch_data_seed_base"] + fork_point * 10 + pair_idx * 2 + side
                train_stats = None
                if branch_steps > 0:
                    train_stats = train_induction_steps(
                        model,
                        optimizer,
                        steps=branch_steps,
                        total_steps=train_cfg["total_steps"],
                        batch_size=train_cfg["batch_size"],
                        data_seed=branch_seed,
                        learning_rate=train_cfg["learning_rate"],
                        warmup_steps=train_cfg["warmup_steps"],
                        grad_clip=train_cfg["grad_clip"],
                        start_step=fork_point,
                    )
                baseline_acc, _ = evaluate_fixed(model, eval_batches)
                branches.append(
                    {
                        "model": model,
                        "baseline_accuracy": baseline_acc,
                        "branch_seed": branch_seed,
                        "train_stats": train_stats,
                    }
                )

            donor_weights = get_head_weights(branches[0]["model"], donor_layer, donor_head)
            unaligned = transplanted_model(branches[1]["model"], donor_weights, target_layer, target_head)
            unaligned_acc, _ = evaluate_fixed(unaligned, eval_batches)

            r_in, r_out = fit_head_procrustes(branches[0]["model"], branches[1]["model"])
            aligned_weights = rotate_head_two_interface(donor_weights, r_in, r_out)
            aligned = transplanted_model(branches[1]["model"], aligned_weights, target_layer, target_head)
            aligned_acc, _ = evaluate_fixed(aligned, eval_batches)

            rows.append(
                {
                    "fork_point": fork_point,
                    "pair_idx": pair_idx,
                    "prefix_stats": prefix_stats,
                    "branch_a_seed": branches[0]["branch_seed"],
                    "branch_b_seed": branches[1]["branch_seed"],
                    "branch_a_baseline_accuracy": branches[0]["baseline_accuracy"],
                    "branch_b_baseline_accuracy": branches[1]["baseline_accuracy"],
                    "unaligned_transplant_accuracy": unaligned_acc,
                    "procrustes_transplant_accuracy": aligned_acc,
                    "branch_a_train_stats": branches[0]["train_stats"],
                    "branch_b_train_stats": branches[1]["train_stats"],
                }
            )

    full_scope = sorted(fork_points) == sorted(exp_cfg["fork_points"]) and args.pairs == exp_cfg["pairs"]
    result = {
        "manifest": run_manifest("shared_trajectory", vars(args), {"config": config}),
        "backend_status": "REAL_BACKEND",
        "scope": "FULL" if full_scope else "DRY_RUN_REDUCED_FORKS_OR_PAIRS",
        "results": rows,
    }
    suffix = "_".join(str(x) for x in fork_points)
    write_json(RESULTS_DIR / f"shared_trajectory_results_{suffix}.json", result)
    if full_scope:
        write_json(RESULTS_DIR / "shared_trajectory_results.json", result)
    return result


def run_stitching_adapters(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    exp_cfg = config["experiments"]["stitching_adapters"]
    eval_batches = _eval_config(config)
    recipient = load_small_model(load_seed_checkpoint(args.recipient_seed))
    donor = load_small_model(load_seed_checkpoint(args.donor_seed))
    donor_head = tuple(config["heads"]["seed0_critical"])
    recipient_head = tuple(config["heads"]["seed1_critical"])
    donor_weights = get_head_weights(donor, *donor_head)
    transplanted = transplanted_model(recipient, donor_weights, *recipient_head)

    def loss_fn(model: StitchedAdapterModel, inputs: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
        logits = model(inputs)
        bsz, seq_len, vocab = logits.shape
        losses = nn.losses.cross_entropy(logits.reshape(bsz * seq_len, vocab), targets.reshape(bsz * seq_len), reduction="none")
        flat_mask = mask.reshape(bsz * seq_len)
        return (losses * flat_mask).sum() / (flat_mask.sum() + 1e-8)

    baseline_acc, _ = evaluate_fixed(recipient, eval_batches)
    transplant_acc, _ = evaluate_fixed(transplanted, eval_batches)

    condition_rows = []
    conditions = args.conditions or exp_cfg["conditions"]
    for condition in conditions:
        adapter_model = StitchedAdapterModel(clone_model(transplanted), condition)
        train_stats = None
        if condition != "none" and args.steps > 0:
            optimizer = optim.AdamW(learning_rate=exp_cfg["learning_rate"], weight_decay=0.0)
            dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=7777)
            grad_mask = adapter_only_grad_mask(adapter_model)
            loss_grad = nn.value_and_grad(adapter_model, loss_fn)
            import time

            t0 = time.perf_counter()
            for inputs, targets, mask in dataset.iter_batches(exp_cfg["batch_size"], args.steps):
                loss, grads = loss_grad(adapter_model, inputs, targets, mask)
                grads = mlx.utils.tree_map(lambda g, m: g * m, grads, grad_mask)
                optimizer.update(adapter_model, grads)
                mx.eval(adapter_model.parameters(), optimizer.state)
            elapsed = time.perf_counter() - t0
            train_stats = {
                "steps": args.steps,
                "elapsed_seconds": elapsed,
                "steps_per_second": args.steps / elapsed if elapsed > 0 else None,
                "final_loss": float(loss.item()),
            }
        adapter_acc, adapter_fracs = evaluate_fixed(adapter_model, eval_batches)
        condition_rows.append(
            {
                "condition": condition,
                "adapter_accuracy": adapter_acc,
                "n_eval_sequences": len(adapter_fracs),
                "train_stats": train_stats,
            }
        )

    result = {
        "manifest": run_manifest("stitching_adapters", vars(args), {"config": config}),
        "backend_status": "REAL_BACKEND",
        "baseline_accuracy": baseline_acc,
        "transplant_accuracy": transplant_acc,
        "steps": args.steps,
        "conditions": condition_rows,
    }
    write_json(RESULTS_DIR / "stitching_adapters_results.json", result)
    return result


def _mean_residual_cka(model_a: GPT, model_b: GPT, eval_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]], n_batches: int = 4) -> dict[str, float]:
    l0_values = []
    l1_values = []
    for inputs_np, _targets_np, _mask_np in eval_batches[:n_batches]:
        inputs = mx.array(inputs_np)
        _, residuals_a = logits_and_residuals(model_a, inputs)
        _, residuals_b = logits_and_residuals(model_b, inputs)
        cka_l0 = linear_cka_mx(residuals_a[1], residuals_b[1])
        cka_l1 = linear_cka_mx(residuals_a[2], residuals_b[2])
        mx.eval(cka_l0, cka_l1)
        l0_values.append(float(cka_l0.item()))
        l1_values.append(float(cka_l1.item()))
    return {
        "post_l0": float(np.mean(l0_values)),
        "post_l1": float(np.mean(l1_values)),
    }


def run_shared_basis_training(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    train_cfg = config["training"]
    exp_cfg = config["experiments"]["shared_basis_training"]
    eval_batches = _eval_config(config)
    lambdas = args.lambdas if args.lambdas is not None else exp_cfg["lambdas"]
    donor_head = tuple(config["heads"]["seed0_critical"])
    recipient_head = tuple(config["heads"]["seed1_critical"])

    rows = []
    for lambda_value in lambdas:
        for pair_idx in range(args.pairs_per_lambda):
            seed_a = args.seed_a + 2 * pair_idx
            seed_b = args.seed_b + 2 * pair_idx
            pair = PairedGPT(seed_a=seed_a, seed_b=seed_b)
            optimizer = optim.AdamW(learning_rate=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
            dataset = InductionDataset(vocab_size=512, seq_len=64, n_bigrams=6, seed=train_cfg["data_seed"] + 171 * pair_idx)

            def loss_fn(model_pair: PairedGPT, inputs: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
                logits_a, residuals_a = logits_and_residuals(model_pair.a, inputs)
                logits_b, residuals_b = logits_and_residuals(model_pair.b, inputs)
                task_loss = 0.5 * (
                    masked_ce_from_logits(logits_a, targets, mask)
                    + masked_ce_from_logits(logits_b, targets, mask)
                )
                cka_l0 = linear_cka_mx(residuals_a[1], residuals_b[1])
                cka_l1 = linear_cka_mx(residuals_a[2], residuals_b[2])
                alignment_loss = 1.0 - 0.5 * (cka_l0 + cka_l1)
                return task_loss + lambda_value * alignment_loss

            loss_grad = nn.value_and_grad(pair, loss_fn)
            import time

            t0 = time.perf_counter()
            last_loss = None
            last_acc_a = None
            last_acc_b = None
            for local_step, (inputs, targets, mask) in enumerate(dataset.iter_batches(train_cfg["batch_size"], args.steps)):
                optimizer.learning_rate = cosine_lr(
                    local_step,
                    train_cfg["total_steps"],
                    train_cfg["learning_rate"],
                    train_cfg["warmup_steps"],
                )
                loss, grads = loss_grad(pair, inputs, targets, mask)
                if train_cfg["grad_clip"] > 0:
                    grads = mlx.utils.tree_map(lambda g: mx.clip(g, -train_cfg["grad_clip"], train_cfg["grad_clip"]), grads)
                optimizer.update(pair, grads)
                mx.eval(pair.parameters(), optimizer.state)
                if local_step == args.steps - 1:
                    last_loss = float(loss.item())
                    last_acc_a = compute_task_accuracy(pair.a, inputs, targets, mask)
                    last_acc_b = compute_task_accuracy(pair.b, inputs, targets, mask)
            elapsed = time.perf_counter() - t0

            baseline_a, _ = evaluate_fixed(pair.a, eval_batches)
            baseline_b, _ = evaluate_fixed(pair.b, eval_batches)
            cka = _mean_residual_cka(pair.a, pair.b, eval_batches)
            donor_weights = get_head_weights(pair.a, *donor_head)
            unaligned = transplanted_model(pair.b, donor_weights, *recipient_head)
            unaligned_acc, _ = evaluate_fixed(unaligned, eval_batches)
            r_in, r_out = fit_head_procrustes(pair.a, pair.b)
            procrustes_weights = rotate_head_two_interface(donor_weights, r_in, r_out)
            aligned = transplanted_model(pair.b, procrustes_weights, *recipient_head)
            procrustes_acc, _ = evaluate_fixed(aligned, eval_batches)

            rows.append(
                {
                    "lambda": lambda_value,
                    "pair_idx": pair_idx,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "steps": args.steps,
                    "train_stats": {
                        "elapsed_seconds": elapsed,
                        "steps_per_second": args.steps / elapsed if elapsed > 0 else None,
                        "final_loss": last_loss,
                        "final_batch_accuracy_a": last_acc_a,
                        "final_batch_accuracy_b": last_acc_b,
                    },
                    "baseline_accuracy_a": baseline_a,
                    "baseline_accuracy_b": baseline_b,
                    "cka": cka,
                    "donor_head": donor_head,
                    "recipient_head": recipient_head,
                    "unaligned_transplant_accuracy": unaligned_acc,
                    "procrustes_transplant_accuracy": procrustes_acc,
                }
            )

    full_scope = args.steps >= train_cfg["total_steps"] and list(lambdas) == exp_cfg["lambdas"] and args.pairs_per_lambda == exp_cfg["pairs_per_lambda"]
    result = {
        "manifest": run_manifest("shared_basis_training", vars(args), {"config": config}),
        "backend_status": "REAL_BACKEND",
        "scope": "FULL" if full_scope else "DRY_RUN_REDUCED_STEPS_OR_LAMBDAS",
        "results": rows,
    }
    out_name = "shared_basis_training_results.json" if full_scope else f"shared_basis_training_results_steps{args.steps}.json"
    write_json(RESULTS_DIR / out_name, result)
    return result


def run_cross_task(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    train_cfg = config["training"]
    eval_batches = _eval_config(config)
    recipient = load_small_model(load_seed_checkpoint(args.recipient_seed))
    donor = create_model(SMALL_CONFIG, seed=args.donor_seed)
    optimizer = optim.AdamW(learning_rate=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])

    # Train an IOI donor with the same architecture. This is a real model, not a placeholder,
    # but it is not an induction-specialized donor.
    dataset = IOIDataset(seed=4242)
    loss_grad = nn.value_and_grad(donor, masked_loss)
    for inputs, targets, mask in dataset.iter_batches(train_cfg["batch_size"], args.steps):
        loss, grads = loss_grad(donor, inputs, targets, mask)
        optimizer.update(donor, grads)
        mx.eval(donor.parameters(), optimizer.state)

    recipient_head = tuple(config["heads"]["seed1_critical"])
    donor_weights = get_head_weights(donor, 0, args.donor_head)
    crossed = transplanted_model(recipient, donor_weights, *recipient_head)
    baseline_acc, _ = evaluate_fixed(recipient, eval_batches)
    crossed_acc, crossed_fracs = evaluate_fixed(crossed, eval_batches)
    result = {
        "manifest": run_manifest("cross_task_transplant", vars(args), {"config": config}),
        "backend_status": "REAL_BACKEND",
        "baseline_accuracy": baseline_acc,
        "cross_task_transplant_accuracy": crossed_acc,
        "n_eval_sequences": len(crossed_fracs),
    }
    write_json(RESULTS_DIR / "cross_task_transplant_results.json", result)
    return result


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke")
    p.add_argument("--steps", type=int, default=20)
    p.set_defaults(func=run_smoke)

    p = sub.add_parser("partial-retrain")
    p.add_argument("--max-steps", type=int, default=20000)
    p.set_defaults(func=run_partial_retrain)

    p = sub.add_parser("shared-trajectory")
    p.add_argument("--fork-points", type=int, nargs="*")
    p.add_argument("--pairs", type=int, default=3)
    p.set_defaults(func=run_shared_trajectory)

    p = sub.add_parser("stitching-adapters")
    p.add_argument("--donor-seed", type=int, default=0)
    p.add_argument("--recipient-seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--conditions", nargs="*")
    p.set_defaults(func=run_stitching_adapters)

    p = sub.add_parser("shared-basis-training")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lambdas", type=float, nargs="*")
    p.add_argument("--pairs-per-lambda", type=int, default=1)
    p.add_argument("--seed-a", type=int, default=0)
    p.add_argument("--seed-b", type=int, default=1)
    p.set_defaults(func=run_shared_basis_training)

    p = sub.add_parser("cross-task")
    p.add_argument("--donor-seed", type=int, default=777)
    p.add_argument("--recipient-seed", type=int, default=1)
    p.add_argument("--donor-head", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.set_defaults(func=run_cross_task)

    args = parser.parse_args()
    result = args.func(args)
    print(f"Wrote result for {args.command}: {result.get('backend_status', result.get('label'))}")


if __name__ == "__main__":
    main()
