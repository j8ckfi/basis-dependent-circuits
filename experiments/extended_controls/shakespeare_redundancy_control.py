"""Redundancy-controlled 19M Shakespeare transplant.

The original 19M Shakespeare result showed that slot-matched L0H7 transplants
drop little while function-matched transplants into each recipient's critical
slot fail. This script addresses the robustness concern directly: first remove
the other Layer-0 heads so the recipient's own critical slot is load-bearing,
then transplant seed0's L0H7 into that critical slot.

This is a real-backend MLX evaluation over the existing Shakespeare
checkpoints. It does not train new models.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shakespeare_portability import (  # noqa: E402
    CHECKPOINTS,
    DONOR_HEAD,
    EVAL_SEED,
    N_EVAL_SAMPLES,
    SEQ_LEN,
    SHAKES_CONFIG,
    TextDataset,
    build_induction_corpus,
    collect_post_l0,
    collect_pre_l0_ln,
    compute_corpus_baselines,
    evaluate_induction_advantage,
    get_head_weights,
    load_model,
    procrustes_orthogonal,
    rotate_head_weights_two_interface,
    transplant_head_into_model,
)
from experiments.extended_controls.common import RESULTS_DIR, ensure_dirs, run_manifest, write_json  # noqa: E402


RESULT_PATH = RESULTS_DIR / "shakespeare_redundancy_control_results.json"
CSV_PATH = RESULTS_DIR / "shakespeare_redundancy_control.csv"
SOURCE_RESULT_PATH = Path(__file__).resolve().parents[2] / "experiments" / "shakespeare_portability_results.json"
CONTROL_SEARCH_SAMPLES = 256


def parse_head(label: str) -> tuple[int, int]:
    label = label.strip()
    if not label.startswith("L") or "H" not in label:
        raise ValueError(f"Bad head label: {label}")
    layer_s, head_s = label[1:].split("H", 1)
    return int(layer_s), int(head_s)


def load_existing_critical_heads() -> dict[int, tuple[int, int]]:
    with SOURCE_RESULT_PATH.open() as f:
        data = json.load(f)
    return {int(seed): parse_head(label) for seed, label in data["per_seed_critical_heads"].items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    summary: dict[str, Any] = {"n_recipients": len(rows)}
    for key in (
        "normal_baseline_advantage",
        "controlled_baseline_advantage",
        "controlled_zero_advantage",
        "controlled_transplant_advantage",
        "controlled_procrustes_advantage",
        "controlled_slot_load_bearing_drop",
        "controlled_transplant_drop",
        "controlled_procrustes_drop",
        "aggressive_critical_only_baseline_advantage",
        "aggressive_critical_only_transplant_drop",
        "normal_function_matched_drop",
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
        summary["all_controlled_slots_load_bearing"] = all(
            row["controlled_slot_load_bearing_drop"] > 0.25 for row in rows
        )
        summary["all_redundancy_controlled_transplants_fail"] = all(
            row["controlled_transplant_drop"] > 0.25 for row in rows
        )
        summary["mean_transplant_drop_fraction_of_controlled_baseline"] = float(
            np.mean([
                row["controlled_transplant_drop"] / row["controlled_baseline_advantage"]
                for row in rows
                if row["controlled_baseline_advantage"] > 0
            ])
        )
        summary["mean_ablated_redundant_heads"] = float(np.mean([row["n_control_ablated_heads"] for row in rows]))
    return summary


def select_redundancy_control(
    model,
    samples: list[dict[str, Any]],
    baselines: dict[str, Any],
    critical: tuple[int, int],
    *,
    min_advantage: float = 0.50,
    min_load_bearing_drop: float = 0.25,
) -> dict[str, Any]:
    """Find the largest other-L0-head ablation set that preserves competence.

    We want to remove as much L0 redundancy as possible without creating a
    degenerate host. The selected set must keep induction advantage above
    ``min_advantage`` and must leave the critical slot load-bearing.
    """
    candidates = [(0, h) for h in range(SHAKES_CONFIG.n_heads) if h != critical[1]]
    best = None
    evaluated = 0
    for size in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            ablated = list(subset)
            ev = evaluate_induction_advantage(model, samples, baselines, ablate_heads=ablated)
            zero_ev = evaluate_induction_advantage(
                model,
                samples,
                baselines,
                ablate_heads=ablated + [critical],
            )
            evaluated += 1
            load_drop = ev["induction_advantage"] - zero_ev["induction_advantage"]
            feasible = ev["induction_advantage"] >= min_advantage and load_drop >= min_load_bearing_drop
            item = {
                "ablated_heads": ablated,
                "baseline_eval": ev,
                "zero_eval": zero_ev,
                "load_bearing_drop": load_drop,
                "feasible": feasible,
            }
            if feasible:
                if best is None:
                    best = item
                else:
                    current_key = (
                        len(item["ablated_heads"]),
                        item["load_bearing_drop"],
                        item["baseline_eval"]["induction_advantage"],
                    )
                    best_key = (
                        len(best["ablated_heads"]),
                        best["load_bearing_drop"],
                        best["baseline_eval"]["induction_advantage"],
                    )
                    if current_key > best_key:
                        best = item

    if best is None:
        ev = evaluate_induction_advantage(model, samples, baselines, ablate_heads=[])
        zero_ev = evaluate_induction_advantage(model, samples, baselines, ablate_heads=[critical])
        best = {
            "ablated_heads": [],
            "baseline_eval": ev,
            "zero_eval": zero_ev,
            "load_bearing_drop": ev["induction_advantage"] - zero_ev["induction_advantage"],
            "feasible": False,
        }
    best["evaluated_subsets"] = evaluated
    best["selection_thresholds"] = {
        "min_advantage": min_advantage,
        "min_load_bearing_drop": min_load_bearing_drop,
    }
    return best


def main() -> None:
    ensure_dirs()
    started = time.perf_counter()

    tds = TextDataset(data_dir=str(Path(__file__).resolve().parents[2] / "data"), seq_len=256, seed=42)
    rng = np.random.default_rng(EVAL_SEED)
    samples = build_induction_corpus(
        tds.data,
        vocab_size=tds.vocab_size,
        seq_len=SEQ_LEN,
        n_samples=N_EVAL_SAMPLES,
        rng=rng,
    )
    baselines = compute_corpus_baselines(tds.data, tds.vocab_size)
    critical_heads = load_existing_critical_heads()

    available_seeds = [seed for seed, path in CHECKPOINTS.items() if path.exists()]
    donor_seed = 0
    if donor_seed not in available_seeds:
        raise FileNotFoundError(CHECKPOINTS[donor_seed])

    models = {seed: load_model(CHECKPOINTS[seed]) for seed in available_seeds}
    donor_model = models[donor_seed]
    donor_weights = get_head_weights(donor_model, *DONOR_HEAD)

    n_align = min(128, len(samples))
    align_inputs_np = np.array([s["window"] for s in samples[:n_align]], dtype=np.int32)
    align_inputs = mx.array(align_inputs_np)
    mx.eval(align_inputs)
    donor_pre = collect_pre_l0_ln(donor_model, align_inputs)
    donor_post = collect_post_l0(donor_model, align_inputs)

    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for recipient_seed in available_seeds:
        if recipient_seed == donor_seed:
            continue
        recipient_model = models[recipient_seed]
        critical = critical_heads[recipient_seed]
        if critical[0] != 0:
            rows.append(
                {
                    "recipient_seed": recipient_seed,
                    "status": "UNAVAILABLE_CRITICAL_HEAD_NOT_IN_L0",
                    "critical_head": f"L{critical[0]}H{critical[1]}",
                }
            )
            continue

        normal_baseline = evaluate_induction_advantage(recipient_model, samples, baselines)
        all_l0_heads = [(0, h) for h in range(SHAKES_CONFIG.n_heads)]
        search_samples = samples[: min(CONTROL_SEARCH_SAMPLES, len(samples))]
        control = select_redundancy_control(recipient_model, search_samples, baselines, critical)
        control_ablations = control["ablated_heads"]
        controlled_baseline = evaluate_induction_advantage(
            recipient_model, samples, baselines, ablate_heads=control_ablations
        )
        controlled_zero = evaluate_induction_advantage(
            recipient_model, samples, baselines, ablate_heads=control_ablations + [critical]
        )

        aggressive_other_l0_heads = [(0, h) for h in range(SHAKES_CONFIG.n_heads) if h != critical[1]]
        aggressive_critical_only = evaluate_induction_advantage(
            recipient_model, samples, baselines, ablate_heads=aggressive_other_l0_heads
        )
        aggressive_critical_zero = evaluate_induction_advantage(
            recipient_model, samples, baselines, ablate_heads=all_l0_heads
        )

        recipient_pre = collect_pre_l0_ln(recipient_model, align_inputs)
        recipient_post = collect_post_l0(recipient_model, align_inputs)
        r_in = procrustes_orthogonal(donor_pre, recipient_pre)
        r_out = procrustes_orthogonal(donor_post, recipient_post)
        aligned_weights = rotate_head_weights_two_interface(donor_weights, r_in, r_out)

        transplanted = transplant_head_into_model(recipient_model, donor_weights, critical[0], critical[1])
        controlled_transplant = evaluate_induction_advantage(
            transplanted, samples, baselines, ablate_heads=control_ablations
        )
        procrustes = transplant_head_into_model(recipient_model, aligned_weights, critical[0], critical[1])
        controlled_procrustes = evaluate_induction_advantage(
            procrustes, samples, baselines, ablate_heads=control_ablations
        )
        aggressive_transplant = evaluate_induction_advantage(
            transplanted, samples, baselines, ablate_heads=aggressive_other_l0_heads
        )

        controlled_drop = (
            controlled_baseline["induction_advantage"] - controlled_transplant["induction_advantage"]
        )
        procrustes_drop = (
            controlled_baseline["induction_advantage"] - controlled_procrustes["induction_advantage"]
        )
        load_bearing_drop = (
            controlled_baseline["induction_advantage"] - controlled_zero["induction_advantage"]
        )
        aggressive_drop = (
            aggressive_critical_only["induction_advantage"] - aggressive_transplant["induction_advantage"]
        )

        row = {
            "recipient_seed": recipient_seed,
            "status": "REAL_BACKEND",
            "critical_head": f"L{critical[0]}H{critical[1]}",
            "control_ablated_heads": [f"L{l}H{h}" for l, h in control_ablations],
            "n_control_ablated_heads": len(control_ablations),
            "control_evaluated_subsets": control["evaluated_subsets"],
            "control_search_n_samples": len(search_samples),
            "control_selection_feasible": control["feasible"],
            "control_selection_thresholds": control["selection_thresholds"],
            "control_screening_baseline_advantage": control["baseline_eval"]["induction_advantage"],
            "control_screening_slot_load_bearing_drop": control["load_bearing_drop"],
            "normal_baseline_accuracy": normal_baseline["model_accuracy"],
            "normal_baseline_advantage": normal_baseline["induction_advantage"],
            "controlled_baseline_accuracy": controlled_baseline["model_accuracy"],
            "controlled_baseline_advantage": controlled_baseline["induction_advantage"],
            "controlled_zero_accuracy": controlled_zero["model_accuracy"],
            "controlled_zero_advantage": controlled_zero["induction_advantage"],
            "controlled_transplant_accuracy": controlled_transplant["model_accuracy"],
            "controlled_transplant_advantage": controlled_transplant["induction_advantage"],
            "controlled_procrustes_accuracy": controlled_procrustes["model_accuracy"],
            "controlled_procrustes_advantage": controlled_procrustes["induction_advantage"],
            "controlled_slot_load_bearing_drop": load_bearing_drop,
            "controlled_transplant_drop": controlled_drop,
            "controlled_procrustes_drop": procrustes_drop,
            "controlled_transplant_drop_fraction": (
                controlled_drop / controlled_baseline["induction_advantage"]
                if controlled_baseline["induction_advantage"] > 0
                else None
            ),
            "aggressive_critical_only_ablated_heads": [f"L{l}H{h}" for l, h in aggressive_other_l0_heads],
            "aggressive_critical_only_baseline_accuracy": aggressive_critical_only["model_accuracy"],
            "aggressive_critical_only_baseline_advantage": aggressive_critical_only["induction_advantage"],
            "aggressive_critical_only_zero_accuracy": aggressive_critical_zero["model_accuracy"],
            "aggressive_critical_only_zero_advantage": aggressive_critical_zero["induction_advantage"],
            "aggressive_critical_only_transplant_accuracy": aggressive_transplant["model_accuracy"],
            "aggressive_critical_only_transplant_advantage": aggressive_transplant["induction_advantage"],
            "aggressive_critical_only_transplant_drop": aggressive_drop,
            "normal_function_matched_drop": normal_baseline["induction_advantage"]
            - evaluate_induction_advantage(
                transplant_head_into_model(recipient_model, donor_weights, critical[0], critical[1]),
                samples,
                baselines,
            )["induction_advantage"],
        }
        rows.append(row)
        csv_rows.append(
            {
                k: (json.dumps(v) if isinstance(v, list) else v)
                for k, v in row.items()
                if k != "status"
            }
        )

    completed = [row for row in rows if row.get("status") == "REAL_BACKEND"]
    result = {
        "manifest": run_manifest(
            "shakespeare_redundancy_control",
            {
                "source_result_path": str(SOURCE_RESULT_PATH),
                "eval_seed": EVAL_SEED,
                "n_eval_samples": len(samples),
                "control_search_samples": CONTROL_SEARCH_SAMPLES,
                "seq_len": SEQ_LEN,
                "donor_seed": donor_seed,
                "donor_head": DONOR_HEAD,
            },
        ),
        "backend_status": "REAL_BACKEND",
        "scope": "SHAKESPEARE_19M_CRITICAL_SLOT_ONLY_CONTROL",
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "summary": summarize(completed),
    }
    write_json(RESULT_PATH, result)
    write_csv(CSV_PATH, csv_rows)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Wrote {RESULT_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
