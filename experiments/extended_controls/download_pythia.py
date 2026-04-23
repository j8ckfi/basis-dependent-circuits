"""Download and verify Pythia checkpoints for extended-control experiments.

This script records unavailable checkpoints explicitly. It never substitutes
adjacent models for missing seed variants.

Use:
    HF_HUB_DISABLE_XET=1 hf auth whoami
    .venv/bin/python -m experiments.extended_controls.download_pythia --dry-run
    .venv/bin/python -m experiments.extended_controls.download_pythia
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import RESULTS_DIR, ensure_dirs, load_config, run_manifest, write_json


def run(cmd: list[str], timeout: int | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return proc.returncode, proc.stdout


def hf_repo_available(repo_id: str) -> tuple[bool, str]:
    code, output = run(["hf", "models", "info", repo_id], timeout=60)
    return code == 0, output


def select_include_patterns(info_output: str, configured_patterns: list[str]) -> list[str]:
    metadata_patterns = [
        p
        for p in configured_patterns
        if p not in {"model.safetensors", "*.safetensors.index.json", "pytorch_model.bin"}
    ]
    try:
        info = json.loads(info_output)
        siblings = {s.get("rfilename") for s in info.get("siblings", [])}
    except Exception:
        siblings = set()

    if "model.safetensors" in siblings:
        return metadata_patterns + ["model.safetensors", "*.safetensors.index.json"]
    if "pytorch_model.bin" in siblings:
        return metadata_patterns + ["pytorch_model.bin"]
    return configured_patterns


def download_model(repo_id: str, include_patterns: list[str], dry_run: bool, max_workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    available, info_output = hf_repo_available(repo_id)
    if not available:
        return {
            "repo_id": repo_id,
            "status": "UNAVAILABLE",
            "reason": "hf models info failed",
            "details": info_output[-2000:],
            "elapsed_seconds": time.perf_counter() - started,
        }

    selected_patterns = select_include_patterns(info_output, include_patterns)
    cmd = ["hf", "download", repo_id, "--max-workers", str(max_workers)]
    for pattern in selected_patterns:
        cmd.extend(["--include", pattern])
    if dry_run:
        cmd.append("--dry-run")

    code, output = run(cmd, timeout=None)
    status = "DRY_RUN_OK" if dry_run and code == 0 else "DOWNLOADED" if code == 0 else "FAILED"
    result = {
        "repo_id": repo_id,
        "status": status,
        "command": cmd,
        "include_patterns": selected_patterns,
        "details": output[-4000:],
        "elapsed_seconds": time.perf_counter() - started,
    }
    if code == 0 and not dry_run:
        cache_name = "models--" + repo_id.replace("/", "--")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
        weight_files = []
        if cache_dir.exists():
            for pattern in ("model.safetensors", "pytorch_model.bin"):
                weight_files.extend(str(p) for p in cache_dir.glob(f"snapshots/**/{pattern}"))
        result["weight_files"] = weight_files
        if not weight_files:
            result["status"] = "FAILED"
            result["reason"] = "download finished but no model.safetensors or pytorch_model.bin was present"
    return result


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--only-scale", choices=["160m", "1.4b"], default=None)
    args = parser.parse_args()

    config = load_config()
    pythia_cfg = config["pythia"]
    rows = []
    for model in pythia_cfg["models"]:
        if args.only_scale and model["scale"] != args.only_scale:
            continue
        row = dict(model)
        result = download_model(model["repo_id"], pythia_cfg["include_patterns"], args.dry_run, args.max_workers)
        row.update(result)
        rows.append(row)
        print(f"{row['repo_id']}: {row['status']}")

    manifest = {
        "manifest": run_manifest("pythia_download", vars(args), {"config_path": str(Path("configs/extended_controls.json"))}),
        "backend_policy": "REAL_BACKEND_OR_UNAVAILABLE",
        "dry_run": args.dry_run,
        "models": rows,
    }
    write_json(RESULTS_DIR / "pythia_model_manifest.json", manifest)

    required_failures = [r for r in rows if r.get("required") and r["status"] not in {"DOWNLOADED", "DRY_RUN_OK"}]
    if required_failures:
        raise SystemExit(f"{len(required_failures)} required Pythia checkpoints failed; see results/extended_controls/pythia_model_manifest.json")


if __name__ == "__main__":
    main()
