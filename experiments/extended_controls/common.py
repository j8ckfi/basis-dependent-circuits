"""Shared configuration, manifests, and filesystem helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "extended_controls.json"
RESULTS_DIR = ROOT / "results" / "extended_controls"
FIGURES_DIR = ROOT / "figures" / "extended_controls"
LOGS_DIR = ROOT / "logs" / "extended_controls"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "extended_controls"


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def json_safe(value: Any) -> Any:
    if callable(value):
        return getattr(value, "__name__", str(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(json_safe(data), f, indent=2, sort_keys=True)
    tmp.replace(path)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - best-effort manifest metadata
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def system_metadata() -> dict[str, Any]:
    return {
        "timestamp": timestamp(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "hardware": command_output(["system_profiler", "SPHardwareDataType"]),
        "gpu": command_output(["system_profiler", "SPDisplaysDataType"]),
    }


def run_manifest(name: str, args: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "name": name,
        "args": json_safe(args),
        "system": system_metadata(),
        "repo_root": str(ROOT),
        "real_backend_policy": "REAL_BACKEND_ONLY_FOR_REPORTED_METRICS",
    }
    if extra:
        manifest.update(extra)
    return manifest
