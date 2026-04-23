"""Local Pythia throughput probe.

This is a real-backend probe when the requested model is downloaded. It does
not perform transplantation and should not be used as a paper metric.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import RESULTS_DIR, ensure_dirs, run_manifest, write_json


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    load_seconds = time.perf_counter() - t0

    input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len), device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
            if device == "mps":
                torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.steps):
            _ = model(input_ids)
            if device == "mps":
                torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = args.batch_size * args.seq_len * args.steps
    result = {
        "manifest": run_manifest("pythia_probe", vars(args)),
        "label": "THROUGHPUT_PROBE_ONLY",
        "backend_status": "REAL_BACKEND",
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "tokenizer_class": tokenizer.__class__.__name__,
        "load_seconds": load_seconds,
        "forward_seconds": elapsed,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else None,
    }
    safe_name = args.model.replace("/", "__")
    write_json(RESULTS_DIR / f"pythia_probe_{safe_name}.json", result)
    print(result)


if __name__ == "__main__":
    main()
