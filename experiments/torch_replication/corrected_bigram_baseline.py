"""Corrected bigram baselines for the Shakespeare/NL induction evals (AUDIT.md S2).

analysis/natural_language_induction_v2.py and experiments/shakespeare_portability.py
compute their bigram baseline as bigram_argmax[window[t-1]] for a target at
window[t+1]. The token actually preceding the target is window[t] (the query
token A), so the implemented baseline is off by one and answers the wrong
question. Because the induction-corpus filter selects positions where (A, B)
already occurred earlier in the window, the corpus is enriched for common
bigrams and the corrected baseline is far higher than the reported one.

This script rebuilds both committed eval corpora deterministically (same seeds)
and reports the implemented baseline (reproducing the committed numbers) next
to the corrected one, without needing any model checkpoints.

Usage: python -m experiments.torch_replication.corrected_bigram_baseline
"""

import json

import numpy as np

from .common import RESULTS_DIR, ROOT


def load_text():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data import TextDataset
    tds = TextDataset(data_dir=str(ROOT / "data"), seq_len=256, seed=42)
    return np.asarray(tds.data), tds.vocab_size


def bigram_tables(data, vocab_size):
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(counts, (data[:-1], data[1:]), 1.0)
    char_argmax = int(np.argmax(np.bincount(data, minlength=vocab_size)))
    return np.argmax(counts, axis=1), char_argmax


def build_corpus(data, seq_len, n_samples, seed, attempts_mult):
    rng = np.random.default_rng(seed)
    samples = []
    max_start = len(data) - seq_len - 2
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * attempts_mult:
        attempts += 1
        start = int(rng.integers(0, max_start))
        window = data[start:start + seq_len + 1]
        t = int(rng.integers(seq_len // 2, seq_len - 1))
        A, B = int(window[t]), int(window[t + 1])
        prior = [i for i in range(1, t) if window[i] == A]
        if not prior:
            continue
        p1 = prior[-1]
        if int(window[p1 + 1]) != B or t - p1 < 5:
            continue
        samples.append({"A": A, "B": B, "prev": int(window[t - 1])})
    return samples


def baselines(samples, bigram_argmax, char_argmax):
    B = np.array([s["B"] for s in samples])
    A = np.array([s["A"] for s in samples])
    prev = np.array([s["prev"] for s in samples])
    return {
        "n_samples": len(samples),
        "char_mode": float((char_argmax == B).mean()),
        "bigram_as_implemented (conditions on window[t-1])":
            float((bigram_argmax[prev] == B).mean()),
        "bigram_corrected (conditions on window[t] = A)":
            float((bigram_argmax[A] == B).mean()),
    }


def main():
    data, vocab_size = load_text()
    bigram_argmax, char_argmax = bigram_tables(data, vocab_size)
    out = {
        "nl_induction_v2_corpus (seed 42, seq 200, N=500)": baselines(
            build_corpus(data, 200, 500, 42, 20), bigram_argmax, char_argmax),
        "portability_corpus (seed 9999, seq 200, N=1024)": baselines(
            build_corpus(data, 200, 1024, 9999, 50), bigram_argmax, char_argmax),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "corrected_bigram_baselines.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
