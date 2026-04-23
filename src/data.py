"""
Data pipeline for developmental interpretability experiments.

For seed divergence experiments, we need CONTROLLED data:
- Same data across all seeds (isolate initialization as the variable)
- Tasks where we know what circuits SHOULD form (induction, IOI-like)
- Simple enough that circuits are tractable, complex enough to be interesting

We provide two data modes:
1. Synthetic: algorithmically generated sequences with known structure
2. Natural: tokenized text (OpenWebText subset) for ecological validity
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


@dataclass
class BatchConfig:
    batch_size: int = 64
    seq_len: int = 256


class InductionDataset:
    """Synthetic data designed to REQUIRE content-matching induction heads.

    Generates sequences with repeated bigrams at RANDOM positions:
        [noise] A B [noise] A B [noise] A B [noise] ...

    The key insight: bigram "A B" appears first at a random position, then
    again later at another random position. When the model sees "A" the
    second time, it must MATCH on token identity (not position) to predict "B".

    This forces:
    - A "previous token" head: given current token A, attend to prior A
    - An "induction" head: having found prior A, copy what followed it (B)

    The repeated bigrams appear at random positions, so a fixed positional
    shortcut is not sufficient. Random token collisions are possible and are
    treated as part of the task distribution.

    Multiple bigrams per sequence = dense learning signal.
    """
    def __init__(
        self,
        vocab_size: int = 50,
        seq_len: int = 128,
        n_bigrams: int = 8,
        seed: int = 0,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_bigrams = n_bigrams
        self.rng = np.random.default_rng(seed)

    def generate_batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
        """Generate a batch with randomly-placed repeated bigrams.

        Returns:
            input_ids: (B, T) token sequences
            targets: (B, T) next-token targets
            induction_mask: (B, T) 1 at positions where target is the B of a repeated A-B
        """
        inputs = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        induction_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        for i in range(batch_size):
            # Start with random noise
            seq = self.rng.integers(0, self.vocab_size, size=self.seq_len + 1)

            # Place repeated bigrams at random positions
            used_positions = set()

            for _ in range(self.n_bigrams):
                A = self.rng.integers(0, self.vocab_size)
                B = self.rng.integers(0, self.vocab_size)

                # Find two non-overlapping positions for the bigram
                # First occurrence must come before second
                attempts = 0
                while attempts < 20:
                    pos1 = self.rng.integers(0, self.seq_len - 10)
                    # Second occurrence: at least 3 positions after first, leave room
                    min_pos2 = pos1 + 3
                    max_pos2 = self.seq_len - 1
                    if min_pos2 >= max_pos2:
                        attempts += 1
                        continue
                    pos2 = self.rng.integers(min_pos2, max_pos2)

                    # Check no overlap with existing bigrams
                    positions_needed = {pos1, pos1 + 1, pos2, pos2 + 1}
                    if positions_needed & used_positions:
                        attempts += 1
                        continue

                    # Place the bigram
                    seq[pos1] = A
                    seq[pos1 + 1] = B
                    seq[pos2] = A
                    seq[pos2 + 1] = B

                    used_positions.update(positions_needed)

                    # Mark second occurrence of B as induction target
                    # At input position pos2, the target is seq[pos2+1] = B
                    induction_mask[i, pos2] = 1.0

                    break

            inputs[i] = seq[:self.seq_len]
            targets[i] = seq[1:self.seq_len + 1]

        return (
            mx.array(inputs),
            mx.array(targets),
            mx.array(induction_mask),
        )

    def iter_batches(self, batch_size: int, n_batches: int) -> Iterator:
        """Yield batches for training."""
        for _ in range(n_batches):
            yield self.generate_batch(batch_size)


class IOIDataset:
    """Indirect Object Identification - simplified synthetic version.

    Generates sequences encoding: "X and Y verb ... X verb2 _"
    Target at _ should be Y (the indirect object).

    This requires:
    - Name detection circuits
    - Positional tracking (which name came where)
    - Inhibition (suppress the repeated name)

    More complex than induction — tests whether multi-component circuits
    form consistently across seeds.
    """
    def __init__(
        self,
        n_names: int = 20,
        n_verbs: int = 10,
        n_fillers: int = 30,
        seq_len: int = 32,
        seed: int = 0,
    ):
        self.n_names = n_names
        self.n_verbs = n_verbs
        self.n_fillers = n_fillers
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

        # Token assignments (offset to avoid overlap)
        self.name_tokens = list(range(0, n_names))
        self.verb_tokens = list(range(n_names, n_names + n_verbs))
        self.filler_tokens = list(range(n_names + n_verbs, n_names + n_verbs + n_fillers))
        self.vocab_size = n_names + n_verbs + n_fillers

    def generate_batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
        """Generate IOI batch.

        Pattern: [fillers] S [filler] IO [fillers] S verb -> IO
        S = subject (repeated name), IO = indirect object (target)
        """
        inputs = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        ioi_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        for i in range(batch_size):
            # Pick two distinct names
            names = self.rng.choice(self.n_names, size=2, replace=False)
            S, IO = int(names[0]), int(names[1])
            verb = int(self.rng.choice(self.verb_tokens))

            # Build sequence: [fill] S [fill] IO [fill] S verb IO [fill...]
            seq = self.rng.choice(self.filler_tokens, size=self.seq_len + 1).astype(np.int32)

            # Place pattern elements
            pos_s1 = 3
            pos_io = 6
            pos_s2 = 10
            pos_verb = 11
            pos_target = 12  # model must predict IO here

            seq[pos_s1] = self.name_tokens[S]
            seq[pos_io] = self.name_tokens[IO]
            seq[pos_s2] = self.name_tokens[S]
            seq[pos_verb] = verb
            seq[pos_target] = self.name_tokens[IO]

            inputs[i] = seq[:-1]
            targets[i] = seq[1:]
            ioi_mask[i, pos_target - 1] = 1.0  # Position where we predict IO

        return (
            mx.array(inputs),
            mx.array(targets),
            mx.array(ioi_mask),
        )

    def iter_batches(self, batch_size: int, n_batches: int) -> Iterator:
        for _ in range(n_batches):
            yield self.generate_batch(batch_size)


class ModularArithmeticDataset:
    """Modular addition: given (a, b), predict (a + b) mod p.

    Known to produce Fourier-based circuits after grokking.
    Useful for studying phase transitions in circuit formation.
    """
    def __init__(self, p: int = 113, seed: int = 0):
        self.p = p
        self.vocab_size = p + 2  # p numbers + '=' token + padding
        self.eq_token = p
        self.rng = np.random.default_rng(seed)

        # Generate all pairs
        self.all_pairs = [(a, b) for a in range(p) for b in range(p)]
        self.rng.shuffle(self.all_pairs)

        # Train/test split (standard: 50% train for grokking)
        split = len(self.all_pairs) // 2
        self.train_pairs = self.all_pairs[:split]
        self.test_pairs = self.all_pairs[split:]

    def generate_batch(self, batch_size: int, split: str = "train") -> tuple[mx.array, mx.array]:
        """Generate batch of (a, =, b) -> (a+b) mod p sequences.

        Input: [a, eq, b]  (length 3)
        Target: (a + b) mod p at the last position
        """
        pairs = self.train_pairs if split == "train" else self.test_pairs
        indices = self.rng.integers(0, len(pairs), size=batch_size)

        inputs = np.zeros((batch_size, 3), dtype=np.int32)
        targets = np.zeros((batch_size,), dtype=np.int32)

        for i, idx in enumerate(indices):
            a, b = pairs[idx]
            inputs[i] = [a, self.eq_token, b]
            targets[i] = (a + b) % self.p

        return mx.array(inputs), mx.array(targets)


class TextDataset:
    """Natural language dataset using a simple byte-pair or character-level tokenizer.

    Downloads a small text corpus (TinyShakespeare or similar) and tokenizes it.
    Character-level keeps us self-contained with no external tokenizer dependencies.

    For circuit mapping on natural language: we want to see what circuits form
    when the model faces real linguistic structure (syntax, semantics, co-reference)
    rather than synthetic patterns.
    """
    def __init__(
        self,
        data_dir: str = "data",
        seq_len: int = 256,
        seed: int = 0,
        corpus: str = "shakespeare",
    ):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        corpus_file = data_path / f"{corpus}.txt"

        if not corpus_file.exists():
            self._download_corpus(corpus, corpus_file)

        # Read and tokenize (character-level)
        with open(corpus_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Build character vocabulary
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

        # Tokenize entire corpus
        self.data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)

        print(f"TextDataset: {len(text):,} chars, vocab_size={self.vocab_size}, "
              f"{len(self.data) // seq_len:,} sequences")

    def _download_corpus(self, corpus: str, path: Path):
        """Download a small text corpus."""
        import urllib.request

        urls = {
            "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        }

        url = urls.get(corpus)
        if url is None:
            raise ValueError(f"Unknown corpus: {corpus}. Available: {list(urls.keys())}")

        print(f"Downloading {corpus} corpus...")
        urllib.request.urlretrieve(url, str(path))
        print(f"Saved to {path}")

    def generate_batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        """Generate a batch of text sequences (no task mask — all positions are targets)."""
        max_start = len(self.data) - self.seq_len - 1
        starts = self.rng.integers(0, max_start, size=batch_size)

        inputs = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, self.seq_len), dtype=np.int32)

        for i, start in enumerate(starts):
            inputs[i] = self.data[start:start + self.seq_len]
            targets[i] = self.data[start + 1:start + self.seq_len + 1]

        return mx.array(inputs), mx.array(targets)

    def iter_batches(self, batch_size: int, n_batches: int) -> Iterator:
        for _ in range(n_batches):
            yield self.generate_batch(batch_size)


if __name__ == "__main__":
    print("=== Induction Dataset ===")
    ds = InductionDataset(vocab_size=50, seq_len=64, seed=42)
    inp, tgt, mask = ds.generate_batch(4)
    print(f"Input shape: {inp.shape}, Target shape: {tgt.shape}")
    print(f"Induction positions per sequence: {mask.sum(axis=1).tolist()}")

    print("\n=== IOI Dataset ===")
    ioi = IOIDataset(seed=42)
    inp, tgt, mask = ioi.generate_batch(4)
    print(f"Input shape: {inp.shape}, Target shape: {tgt.shape}")
    print(f"IOI positions per sequence: {mask.sum(axis=1).tolist()}")

    print("\n=== Modular Arithmetic Dataset ===")
    mod = ModularArithmeticDataset(p=113, seed=42)
    inp, tgt = mod.generate_batch(4)
    print(f"Input shape: {inp.shape}, Target shape: {tgt.shape}")
    print(f"Targets: {tgt.tolist()}")
