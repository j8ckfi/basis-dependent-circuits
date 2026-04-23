"""Thin training entrypoint for the extended-control harness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.extended_controls.run_mlx_experiment import main


if __name__ == "__main__":
    main()
