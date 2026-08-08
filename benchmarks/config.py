"""Per-machine benchmark settings.

Every value can be overridden by an environment variable of the same name prefixed with
NACHO_, so a different machine needs no edits:

    NACHO_MATRIX_DIR=/data/suitesparse python benchmarks/smoke.py

Paths resolve relative to this file rather than the working directory, so the scripts can
be run from anywhere.
"""

import os
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(f"NACHO_{name}", default)).expanduser()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(f"NACHO_{name}", default))


# Directory holding the SuiteSparse .mtx files.
MATRIX_DIR = _path("MATRIX_DIR", Path("/scratch/atharva/suitesparse/"))

# Index of the SuiteSparse matrices: name, nnz, percent_nnz, total_elements, rows, columns.
STATS_CSV = _path("STATS_CSV", BENCHMARKS_DIR / "suitesparse_stats.csv")

# Where .npz result files and plots are written.
RESULTS_DIR = _path("RESULTS_DIR", REPO_ROOT / "benchmark_results")

# Timed iterations per measurement, and how many of the slowest/fastest to trim.
ITER_COUNT = _int("ITER_COUNT", 14)
TRIM = _int("TRIM", 2)

# Default CUDA launch geometry for the generated GPU kernels.
NUM_BLOCKS = _int("NUM_BLOCKS", 256)
THREADS_PER_BLOCK = _int("THREADS_PER_BLOCK", 256)
