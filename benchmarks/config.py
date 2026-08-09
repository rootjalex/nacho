"""Per-machine benchmark settings.

Every value can be overridden by an environment variable of the same name prefixed with
NACHO_, so a different machine needs no edits:

    NACHO_SUITESPARSE_DIR=/data/suitesparse python benchmarks/csr_add.py

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
SUITESPARSE_DIR = _path("SUITESPARSE_DIR", Path("/scratch/atharva/suitesparse/"))

# Index of the SuiteSparse matrices: name, nnz, percent_nnz, total_elements, rows, columns.
STATS_CSV = _path("STATS_CSV", BENCHMARKS_DIR / "suitesparse_stats.csv")

# Directory holding the FROSTT .tns tensors.
FROSTT_DIR = _path("FROSTT_DIR", Path("/scratch/frostt/"))

# FROSTT tensors to benchmark: (name, file stem, (I, J, K)). Coordinates must already be
# sorted lexicographically and free of duplicates, which is what the *_sorted stems are:
# datasets/sort_frostt.py writes them from the downloaded files.
FROSTT_TENSORS = [
    # ("nell-2", "nell-2_sorted", (12092, 9184, 28818)),
    ("nell-1", "nell-1_sorted", (2902330, 2143368, 25495389)),
    # ("darpa", "1998DARPA_sorted", (22476, 22476, 23776223)),
    # ("fb-m", "fb-m_sorted", (23344784, 23344784, 166)),
]

# Where .npz result files and plots are written.
RESULTS_DIR = _path("RESULTS_DIR", REPO_ROOT / "benchmark_results")

# Timed iterations per measurement, and how many of the slowest/fastest to trim.
ITER_COUNT = _int("ITER_COUNT", 14)
TRIM = _int("TRIM", 2)

# Default CUDA launch geometry for the generated GPU kernels.
NUM_BLOCKS = _int("NUM_BLOCKS", 256)
THREADS_PER_BLOCK = _int("THREADS_PER_BLOCK", 256)


def require_dataset_dir(path, dataset, env_var):
    """Fail loudly when a dataset directory is missing, naming the variable that sets it."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"{dataset} directory not found: {path}\n"
            f"Set {env_var} to the directory holding the {dataset} files.")
    return path
