#!/usr/bin/env python3
"""
Run nacho runtime benchmarks against SuiteSparse matrices.

Usage:
    python tests/run_benchmarks.py                  # all benchmarks, full sweep
    python tests/run_benchmarks.py csr_add          # just CSR add
    python tests/run_benchmarks.py spgemm -s 982 -e 988   # SpGEMM on a range
    python tests/run_benchmarks.py --quick           # quick smoke test (~5 matrix pairs)

Run from the runtime/ directory:
    cd runtime && conda activate nacho && module load cuda
    python tests/run_benchmarks.py
"""

import argparse
import subprocess
import sys
import os

# Ensure tests/ is on the path so existing modules import cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ensure_cuda():
    """Load CUDA via the module system if nvcc isn't already on PATH."""
    if os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'):
        return
    if any(os.path.isfile(os.path.join(d, 'nvcc')) for d in os.environ.get('PATH', '').split(':')):
        return

    # Source the module system and load cuda, then apply the env changes
    for init in ['/etc/profile.d/modules.sh', '/usr/share/modules/init/bash']:
        if os.path.isfile(init):
            try:
                result = subprocess.run(
                    ['bash', '-c', f'source {init} && module load cuda 2>/dev/null && env'],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        key, _, val = line.partition('=')
                        if key and val:
                            os.environ[key] = val
                    return
            except (subprocess.TimeoutExpired, OSError):
                continue

    print("Warning: could not find CUDA. Set CUDA_HOME or run 'module load cuda' first.",
          file=sys.stderr)


ensure_cuda()


def get_num_matrices():
    from parser import matrix_list
    return len(matrix_list())


def run_csr_add(start, end, save_and_plot):
    from coo_and_csr import benchmark_csr_add
    print(f"\n{'='*60}")
    print(f"CSR Add benchmark  (matrices {start}–{end})")
    print(f"{'='*60}")
    benchmark_csr_add(start, end, save_and_plot=save_and_plot)


def run_coo_add(start, end, save_and_plot):
    from coo_and_csr import benchmark_coo_add
    print(f"\n{'='*60}")
    print(f"COO Add benchmark  (matrices {start}–{end})")
    print(f"{'='*60}")
    benchmark_coo_add(start, end, save_and_plot=save_and_plot)


def run_spgemm(start, end, save_and_plot):
    from spgemm import spgemm
    print(f"\n{'='*60}")
    print(f"SpGEMM benchmark  (matrices {start}–{end})")
    print(f"{'='*60}")
    spgemm(start, end, save_and_plot=save_and_plot)


def run_sparse_vectors(start, end, _save_and_plot):
    from parser import matrix_list
    from sparse_vectors import test_mergepath
    df = matrix_list()
    n = len(df)
    end = min(end, n - 2)  # needs triplets of matrices
    print(f"\n{'='*60}")
    print(f"Sparse vector (a*b+c) benchmark  (matrices {start}–{end})")
    print(f"{'='*60}")
    failed = []
    for i in range(start, end):
        print(f"iteration {i}")
        result = test_mergepath(
            0, 0,
            df.iloc[i]['name'],
            df.iloc[i + 1]['name'],
            df.iloc[i + 2]['name'],
        )
        if result is None:
            failed.append(i)
            continue
        ans, full, partial, no, total_nnz = result
        if not ans:
            failed.append(i)
        else:
            print(f"  nnz={total_nnz}  full={full}  partial={partial}  nofusion={no}")
    if failed:
        print(f"FAILED iterations: {failed}")


def run_broadcast(_start, _end, _save_and_plot):
    from broadcasts import benchmark_broadcast
    print(f"\n{'='*60}")
    print(f"Broadcast (x*A) benchmark")
    print(f"{'='*60}")
    ans, xa_time, csr_time = benchmark_broadcast()
    print(f"  correct={ans}  broadcast={xa_time:.3f}ms  csr_add={csr_time:.3f}ms")


BENCHMARKS = {
    'csr_add':        run_csr_add,
    'coo_add':        run_coo_add,
    'spgemm':         run_spgemm,
    'sparse_vectors': run_sparse_vectors,
    'broadcast':      run_broadcast,
}

QUICK_RANGE = (982, 988)
DEFAULT_START = 0


def main():
    parser = argparse.ArgumentParser(
        description='Run nacho runtime benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
benchmarks:
  csr_add          CSR sparse matrix addition (manual vs cuSPARSE vs PyTorch)
  coo_add          COO sparse matrix addition (manual vs PyTorch)
  spgemm           Sparse matrix-matrix multiply (manual vs cuSPARSE)
  sparse_vectors   Sparse vector a*b+c with different fusion strategies
  broadcast        Broadcast x*A correctness + timing
  all              Run all benchmarks (default)

examples:
  python tests/run_benchmarks.py --quick
  python tests/run_benchmarks.py csr_add spgemm
  python tests/run_benchmarks.py spgemm -s 500 -e 600
  python tests/run_benchmarks.py --no-plot
""",
    )
    parser.add_argument(
        'benchmarks', nargs='*', default=['all'],
        help='Which benchmarks to run (default: all)',
    )
    parser.add_argument(
        '-s', '--start', type=int, default=None,
        help='Start matrix index (default: 0, or 982 with --quick)',
    )
    parser.add_argument(
        '-e', '--end', type=int, default=None,
        help='End matrix index (default: all matrices, or 988 with --quick)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick smoke test on a small range of matrices (982–988)',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip saving plots and .npz files',
    )
    args = parser.parse_args()

    # Resolve which benchmarks to run
    if 'all' in args.benchmarks:
        to_run = list(BENCHMARKS.keys())
    else:
        to_run = []
        for b in args.benchmarks:
            if b not in BENCHMARKS:
                print(f"Unknown benchmark: {b}")
                print(f"Available: {', '.join(BENCHMARKS.keys())}")
                sys.exit(1)
            to_run.append(b)

    # Resolve matrix range
    num_matrices = get_num_matrices()
    if args.quick:
        start = args.start if args.start is not None else QUICK_RANGE[0]
        end = args.end if args.end is not None else QUICK_RANGE[1]
    else:
        start = args.start if args.start is not None else DEFAULT_START
        end = args.end if args.end is not None else num_matrices

    save_and_plot = not args.no_plot

    print(f"Benchmarks: {', '.join(to_run)}")
    print(f"Matrix range: {start}–{end} (of {num_matrices} available)")
    if not save_and_plot:
        print("Plots disabled")

    for name in to_run:
        BENCHMARKS[name](start, end, save_and_plot)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == '__main__':
    main()
