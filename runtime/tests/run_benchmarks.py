#!/usr/bin/env python3
"""
Run nacho runtime benchmarks against SuiteSparse matrices.

Usage:
    python runtime/tests/run_benchmarks.py                       # all benchmarks, full sweep
    python runtime/tests/run_benchmarks.py csr_add               # just CSR add
    python runtime/tests/run_benchmarks.py spgemm -s 982 -e 988  # SpGEMM on a range
    python runtime/tests/run_benchmarks.py --quick               # quick smoke test
    python runtime/tests/run_benchmarks.py --no-continue csr_add  # start fresh, ignore existing CSV
"""

import argparse
import subprocess
import sys
import os

import pandas as pd

# Ensure tests/ is on the path so existing modules import cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ensure_cuda():
    """Load CUDA via the module system if nvcc isn't already on PATH."""
    if os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'):
        return
    if any(os.path.isfile(os.path.join(d, 'nvcc')) for d in os.environ.get('PATH', '').split(':')):
        return

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

from tqdm import tqdm

SAVE_EVERY = 20  # flush CSV every N iterations


def get_num_matrices():
    from parser import matrix_list
    return len(matrix_list())


def _csv_path(name):
    from plotter import _path
    return _path(name, "csv")


def _save_rows(rows, csv_name):
    """Save accumulated rows to CSV (overwrites)."""
    if rows:
        pd.DataFrame(rows).to_csv(_csv_path(csv_name), index=False)


def _load_done_indices(csv_name):
    """Load completed indices from an existing CSV. Returns a set of idx values."""
    path = _csv_path(csv_name)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if 'idx' in df.columns:
            done = set(df['idx'].tolist())
            return done, df.to_dict('records')
    return set(), []


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_csr_add(start, end, save_and_plot, continue_mode=False):
    import torch
    import nanobind_cuda_example
    from parser import matrix_list, parse_matrix
    from coo_and_csr import csr_add, torch_add, failure_reason
    from plotter import plot

    csv_name = f"csr_add_{start}-{end}"
    df = matrix_list()
    failed = []
    skip = {611}

    if continue_mode:
        done, rows = _load_done_indices(csv_name)
        if done:
            tqdm.write(f"Continuing: {len(done)} pairs already done")
    else:
        done, rows = set(), []

    indices = [i for i in range(start + 1, end) if i not in skip and i not in done]
    for i in tqdm(indices, desc="CSR Add", unit="pair"):
        try:
            A = parse_matrix(df.iloc[i - 1]['name'])
            B = parse_matrix(df.iloc[i]['name'])
            M = min(A.size(0), B.size(0))
            N = max(A.size(1), B.size(1))

            A_t = torch.sparse_csr_tensor(A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]], A.values()[:A.crow_indices()[M]], (M, N))
            B_t = torch.sparse_csr_tensor(B.crow_indices()[:M+1], B.col_indices()[:B.crow_indices()[M]], B.values()[:B.crow_indices()[M]], (M, N))
            plus_row = A_t.crow_indices() + B_t.crow_indices()

            A_CSR = nanobind_cuda_example.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(), torch.tensor([M, N], dtype=torch.int32))
            B_CSR = nanobind_cuda_example.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(), torch.tensor([M, N], dtype=torch.int32))

            C_pytorch, pytorch = torch_add(A_t, B_t)
            C_cusparse, cusparse = csr_add(A_CSR, B_CSR, True)
            C_manual, manual = csr_add(A_CSR, B_CSR, False)

            ans = (torch.equal(C_cusparse.indptr, C_manual.indptr)
                   and torch.equal(C_cusparse.indices, C_manual.indices)
                   and torch.equal(C_cusparse.data, C_manual.data)
                   and torch.equal(C_pytorch.crow_indices(), C_manual.indptr)
                   and torch.equal(C_pytorch.col_indices(), C_manual.indices)
                   and torch.equal(C_pytorch.values(), C_manual.data))

            if not ans:
                tqdm.write(f"FAILED at {i}: {df.iloc[i-1]['name']} x {df.iloc[i]['name']}")
                failed.append(i)
                failure_reason(C_manual, C_cusparse)

            rows.append({
                "idx": i,
                "matrix_a": df.iloc[i - 1]['name'],
                "matrix_b": df.iloc[i]['name'],
                "nnz": plus_row.max().item(),
                "manual_ms": manual,
                "cusparse_ms": cusparse,
                "pytorch_ms": pytorch,
                "correct": ans,
            })

            if len(rows) % SAVE_EVERY == 0:
                _save_rows(rows, csv_name)

        except (RuntimeError, MemoryError) as e:
            tqdm.write(f"Skipping {i} ({df.iloc[i-1]['name']} x {df.iloc[i]['name']}): {e}")
            torch.cuda.empty_cache()
            continue

    _save_rows(rows, csv_name)
    if save_and_plot and rows:
        rdf = pd.DataFrame(rows)
        plot(rdf["nnz"].tolist(), rdf["manual_ms"].tolist(),
             rdf["cusparse_ms"].tolist(), rdf["pytorch_ms"].tolist(), csv_name)
    if failed:
        print(f"Failed: {failed}")


def run_coo_add(start, end, save_and_plot, continue_mode=False):
    import torch
    import nanobind_cuda_example
    from parser import matrix_list, parse_matrix
    from coo_and_csr import coo_add
    from plotter import plot

    csv_name = f"coo_add_{start}-{end}"
    df = matrix_list()
    failed = []

    if continue_mode:
        done, rows = _load_done_indices(csv_name)
        if done:
            tqdm.write(f"Continuing: {len(done)} pairs already done")
    else:
        done, rows = set(), []

    indices = [i for i in range(start + 1, end) if i not in done]
    for i in tqdm(indices, desc="COO Add", unit="pair"):
        try:
            A = parse_matrix(df.iloc[i - 1]['name'], True).coalesce()
            B = parse_matrix(df.iloc[i]['name'], True).coalesce()
            M = max(A.size(0), B.size(0))
            N = max(A.size(1), B.size(1))

            A_t = torch.sparse_coo_tensor(A.indices(), A.values(), (M, N)).coalesce()
            B_t = torch.sparse_coo_tensor(B.indices(), B.values(), (M, N)).coalesce()

            A_COO = nanobind_cuda_example.COO(A.indices()[0], A.indices()[1], A.values(), torch.tensor([M, N], dtype=torch.int32))
            B_COO = nanobind_cuda_example.COO(B.indices()[0], B.indices()[1], B.values(), torch.tensor([M, N], dtype=torch.int32))

            C_pytorch, pytorch = coo_add(A_t, B_t, True)
            C_manual, manual = coo_add(A_COO, B_COO, False)

            ans = (torch.equal(C_pytorch.indices()[0], C_manual.row)
                   and torch.equal(C_pytorch.indices()[1], C_manual.col)
                   and torch.equal(C_pytorch.values(), C_manual.data))

            if not ans:
                tqdm.write(f"FAILED at {i}: {df.iloc[i-1]['name']} x {df.iloc[i]['name']}")
                failed.append(i)

            rows.append({
                "idx": i,
                "matrix_a": df.iloc[i - 1]['name'],
                "matrix_b": df.iloc[i]['name'],
                "nnz": A_COO.data.numel() + B_COO.data.numel(),
                "manual_ms": manual,
                "pytorch_ms": pytorch,
                "correct": ans,
            })

            if len(rows) % SAVE_EVERY == 0:
                _save_rows(rows, csv_name)

        except (RuntimeError, MemoryError) as e:
            tqdm.write(f"Skipping {i} ({df.iloc[i-1]['name']} x {df.iloc[i]['name']}): {e}")
            torch.cuda.empty_cache()
            continue

    _save_rows(rows, csv_name)
    if save_and_plot and rows:
        rdf = pd.DataFrame(rows)
        plot(rdf["nnz"].tolist(), rdf["manual_ms"].tolist(),
             [], rdf["pytorch_ms"].tolist(), csv_name)
    if failed:
        print(f"Failed: {failed}")


def run_spgemm(start, end, save_and_plot, continue_mode=False):
    import torch
    import nanobind_cuda_example
    from parser import matrix_list, parse_matrix
    from spgemm import spgemm_benchmark
    from coo_and_csr import failure_reason
    from plotter import plot

    csv_name = f"spgemm_{start}-{end}"
    df = matrix_list()
    failed = []
    skip = {611}

    if continue_mode:
        done, rows = _load_done_indices(csv_name)
        if done:
            tqdm.write(f"Continuing: {len(done)} pairs already done")
    else:
        done, rows = set(), []

    indices = [i for i in range(start + 1, end) if i not in skip and i not in done]
    for i in tqdm(indices, desc="SpGEMM", unit="pair"):
        try:
            A = parse_matrix(df.iloc[i - 1]['name'])
            B = parse_matrix(df.iloc[i]['name'])

            # AxA benchmark for square matrices
            if A.size(0) == A.size(1):
                M = A.size(0)
                A_CSR = nanobind_cuda_example.CSR(A.crow_indices(), A.col_indices(), A.values(), torch.tensor([M, M], dtype=torch.int32))
                C_cusparse, cusparse = spgemm_benchmark(A_CSR, A_CSR, True)
                C_manual, manual = spgemm_benchmark(A_CSR, A_CSR, False)
                ans = (torch.equal(C_cusparse.indptr, C_manual.indptr)
                       and torch.equal(C_cusparse.indices, C_manual.indices))
                if not ans:
                    tqdm.write(f"FAILED AxA at {i}: {df.iloc[i-1]['name']}")
                    failed.append(i)
                    failure_reason(C_manual, C_cusparse)
                    break
                rows.append({
                    "idx": i, "type": "AxA",
                    "matrix_a": df.iloc[i - 1]['name'], "matrix_b": df.iloc[i - 1]['name'],
                    "nnz": A.crow_indices().max().item() * 2,
                    "manual_ms": manual, "cusparse_ms": cusparse, "correct": ans,
                })

            # AxB benchmark
            M = A.size(0)
            K = min(A.size(1), B.size(0))
            if K != A.size(1):
                continue
            N = B.size(1)

            A_t = torch.sparse_csr_tensor(A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]], A.values()[:A.crow_indices()[M]], (M, K))
            B_t = torch.sparse_csr_tensor(B.crow_indices()[:K+1], B.col_indices()[:B.crow_indices()[K]], B.values()[:B.crow_indices()[K]], (K, N))
            plus_row = A_t.crow_indices().max() + B_t.crow_indices().max()

            A_CSR = nanobind_cuda_example.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(), torch.tensor([M, K], dtype=torch.int32))
            B_CSR = nanobind_cuda_example.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(), torch.tensor([K, N], dtype=torch.int32))

            C_cusparse, cusparse = spgemm_benchmark(A_CSR, B_CSR, True)
            C_manual, manual = spgemm_benchmark(A_CSR, B_CSR, False)

            ans = (torch.equal(C_cusparse.indptr, C_manual.indptr)
                   and torch.equal(C_cusparse.indices, C_manual.indices))
            if not ans:
                tqdm.write(f"FAILED at {i}: {df.iloc[i-1]['name']} x {df.iloc[i]['name']}")
                failed.append(i)
                failure_reason(C_manual, C_cusparse)
                break

            rows.append({
                "idx": i, "type": "AxB",
                "matrix_a": df.iloc[i - 1]['name'], "matrix_b": df.iloc[i]['name'],
                "nnz": plus_row.max().item(),
                "manual_ms": manual, "cusparse_ms": cusparse, "correct": ans,
            })

            if len(rows) % SAVE_EVERY == 0:
                _save_rows(rows, csv_name)

        except (RuntimeError, MemoryError) as e:
            tqdm.write(f"Skipping {i} ({df.iloc[i-1]['name']} x {df.iloc[i]['name']}): {e}")
            torch.cuda.empty_cache()
            continue

    _save_rows(rows, csv_name)
    if save_and_plot and rows:
        rdf = pd.DataFrame(rows)
        plot(rdf["nnz"].tolist(), rdf["manual_ms"].tolist(),
             rdf["cusparse_ms"].tolist(), [], csv_name)
    if failed:
        print(f"Failed: {failed}")


def run_sparse_vectors(start, end, save_and_plot, continue_mode=False):
    from parser import matrix_list
    from sparse_vectors import test_mergepath
    from plotter import plot_bar_graph_2

    csv_name = f"sparse_vectors_{start}-{end}"
    df = matrix_list()
    n = len(df)
    end = min(end, n - 2)
    failed = []

    if continue_mode:
        done, rows = _load_done_indices(csv_name)
        if done:
            tqdm.write(f"Continuing: {len(done)} triplets already done")
    else:
        done, rows = set(), []

    indices = [i for i in range(start, end) if i not in done]
    for i in tqdm(indices, desc="Sparse Vectors", unit="triplet"):
        try:
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
            rows.append({
                "idx": i,
                "matrix_a": df.iloc[i]['name'],
                "matrix_b": df.iloc[i + 1]['name'],
                "matrix_c": df.iloc[i + 2]['name'],
                "total_nnz": total_nnz,
                "full_mergepath_ms": full[0],
                "full_precompute_ms": full[1],
                "full_compute_ms": full[2],
                "partial_mergepath_ms": partial[0],
                "partial_precompute_ms": partial[1],
                "partial_compute_ms": partial[2],
                "nofusion_mergepath_ms": no[0],
                "nofusion_precompute_ms": no[1],
                "nofusion_compute_ms": no[2],
                "correct": ans,
            })

            if len(rows) % SAVE_EVERY == 0:
                _save_rows(rows, csv_name)

        except (RuntimeError, MemoryError) as e:
            import torch
            tqdm.write(f"Skipping {i} ({df.iloc[i]['name']}): {e}")
            torch.cuda.empty_cache()
            continue

    _save_rows(rows, csv_name)
    if save_and_plot and rows:
        full_lb = [[r["full_mergepath_ms"], r["full_precompute_ms"], r["full_compute_ms"]] for r in rows]
        partial_lb = [[r["partial_mergepath_ms"], r["partial_precompute_ms"], r["partial_compute_ms"]] for r in rows]
        single_lb = [[r["nofusion_mergepath_ms"], r["nofusion_precompute_ms"], r["nofusion_compute_ms"]] for r in rows]
        lengths = [r["total_nnz"] for r in rows]
        plot_bar_graph_2(0, lengths, full_lb, partial_lb, single_lb, csv_name)
    if failed:
        print(f"Failed: {failed}")


def run_broadcast(_start, _end, _save_and_plot, continue_mode=False):
    from broadcasts import benchmark_broadcast
    print("Running broadcast (x*A) benchmark...")
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
  python runtime/tests/run_benchmarks.py --quick
  python runtime/tests/run_benchmarks.py csr_add spgemm
  python runtime/tests/run_benchmarks.py spgemm -s 500 -e 600
  python runtime/tests/run_benchmarks.py --no-continue csr_add
  python runtime/tests/run_benchmarks.py --no-plot
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
        help='Skip saving plots',
    )
    parser.add_argument(
        '--no-continue', dest='continue_mode', action='store_false', default=True,
        help='Start fresh instead of resuming from existing CSV',
    )
    args = parser.parse_args()

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
    if not args.continue_mode:
        print("Fresh run: ignoring any existing CSV results")
    if not save_and_plot:
        print("Plots disabled")
    print()

    for name in to_run:
        BENCHMARKS[name](start, end, save_and_plot, continue_mode=args.continue_mode)

    print("\nDone.")


if __name__ == '__main__':
    main()
