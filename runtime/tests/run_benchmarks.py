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


def _subprocess_nacho_timing(matrix_a, matrix_b, fmt, M, N, timeout=120, matrix_c=None):
    """Run a nacho kernel in a completely separate process.

    nanobind calls abort() on unrecoverable errors (e.g. buffer overflow on
    large matrices).  abort() sends SIGABRT which kills the process — Python
    try/except cannot catch it, and os.fork() breaks CUDA contexts.

    Instead, we spawn a fresh Python interpreter that loads its own CUDA
    context, parses the matrices, runs the nacho kernel, and prints the
    timing to stdout.  If it crashes, subprocess.run() returns non-zero
    and we skip.
    """
    # Build the kernel call based on format
    if fmt == "csr":
        kernel_code = f'''\
from coo_and_csr import nacho_csr_add
A = parse_matrix("{matrix_a}")
B = parse_matrix("{matrix_b}")
A_t = torch.sparse_csr_tensor(
    A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]],
    A.values()[:A.crow_indices()[M]], (M, N))
B_t = torch.sparse_csr_tensor(
    B.crow_indices()[:M+1], B.col_indices()[:B.crow_indices()[M]],
    B.values()[:B.crow_indices()[M]], (M, N))
A_CSR = nacho_runtime.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
B_CSR = nacho_runtime.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
_, ms = nacho_csr_add(A_CSR, B_CSR)'''
    elif fmt == "coo":
        kernel_code = f'''\
from coo_and_csr import nacho_coo_add
A = parse_matrix("{matrix_a}", True).coalesce()
B = parse_matrix("{matrix_b}", True).coalesce()
A_t = torch.sparse_coo_tensor(A.indices(), A.values(), (M, N)).coalesce()
B_t = torch.sparse_coo_tensor(B.indices(), B.values(), (M, N)).coalesce()
A_COO = nacho_runtime.COO(A_t.indices()[0], A_t.indices()[1], A_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
B_COO = nacho_runtime.COO(B_t.indices()[0], B_t.indices()[1], B_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
_, ms = nacho_coo_add(A_COO, B_COO)'''
    elif fmt == "coo_mul":
        kernel_code = f'''\
from coo_and_csr import nacho_coo_mul
A = parse_matrix("{matrix_a}", True).coalesce()
B = parse_matrix("{matrix_b}", True).coalesce()
A_t = torch.sparse_coo_tensor(A.indices(), A.values(), (M, N)).coalesce()
B_t = torch.sparse_coo_tensor(B.indices(), B.values(), (M, N)).coalesce()
A_COO = nacho_runtime.COO(A_t.indices()[0], A_t.indices()[1], A_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
B_COO = nacho_runtime.COO(B_t.indices()[0], B_t.indices()[1], B_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
_, ms = nacho_coo_mul(A_COO, B_COO)'''
    elif fmt == "csr_3":
        kernel_code = f'''\
from coo_and_csr import nacho_csr_add_3
A = parse_matrix("{matrix_a}")
B = parse_matrix("{matrix_b}")
C = parse_matrix("{matrix_c}")
A_t = torch.sparse_csr_tensor(
    A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]],
    A.values()[:A.crow_indices()[M]], (M, N))
B_t = torch.sparse_csr_tensor(
    B.crow_indices()[:M+1], B.col_indices()[:B.crow_indices()[M]],
    B.values()[:B.crow_indices()[M]], (M, N))
C_t = torch.sparse_csr_tensor(
    C.crow_indices()[:M+1], C.col_indices()[:C.crow_indices()[M]],
    C.values()[:C.crow_indices()[M]], (M, N))
A_CSR = nacho_runtime.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
B_CSR = nacho_runtime.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
C_CSR = nacho_runtime.CSR(C_t.crow_indices(), C_t.col_indices(), C_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
_, ms = nacho_csr_add_3(A_CSR, B_CSR, C_CSR)'''
    elif fmt == "csr_3_unfused":
        kernel_code = f'''\
from coo_and_csr import nacho_csr_add_3_unfused
A = parse_matrix("{matrix_a}")
B = parse_matrix("{matrix_b}")
C = parse_matrix("{matrix_c}")
A_t = torch.sparse_csr_tensor(
    A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]],
    A.values()[:A.crow_indices()[M]], (M, N))
B_t = torch.sparse_csr_tensor(
    B.crow_indices()[:M+1], B.col_indices()[:B.crow_indices()[M]],
    B.values()[:B.crow_indices()[M]], (M, N))
C_t = torch.sparse_csr_tensor(
    C.crow_indices()[:M+1], C.col_indices()[:C.crow_indices()[M]],
    C.values()[:C.crow_indices()[M]], (M, N))
A_CSR = nacho_runtime.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
B_CSR = nacho_runtime.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
C_CSR = nacho_runtime.CSR(C_t.crow_indices(), C_t.col_indices(), C_t.values(),
                           torch.tensor([M, N], dtype=torch.int32))
_, ms = nacho_csr_add_3_unfused(A_CSR, B_CSR, C_CSR)'''
    else:
        return None

    script = f'''\
import sys; sys.path.insert(0, "runtime/tests")
import nacho_runtime
from parser import parse_matrix
import torch
M, N = {M}, {N}
{kernel_code}
print(ms)
'''
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


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


def _load_done_rows(csv_name):
    """Load completed rows from CSV as a dict keyed by idx.

    Returns {idx: row_dict} so callers can check which columns are populated
    and only run missing sections.
    """
    path = _csv_path(csv_name)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if 'idx' in df.columns:
            return {int(r['idx']): r for r in df.to_dict('records')}
    return {}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_csr_add(start, end, save_and_plot, continue_mode=False, csv_name=None):
    import torch
    import nacho_runtime
    from parser import matrix_list, parse_matrix
    from coo_and_csr import csr_add, torch_add, failure_reason
    from plotter import plot

    csv_name = csv_name or f"csr_add_{start}-{end}"
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

            A_CSR = nacho_runtime.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(), torch.tensor([M, N], dtype=torch.int32))
            B_CSR = nacho_runtime.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(), torch.tensor([M, N], dtype=torch.int32))

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


def run_coo_add(start, end, save_and_plot, continue_mode=False, csv_name=None):
    import torch
    import nacho_runtime
    from parser import matrix_list, parse_matrix
    from coo_and_csr import coo_add
    from plotter import plot

    csv_name = csv_name or f"coo_add_{start}-{end}"
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

            A_COO = nacho_runtime.COO(A.indices()[0], A.indices()[1], A.values(), torch.tensor([M, N], dtype=torch.int32))
            B_COO = nacho_runtime.COO(B.indices()[0], B.indices()[1], B.values(), torch.tensor([M, N], dtype=torch.int32))

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


def run_spgemm(start, end, save_and_plot, continue_mode=False, csv_name=None):
    import torch
    import nacho_runtime
    from parser import matrix_list, parse_matrix
    from spgemm import spgemm_benchmark
    from coo_and_csr import failure_reason
    from plotter import plot

    csv_name = csv_name or f"spgemm_{start}-{end}"
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
                A_CSR = nacho_runtime.CSR(A.crow_indices(), A.col_indices(), A.values(), torch.tensor([M, M], dtype=torch.int32))
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

            A_CSR = nacho_runtime.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(), torch.tensor([M, K], dtype=torch.int32))
            B_CSR = nacho_runtime.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(), torch.tensor([K, N], dtype=torch.int32))

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


def run_sparse_vectors(start, end, save_and_plot, continue_mode=False, csv_name=None):
    from parser import matrix_list
    from sparse_vectors import test_mergepath
    from plotter import plot_bar_graph_2

    csv_name = csv_name or f"sparse_vectors_{start}-{end}"
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


def run_broadcast(_start, _end, _save_and_plot, continue_mode=False, csv_name=None):
    from broadcasts import benchmark_broadcast
    print("Running broadcast (x*A) benchmark...")
    ans, xa_time, csr_time = benchmark_broadcast()
    print(f"  correct={ans}  broadcast={xa_time:.3f}ms  csr_add={csr_time:.3f}ms")


def _has_col(row_dict, col):
    """Check if a column is present and not NaN in a row dict."""
    v = row_dict.get(col)
    if v is None:
        return False
    try:
        import math
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def run_nacho_comparison(start, end, save_and_plot, continue_mode=False, csv_name=None):
    """Benchmark nacho-generated CSR add, COO add, and COO mul against cuSPARSE & PyTorch."""
    import torch
    import nacho_runtime
    from parser import matrix_list, parse_matrix
    from coo_and_csr import csr_add, coo_add, torch_add, nacho_csr_add, nacho_coo_add, nacho_coo_mul, pytorch_coo_mul, _remove_zeros_coo, failure_reason, csr_add_3_cusparse_unfused
    from plotter import plot

    csv_name = csv_name or f"nacho_comparison_{start}-{end}"
    df = matrix_list()
    skip = {611}

    # Column-aware continue mode: load existing rows keyed by idx so we can
    # skip sections that are already populated (e.g. csr_add / coo_add) and
    # only collect data for missing sections (e.g. coo_mul).
    if continue_mode:
        existing = _load_done_rows(csv_name)
        if existing:
            tqdm.write(f"Continuing: {len(existing)} pairs in existing CSV")
    else:
        existing = {}

    indices = [i for i in range(start + 1, end) if i not in skip]
    for i in tqdm(indices, desc="Nacho Comparison", unit="pair"):
        prev = existing.get(i, {})
        need_csr_nacho = not _has_col(prev, 'csr_nacho_ms')
        need_csr_baseline = (not _has_col(prev, 'csr_cusparse_ms')
                             or not _has_col(prev, 'csr_manual_ms')
                             or not _has_col(prev, 'csr_pytorch_ms'))
        need_csr = need_csr_nacho or need_csr_baseline
        need_coo_add = not _has_col(prev, 'coo_nacho_ms')
        need_coo_mul = not _has_col(prev, 'coo_mul_nacho_ms')
        need_csr_3way = not _has_col(prev, 'csr_3way_fused_ms')

        if not need_csr and not need_coo_add and not need_coo_mul and not need_csr_3way:
            continue  # fully done

        # Start with existing data so we don't lose already-collected columns
        row = dict(prev)
        row["idx"] = i
        row["matrix_a"] = df.iloc[i - 1]['name']
        row["matrix_b"] = df.iloc[i]['name']

        # --- CSR add ---
        if need_csr:
            try:
                A_csr = parse_matrix(df.iloc[i - 1]['name'])
                B_csr = parse_matrix(df.iloc[i]['name'])
                M = min(A_csr.size(0), B_csr.size(0))
                N = max(A_csr.size(1), B_csr.size(1))

                A_t = torch.sparse_csr_tensor(
                    A_csr.crow_indices()[:M+1],
                    A_csr.col_indices()[:A_csr.crow_indices()[M]],
                    A_csr.values()[:A_csr.crow_indices()[M]], (M, N))
                B_t = torch.sparse_csr_tensor(
                    B_csr.crow_indices()[:M+1],
                    B_csr.col_indices()[:B_csr.crow_indices()[M]],
                    B_csr.values()[:B_csr.crow_indices()[M]], (M, N))

                A_CSR = nacho_runtime.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(),
                                          torch.tensor([M, N], dtype=torch.int32))
                B_CSR = nacho_runtime.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(),
                                          torch.tensor([M, N], dtype=torch.int32))

                # Only run expensive nacho subprocess if nacho data is missing
                if need_csr_nacho:
                    nacho_ms = _subprocess_nacho_timing(df.iloc[i-1]['name'], df.iloc[i]['name'], 'csr', M, N)
                    if nacho_ms is None:
                        tqdm.write(f"nacho CSR crash at {i}, skipping nacho timing")
                else:
                    nacho_ms = prev.get('csr_nacho_ms')

                C_manual, manual_ms = csr_add(A_CSR, B_CSR, False)
                C_cusparse, cusparse_ms = csr_add(A_CSR, B_CSR, True)
                C_pytorch, pytorch_ms = torch_add(A_t, B_t)

                csr_correct = (torch.equal(C_manual.indptr, C_cusparse.indptr)
                               and torch.equal(C_manual.indices, C_cusparse.indices)
                               and torch.allclose(C_manual.data, C_cusparse.data, rtol=1e-4, atol=1e-5))
                if not csr_correct:
                    tqdm.write(f"CSR add mismatch at {i}: {df.iloc[i-1]['name']} x {df.iloc[i]['name']}")
                    failure_reason(C_manual, C_cusparse)

                plus_row = A_t.crow_indices() + B_t.crow_indices()
                row.update({
                    "nnz_csr": plus_row.max().item(),
                    "csr_nacho_ms": nacho_ms,
                    "csr_manual_ms": manual_ms,
                    "csr_cusparse_ms": cusparse_ms,
                    "csr_pytorch_ms": pytorch_ms,
                    "csr_correct": csr_correct,
                })
            except (RuntimeError, MemoryError) as e:
                tqdm.write(f"CSR skip {i}: {e}")
                torch.cuda.empty_cache()

        # --- COO add ---
        if need_coo_add:
            try:
                A_coo = parse_matrix(df.iloc[i - 1]['name'], True).coalesce()
                B_coo = parse_matrix(df.iloc[i]['name'], True).coalesce()
                M = max(A_coo.size(0), B_coo.size(0))
                N = max(A_coo.size(1), B_coo.size(1))

                A_t_coo = torch.sparse_coo_tensor(A_coo.indices(), A_coo.values(), (M, N)).coalesce()
                B_t_coo = torch.sparse_coo_tensor(B_coo.indices(), B_coo.values(), (M, N)).coalesce()

                A_COO = nacho_runtime.COO(A_coo.indices()[0], A_coo.indices()[1], A_coo.values(),
                                          torch.tensor([M, N], dtype=torch.int32))
                B_COO = nacho_runtime.COO(B_coo.indices()[0], B_coo.indices()[1], B_coo.values(),
                                          torch.tensor([M, N], dtype=torch.int32))

                coo_nacho_ms = _subprocess_nacho_timing(df.iloc[i-1]['name'], df.iloc[i]['name'], 'coo', M, N)
                if coo_nacho_ms is None:
                    tqdm.write(f"nacho COO crash at {i}, skipping nacho timing")
                C_coo_manual, coo_manual_ms = coo_add(A_COO, B_COO, False)
                C_coo_pytorch, coo_pytorch_ms = coo_add(A_t_coo, B_t_coo, True)

                coo_correct = (torch.equal(C_coo_pytorch.indices()[0], C_coo_manual.row)
                               and torch.equal(C_coo_pytorch.indices()[1], C_coo_manual.col)
                               and torch.equal(C_coo_pytorch.values(), C_coo_manual.data))
                if not coo_correct:
                    tqdm.write(f"COO add mismatch at {i}: {df.iloc[i-1]['name']} x {df.iloc[i]['name']}")

                row.update({
                    "nnz_coo": A_COO.data.numel() + B_COO.data.numel(),
                    "coo_nacho_ms": coo_nacho_ms,
                    "coo_manual_ms": coo_manual_ms,
                    "coo_pytorch_ms": coo_pytorch_ms,
                    "coo_correct": coo_correct,
                })
            except (RuntimeError, MemoryError) as e:
                tqdm.write(f"COO add skip {i}: {e}")
                torch.cuda.empty_cache()

        # --- COO mul (element-wise) ---
        if need_coo_mul:
            try:
                A_coo = parse_matrix(df.iloc[i - 1]['name'], True).coalesce()
                B_coo = parse_matrix(df.iloc[i]['name'], True).coalesce()
                M = max(A_coo.size(0), B_coo.size(0))
                N = max(A_coo.size(1), B_coo.size(1))

                A_t_coo = torch.sparse_coo_tensor(A_coo.indices(), A_coo.values(), (M, N)).coalesce()
                B_t_coo = torch.sparse_coo_tensor(B_coo.indices(), B_coo.values(), (M, N)).coalesce()

                A_COO = nacho_runtime.COO(A_coo.indices()[0], A_coo.indices()[1], A_coo.values(),
                                          torch.tensor([M, N], dtype=torch.int32))
                B_COO = nacho_runtime.COO(B_coo.indices()[0], B_coo.indices()[1], B_coo.values(),
                                          torch.tensor([M, N], dtype=torch.int32))

                coo_mul_nacho_ms = _subprocess_nacho_timing(df.iloc[i-1]['name'], df.iloc[i]['name'], 'coo_mul', M, N)
                if coo_mul_nacho_ms is None:
                    tqdm.write(f"nacho COO mul crash at {i}, skipping nacho timing")
                C_coo_mul_pytorch, coo_mul_pytorch_ms = pytorch_coo_mul(A_t_coo, B_t_coo)

                # Correctness check: remove zeros from both results before comparing,
                # since SuiteSparse matrices can have explicit zeros that produce
                # zero-valued entries in the mul output.
                coo_mul_correct = None
                C_coo_mul_nacho, _ = nacho_coo_mul(A_COO, B_COO)
                if C_coo_mul_nacho is not None:
                    # Remove zeros from nacho result
                    nacho_mask = C_coo_mul_nacho.data != 0
                    nacho_row = C_coo_mul_nacho.row[nacho_mask]
                    nacho_col = C_coo_mul_nacho.col[nacho_mask]
                    nacho_vals = C_coo_mul_nacho.data[nacho_mask]
                    # PyTorch result already has zeros removed by _remove_zeros_coo
                    pt = C_coo_mul_pytorch
                    coo_mul_correct = (torch.equal(pt.indices()[0], nacho_row)
                                       and torch.equal(pt.indices()[1], nacho_col)
                                       and torch.equal(pt.values(), nacho_vals))
                    if not coo_mul_correct:
                        tqdm.write(f"COO mul mismatch at {i}: {df.iloc[i-1]['name']} x {df.iloc[i]['name']}")

                row.update({
                    "nnz_coo_mul": A_COO.data.numel() + B_COO.data.numel(),
                    "coo_mul_nacho_ms": coo_mul_nacho_ms,
                    "coo_mul_pytorch_ms": coo_mul_pytorch_ms,
                    "coo_mul_correct": coo_mul_correct,
                })
            except (RuntimeError, MemoryError) as e:
                tqdm.write(f"COO mul skip {i}: {e}")
                torch.cuda.empty_cache()

        # --- CSR 3-way add (A+B+C) ---
        if need_csr_3way and i + 1 < len(df) and i + 1 not in skip:
            try:
                A_csr = parse_matrix(df.iloc[i - 1]['name'])
                B_csr = parse_matrix(df.iloc[i]['name'])
                C_csr = parse_matrix(df.iloc[i + 1]['name'])
                M = min(A_csr.size(0), B_csr.size(0), C_csr.size(0))
                N = max(A_csr.size(1), B_csr.size(1), C_csr.size(1))

                A_t = torch.sparse_csr_tensor(
                    A_csr.crow_indices()[:M+1],
                    A_csr.col_indices()[:A_csr.crow_indices()[M]],
                    A_csr.values()[:A_csr.crow_indices()[M]], (M, N))
                B_t = torch.sparse_csr_tensor(
                    B_csr.crow_indices()[:M+1],
                    B_csr.col_indices()[:B_csr.crow_indices()[M]],
                    B_csr.values()[:B_csr.crow_indices()[M]], (M, N))
                C_t = torch.sparse_csr_tensor(
                    C_csr.crow_indices()[:M+1],
                    C_csr.col_indices()[:C_csr.crow_indices()[M]],
                    C_csr.values()[:C_csr.crow_indices()[M]], (M, N))

                A_CSR = nacho_runtime.CSR(A_t.crow_indices(), A_t.col_indices(), A_t.values(),
                                           torch.tensor([M, N], dtype=torch.int32))
                B_CSR = nacho_runtime.CSR(B_t.crow_indices(), B_t.col_indices(), B_t.values(),
                                           torch.tensor([M, N], dtype=torch.int32))
                C_CSR = nacho_runtime.CSR(C_t.crow_indices(), C_t.col_indices(), C_t.values(),
                                           torch.tensor([M, N], dtype=torch.int32))

                # Fused nacho (subprocess for abort-safety)
                nacho_fused_ms = _subprocess_nacho_timing(
                    df.iloc[i-1]['name'], df.iloc[i]['name'], 'csr_3', M, N,
                    matrix_c=df.iloc[i+1]['name'])
                if nacho_fused_ms is None:
                    tqdm.write(f"nacho CSR 3-way fused crash at {i}, skipping")

                # Unfused nacho (subprocess for abort-safety)
                nacho_unfused_ms = _subprocess_nacho_timing(
                    df.iloc[i-1]['name'], df.iloc[i]['name'], 'csr_3_unfused', M, N,
                    matrix_c=df.iloc[i+1]['name'])
                if nacho_unfused_ms is None:
                    tqdm.write(f"nacho CSR 3-way unfused crash at {i}, skipping")

                # Unfused cuSPARSE (direct call)
                _, cusparse_unfused_ms = csr_add_3_cusparse_unfused(A_CSR, B_CSR, C_CSR)

                total_nnz = (A_t.crow_indices().max() + B_t.crow_indices().max()
                             + C_t.crow_indices().max()).item()
                row.update({
                    "matrix_c": df.iloc[i + 1]['name'],
                    "nnz_csr_3way": total_nnz,
                    "csr_3way_fused_ms": nacho_fused_ms,
                    "csr_3way_unfused_ms": nacho_unfused_ms,
                    "csr_3way_cusparse_ms": cusparse_unfused_ms,
                })
            except (RuntimeError, MemoryError) as e:
                tqdm.write(f"CSR 3-way skip {i}: {e}")
                torch.cuda.empty_cache()

        existing[i] = row
        # Save every row — nanobind aborts on large matrices can't be caught
        _save_rows(list(existing.values()), csv_name)

    _save_rows(list(existing.values()), csv_name)

    if save_and_plot and existing:
        rdf = pd.DataFrame(list(existing.values()))
        # CSR add plot: nacho vs cusparse vs pytorch
        if 'nnz_csr' in rdf.columns:
            csr_rows = rdf.dropna(subset=["nnz_csr"])
            if not csr_rows.empty:
                plot(csr_rows["nnz_csr"].tolist(),
                     csr_rows["csr_nacho_ms"].tolist(),
                     csr_rows["csr_cusparse_ms"].tolist(),
                     csr_rows["csr_pytorch_ms"].tolist(),
                     f"nacho_csr_add_{start}-{end}",
                     labels=("Nacho", "cuSPARSE", "PyTorch"))
        # COO add plot: nacho vs manual vs pytorch
        if 'nnz_coo' in rdf.columns:
            coo_rows = rdf.dropna(subset=["nnz_coo"])
            if not coo_rows.empty:
                plot(coo_rows["nnz_coo"].tolist(),
                     coo_rows["coo_nacho_ms"].tolist(),
                     [],
                     coo_rows["coo_pytorch_ms"].tolist(),
                     f"nacho_coo_add_{start}-{end}",
                     labels=("Nacho", "", "PyTorch"))
        # COO mul plot: nacho vs pytorch
        if 'nnz_coo_mul' in rdf.columns:
            coo_mul_rows = rdf.dropna(subset=["nnz_coo_mul"])
            if not coo_mul_rows.empty:
                plot(coo_mul_rows["nnz_coo_mul"].tolist(),
                     coo_mul_rows["coo_mul_nacho_ms"].tolist(),
                     [],
                     coo_mul_rows["coo_mul_pytorch_ms"].tolist(),
                     f"nacho_coo_mul_{start}-{end}",
                     labels=("Nacho", "", "PyTorch"))
        # CSR 3-way add plot: fused nacho vs unfused nacho vs unfused cuSPARSE
        if 'nnz_csr_3way' in rdf.columns:
            csr3_rows = rdf.dropna(subset=["nnz_csr_3way"])
            if not csr3_rows.empty:
                plot(csr3_rows["nnz_csr_3way"].tolist(),
                     csr3_rows["csr_3way_fused_ms"].tolist(),
                     csr3_rows["csr_3way_unfused_ms"].tolist(),
                     csr3_rows["csr_3way_cusparse_ms"].tolist(),
                     f"nacho_csr_3way_add_{start}-{end}",
                     labels=("Nacho Fused", "Nacho Unfused", "cuSPARSE Unfused"))


def _replot_from_csv(csv_name, benchmark):
    """Regenerate plots from an existing CSV without re-running benchmarks."""
    from plotter import plot, plot_bar_graph_2

    path = _csv_path(csv_name)
    if not os.path.isfile(path):
        print(f"CSV not found: {path}")
        return
    rdf = pd.read_csv(path)
    print(f"Replotting {benchmark} from {path} ({len(rdf)} rows)")

    if benchmark == 'csr_add':
        plot(rdf["nnz"].tolist(), rdf["manual_ms"].tolist(),
             rdf["cusparse_ms"].tolist(), rdf["pytorch_ms"].tolist(), csv_name)
    elif benchmark == 'coo_add':
        plot(rdf["nnz"].tolist(), rdf["manual_ms"].tolist(),
             [], rdf["pytorch_ms"].tolist(), csv_name)
    elif benchmark == 'spgemm':
        plot(rdf["nnz"].tolist(), rdf["manual_ms"].tolist(),
             rdf["cusparse_ms"].tolist(), [], csv_name)
    elif benchmark == 'sparse_vectors':
        full_lb = rdf[["full_mergepath_ms", "full_precompute_ms", "full_compute_ms"]].values.tolist()
        partial_lb = rdf[["partial_mergepath_ms", "partial_precompute_ms", "partial_compute_ms"]].values.tolist()
        single_lb = rdf[["nofusion_mergepath_ms", "nofusion_precompute_ms", "nofusion_compute_ms"]].values.tolist()
        plot_bar_graph_2(0, rdf["total_nnz"].tolist(), full_lb, partial_lb, single_lb, csv_name)
    elif benchmark == 'nacho_comparison':
        if 'nnz_csr' in rdf.columns:
            csr = rdf.dropna(subset=["nnz_csr"])
            if not csr.empty:
                plot(csr["nnz_csr"].tolist(), csr["csr_nacho_ms"].tolist(),
                     csr["csr_cusparse_ms"].tolist(), csr["csr_pytorch_ms"].tolist(),
                     csv_name.replace("nacho_comparison", "nacho_csr_add"),
                     labels=("Nacho", "cuSPARSE", "PyTorch"))
                # Nacho vs cuSPARSE only
                csr2 = rdf.dropna(subset=["nnz_csr", "csr_nacho_ms", "csr_cusparse_ms"])
                if not csr2.empty:
                    plot(csr2["nnz_csr"].tolist(), csr2["csr_nacho_ms"].tolist(),
                         csr2["csr_cusparse_ms"].tolist(), [],
                         csv_name.replace("nacho_comparison", "nacho_vs_cusparse_csr_add"),
                         labels=("Nacho", "cuSPARSE", ""))
        if 'nnz_coo' in rdf.columns:
            coo = rdf.dropna(subset=["nnz_coo"])
            if not coo.empty:
                plot(coo["nnz_coo"].tolist(), coo["coo_nacho_ms"].tolist(),
                     [], coo["coo_pytorch_ms"].tolist(),
                     csv_name.replace("nacho_comparison", "nacho_coo_add"),
                     labels=("Nacho", "", "PyTorch"))
        if 'nnz_coo_mul' in rdf.columns:
            coo_mul = rdf.dropna(subset=["nnz_coo_mul"])
            if not coo_mul.empty:
                plot(coo_mul["nnz_coo_mul"].tolist(), coo_mul["coo_mul_nacho_ms"].tolist(),
                     [], coo_mul["coo_mul_pytorch_ms"].tolist(),
                     csv_name.replace("nacho_comparison", "nacho_coo_mul"),
                     labels=("Nacho", "", "PyTorch"))
        if 'nnz_csr_3way' in rdf.columns:
            csr3 = rdf.dropna(subset=["nnz_csr_3way"])
            if not csr3.empty:
                plot(csr3["nnz_csr_3way"].tolist(),
                     csr3["csr_3way_fused_ms"].tolist(),
                     csr3["csr_3way_unfused_ms"].tolist(),
                     csr3["csr_3way_cusparse_ms"].tolist(),
                     csv_name.replace("nacho_comparison", "nacho_csr_3way_add"),
                     labels=("Nacho Fused", "Nacho Unfused", "cuSPARSE Unfused"))


BENCHMARKS = {
    'csr_add':           run_csr_add,
    'coo_add':           run_coo_add,
    'spgemm':            run_spgemm,
    'sparse_vectors':    run_sparse_vectors,
    'broadcast':         run_broadcast,
    'nacho_comparison':  run_nacho_comparison,
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
  broadcast         Broadcast x*A correctness + timing
  nacho_comparison  Nacho-generated CSR add, COO add & COO mul vs cuSPARSE & PyTorch
  all               Run all benchmarks (default)

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
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output CSV name (without .csv extension). Overrides the default '
             '<benchmark>_<start>-<end> naming scheme. Continue mode resumes '
             'from this file.',
    )
    parser.add_argument(
        '--replot', action='store_true',
        help='Regenerate plots from existing CSV data without running benchmarks',
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

    if args.replot:
        for name in to_run:
            if name == 'broadcast':
                continue
            csv_name = args.output or f"{name}_{start}-{end}"
            _replot_from_csv(csv_name, name)
        print("\nDone.")
        return

    print(f"Benchmarks: {', '.join(to_run)}")
    print(f"Matrix range: {start}–{end} (of {num_matrices} available)")
    if not args.continue_mode:
        print("Fresh run: ignoring any existing CSV results")
    if not save_and_plot:
        print("Plots disabled")
    print()

    for name in to_run:
        BENCHMARKS[name](start, end, save_and_plot, continue_mode=args.continue_mode,
                         csv_name=args.output)

    print("\nDone.")


if __name__ == '__main__':
    main()
