"""Sparse matrix product: the generated kernel against cuSPARSE SpGEMM.

Contracts j out of a[i,j] * b[j,k]. Each iteration measures up to two products: A @ A when
A is square, and A @ B for the consecutive pair, with B sliced to its first K rows so the
inner dimensions line up. A pair whose dimensions cannot be reconciled is skipped.

    python benchmarks/spgemm.py --start 0 --end 1300
"""

import torch

import nacho

from common.compare import csr_structure_equal, csr_structure_failure_reason, summarize
from common.csr import as_nacho_csr, to_baseline_csr
from common.parser import matrix_list, parse_matrix
from common.plotter import plot_scatter
from common.timing import flush_gpu_state, gpu_time, launch_args, parse_sweep_args


# Products whose intermediate does not fit on the device. They are listed rather than
# detected because a product that runs out cannot release what it already allocated, so
# every later iteration would see less memory than the one before.
SKIP_INDICES = frozenset({1057, 1063, 1094, 1100, 1101, 1102, 1103, 1104, 1105,
                          1164, 1213, 1218, 1246, 1247, 1273, 1292})


def _time_product(label, run):
    """(result, milliseconds), or (None, None) if the product could not be computed."""
    try:
        return gpu_time(run)
    except (RuntimeError, MemoryError) as error:
        print(f"  {label:<9} did not complete: {error}")
        return None, None


def _measure_pair(A_torch, B_torch, launch):
    """Time nacho and cuSPARSE on one product and report whether they agree."""
    A_csr = nacho.to_csr(A_torch, "cuda")
    B_csr = nacho.to_csr(B_torch, "cuda")
    A_base = to_baseline_csr(A_torch, "cuda")
    B_base = to_baseline_csr(B_torch, "cuda")

    result, nacho_ms = _time_product("nacho", lambda: nacho.gpu_spgemm_f32(A_csr, B_csr, *launch))
    flush_gpu_state()
    reference, cusparse_ms = _time_product(
        "cuSPARSE", lambda: nacho.gpu_spgemm_cusparse_f32(A_base, B_base))
    flush_gpu_state()

    correct = True
    if result is not None and reference is not None:
        correct = csr_structure_equal(result, as_nacho_csr(reference))
        print(f"  nacho     {nacho_ms:.4f} ms   correct={correct}")
        print(f"  cuSPARSE  {cusparse_ms:.4f} ms   speedup={cusparse_ms/nacho_ms:.3f}x")
        if not correct:
            csr_structure_failure_reason(result, as_nacho_csr(reference))

    del A_csr, B_csr, A_base, B_base, result, reference
    flush_gpu_state()
    return nacho_ms, cusparse_ms, correct


def benchmark_spgemm(start, end, save_and_plot=True):
    """Nacho vs cuSPARSE over the matrix list, squaring and pairing as shapes allow."""
    launch = launch_args("cuda")
    df = matrix_list()

    nnz_totals, nacho_runtimes, cusparse_runtimes, failed = [], [], [], []

    def record(nnz, nacho_ms, cusparse_ms, correct, index):
        nnz_totals.append(nnz)
        nacho_runtimes.append(nacho_ms)
        cusparse_runtimes.append(cusparse_ms)
        if not correct:
            print(f"  FAILED at {index}")
            failed.append(index)

    for i in range(start + 1, end):
        print(f"\nIteration {i}")
        if i in SKIP_INDICES:
            print("  skipped, intermediate does not fit on the device")
            continue

        flush_gpu_state()
        A = parse_matrix(df.iloc[i - 1]["name"])
        B = parse_matrix(df.iloc[i]["name"])

        if A.size(0) == A.size(1):
            print(f"  A x A  {df.iloc[i-1]['name']}  M={A.size(0)}")
            nacho_ms, cusparse_ms, correct = _measure_pair(A, A, launch)
            record(2 * int(A.crow_indices().max()), nacho_ms, cusparse_ms, correct, i)
            flush_gpu_state()

        rows, inner, cols = A.size(0), min(A.size(1), B.size(0)), B.size(1)
        if inner != A.size(1):
            print("  A x B  skipped, inner dimensions do not match")
            del A, B
            continue

        # A is already (rows, inner); B keeps only the rows the product reaches.
        B_sliced = torch.sparse_csr_tensor(
            B.crow_indices()[:inner + 1],
            B.col_indices()[:B.crow_indices()[inner]],
            B.values()[:B.crow_indices()[inner]], (inner, cols))
        A_shaped = torch.sparse_csr_tensor(A.crow_indices(), A.col_indices(), A.values(),
                                           (rows, inner))

        print(f"  A x B  M={rows}  K={inner}  N={cols}")
        nacho_ms, cusparse_ms, correct = _measure_pair(A_shaped, B_sliced, launch)
        record(int(A_shaped.crow_indices().max()) + int(B_sliced.crow_indices().max()),
               nacho_ms, cusparse_ms, correct, i)

        del A, B, A_shaped, B_sliced
        flush_gpu_state()

    if save_and_plot:
        plot_scatter(f"spgemm_gpu_{start}-{end}", nnz_totals, "Total nnz (nnzA + nnzB)",
                     nacho_runtimes, cusparse=cusparse_runtimes)

    both = [(n, c) for n, c in zip(nacho_runtimes, cusparse_runtimes)
            if n is not None and c is not None]
    summarize(f"gpu spgemm {start}-{end}", [n for n, _ in both],
              "cuSPARSE", [c for _, c in both], failed)
    print(f"  incomplete: nacho {sum(t is None for t in nacho_runtimes)}, "
          f"cuSPARSE {sum(t is None for t in cusparse_runtimes)}, "
          f"of {len(nnz_totals)} products")
    return failed


def main():
    args = parse_sweep_args(__doc__.splitlines()[0], end_default=1300, gpu_only=True)
    benchmark_spgemm(args.start, args.end, save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
