"""Masked sparse matrix product: the fused kernel against unfused equivalents.

Computes c[i,k] * sum_j a[i,j] * b[j,k] as one generated kernel, so the product is only
formed where the mask has an entry and the intermediate A @ B is never materialised in
full. The comparisons materialise it:

  Nacho unfused   spgemm(A, B) then csr_mul with C
  cuSPARSE        torch.sparse.mm(A, B) then an elementwise multiply

Each iteration measures up to two products: A, A, A when A is square, and the consecutive
triple with each operand sliced so the dimensions line up. A triple that cannot be
reconciled is skipped.

Generated for the GPU only, and the default range stops short of the full matrix list for
the same reason as spgemm: the intermediate is sized by the multiply count.

    python benchmarks/sssmm.py --start 0 --end 1300
"""

import torch

import nacho

from common.compare import csr_allclose, summarize
from common.parser import matrix_list, parse_matrix
from common.plotter import plot_scatter
from common.timing import flush_gpu_state, gpu_time, launch_args, parse_sweep_args


def _time_product(label, run):
    """(result, milliseconds), or (None, None) if the product could not be computed."""
    try:
        return gpu_time(run)
    except (RuntimeError, MemoryError) as error:
        print(f"  {label:<14} did not complete: {error}")
        return None, None


def _measure_triple(A_torch, B_torch, C_torch, launch):
    """Time the fused kernel and both unfused routes on one masked product."""
    A_csr = nacho.to_csr(A_torch, "cuda")
    B_csr = nacho.to_csr(B_torch, "cuda")
    C_csr = nacho.to_csr(C_torch, "cuda")

    fused, fused_ms = _time_product(
        "nacho fused", lambda: nacho.gpu_sssmm_f32(A_csr, B_csr, C_csr, *launch))
    _, unfused_ms = _time_product(
        "nacho unfused",
        lambda: nacho.gpu_csr_mul_f32(nacho.gpu_spgemm_f32(A_csr, B_csr, *launch),
                                      C_csr, *launch))
    reference, cusparse_ms = _time_product(
        "cuSPARSE", lambda: torch.sparse.mm(A_torch, B_torch) * C_torch)

    correct = True
    if fused is not None and reference is not None:
        correct = csr_allclose(fused, reference.to_sparse_csr())
        print(f"  nacho fused    {fused_ms:.4f} ms   correct={correct}")
        if unfused_ms is not None:
            print(f"  nacho unfused  {unfused_ms:.4f} ms   "
                  f"speedup_fused={unfused_ms/fused_ms:.3f}x")
        print(f"  cuSPARSE       {cusparse_ms:.4f} ms   "
              f"speedup_fused={cusparse_ms/fused_ms:.3f}x")

    del A_csr, B_csr, C_csr, fused, reference
    return fused_ms, unfused_ms, cusparse_ms, correct


def _slice_rows(matrix, rows, cols):
    """The matrix restricted to its first `rows` rows and declared `cols` wide."""
    nnz = matrix.crow_indices()[rows]
    return torch.sparse_csr_tensor(
        matrix.crow_indices()[:rows + 1],
        matrix.col_indices()[:nnz],
        matrix.values()[:nnz], (rows, cols))


def benchmark_sssmm(start, end, save_and_plot=True):
    """Fused vs unfused vs cuSPARSE over the matrix list."""
    launch = launch_args("cuda")
    df = matrix_list()

    nnz_totals, fused_times, unfused_times, cusparse_times, failed = [], [], [], [], []

    def record(nnz, fused_ms, unfused_ms, cusparse_ms, correct, index):
        nnz_totals.append(nnz)
        fused_times.append(fused_ms)
        unfused_times.append(unfused_ms)
        cusparse_times.append(cusparse_ms)
        if not correct:
            print(f"  FAILED at {index}")
            failed.append(index)

    for i in range(start + 1, end - 1):
        flush_gpu_state()
        print(f"\nIteration {i}")
        A = parse_matrix(df.iloc[i - 1]["name"])
        B = parse_matrix(df.iloc[i]["name"])
        C = parse_matrix(df.iloc[i + 1]["name"])

        if A.size(0) == A.size(1):
            print(f"  A, A, A  {df.iloc[i-1]['name']}  M={A.size(0)}")
            record(3 * int(A.crow_indices().max()),
                   *_measure_triple(A, A, A, launch), i)
            flush_gpu_state()

        inner = min(A.size(1), B.size(0))
        if inner != A.size(1):
            print("  A, B, C  skipped, inner dimensions do not match")
            del A, B, C
            continue
        rows = min(A.size(0), C.size(0))
        cols = max(B.size(1), C.size(1))

        A_shaped = _slice_rows(A, rows, inner)
        B_shaped = _slice_rows(B, inner, cols)
        C_shaped = _slice_rows(C, rows, cols)

        print(f"  A, B, C  M={rows}  K={inner}  N={cols}")
        record(int(A_shaped.crow_indices().max()) + int(B_shaped.crow_indices().max())
               + int(C_shaped.crow_indices().max()),
               *_measure_triple(A_shaped, B_shaped, C_shaped, launch), i)

        del A, B, C, A_shaped, B_shaped, C_shaped
        flush_gpu_state()

    if save_and_plot:
        plot_scatter(f"sssmm_gpu_{start}-{end}", nnz_totals,
                     "Total nnz (nnzA + nnzB + nnzC)", fused_times,
                     cusparse=cusparse_times, unfused=unfused_times)

    for label, series in (("Nacho unfused", unfused_times), ("cuSPARSE", cusparse_times)):
        both = [(f, o) for f, o in zip(fused_times, series)
                if f is not None and o is not None]
        summarize(f"gpu sssmm {start}-{end}", [f for f, _ in both],
                  label, [o for _, o in both], failed)
    print(f"  incomplete: fused {sum(t is None for t in fused_times)}, "
          f"unfused {sum(t is None for t in unfused_times)}, "
          f"cuSPARSE {sum(t is None for t in cusparse_times)}, "
          f"of {len(nnz_totals)} products")
    return failed


def main():
    args = parse_sweep_args(__doc__.splitlines()[0], end_default=1300, gpu_only=True)
    benchmark_sssmm(args.start, args.end, save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
