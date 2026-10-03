"""Sparse matrix vector multiplication: the generated kernel against:
- cuSPARSE
- pyTorch's SpMV.

Contracts j out of a[i,j] * b[j]. This is a column-major format traversal of the
sparse matrix. Outputs to a dense vector y[i]. This is because cuSPARSE produces
a dense output, so we designed Nacho's SpMV kernel to similarly produce a dense
output vector.

    python benchmarks/spmv.py --start 0 --end 1300
"""

import torch

import nacho

from common.compare import summarize
from common.parser import matrix_list, parse_matrix
from common.plotter import plot_scatter
from common.timing import flush_gpu_state, gpu_time, launch_args, parse_sweep_args
from common.csr import to_baseline_csr

# Products whose intermediate does not fit on the device. They are listed rather than
# detected because a product that runs out cannot release what it already allocated, so
# every later iteration would see less memory than the one before.
SKIP_INDICES = frozenset()


def dense_failure_reason(result, reference):
    """Print the largest difference between two dense results."""
    error = (result - reference).abs()
    index = error.argmax().item()
    print(
        f"  max error at {index}: "
        f"nacho={result[index]}  "
        f"ref={reference[index]}  "
        f"abs_error={error[index]}"
    )


def dense_vector(length, device="cuda"):
    """Create a dense float32 vector."""

    return torch.randn(length, dtype=torch.float32, device=device)


def _time_product(label, run):
    """(result, milliseconds), or (None, None) if the product could not be computed."""
    try:
        return gpu_time(run)
    except (RuntimeError, MemoryError) as error:
        print(f"  {label:<9} did not complete: {error}")
        return None, None


def _measure_pair(A_coo, x_torch, launch):
    """Time nacho and pyTorch on one product and report whether they agree."""

    # ------------------------------------------------------------------
    # Matrix/vector representations for Nacho, cuSPARSE, and pyTorch
    # ------------------------------------------------------------------

    # Change COO(A) to CSR(A^T) to represent CSC(A) for Nacho
    # CSR(A^T) represents CSC(A) for Nacho
    A_csr_t = A_coo.transpose(0, 1).coalesce().to_sparse_csr()

    # Nacho: representation of CSC(A).
    A_nacho = nacho.to_csr(A_csr_t, "cuda")

    # cuSPARSE: baseline representation of CSC(A).
    A_cusparse = to_baseline_csr(A_csr_t, "cuda")

    # Nacho: Convert torch dense vector to nacho DenseVector_gpu
    x_nacho = nacho.to_dense_vector(x_torch, "cuda")

    # PyTorch: Change COO(A) to CSC(A)
    A_torch = A_coo.to_sparse_csc()

    # torch.sparse.mm expects a 2D tensor, so we unsqueeze the vector to make
    # it a column vector and then squeeze the result back to a 1D tensor.
    # We don't include this in the timing because it's not part of the actual
    # SpMV operation.
    x_matrix = x_torch.unsqueeze(1)

    # ------------------------------------------------------------------
    # Nacho
    # ------------------------------------------------------------------

    result_nacho, nacho_ms = _time_product(
        "nacho", lambda: nacho.gpu_spmv_f32(A_nacho, x_nacho, *launch)
    )

    # Extract Nacho output values as a PyTorch (m,) tensor
    if result_nacho is not None:
        result_nacho = result_nacho.values

    # ------------------------------------------------------------------
    # PyTorch
    # ------------------------------------------------------------------
    reference, torch_ms = _time_product(
        "torch", lambda: torch.sparse.mm(A_torch, x_matrix)
    )

    # PyTorch returns an (m, 1) column vector. Flatten it to match Nacho's (m,) output.
    if reference is not None:
        reference = reference.squeeze(1)

    # ------------------------------------------------------------------
    # cuSPARSE
    # ------------------------------------------------------------------

    # output as a (m,) tensor
    result_cusparse, cusparse_ms = _time_product(
        "cusparse",
        lambda: nacho.gpu_spmv_cusparse_f32(
            A_cusparse,
            x_torch,
        ),
    )

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    correct = True

    # Nacho vs PyTorch
    if result_nacho is not None and reference is not None:
        # default tolerances  atol=1e-8
        nacho_correct = torch.allclose(
            result_nacho,
            reference,
            rtol=1e-4,
            atol=1e-5,
        )
        print(f"  nacho     {nacho_ms:.4f} ms   correct={nacho_correct}")
        print(f"  torch     {torch_ms:.4f} ms   speedup={torch_ms/nacho_ms:.3f}x")
        if not nacho_correct:
            dense_failure_reason(result_nacho, reference)

        correct &= nacho_correct

    # cuSPARSE vs PyTorch
    if result_cusparse is not None and reference is not None:
        cusparse_correct = torch.allclose(
            # default tolerances: rtol=1e-5, atol=1e-8
            result_cusparse,
            reference,
            rtol=1e-4,
            atol=1e-5,
        )
        print(f"  cusparse  {cusparse_ms:.4f} ms   speedup={cusparse_ms/nacho_ms:.3f}x")
        if not cusparse_correct:
            dense_failure_reason(result_cusparse, reference)

        correct &= cusparse_correct

    del (
        A_csr_t,
        A_nacho,
        A_torch,
        A_cusparse,
        x_nacho,
        x_torch,
        x_matrix,
        reference,
        result_cusparse,
        result_nacho,
    )
    # flush_gpu_state()
    return (nacho_ms, cusparse_ms, torch_ms, correct)


def benchmark_spmv(start, end, save_and_plot=True):
    """Nacho vs pyTorch on a sweep of sparse matrices. Returns a list of
    (matrix, nacho_ms, torch_ms, correct)."""

    launch = launch_args("cuda")
    df = matrix_list()

    nnz_totals, nacho_runtimes, pytorch_runtimes, cusparse_runtimes, failed = (
        [],
        [],
        [],
        [],
        [],
    )

    def record(nnz, nacho_ms, cusparse_ms, pytorch_ms, correct, index):
        nnz_totals.append(nnz)
        nacho_runtimes.append(nacho_ms)
        cusparse_runtimes.append(cusparse_ms)
        pytorch_runtimes.append(pytorch_ms)
        if not correct:
            print(f"  FAILED at {index}")
            failed.append(index)

    for i in range(start, end):
        print(f"\nIteration {i}")
        if i in SKIP_INDICES:
            print("  skipped, intermediate does not fit on the device")
            continue

        flush_gpu_state()

        # We start with a COO representation of the matrix because it's the most
        # general and can be converted to any other format. We then convert it
        # to the appropriate formats for Nacho, cuSPARSE, and PyTorch.
        A_coo = parse_matrix(df.iloc[i]["name"], return_coo=True)
        x = dense_vector(A_coo.shape[1], device="cuda")

        nacho_ms, cusparse_ms, pytorch_ms, correct = _measure_pair(A_coo, x, launch)
        record(A_coo._nnz(), nacho_ms, cusparse_ms, pytorch_ms, correct, i)

        del A_coo, x
        # flush_gpu_state()

    # Plotting and summarizing results
    if save_and_plot:
        plot_scatter(
            f"spmv_gpu_{start}-{end}",
            nnz_totals,
            "nnz(A)",
            nacho=nacho_runtimes,
            cusparse=cusparse_runtimes,
            pytorch=pytorch_runtimes,
        )

    # Nacho vs PyTorch
    both = [
        (n, c)
        for n, c in zip(nacho_runtimes, pytorch_runtimes)
        if n is not None and c is not None
    ]
    summarize(
        f"gpu spmv {start}-{end}",
        [n for n, _ in both],
        "pyTorch",
        [c for _, c in both],
        failed,
    )

    # Nacho vs cuSPARSE
    both = [
        (n, c)
        for n, c in zip(nacho_runtimes, cusparse_runtimes)
        if n is not None and c is not None
    ]
    summarize(
        f"gpu spmv {start}-{end}",
        [n for n, _ in both],
        "cuSPARSE",
        [c for _, c in both],
        failed,
    )

    print(
        f"  incomplete: nacho {sum(t is None for t in nacho_runtimes)}, "
        f"pytorch {sum(t is None for t in pytorch_runtimes)}, "
        f"cuSPARSE {sum(t is None for t in cusparse_runtimes)}, "
        f"of {len(nnz_totals)} products"
    )

    return failed


def main():
    args = parse_sweep_args(__doc__.splitlines()[0], end_default=1300, gpu_only=True)
    benchmark_spmv(args.start, args.end, save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
