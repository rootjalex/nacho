"""Elementwise CSR multiply: the generated nacho kernel against PyTorch sparse.

Walks consecutive pairs in the SuiteSparse list, clipping each pair to a common shape,
and plots runtime against the combined nnz of the two operands.

A product cancels wherever both operands are non-zero but the values multiply to zero.
Nacho keeps those positions stored; torch prunes them. Both sides are pruned before the
correctness comparison.

    python benchmarks/csr_mul.py --device cpu --start 0 --end 1600
"""

import torch

import nacho

from common.compare import csr_as_coo_equal, prune_zeros, summarize
from common.parser import matrix_list, parse_matrix
from common.plotter import plot_scatter
from common.timing import flush_gpu_state, launch_args, parse_sweep_args, timer_for


def to_int32_if_safe(tensor):
    if tensor.numel() == 0 or tensor.max().item() <= torch.iinfo(torch.int32).max:
        return tensor.to(torch.int32)
    return tensor


def _clip_to_common_shape(sparse, rows, cols, to_cpu=False):
    """Restrict a CSR matrix to its first `rows` rows and declare it `cols` wide."""
    nnz = sparse.crow_indices()[rows]
    indptr = sparse.crow_indices()[:rows + 1]
    indices = sparse.col_indices()[:nnz]
    values = sparse.values()[:nnz]
    if to_cpu:
        indptr, indices, values = indptr.cpu(), indices.cpu(), values.cpu()
    return torch.sparse_csr_tensor(to_int32_if_safe(indptr), to_int32_if_safe(indices),
                                   values, (rows, cols))


def benchmark_csr_mul(start, end, device="cpu", save_and_plot=True):
    """Nacho vs PyTorch elementwise CSR multiply over consecutive matrix pairs."""
    on_gpu = device == "cuda"
    kernel = nacho.gpu_csr_mul_f32 if on_gpu else nacho.cpu_csr_mul_f32
    measure = timer_for(device)
    launch = launch_args(device)

    df = matrix_list()
    nnz_totals, nacho_runtimes, pytorch_runtimes, failed = [], [], [], []

    for i in range(start + 1, end):
        print(f"\nIteration {i}")
        A_raw = parse_matrix(df.iloc[i - 1]["name"], device=device)
        B_raw = parse_matrix(df.iloc[i]["name"], device=device)

        rows = min(A_raw.size(0), B_raw.size(0))
        cols = max(A_raw.size(1), B_raw.size(1))

        A_torch = _clip_to_common_shape(A_raw, rows, cols, to_cpu=not on_gpu)
        B_torch = _clip_to_common_shape(B_raw, rows, cols, to_cpu=not on_gpu)
        A_csr = nacho.to_csr(A_torch, device)
        B_csr = nacho.to_csr(B_torch, device)

        nnz_a, nnz_b = A_csr.values.numel(), B_csr.values.numel()
        print(f"  M={rows}  N={cols}  nnzA={nnz_a}  nnzB={nnz_b}")
        print(f"  A={df.iloc[i-1]['name']}  B={df.iloc[i]['name']}")

        kernel(A_csr, B_csr, *launch)
        reference, pytorch_ms = measure(lambda: A_torch * B_torch)
        result, nacho_ms = measure(lambda: kernel(A_csr, B_csr, *launch))

        correct = csr_as_coo_equal(result, prune_zeros(reference))
        print(f"  Nacho    {nacho_ms:.4f} ms   correct={correct}")
        print(f"  PyTorch  {pytorch_ms:.4f} ms   speedup={pytorch_ms/nacho_ms:.3f}x")

        if not correct:
            print(f"  FAILED at {i}")
            failed.append(i)

        nnz_totals.append(nnz_a + nnz_b)
        nacho_runtimes.append(nacho_ms)
        pytorch_runtimes.append(pytorch_ms)

        del A_torch, B_torch, A_csr, B_csr, reference, result
        if on_gpu:
            flush_gpu_state()

    tag = "gpu" if on_gpu else "cpu"
    if save_and_plot:
        plot_scatter(f"csr_mul_{tag}_{start}-{end}", nnz_totals,
                     "Total nnz (nnzA + nnzB)", nacho_runtimes,
                     pytorch=pytorch_runtimes)

    summarize(f"{tag} csr_mul {start}-{end}", nacho_runtimes,
              "PyTorch", pytorch_runtimes, failed)
    return failed


def main():
    args = parse_sweep_args(__doc__.splitlines()[0])
    for device in args.devices:
        benchmark_csr_mul(args.start, args.end, device=device,
                          save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
