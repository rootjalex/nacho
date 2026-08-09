"""Elementwise COO multiply: the generated nacho kernel against PyTorch sparse.

Walks consecutive pairs in the SuiteSparse list as coalesced COO tensors over a common
shape, and plots runtime against the combined nnz of the two operands.

A product cancels wherever both operands are non-zero but the values multiply to zero.
Nacho keeps those positions stored; torch prunes them. Both sides are pruned before the
correctness comparison, outside the timed region.

    python benchmarks/coo_mul.py --device cpu --start 0 --end 1600
"""

import torch

import nacho

from common.compare import coo_equal, prune_zeros, summarize
from common.parser import matrix_list, parse_matrix
from common.plotter import plot_scatter
from common.timing import flush_gpu_state, parse_sweep_args, timer_for


def _load_coo_pair(df, i, device):
    """Both matrices as coalesced torch COO tensors over one shared shape."""
    A = parse_matrix(df.iloc[i - 1]["name"], return_coo=True, device=device).coalesce()
    B = parse_matrix(df.iloc[i]["name"], return_coo=True, device=device).coalesce()

    rows = max(A.size(0), B.size(0))
    cols = max(A.size(1), B.size(1))
    return (torch.sparse_coo_tensor(A.indices(), A.values(), (rows, cols)).coalesce(),
            torch.sparse_coo_tensor(B.indices(), B.values(), (rows, cols)).coalesce(),
            rows, cols)


def benchmark_coo_mul(start, end, device="cpu", save_and_plot=True):
    """Nacho vs PyTorch elementwise COO multiply over consecutive matrix pairs."""
    on_gpu = device == "cuda"
    kernel = nacho.gpu_coo_mul_f32 if on_gpu else nacho.cpu_coo_mul_f32
    measure = timer_for(device)

    df = matrix_list()
    nnz_totals, nacho_runtimes, pytorch_runtimes, failed = [], [], [], []

    for i in range(start + 1, end):
        print(f"\nIteration {i}")
        A_torch, B_torch, rows, cols = _load_coo_pair(df, i, device)
        A_coo = nacho.to_coo(A_torch, device)
        B_coo = nacho.to_coo(B_torch, device)

        nnz_a, nnz_b = A_coo.values.numel(), B_coo.values.numel()
        print(f"  M={rows}  N={cols}  nnzA={nnz_a}  nnzB={nnz_b}")
        print(f"  A={df.iloc[i-1]['name']}  B={df.iloc[i]['name']}")

        reference, pytorch_ms = measure(lambda: (A_torch * B_torch).coalesce())
        result, nacho_ms = measure(lambda: kernel(A_coo, B_coo))

        correct = coo_equal(result, prune_zeros(reference), prune=True)
        print(f"  Nacho    {nacho_ms:.4f} ms   correct={correct}")
        print(f"  PyTorch  {pytorch_ms:.4f} ms   speedup={pytorch_ms/nacho_ms:.3f}x")

        if not correct:
            print(f"  FAILED at {i}")
            failed.append(i)

        nnz_totals.append(nnz_a + nnz_b)
        nacho_runtimes.append(nacho_ms)
        pytorch_runtimes.append(pytorch_ms)

        del A_torch, B_torch, A_coo, B_coo, reference, result
        if on_gpu:
            flush_gpu_state()

    tag = "gpu" if on_gpu else "cpu"
    if save_and_plot:
        plot_scatter(f"coo_mul_{tag}_{start}-{end}", nnz_totals,
                     "Total nnz (nnzA + nnzB)", nacho_runtimes,
                     pytorch=pytorch_runtimes)

    summarize(f"{tag} coo_mul {start}-{end}", nacho_runtimes,
              "PyTorch", pytorch_runtimes, failed)
    return failed


def main():
    args = parse_sweep_args(__doc__.splitlines()[0])
    for device in args.devices:
        benchmark_coo_mul(args.start, args.end, device=device,
                          save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
