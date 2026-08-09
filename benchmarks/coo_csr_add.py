"""Mixed-format addition, CSR + COO -> CSR: the generated nacho kernel against PyTorch.

The two operands are stored differently, so nacho merges a row-offset traversal against
a coordinate traversal in one pass. PyTorch has no mixed-format add, so the reference
converts the COO side to CSR first, and that conversion is part of its measured time.

Plots runtime against the combined nnz of the two operands.

    python benchmarks/coo_csr_add.py --device cpu --start 0 --end 1600
"""

import torch

import nacho

from common.compare import csr_equal, csr_failure_reason, summarize
from common.parser import matrix_list, parse_matrix
from common.plotter import plot_scatter
from common.timing import flush_gpu_state, launch_args, parse_sweep_args, timer_for


def _to_int32_if_safe(tensor):
    if tensor.numel() == 0 or tensor.max().item() <= torch.iinfo(torch.int32).max:
        return tensor.to(torch.int32)
    return tensor


def _load_mixed_pair(df, i, device):
    """The left operand as CSR and the right as COO, over one shared shape."""
    A = parse_matrix(df.iloc[i - 1]["name"], device=device)
    B = parse_matrix(df.iloc[i]["name"], return_coo=True, device=device).coalesce()

    rows = min(A.size(0), B.size(0))
    cols = max(A.size(1), B.size(1))

    nnz = A.crow_indices()[rows]
    A_torch = torch.sparse_csr_tensor(
        _to_int32_if_safe(A.crow_indices()[:rows + 1]),
        _to_int32_if_safe(A.col_indices()[:nnz]),
        A.values()[:nnz], (rows, cols))

    # The COO side has no row offsets to truncate, so drop the entries past the clip.
    within = B.indices()[0] < rows
    B_torch = torch.sparse_coo_tensor(B.indices()[:, within], B.values()[within],
                                      (rows, cols)).coalesce()
    return A_torch, B_torch, rows, cols


def benchmark_coo_csr_add(start, end, device="cpu", save_and_plot=True):
    """Nacho vs PyTorch mixed CSR+COO add over consecutive matrix pairs."""
    on_gpu = device == "cuda"
    kernel = nacho.gpu_coo_csr_add_f32 if on_gpu else nacho.cpu_coo_csr_add_f32
    measure = timer_for(device)
    launch = launch_args(device)

    df = matrix_list()
    nnz_totals, nacho_runtimes, pytorch_runtimes, failed = [], [], [], []

    for i in range(start + 1, end):
        print(f"\nIteration {i}")
        A_torch, B_torch, rows, cols = _load_mixed_pair(df, i, device)
        A_csr = nacho.to_csr(A_torch, device)
        B_coo = nacho.to_coo(B_torch, device)

        nnz_a, nnz_b = A_csr.values.numel(), B_coo.values.numel()
        print(f"  M={rows}  N={cols}  nnzA={nnz_a}  nnzB={nnz_b}")
        print(f"  A={df.iloc[i-1]['name']} (CSR)  B={df.iloc[i]['name']} (COO)")

        reference, pytorch_ms = measure(lambda: A_torch + B_torch.to_sparse_csr())
        result, nacho_ms = measure(lambda: kernel(A_csr, B_coo, *launch))

        correct = csr_equal(result, reference)
        print(f"  Nacho    {nacho_ms:.4f} ms   correct={correct}")
        print(f"  PyTorch  {pytorch_ms:.4f} ms   speedup={pytorch_ms/nacho_ms:.3f}x")

        if not correct:
            print(f"  FAILED at {i}")
            failed.append(i)
            csr_failure_reason(result, reference)

        nnz_totals.append(nnz_a + nnz_b)
        nacho_runtimes.append(nacho_ms)
        pytorch_runtimes.append(pytorch_ms)

        del A_torch, B_torch, A_csr, B_coo, reference, result
        if on_gpu:
            flush_gpu_state()

    tag = "gpu" if on_gpu else "cpu"
    if save_and_plot:
        plot_scatter(f"coo_csr_add_{tag}_{start}-{end}", nnz_totals,
                     "Total nnz (nnzA + nnzB)", nacho_runtimes,
                     pytorch=pytorch_runtimes)

    summarize(f"{tag} coo_csr_add {start}-{end}", nacho_runtimes,
              "PyTorch", pytorch_runtimes, failed)
    return failed


def main():
    args = parse_sweep_args(__doc__.splitlines()[0])
    for device in args.devices:
        benchmark_coo_csr_add(args.start, args.end, device=device,
                              save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
