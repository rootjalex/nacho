"""Two-way CSR addition: the generated nacho kernel against its comparison points.

Walks consecutive pairs in the SuiteSparse list, clipping each pair to a common shape,
and plots runtime against the combined nnz of the two operands.

Series per device:
  GPU  nacho, cuSPARSE, Taco
  CPU  nacho, Taco, Intel MKL (PyTorch's CPU sparse backend)

Taco here is the row-parallel kernel in baselines/: the same addition with no merge-path
partitioning of work between threads, which is what Taco generates.

    python benchmarks/csr_add.py --device cpu --start 0 --end 1600
"""

import torch

import nacho

from common.compare import csr_equal, csr_failure_reason, summarize
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


def _to_baseline_csr(csr, device):
    """The same buffers as a baseline CSR, which the hand-written kernels take."""
    cls = nacho.BaselineCSR_cpu if device == "cpu" else nacho.BaselineCSR_gpu
    return cls(
        csr.crow_indices().to(dtype=torch.int32, device=device),
        csr.col_indices().to(dtype=torch.int32, device=device),
        csr.values().to(dtype=torch.float32, device=device),
        torch.tensor([csr.shape[0], csr.shape[1]], dtype=torch.int32),
    )


def benchmark_csr_add(start, end, device="cpu", save_and_plot=True):
    """Nacho against cuSPARSE and Taco on the GPU, Taco and Intel MKL on the CPU."""
    on_gpu = device == "cuda"
    kernel = nacho.gpu_csr_add_f32 if on_gpu else nacho.cpu_csr_add_f32
    taco_kernel = nacho.gpu_csr_add_taco_f32 if on_gpu else nacho.cpu_csr_add_taco_f32
    measure = timer_for(device)
    launch = launch_args(device)

    df = matrix_list()
    nnz_totals, nacho_runtimes, failed = [], [], []
    taco_runtimes, cusparse_runtimes, mkl_runtimes = [], [], []

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
        A_base = _to_baseline_csr(A_torch, device)
        B_base = _to_baseline_csr(B_torch, device)

        # An add touches every non-zero of both operands, so their sum is the work done.
        combined_nnz = int((A_torch.crow_indices() + B_torch.crow_indices()).max().item())
        print(f"  M={rows}  N={cols}  nnz={combined_nnz}")
        print(f"  A={df.iloc[i-1]['name']}  B={df.iloc[i]['name']}")

        reference, torch_ms = measure(lambda: A_torch + B_torch)
        result, nacho_ms = measure(lambda: kernel(A_csr, B_csr, *launch))
        _, taco_ms = measure(lambda: taco_kernel(A_base, B_base))

        correct = csr_equal(result, reference)
        print(f"  Nacho     {nacho_ms:.4f} ms   correct={correct}")
        print(f"  Taco      {taco_ms:.4f} ms   speedup={taco_ms/nacho_ms:.3f}x")

        if on_gpu:
            _, cusparse_ms = measure(lambda: nacho.gpu_csr_add_cusparse_f32(A_base, B_base))
            cusparse_runtimes.append(cusparse_ms)
            print(f"  cuSPARSE  {cusparse_ms:.4f} ms   speedup={cusparse_ms/nacho_ms:.3f}x")
        else:
            # torch's CPU sparse ops go through Intel MKL, so this is the MKL series.
            mkl_runtimes.append(torch_ms)
            print(f"  IntelMKL  {torch_ms:.4f} ms   speedup={torch_ms/nacho_ms:.3f}x")

        if not correct:
            print(f"  FAILED at {i}")
            failed.append(i)
            csr_failure_reason(result, reference)

        nnz_totals.append(combined_nnz)
        nacho_runtimes.append(nacho_ms)
        taco_runtimes.append(taco_ms)

        del A_torch, B_torch, A_csr, B_csr, A_base, B_base, reference, result
        if on_gpu:
            flush_gpu_state()

    tag = "gpu" if on_gpu else "cpu"
    if save_and_plot:
        plot_scatter(f"csr_add_{tag}_{start}-{end}", nnz_totals,
                     "Total nnz (nnzA + nnzB)", nacho_runtimes,
                     cusparse=cusparse_runtimes or None,
                     pytorch=mkl_runtimes or None,
                     taco=taco_runtimes,
                     pytorch_as_mkl=True)

    summarize(f"{tag} csr_add {start}-{end}", nacho_runtimes,
              "Taco", taco_runtimes, failed)
    if on_gpu:
        summarize(f"{tag} csr_add {start}-{end}", nacho_runtimes,
                  "cuSPARSE", cusparse_runtimes, failed)
    else:
        summarize(f"{tag} csr_add {start}-{end}", nacho_runtimes,
                  "Intel MKL", mkl_runtimes, failed)
    return failed


def main():
    args = parse_sweep_args(__doc__.splitlines()[0])
    for device in args.devices:
        benchmark_csr_add(args.start, args.end, device=device,
                          save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
