"""Three-way CSR addition: the fused nacho kernel against two chained two-way adds.

A + B + C is a single generated kernel (csr_add_3), which avoids materializing the
intermediate A + B. The comparison is that intermediate being materialized, by chaining
the generated csr_add twice. PyTorch is used only as the correctness reference.

    python benchmarks/csr_add_3.py --device cpu --start 0 --end 1600
"""

import argparse
import gc
import time as _time

import numpy as np
import torch

import nacho

import config
from common.compare import csr_equal, csr_failure_reason
from common.parser import matrix_list, parse_matrix, to_csr
from common.plotter import plot, plot_scatter


def flush_gpu_state():
    gc.collect()
    torch.cuda.empty_cache()
    torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device="cuda").zero_()
    torch.cuda.synchronize()


def _gpu_time(fn):
    """Run fn() config.ITER_COUNT times; return (last result, trimmed mean ms)."""
    torch.cuda.synchronize()
    times = []
    result = None
    for _ in range(config.ITER_COUNT):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return result, float(np.sort(np.array(times))[config.TRIM:-config.TRIM].mean())


def _cpu_time(fn):
    """Run fn() config.ITER_COUNT times; return (last result, trimmed mean ms)."""
    times = []
    result = None
    for _ in range(config.ITER_COUNT):
        started = _time.perf_counter()
        result = fn()
        times.append((_time.perf_counter() - started) * 1000)
    return result, float(np.sort(np.array(times))[config.TRIM:-config.TRIM].mean())


def time_fused_gpu(A, B, C):
    return _gpu_time(lambda: nacho.gpu_csr_add_3_f32(A, B, C))


def time_unfused_gpu(A, B, C):
    return _gpu_time(lambda: nacho.gpu_csr_add_f32(nacho.gpu_csr_add_f32(A, B), C))


def time_fused_cpu(A, B, C):
    return _cpu_time(lambda: nacho.cpu_csr_add_3_f32(A, B, C))


def time_unfused_cpu(A, B, C):
    return _cpu_time(lambda: nacho.cpu_csr_add_f32(nacho.cpu_csr_add_f32(A, B), C))


def to_int32_if_safe(tensor):
    if tensor.numel() == 0 or tensor.max().item() <= torch.iinfo(torch.int32).max:
        return tensor.to(torch.int32)
    return tensor


def _print_summary(title, fused_runtimes, unfused_runtimes):
    if not fused_runtimes:
        return

    from scipy.stats import gmean

    rule = "-" * 70
    print(f"\n{rule}\n  {title}  ({len(fused_runtimes)} matrix triples)\n{rule}")

    for label, values in (("Nacho fused", np.array(fused_runtimes)),
                          ("Nacho unfused x2", np.array(unfused_runtimes))):
        print(f"    {label:<20} mean={values.mean():.4f}  "
              f"median={np.median(values):.4f}  std={values.std():.4f}")

    speedups = np.array(unfused_runtimes) / np.array(fused_runtimes)
    print("\n  Speedup of fused over unfused:")
    print(f"      Mean:    {speedups.mean():.4f}x")
    print(f"      Geomean: {gmean(speedups):.4f}x")
    print(f"      Median:  {np.median(speedups):.4f}x")
    print(f"      Min:     {speedups.min():.4f}x  (idx {speedups.argmin()})")
    print(f"      Max:     {speedups.max():.4f}x  (idx {speedups.argmax()})")
    print(f"      % faster: {np.mean(speedups > 1.0) * 100:.1f}%")
    print(rule)


def _clip_to_common_shape(sparse, rows, cols):
    """Restrict a CSR matrix to its first `rows` rows and declare it `cols` wide."""
    nnz = sparse.crow_indices()[rows]
    return torch.sparse_csr_tensor(
        sparse.crow_indices()[:rows + 1],
        sparse.col_indices()[:nnz],
        sparse.values()[:nnz],
        (rows, cols),
    )


def benchmark_csr_add_3_gpu(start, end, save_and_plot=True):
    """Fused vs unfused three-way add on the GPU, over consecutive matrix triples."""
    df = matrix_list()

    nnz_totals, fused_runtimes, unfused_runtimes, failed = [], [], [], []

    for i in range(start + 1, end - 1):
        print(f"\nIteration {i}")
        A = parse_matrix(df.iloc[i - 1]["name"])
        B = parse_matrix(df.iloc[i]["name"])
        C = parse_matrix(df.iloc[i + 1]["name"])

        rows = min(A.size(0), B.size(0), C.size(0))
        cols = max(A.size(1), B.size(1), C.size(1))

        A_torch = _clip_to_common_shape(A, rows, cols)
        B_torch = _clip_to_common_shape(B, rows, cols)
        C_torch = _clip_to_common_shape(C, rows, cols)

        A_csr = to_csr(A_torch, "cuda")
        B_csr = to_csr(B_torch, "cuda")
        C_csr = to_csr(C_torch, "cuda")

        nnz_a, nnz_b, nnz_c = (A_csr.values.numel(), B_csr.values.numel(), C_csr.values.numel())
        print(f"  M={rows}  N={cols}  nnzA={nnz_a}  nnzB={nnz_b}  nnzC={nnz_c}")
        print(f"  {df.iloc[i-1]['name']}  +  {df.iloc[i]['name']}  +  {df.iloc[i+1]['name']}")

        reference = A_torch + B_torch + C_torch

        fused, fused_ms = time_fused_gpu(A_csr, B_csr, C_csr)
        unfused, unfused_ms = time_unfused_gpu(A_csr, B_csr, C_csr)

        fused_ok = csr_equal(fused, reference)
        unfused_ok = csr_equal(unfused, reference)

        print(f"  nnz_out={reference.values().numel()}")
        print(f"  Fused nacho      {fused_ms:.4f} ms   correct={fused_ok}")
        print(f"  Unfused nacho x2 {unfused_ms:.4f} ms   correct={unfused_ok}   "
              f"speedup_fused={unfused_ms/fused_ms:.4f}x")

        if not (fused_ok and unfused_ok):
            print(f"  FAILED at {i}")
            failed.append(i)
            if not fused_ok:
                print("    fused:")
                csr_failure_reason(fused, reference)
            if not unfused_ok:
                print("    unfused:")
                csr_failure_reason(unfused, reference)

        nnz_totals.append(nnz_a + nnz_b + nnz_c)
        fused_runtimes.append(fused_ms)
        unfused_runtimes.append(unfused_ms)

        del A_torch, B_torch, C_torch, A_csr, B_csr, C_csr, reference, fused, unfused
        flush_gpu_state()

    if save_and_plot:
        plot_scatter(f"csr_add3_gpu_{start}-{end}", nnz_totals,
                     "Total nnz (nnzA + nnzB + nnzC)", fused_runtimes,
                     unfused=unfused_runtimes)

    _print_summary("GPU 3-way CSR add", fused_runtimes, unfused_runtimes)
    print(f"failed: {failed}")
    return failed


def benchmark_csr_add_3_cpu(start, end, save_and_plot=True):
    """Fused vs unfused three-way add on the CPU, over consecutive matrix triples."""
    df = matrix_list()

    nnz_totals, fused_runtimes, unfused_runtimes, failed = [], [], [], []

    for i in range(start + 1, end - 1):
        print(f"\nIteration {i}")
        A = parse_matrix(df.iloc[i - 1]["name"], device="cpu")
        B = parse_matrix(df.iloc[i]["name"], device="cpu")
        C = parse_matrix(df.iloc[i + 1]["name"], device="cpu")

        rows = min(A.size(0), B.size(0), C.size(0))
        cols = max(A.size(1), B.size(1), C.size(1))

        def _to_cpu_torch(sparse):
            nnz = sparse.crow_indices()[rows]
            return torch.sparse_csr_tensor(
                to_int32_if_safe(sparse.crow_indices()[:rows + 1].cpu()),
                to_int32_if_safe(sparse.col_indices()[:nnz].cpu()),
                sparse.values()[:nnz].cpu(), (rows, cols))

        A_torch = _to_cpu_torch(A)
        B_torch = _to_cpu_torch(B)
        C_torch = _to_cpu_torch(C)

        A_csr = to_csr(A_torch, "cpu")
        B_csr = to_csr(B_torch, "cpu")
        C_csr = to_csr(C_torch, "cpu")

        nnz_a, nnz_b, nnz_c = (A_csr.values.numel(), B_csr.values.numel(), C_csr.values.numel())
        print(f"  M={rows}  N={cols}  nnzA={nnz_a}  nnzB={nnz_b}  nnzC={nnz_c}")
        print(f"  {df.iloc[i-1]['name']}  +  {df.iloc[i]['name']}  +  {df.iloc[i+1]['name']}")

        reference = A_torch + B_torch + C_torch

        fused, fused_ms = time_fused_cpu(A_csr, B_csr, C_csr)
        unfused, unfused_ms = time_unfused_cpu(A_csr, B_csr, C_csr)

        fused_ok = csr_equal(fused, reference)
        unfused_ok = csr_equal(unfused, reference)

        print(f"  Fused nacho      {fused_ms:.4f} ms   correct={fused_ok}")
        print(f"  Unfused nacho x2 {unfused_ms:.4f} ms   correct={unfused_ok}   "
              f"speedup_fused={unfused_ms/fused_ms:.4f}x")

        if not (fused_ok and unfused_ok):
            print(f"  FAILED at {i}")
            failed.append(i)
            if not fused_ok:
                print("    fused:")
                csr_failure_reason(fused, reference)
            if not unfused_ok:
                print("    unfused:")
                csr_failure_reason(unfused, reference)

        nnz_totals.append(nnz_a + nnz_b + nnz_c)
        fused_runtimes.append(fused_ms)
        unfused_runtimes.append(unfused_ms)

        del A_torch, B_torch, C_torch, A_csr, B_csr, C_csr, reference, fused, unfused

    if save_and_plot:
        plot(nnz_totals, fused_runtimes, unfused_runtimes, None,
             f"csr_add_3_cpu_{start}-{end}")

    _print_summary("CPU 3-way CSR add", fused_runtimes, unfused_runtimes)
    print(f"failed: {failed}")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                        help="which device's benchmark to run (default: both)")
    parser.add_argument("--start", type=int, default=0,
                        help="first index into the matrix list (default: 0)")
    parser.add_argument("--end", type=int, default=1600,
                        help="one past the last index into the matrix list (default: 1600)")
    parser.add_argument("--no-plot", action="store_true",
                        help="print results without writing figures or .npz files")
    args = parser.parse_args()

    if args.device in ("cpu", "both"):
        benchmark_csr_add_3_cpu(args.start, args.end, save_and_plot=not args.no_plot)
    if args.device in ("cuda", "both"):
        benchmark_csr_add_3_gpu(args.start, args.end, save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
