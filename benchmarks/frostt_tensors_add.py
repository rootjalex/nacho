"""Three-dimensional addition on FROSTT tensors: CSF3 and COO3D against PyTorch.

Each tensor is added to a coordinate-shifted copy of itself, so both operands have the
same nnz and a controlled amount of overlap. The same addition is run three ways:

  csf_add     generated kernel over the CSF3 level tree
  coo3d_add   generated kernel over a flat coordinate list
  PyTorch     native 3D sparse add, then coalesce

Comparing the two nacho kernels shows what the level tree buys over a flat coordinate
list at the same nnz.

Each result is checked against the next: csf_add against coo3d_add, then coo3d_add
against PyTorch. Coordinates have to agree exactly and values to within a relative
tolerance. Holding two results at once roughly doubles peak memory on the largest
tensors, so --no-check skips it.

Tensors and their location come from config.FROSTT_TENSORS / config.FROSTT_DIR, and
their coordinates must already be sorted lexicographically.

    python benchmarks/frostt_tensors_add.py --device cpu
"""

import argparse
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import nacho

import config
from common.compare import (coo3_result_to_coordinates, coordinates_equal,
                            csf3_result_to_coordinates)
from common.frostt import iter_shifted_pairs, to_torch
from common.timing import cpu_time, gpu_time, launch_args


def _release(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _check(label, actual, expected):
    """Report whether two (coordinates, values) pairs agree, and how they diverge."""
    ok, reason = coordinates_equal(actual, expected)
    print(f"  [check {label}] {'PASSED' if ok else 'FAILED — ' + reason}")
    return ok


def benchmark_frostt_add(device="cpu", save_and_plot=True, check=True):
    """Time csf_add, coo3d_add and PyTorch on each configured FROSTT tensor."""
    on_gpu = device == "cuda"
    measure = gpu_time if on_gpu else cpu_time
    launch = launch_args(device)
    csf_kernel = nacho.gpu_csf_add_f32 if on_gpu else nacho.cpu_csf_add_f32
    coo3d_kernel = nacho.gpu_coo3d_add_f32 if on_gpu else nacho.cpu_coo3d_add_f32

    names, csf_times, coo3d_times, pytorch_times, nnzs = [], [], [], [], []
    failed = []

    for name, dims, a, b in iter_shifted_pairs(device):
        a_coordinates, a_values = a
        b_coordinates, b_values = b
        correct = True

        A_csf = nacho.to_csf3(a_coordinates, a_values, dims, device)
        B_csf = nacho.to_csf3(b_coordinates, b_values, dims, device)
        csf_result, csf_ms = measure(lambda: csf_kernel(A_csf, B_csf, *launch))
        # The result borrows nothing from its operands, so they go before the next stage
        # allocates. Only the expanded coordinates are kept, and only when checking.
        csf_coordinates = csf3_result_to_coordinates(csf_result) if check else None
        del A_csf, B_csf, csf_result
        _release(device)
        print(f"  csf_add    {csf_ms:.2f} ms")

        A_coo3 = nacho.to_coo3(a_coordinates, a_values, dims, device)
        B_coo3 = nacho.to_coo3(b_coordinates, b_values, dims, device)
        coo3d_result, coo3d_ms = measure(lambda: coo3d_kernel(A_coo3, B_coo3, *launch))
        coo3d_coordinates = coo3_result_to_coordinates(coo3d_result) if check else None
        del A_coo3, B_coo3, coo3d_result
        _release(device)
        print(f"  coo3d_add  {coo3d_ms:.2f} ms")

        if check:
            correct &= _check("csf_add vs coo3d_add", csf_coordinates, coo3d_coordinates)
            del csf_coordinates
            _release(device)

        A_torch = to_torch(a_coordinates, a_values, dims, device)
        B_torch = to_torch(b_coordinates, b_values, dims, device)
        reference, pytorch_ms = measure(lambda: (A_torch + B_torch).coalesce())
        del A_torch, B_torch
        print(f"  PyTorch    {pytorch_ms:.2f} ms")

        if check:
            correct &= _check("coo3d_add vs PyTorch", coo3d_coordinates,
                              (reference.indices(), reference.values()))
            del coo3d_coordinates
        del reference
        _release(device)

        print(f"  speedup vs PyTorch: csf={pytorch_ms/csf_ms:.2f}x  "
              f"coo3d={pytorch_ms/coo3d_ms:.2f}x")
        if not correct:
            failed.append(name)

        names.append(name)
        nnzs.append(len(a_values) + len(b_values))
        csf_times.append(csf_ms)
        coo3d_times.append(coo3d_ms)
        pytorch_times.append(pytorch_ms)

        del a_coordinates, a_values, b_coordinates, b_values
        _release(device)

    if not names:
        print("\nNo tensors were benchmarked.")
        return names, csf_times, coo3d_times, pytorch_times

    _print_summary(device, names, csf_times, coo3d_times, pytorch_times, failed)
    if save_and_plot:
        _plot_speedups(device, names, nnzs, csf_times, coo3d_times, pytorch_times)
    return names, csf_times, coo3d_times, pytorch_times


def _print_summary(device, names, csf_times, coo3d_times, pytorch_times, failed):
    csf = np.array(csf_times)
    coo3d = np.array(coo3d_times)
    pytorch = np.array(pytorch_times)

    rule = "=" * 72
    print(f"\n{rule}\n  FROSTT 3D addition on {device}  ({len(names)} tensors)\n{rule}")
    print(f"  {'tensor':<18}{'csf (ms)':>12}{'coo3d (ms)':>12}"
          f"{'torch (ms)':>12}{'csf sp':>9}{'coo3d sp':>10}")
    for i, name in enumerate(names):
        print(f"  {name:<18}{csf[i]:>12.2f}{coo3d[i]:>12.2f}{pytorch[i]:>12.2f}"
              f"{pytorch[i]/csf[i]:>9.2f}{pytorch[i]/coo3d[i]:>10.2f}")
    print(f"\n  geomean speedup over PyTorch: "
          f"csf={np.exp(np.mean(np.log(pytorch / csf))):.2f}x  "
          f"coo3d={np.exp(np.mean(np.log(pytorch / coo3d))):.2f}x")
    print(f"  geomean csf over coo3d: {np.exp(np.mean(np.log(coo3d / csf))):.2f}x")
    print(f"  mismatches: {len(failed)} {failed if failed else ''}")
    print(rule)


def _plot_speedups(device, names, nnzs, csf_times, coo3d_times, pytorch_times):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = str(config.RESULTS_DIR / f"frostt_tensors_add_{device}")

    np.savez(f"{stem}.npz", names=names, nnz=nnzs, csf=csf_times,
             coo3d=coo3d_times, pytorch=pytorch_times)

    pytorch = np.array(pytorch_times)
    positions = np.arange(len(names))
    width = 0.38

    figure, axes = plt.subplots(figsize=(max(4.0, 1.1 * len(names)), 2.6))
    axes.bar(positions - width / 2, pytorch / np.array(csf_times), width,
             label="CSF3", color="#88CCEE")
    axes.bar(positions + width / 2, pytorch / np.array(coo3d_times), width,
             label="COO3D", color="#332288")
    axes.axhline(1.0, color="grey", linewidth=0.8)

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=30, ha="right")
    axes.set_ylabel("Speedup over PyTorch")
    axes.set_title(f"FROSTT 3D addition ({device})")
    axes.legend()
    figure.tight_layout()
    figure.savefig(f"{stem}.png", dpi=200)
    print(f"Saved: {stem}.png")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                        help="which device's benchmark to run (default: both)")
    parser.add_argument("--no-plot", action="store_true",
                        help="print results without writing figures or .npz files")
    parser.add_argument("--no-check", action="store_true",
                        help="time only, skipping the correctness comparison and the "
                             "second result it has to keep alive")
    args = parser.parse_args()

    for device in (["cpu", "cuda"] if args.device == "both" else [args.device]):
        benchmark_frostt_add(device=device, save_and_plot=not args.no_plot,
                             check=not args.no_check)


if __name__ == "__main__":
    main()
