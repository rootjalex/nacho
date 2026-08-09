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

import numpy as np
import torch

import nacho

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


def benchmark_frostt_add(device="cpu", check=True):
    """Time csf_add, coo3d_add and PyTorch on each configured FROSTT tensor."""
    on_gpu = device == "cuda"
    measure = gpu_time if on_gpu else cpu_time
    launch = launch_args(device)
    csf_kernel = nacho.gpu_csf_add_f32 if on_gpu else nacho.cpu_csf_add_f32
    coo3d_kernel = nacho.gpu_coo3d_add_f32 if on_gpu else nacho.cpu_coo3d_add_f32

    names, csf_times, coo3d_times, pytorch_times = [], [], [], []
    failed = []

    for name, dims, a, b in iter_shifted_pairs(device):
        a_coordinates, a_values = a
        b_coordinates, b_values = b
        correct = True

        def stage(label, run):
            """(result, milliseconds) for one implementation, (None, None) if it cannot run.
            """
            try:
                return run()
            except (RuntimeError, MemoryError) as error:
                print(f"  {label:<10} did not complete: {str(error).splitlines()[0]}")
                _release(device)
                return None, None

        def timed_csf():
            A_csf = nacho.to_csf3(a_coordinates, a_values, dims, device)
            B_csf = nacho.to_csf3(b_coordinates, b_values, dims, device)
            result, elapsed = measure(lambda: csf_kernel(A_csf, B_csf, *launch))
            # The result borrows nothing from its operands, so they go before the next
            # stage allocates. Only the expanded coordinates are kept, when checking.
            coordinates = csf3_result_to_coordinates(result) if check else None
            del A_csf, B_csf, result
            return coordinates, elapsed

        def timed_coo3d():
            A_coo3 = nacho.to_coo3(a_coordinates, a_values, dims, device)
            B_coo3 = nacho.to_coo3(b_coordinates, b_values, dims, device)
            result, elapsed = measure(lambda: coo3d_kernel(A_coo3, B_coo3, *launch))
            coordinates = coo3_result_to_coordinates(result) if check else None
            del A_coo3, B_coo3, result
            return coordinates, elapsed

        def timed_pytorch():
            A_torch = to_torch(a_coordinates, a_values, dims, device)
            B_torch = to_torch(b_coordinates, b_values, dims, device)
            reference, elapsed = measure(lambda: (A_torch + B_torch).coalesce())
            del A_torch, B_torch
            coordinates = (reference.indices(), reference.values()) if check else None
            del reference
            return coordinates, elapsed

        csf_coordinates, csf_ms = stage("csf_add", timed_csf)
        _release(device)
        if csf_ms is not None:
            print(f"  csf_add    {csf_ms:.2f} ms")

        coo3d_coordinates, coo3d_ms = stage("coo3d_add", timed_coo3d)
        _release(device)
        if coo3d_ms is not None:
            print(f"  coo3d_add  {coo3d_ms:.2f} ms")

        if check and csf_coordinates is not None and coo3d_coordinates is not None:
            correct &= _check("csf_add vs coo3d_add", csf_coordinates, coo3d_coordinates)
        del csf_coordinates
        _release(device)

        pytorch_coordinates, pytorch_ms = stage("PyTorch", timed_pytorch)
        _release(device)
        if pytorch_ms is not None:
            print(f"  PyTorch    {pytorch_ms:.2f} ms")

        if check and coo3d_coordinates is not None and pytorch_coordinates is not None:
            correct &= _check("coo3d_add vs PyTorch", coo3d_coordinates, pytorch_coordinates)
        del coo3d_coordinates, pytorch_coordinates
        _release(device)

        if pytorch_ms is not None and csf_ms is not None and coo3d_ms is not None:
            print(f"  speedup vs PyTorch: csf={pytorch_ms/csf_ms:.2f}x  "
                  f"coo3d={pytorch_ms/coo3d_ms:.2f}x")
        if not correct:
            failed.append(name)

        names.append(name)
        csf_times.append(csf_ms)
        coo3d_times.append(coo3d_ms)
        pytorch_times.append(pytorch_ms)

        del a_coordinates, a_values, b_coordinates, b_values
        _release(device)

    if not names:
        print("\nNo tensors were benchmarked.")
        return names, csf_times, coo3d_times, pytorch_times

    _print_summary(device, names, csf_times, coo3d_times, pytorch_times, failed)
    return names, csf_times, coo3d_times, pytorch_times


def _geomean(numerators, denominators):
    """Geometric mean of the ratio over the tensors where both sides completed."""
    ratios = [n / d for n, d in zip(numerators, denominators)
              if n is not None and d is not None]
    return np.exp(np.mean(np.log(ratios))) if ratios else float("nan")


def _print_summary(device, names, csf_times, coo3d_times, pytorch_times, failed):
    def cell(value, width):
        return f"{'-':>{width}}" if value is None else f"{value:>{width}.2f}"

    def ratio(numerator, denominator, width):
        if numerator is None or denominator is None:
            return f"{'-':>{width}}"
        return f"{numerator / denominator:>{width}.2f}"

    rule = "=" * 72
    print(f"\n{rule}\n  FROSTT 3D addition on {device}  ({len(names)} tensors)\n{rule}")
    print(f"  {'tensor':<18}{'csf (ms)':>12}{'coo3d (ms)':>12}"
          f"{'torch (ms)':>12}{'csf sp':>9}{'coo3d sp':>10}")
    for i, name in enumerate(names):
        print(f"  {name:<18}{cell(csf_times[i], 12)}{cell(coo3d_times[i], 12)}"
              f"{cell(pytorch_times[i], 12)}"
              f"{ratio(pytorch_times[i], csf_times[i], 9)}"
              f"{ratio(pytorch_times[i], coo3d_times[i], 10)}")
    print(f"\n  geomean speedup over PyTorch: "
          f"csf={_geomean(pytorch_times, csf_times):.2f}x  "
          f"coo3d={_geomean(pytorch_times, coo3d_times):.2f}x")
    print(f"  geomean csf over coo3d: {_geomean(coo3d_times, csf_times):.2f}x")
    print("  incomplete: " + ", ".join(
        f"{label} {sum(time is None for time in series)}"
        for label, series in (("csf", csf_times), ("coo3d", coo3d_times),
                              ("torch", pytorch_times)))
          + f", of {len(names)} tensors")
    print(f"  mismatches: {len(failed)} {failed if failed else ''}")
    print(rule)



def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                        help="which device's benchmark to run (default: both)")
    parser.add_argument("--no-check", action="store_true",
                        help="time only, skipping the correctness comparison and the "
                             "second result it has to keep alive")
    args = parser.parse_args()

    for device in (["cpu", "cuda"] if args.device == "both" else [args.device]):
        benchmark_frostt_add(device=device, check=not args.no_check)


if __name__ == "__main__":
    main()
