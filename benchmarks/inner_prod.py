"""Inner product of two FROSTT tensors, over both 3D layouts, against PyTorch.

Contracts every dimension of a[i,j,k] * b[i,j,k] down to one value. Each tensor is paired
with a coordinate-shifted copy of itself, so both operands have the same nnz and a
controlled overlap. The same contraction runs three ways:

  inner_prod      generated kernel over the CSF3 level tree
  inner_prod_coo  generated kernel over a flat coordinate list
  PyTorch         elementwise multiply, then sum

Comparing the two nacho kernels shows what the level tree buys at equal nnz.

Unlike the other reductions this one reduces its innermost loop, so the compute kernel
accumulates directly with atomic adds rather than materialising an intermediate.

    python benchmarks/inner_prod.py --device cpu
"""

import argparse
import gc

import numpy as np
import torch

import nacho

import config
from common.frostt import iter_shifted_pairs, to_torch
from common.timing import cpu_time, gpu_time, launch_args

TOLERANCE = 1e-3


def _release(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _scalar(result):
    """The single value a fully reduced result carries."""
    values = result.values
    if values.is_cuda:
        torch.cuda.synchronize()
    return float(values.cpu().numpy()[0])


def benchmark_inner_prod(device="cpu", save_and_plot=True):
    """Time both layouts against PyTorch on each configured FROSTT tensor."""
    on_gpu = device == "cuda"
    measure = gpu_time if on_gpu else cpu_time
    launch = launch_args(device)
    csf_kernel = nacho.gpu_inner_prod_f32 if on_gpu else nacho.cpu_inner_prod_f32
    coo_kernel = nacho.gpu_inner_prod_coo_f32 if on_gpu else nacho.cpu_inner_prod_coo_f32

    names, csf_times, coo_times, pytorch_times, nnzs, failed = [], [], [], [], [], []

    for name, dims, a, b in iter_shifted_pairs(device):
        a_coordinates, a_values = a
        b_coordinates, b_values = b

        def stage(label, run):
            """(value, milliseconds) for one implementation, (None, None) if it cannot run.

            The largest tensors defeat an implementation without defeating the others:
            torch cannot hold a shape whose element count overflows, and any of the three
            can exhaust the device. Losing one is a gap in that series, not the end of
            the sweep.
            """
            try:
                return run()
            except (RuntimeError, MemoryError) as error:
                print(f"  {label:<15} did not complete: {str(error).splitlines()[0]}")
                _release(device)
                return None, None

        def timed_csf():
            A_csf = nacho.to_csf3(a_coordinates, a_values, dims, device)
            B_csf = nacho.to_csf3(b_coordinates, b_values, dims, device)
            result, elapsed = measure(lambda: csf_kernel(A_csf, B_csf, *launch))
            product = _scalar(result)
            del A_csf, B_csf, result
            return product, elapsed

        def timed_coo():
            A_coo3 = nacho.to_coo3(a_coordinates, a_values, dims, device)
            B_coo3 = nacho.to_coo3(b_coordinates, b_values, dims, device)
            result, elapsed = measure(lambda: coo_kernel(A_coo3, B_coo3, *launch))
            product = _scalar(result)
            del A_coo3, B_coo3, result
            return product, elapsed

        def timed_pytorch():
            A_torch = to_torch(a_coordinates, a_values, dims, device)
            B_torch = to_torch(b_coordinates, b_values, dims, device)
            reference, elapsed = measure(
                lambda: (A_torch * B_torch).coalesce().values().sum())
            value = float(reference.cpu())
            del A_torch, B_torch, reference
            return value, elapsed

        csf_product, csf_ms = stage("inner_prod", timed_csf)
        _release(device)
        coo_product, coo_ms = stage("inner_prod_coo", timed_coo)
        _release(device)
        expected, pytorch_ms = stage("PyTorch", timed_pytorch)
        _release(device)

        # Summation order differs between all three, so compare relatively.
        def matches(product, target):
            return abs(product - target) <= TOLERANCE * max(1.0, abs(target))

        def report(label, elapsed, product):
            if elapsed is None:
                return
            agreement = ("" if expected is None
                         else f"   correct={matches(product, expected)}")
            print(f"  {label:<15} {elapsed:.2f} ms   value={product:.6g}{agreement}")

        report("inner_prod", csf_ms, csf_product)
        report("inner_prod_coo", coo_ms, coo_product)
        if pytorch_ms is not None:
            print(f"  {'PyTorch':<15} {pytorch_ms:.2f} ms   value={expected:.6g}")

        if expected is not None:
            correct = all(matches(product, expected)
                          for product in (csf_product, coo_product) if product is not None)
        elif csf_product is not None and coo_product is not None:
            # No reference to check against, but the two layouts still check each other.
            correct = matches(csf_product, coo_product)
            print(f"  [check inner_prod vs inner_prod_coo] "
                  f"{'PASSED' if correct else 'FAILED'}")
        else:
            correct = True

        if pytorch_ms is not None and csf_ms is not None and coo_ms is not None:
            print(f"  speedup vs PyTorch: csf={pytorch_ms/csf_ms:.2f}x  "
                  f"coo={pytorch_ms/coo_ms:.2f}x")
        if not correct:
            failed.append(name)

        names.append(name)
        nnzs.append(len(a_values) + len(b_values))
        csf_times.append(csf_ms)
        coo_times.append(coo_ms)
        pytorch_times.append(pytorch_ms)

        del a_coordinates, a_values, b_coordinates, b_values
        _release(device)

    if not names:
        print("\nNo tensors were benchmarked.")
        return failed

    _print_summary(device, names, csf_times, coo_times, pytorch_times, failed)
    if save_and_plot:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RESULTS_DIR / f"inner_prod_{device}.npz"
        # nan where an implementation could not run, so the series stay aligned by tensor.
        np.savez(path, names=names, nnz=nnzs,
                 csf=np.array(csf_times, dtype=float),
                 coo=np.array(coo_times, dtype=float),
                 pytorch=np.array(pytorch_times, dtype=float))
        print(f"Saved: {path}")
    return failed


def _geomean(numerators, denominators):
    """Geometric mean of the ratio over the tensors where both sides completed."""
    ratios = [n / d for n, d in zip(numerators, denominators)
              if n is not None and d is not None]
    return np.exp(np.mean(np.log(ratios))) if ratios else float("nan")


def _print_summary(device, names, csf_times, coo_times, pytorch_times, failed):
    def cell(value, width):
        return f"{'-':>{width}}" if value is None else f"{value:>{width}.2f}"

    def ratio(numerator, denominator, width):
        if numerator is None or denominator is None:
            return f"{'-':>{width}}"
        return f"{numerator / denominator:>{width}.2f}"

    rule = "=" * 78
    print(f"\n{rule}\n  FROSTT inner product on {device}  ({len(names)} tensors)\n{rule}")
    print(f"  {'tensor':<18}{'csf (ms)':>12}{'coo (ms)':>12}{'torch (ms)':>12}"
          f"{'csf sp':>9}{'coo sp':>9}")
    for i, name in enumerate(names):
        print(f"  {name:<18}{cell(csf_times[i], 12)}{cell(coo_times[i], 12)}"
              f"{cell(pytorch_times[i], 12)}"
              f"{ratio(pytorch_times[i], csf_times[i], 9)}"
              f"{ratio(pytorch_times[i], coo_times[i], 9)}")
    print(f"\n  geomean speedup over PyTorch: "
          f"csf={_geomean(pytorch_times, csf_times):.2f}x  "
          f"coo={_geomean(pytorch_times, coo_times):.2f}x")
    print(f"  geomean csf over coo: {_geomean(coo_times, csf_times):.2f}x")
    print("  incomplete: " + ", ".join(
        f"{label} {sum(time is None for time in series)}"
        for label, series in (("csf", csf_times), ("coo", coo_times),
                              ("torch", pytorch_times)))
          + f", of {len(names)} tensors")
    print(f"  mismatches: {len(failed)} {failed if failed else ''}")
    print(rule)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                        help="which device's benchmark to run (default: both)")
    parser.add_argument("--no-plot", action="store_true",
                        help="print results without writing .npz files")
    args = parser.parse_args()

    for device in (["cpu", "cuda"] if args.device == "both" else [args.device]):
        benchmark_inner_prod(device=device, save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
