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

        A_csf = nacho.to_csf3(a_coordinates, a_values, dims, device)
        B_csf = nacho.to_csf3(b_coordinates, b_values, dims, device)
        result, csf_ms = measure(lambda: csf_kernel(A_csf, B_csf, *launch))
        csf_product = _scalar(result)
        del A_csf, B_csf, result
        _release(device)

        A_coo3 = nacho.to_coo3(a_coordinates, a_values, dims, device)
        B_coo3 = nacho.to_coo3(b_coordinates, b_values, dims, device)
        result, coo_ms = measure(lambda: coo_kernel(A_coo3, B_coo3, *launch))
        coo_product = _scalar(result)
        del A_coo3, B_coo3, result
        _release(device)

        A_torch = to_torch(a_coordinates, a_values, dims, device)
        B_torch = to_torch(b_coordinates, b_values, dims, device)
        reference, pytorch_ms = measure(lambda: (A_torch * B_torch).coalesce().values().sum())
        expected = float(reference.cpu())
        del A_torch, B_torch, reference
        _release(device)

        # Summation order differs between all three, so compare relatively.
        def matches(product):
            return abs(product - expected) <= TOLERANCE * max(1.0, abs(expected))

        print(f"  inner_prod      {csf_ms:.2f} ms   value={csf_product:.6g}   "
              f"correct={matches(csf_product)}")
        print(f"  inner_prod_coo  {coo_ms:.2f} ms   value={coo_product:.6g}   "
              f"correct={matches(coo_product)}")
        print(f"  PyTorch         {pytorch_ms:.2f} ms   value={expected:.6g}")
        print(f"  speedup vs PyTorch: csf={pytorch_ms/csf_ms:.2f}x  "
              f"coo={pytorch_ms/coo_ms:.2f}x")
        if not (matches(csf_product) and matches(coo_product)):
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
        np.savez(path, names=names, nnz=nnzs, csf=csf_times, coo=coo_times,
                 pytorch=pytorch_times)
        print(f"Saved: {path}")
    return failed


def _print_summary(device, names, csf_times, coo_times, pytorch_times, failed):
    csf = np.array(csf_times)
    coo = np.array(coo_times)
    pytorch = np.array(pytorch_times)

    rule = "=" * 78
    print(f"\n{rule}\n  FROSTT inner product on {device}  ({len(names)} tensors)\n{rule}")
    print(f"  {'tensor':<18}{'csf (ms)':>12}{'coo (ms)':>12}{'torch (ms)':>12}"
          f"{'csf sp':>9}{'coo sp':>9}")
    for i, name in enumerate(names):
        print(f"  {name:<18}{csf[i]:>12.2f}{coo[i]:>12.2f}{pytorch[i]:>12.2f}"
              f"{pytorch[i]/csf[i]:>9.2f}{pytorch[i]/coo[i]:>9.2f}")
    print(f"\n  geomean speedup over PyTorch: "
          f"csf={np.exp(np.mean(np.log(pytorch / csf))):.2f}x  "
          f"coo={np.exp(np.mean(np.log(pytorch / coo))):.2f}x")
    print(f"  geomean csf over coo: {np.exp(np.mean(np.log(coo / csf))):.2f}x")
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
