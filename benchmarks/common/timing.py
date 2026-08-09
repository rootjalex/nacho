"""Timing helpers shared by the benchmark scripts.

Every measurement runs config.ITER_COUNT times and reports a trimmed mean, dropping
config.TRIM samples from each tail so a stray slow run does not dominate.
"""

import argparse
import gc
import time

import numpy as np
import torch

import config


def flush_gpu_state():
    """Drop cached allocations and push stale data out of L2 between measurements."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device="cuda").zero_()
    torch.cuda.synchronize()


def _trimmed_mean(times):
    return float(np.sort(np.array(times))[config.TRIM:-config.TRIM].mean())


def gpu_time(fn):
    """Run fn() on the GPU; return (last result, trimmed mean ms) from CUDA events."""
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
    return result, _trimmed_mean(times)


def cpu_time(fn):
    """Run fn() on the host; return (last result, trimmed mean ms)."""
    times = []
    result = None
    for _ in range(config.ITER_COUNT):
        started = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - started) * 1000)
    return result, _trimmed_mean(times)


def timer_for(device):
    return gpu_time if device == "cuda" else cpu_time


def launch_args(device):
    """Trailing arguments the generated GPU kernels take, empty on the CPU.

    Spliced into a call as `kernel(a, b, *launch_args(device))`. The generated wrappers
    carry their own defaults, so passing these explicitly is what keeps every script on
    the geometry in config rather than on whatever the compiler emitted.
    """
    return (config.NUM_BLOCKS, config.THREADS_PER_BLOCK) if device == "cuda" else ()


def parse_sweep_args(description, end_default=1600, gpu_only=False):
    """The argument set every matrix-pair sweep takes.

    end_default varies per benchmark: the reductions run out of memory well before the end
    of the matrix list, so they stop earlier. gpu_only drops the device choice for kernels
    generated only for CUDA.
    """
    parser = argparse.ArgumentParser(description=description)
    if not gpu_only:
        parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                            help="which device's benchmark to run (default: both)")
    parser.add_argument("--start", type=int, default=0,
                        help="first index into the matrix list (default: 0)")
    parser.add_argument("--end", type=int, default=end_default,
                        help=f"one past the last index into the matrix list "
                             f"(default: {end_default})")
    parser.add_argument("--no-plot", action="store_true",
                        help="print results without writing figures or .npz files")
    args = parser.parse_args()
    if gpu_only:
        args.device = "cuda"
    args.devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    return args
