"""CSR addition over synthetic power-law matrices, as a speedup heatmap.

Where the SuiteSparse sweeps show what happens on real matrices, this controls the one
property that decides how uneven the work is. A power-law degree sequence with a small
exponent gives a few very long rows among many short ones; a large exponent gives rows of
similar length. Every pair of exponents is benchmarked, so the heatmap reads as: how much
does nacho's merge-path partitioning buy as the two operands' row-length skew varies.

The comparison is the load balancing, not the library: both sides are the same addition,
the baseline being the row-parallel kernel in baselines/ that assigns one row per thread.
On the GPU cuSPARSE stands in for it, being the vendor's row-parallel implementation.

    python benchmarks/powerlaw_heatmap_csr_add.py --device both --n 100000
"""

import argparse

import networkx as nx
import numpy as np
import torch

import nacho

from common.compare import csr_close
from common.csr import as_nacho_csr, to_baseline_csr
from common.plotter import plot_heatmap
from common.timing import flush_gpu_state, launch_args, timer_for

SEED = 42

# Power-law exponents to cross. Below about 2 the degree sequence is dominated by a few
# enormous rows; by 4 the rows are nearly uniform.
EXPONENTS = np.linspace(1.5, 4.0, num=8)


def generate_matrix(n, exponent, device):
    """An n x n symmetric matrix whose row lengths follow a power law of `exponent`."""
    degrees = np.round(nx.utils.powerlaw_sequence(n, exponent=exponent, seed=SEED))
    graph = nx.expected_degree_graph(degrees.astype(int), selfloops=False, seed=SEED)
    scipy_csr = nx.to_scipy_sparse_array(graph, format="csr")

    return torch.sparse_csr_tensor(
        torch.tensor(scipy_csr.indptr, dtype=torch.int32, device=device),
        torch.tensor(scipy_csr.indices, dtype=torch.int32, device=device),
        torch.tensor(scipy_csr.data, dtype=torch.float32, device=device),
        size=scipy_csr.shape, device=device)


def evaluate_powerlaw_matrices(n, device):
    """Speedup of nacho over the row-parallel baseline for every pair of exponents."""
    on_gpu = device == "cuda"
    kernel = nacho.gpu_csr_add_f32 if on_gpu else nacho.cpu_csr_add_f32
    baseline = (nacho.gpu_csr_add_cusparse_f32 if on_gpu else nacho.cpu_csr_add_taco_f32)
    measure = timer_for(device)
    launch = launch_args(device)

    speedups = np.zeros((len(EXPONENTS), len(EXPONENTS)))
    failed = []

    for i, exponent_a in enumerate(EXPONENTS):
        A_torch = generate_matrix(n, exponent_a, device)
        A_csr = nacho.to_csr(A_torch, device)
        A_base = to_baseline_csr(A_torch, device)

        for j, exponent_b in enumerate(EXPONENTS):
            B_torch = generate_matrix(n, exponent_b, device)
            B_csr = nacho.to_csr(B_torch, device)
            B_base = to_baseline_csr(B_torch, device)
            del B_torch

            result, nacho_ms = measure(lambda: kernel(A_csr, B_csr, *launch))
            reference, baseline_ms = measure(lambda: baseline(A_base, B_base))

            correct = csr_close(result, as_nacho_csr(reference, device))
            speedups[i, j] = baseline_ms / nacho_ms
            print(f"  exponents A={exponent_a:.2f} B={exponent_b:.2f}   "
                  f"nacho={nacho_ms:.4f} ms  baseline={baseline_ms:.4f} ms  "
                  f"speedup={speedups[i, j]:.3f}x  correct={correct}")
            if not correct:
                failed.append((round(exponent_a, 2), round(exponent_b, 2)))

            del B_csr, B_base, result, reference
            if on_gpu:
                flush_gpu_state()

        del A_torch, A_csr, A_base
        if on_gpu:
            flush_gpu_state()

    return speedups, failed


def benchmark_powerlaw_csr_add(n, device="cpu", save_and_plot=True):
    """Run the exponent grid on one device and plot it."""
    baseline_name = "cuSPARSE" if device == "cuda" else "Taco"
    print(f"\nGenerating {n} x {n} power-law matrices on {device}, "
          f"nacho vs {baseline_name}")

    speedups, failed = evaluate_powerlaw_matrices(n, device)

    rule = "=" * 66
    print(f"\n{rule}\n  csr_add over power-law matrices on {device}  (n={n})\n{rule}")
    print(f"  speedup over {baseline_name}: min={speedups.min():.2f}x  "
          f"max={speedups.max():.2f}x  geomean={np.exp(np.mean(np.log(speedups))):.2f}x")
    print(f"  mismatches: {len(failed)} {failed if failed else ''}")
    print(rule)

    if save_and_plot:
        plot_heatmap(f"powerlaw_csr_add_{device}_{n}", speedups, EXPONENTS,
                     "Matrix B Exponent", "Matrix A Exponent")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                        help="which device's benchmark to run (default: both)")
    parser.add_argument("--n", type=int, default=100000,
                        help="rows and columns of the generated matrices (default: 100000)")
    parser.add_argument("--no-plot", action="store_true",
                        help="print results without writing figures or .npz files")
    args = parser.parse_args()

    for device in (["cpu", "cuda"] if args.device == "both" else [args.device]):
        benchmark_powerlaw_csr_add(args.n, device=device,
                                   save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
