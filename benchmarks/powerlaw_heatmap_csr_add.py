"""CSR addition over synthetic power-law matrices, as a speedup heatmap.

Where the SuiteSparse sweeps show what happens on real matrices, this controls the one
property that decides how uneven the work is. A power-law row-length distribution with a
small exponent gives a few very long rows among many short ones; a large exponent gives
rows of similar length. Every pair of exponents is benchmarked, so the heatmap reads as:
how much does nacho's merge-path partitioning buy as the two operands' row-length skew
varies.

Every matrix carries the same number of nonzeros whatever the exponent, so a cell differs
from its neighbours only in how those nonzeros are spread over the rows. What decides the
outcome is the heaviest row measured against one thread's share of the work: below about
exponent 2.5 the heaviest row exceeds it and the row-parallel baseline stalls behind the
thread that owns it, above that the rows are even enough that there is no imbalance left
to remove.

The comparison is the load balancing, not the library: both sides are the same addition,
the baseline being the row-parallel kernel in baselines/ that assigns one row per thread.
On the GPU cuSPARSE stands in for it, being the vendor's row-parallel implementation.

    python benchmarks/powerlaw_heatmap_csr_add.py --device both
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

# Power-law exponents to cross. Below about 2 the row lengths are dominated by a few
# enormous rows; by 4 the rows are nearly uniform.
EXPONENTS = np.linspace(1.5, 2.5, num=8)

# Shape and weight of every generated matrix. COLS is far larger than NNZ, so the heaviest
# row is free to hold most of the matrix however long the power-law tail asks for; ROWS is
# small enough that one such row outweighs a whole thread's share of the work.
ROWS = 10_000
COLS = 1_000_000
NNZ = 2_000_000


def generate_matrix(exponent, device, seed=SEED):
    """A ROWS x COLS matrix of about NNZ nonzeros, row lengths following a power law.

    The lengths are drawn for `exponent` and then rescaled to sum to NNZ, so the exponent
    sets how unevenly the nonzeros are spread over the rows without changing how many of
    them there are. Columns within a row are drawn uniformly, sorted, and deduplicated,
    which is what leaves the count slightly under NNZ at the skewed end.
    """
    lengths = np.asarray(nx.utils.powerlaw_sequence(ROWS, exponent=exponent, seed=seed))
    lengths = np.clip(np.round(lengths / lengths.sum() * NNZ), 0, COLS).astype(np.int64)

    generator = np.random.default_rng(seed)
    rows = np.repeat(np.arange(ROWS, dtype=np.int64), lengths)
    columns = generator.integers(0, COLS, size=int(lengths.sum()), dtype=np.int64)

    order = np.lexsort((columns, rows))
    rows, columns = rows[order], columns[order]
    distinct = np.ones(len(columns), dtype=bool)
    distinct[1:] = (rows[1:] != rows[:-1]) | (columns[1:] != columns[:-1])
    rows, columns = rows[distinct], columns[distinct]

    offsets = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=ROWS))])
    return torch.sparse_csr_tensor(
        torch.tensor(offsets, dtype=torch.int32, device=device),
        torch.tensor(columns, dtype=torch.int32, device=device),
        torch.tensor(generator.random(len(columns)) + 1.0, dtype=torch.float32,
                     device=device),
        size=(ROWS, COLS), device=device)


def evaluate_powerlaw_matrices(device):
    """Speedup of nacho over the row-parallel baseline for every pair of exponents."""
    on_gpu = device == "cuda"
    kernel = nacho.gpu_csr_add_f32 if on_gpu else nacho.cpu_csr_add_f32
    baseline = (nacho.gpu_csr_add_cusparse_f32 if on_gpu else nacho.cpu_csr_add_taco_f32)
    measure = timer_for(device)
    launch = launch_args(device)

    speedups = np.zeros((len(EXPONENTS), len(EXPONENTS)))
    failed = []

    for i, exponent_a in enumerate(EXPONENTS):
        A_torch = generate_matrix(exponent_a, device)
        A_csr = nacho.to_csr(A_torch, device)
        A_base = to_baseline_csr(A_torch, device)

        for j, exponent_b in enumerate(EXPONENTS):
            # A second seed, so that the diagonal pairs two different matrices rather than
            # adding one to a copy of itself.
            B_torch = generate_matrix(exponent_b, device)
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


def benchmark_powerlaw_csr_add(device="cpu", save_and_plot=True):
    """Run the exponent grid on one device and plot it."""
    baseline_name = "cuSPARSE" if device == "cuda" else "Taco"
    print(f"\nGenerating {ROWS} x {COLS} power-law matrices of {NNZ} nonzeros on {device}, "
          f"nacho vs {baseline_name}")

    speedups, failed = evaluate_powerlaw_matrices(device)

    rule = "=" * 66
    print(f"\n{rule}\n  csr_add over power-law matrices on {device}  "
          f"({ROWS} x {COLS}, {NNZ} nonzeros)\n{rule}")
    print(f"  speedup over {baseline_name}: min={speedups.min():.2f}x  "
          f"max={speedups.max():.2f}x  geomean={np.exp(np.mean(np.log(speedups))):.2f}x")
    print(f"  mismatches: {len(failed)} {failed if failed else ''}")
    print(rule)

    if save_and_plot:
        plot_heatmap(f"powerlaw_csr_add_{device}_{ROWS}x{COLS}", speedups, EXPONENTS,
                     "Matrix B Exponent", "Matrix A Exponent")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                        help="which device's benchmark to run (default: both)")
    parser.add_argument("--no-plot", action="store_true",
                        help="print results without writing figures or .npz files")
    args = parser.parse_args()

    for device in (["cpu", "cuda"] if args.device == "both" else [args.device]):
        benchmark_powerlaw_csr_add(device=device, save_and_plot=not args.no_plot)


if __name__ == "__main__":
    main()
