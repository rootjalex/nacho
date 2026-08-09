"""CPU speedup heatmap: recursive vs single-phase partitioning for the DCSR product.

Both kernels are generated from the same expression; they differ only in whether work is
re-partitioned at each sparse intersection level (dcsr_mul) or partitioned once over the
whole loop nest (dcsr_mul_without_recursive_partitioning).

Grid:
  y-axis: row_density  (fraction of non-empty rows, log scale)
  x-axis: num_rows     (total rows, log scale)
  color:  speedup = single-phase time / recursive time  (>1 means recursive is faster)

The `skew` parameter (0..1) controls what fraction of A's active rows are also active
in B:
  skew=0.0  -> no rows intersect between A and B (empty result)
  skew=1.0  -> all of A's active rows also appear in B (maximum intersection)
  skew=None -> independent random rows (natural overlap ~ row_density)

    python benchmarks/dcsr_partitioning_heatmap.py --trials 3
"""

import argparse
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import nacho

import config


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _build_dcsr_arrays(row_ids, num_rows, num_cols, col_density, dtype):
    """Given sorted row_ids, sample column indices and return DCSR arrays."""
    active_rows = row_ids.numel()
    if active_rows == 0:
        return (row_ids,
                torch.zeros(1, dtype=dtype),
                torch.zeros(0, dtype=dtype),
                torch.zeros(0, dtype=torch.float32),
                torch.tensor([num_rows, num_cols], dtype=dtype))

    col_mask = torch.rand(active_rows, num_cols) < col_density
    nnz_per_row = col_mask.sum(dim=1, dtype=torch.int64)
    row_ptr = torch.zeros(active_rows + 1, dtype=torch.int64)
    row_ptr[1:] = nnz_per_row.cumsum(0)

    nnz = int(row_ptr[-1].item())
    col_idx = col_mask.nonzero(as_tuple=False)[:, 1].to(dtype)
    values = torch.randn(nnz, dtype=torch.float32)
    shape = torch.tensor([num_rows, num_cols], dtype=dtype)
    return row_ids, row_ptr.to(dtype), col_idx, values, shape


def get_synthetic_dcsr_cpu(num_rows, num_cols, row_density, col_density, dtype=torch.int32):
    """A single DCSR matrix with independently sampled active rows."""
    row_mask = torch.rand(num_rows) < row_density
    row_ids = row_mask.nonzero(as_tuple=False).squeeze(-1).to(dtype)
    return _build_dcsr_arrays(row_ids, num_rows, num_cols, col_density, dtype)


def get_synthetic_dcsr_pair_cpu(num_rows, num_cols, row_density, col_density,
                                skew=None, dtype=torch.int32):
    """A pair (A, B) of DCSR matrices with controlled row intersection.

    Both have approximately num_rows * row_density active rows.
    """
    if skew is None:
        return (get_synthetic_dcsr_cpu(num_rows, num_cols, row_density, col_density, dtype),
                get_synthetic_dcsr_cpu(num_rows, num_cols, row_density, col_density, dtype))

    row_mask_a = torch.rand(num_rows) < row_density
    row_ids_a = row_mask_a.nonzero(as_tuple=False).squeeze(-1).to(dtype)
    active_a = row_ids_a.numel()

    total_b = max(1, int(num_rows * row_density))
    shared_count = min(active_a, int(round(skew * min(active_a, total_b))))
    extra_count = max(0, total_b - shared_count)

    shared_rows = torch.zeros(0, dtype=dtype)
    if shared_count > 0 and active_a > 0:
        shared_rows = row_ids_a[torch.randperm(active_a)[:shared_count]]

    extra_rows = torch.zeros(0, dtype=dtype)
    inactive_rows = (~row_mask_a).nonzero(as_tuple=False).squeeze(-1).to(dtype)
    if extra_count > 0 and inactive_rows.numel() > 0:
        take = min(extra_count, inactive_rows.numel())
        extra_rows = inactive_rows[torch.randperm(inactive_rows.numel())[:take]]

    row_ids_b, _ = torch.sort(torch.cat([shared_rows, extra_rows]))

    return (_build_dcsr_arrays(row_ids_a, num_rows, num_cols, col_density, dtype),
            _build_dcsr_arrays(row_ids_b, num_rows, num_cols, col_density, dtype))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_fn_ms(fn, *args):
    times = []
    for _ in range(config.ITER_COUNT):
        started = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - started) * 1e3)
    sorted_times = np.sort(times)
    trimmed = sorted_times[config.TRIM: len(sorted_times) - config.TRIM]
    return float(trimmed.mean()) if trimmed.size > 0 else float(sorted_times.mean())


def generate_heatmap(row_density_list, num_rows_list, nnz_target=5_000_000,
                     col_density=0.5, skew=None, n_trials=3):
    """Median speedup of recursive over single-phase at each (row_density, num_rows).

    Column count is chosen per cell to hold total nnz near nnz_target, so cells differ in
    shape rather than in total work.
    """
    speedup_matrix = np.zeros((len(row_density_list), len(num_rows_list)))

    for i, row_density in enumerate(row_density_list):
        for j, num_rows in enumerate(num_rows_list):
            num_active = max(1, int(num_rows * row_density))
            num_cols = max(1, int(nnz_target / (num_active * col_density)))

            trial_speedups = []
            for _ in range(n_trials):
                try:
                    (ri_a, rp_a, ci_a, v_a, sh_a), \
                        (ri_b, rp_b, ci_b, v_b, sh_b) = get_synthetic_dcsr_pair_cpu(
                            num_rows, num_cols, row_density, col_density, skew=skew)

                    A = nacho.DCSR_cpu(ri_a, rp_a, ci_a, v_a, sh_a)
                    B = nacho.DCSR_cpu(ri_b, rp_b, ci_b, v_b, sh_b)

                    recursive_ms = time_fn_ms(nacho.cpu_dcsr_mul_f32, A, B)
                    single_ms = time_fn_ms(
                        nacho.cpu_dcsr_mul_without_recursive_partitioning_f32, A, B)
                    if recursive_ms > 0:
                        trial_speedups.append(single_ms / recursive_ms)
                except Exception as error:  # noqa: BLE001 - one bad cell should not stop the sweep
                    print(f"  SKIP row_density={row_density:.2e}, num_rows={num_rows}: {error}")

            speedup_matrix[i, j] = float(np.median(trial_speedups)) if trial_speedups else 1.0
            skew_label = f"{skew:.2f}" if skew is not None else "random"
            print(f"  row_density={row_density:.2e}  num_rows={num_rows:6d}  "
                  f"skew={skew_label}  speedup={speedup_matrix[i, j]:.2f}x")

    return speedup_matrix


def plot_speedup_heatmap(speedup_matrix, row_density_list, num_rows_list, title,
                         save_path=None):
    figure, axes = plt.subplots(figsize=(12, 8))

    data = np.clip(speedup_matrix, 1e-2, None)
    norm = mcolors.LogNorm(vmin=max(data.min(), 1e-2), vmax=max(data.max(), 1.01))

    sns.heatmap(data, ax=axes, norm=norm, cmap="RdYlGn",
                xticklabels=[f"{n:.0e}" for n in num_rows_list],
                yticklabels=[f"{d:.1e}" for d in row_density_list],
                annot=True, fmt=".1f", linewidths=0.4, linecolor="gray",
                cbar_kws={"label": "Speedup (single-phase time / recursive time)"})

    axes.set_xlabel("num_rows")
    axes.set_ylabel("row_density")
    axes.set_title(title)
    plt.tight_layout()

    if save_path:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=3,
                        help="matrix pairs sampled per grid cell (default: 3)")
    parser.add_argument("--nnz-target", type=int, default=5_000_000,
                        help="approximate nnz per operand (default: 5000000)")
    parser.add_argument("--col-density", type=float, default=0.5,
                        help="fraction of non-zeros within an active row (default: 0.5)")
    parser.add_argument("--skew", type=float, default=None,
                        help="fraction of A's active rows also active in B "
                             "(default: independent random rows)")
    args = parser.parse_args()

    row_density_list = [10 ** (-x) for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
    num_rows_list = [int(10 ** x) for x in [2, 3, 4, 5, 6]]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    skew_tag = f"{args.skew:.1f}" if args.skew is not None else "random"
    print(f"\n=== skew={skew_tag} ===")

    speedup = generate_heatmap(row_density_list, num_rows_list,
                               nnz_target=args.nnz_target, col_density=args.col_density,
                               skew=args.skew, n_trials=args.trials)

    plot_speedup_heatmap(
        speedup, row_density_list, num_rows_list,
        title=(f"CPU speedup: recursive vs single-phase partitioning "
               f"(skew={skew_tag}, >1 = recursive faster)"),
        save_path=str(config.RESULTS_DIR / f"dcsr_partitioning_speedup_skew{skew_tag}.png"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
