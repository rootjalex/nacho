"""Load-balance comparison for the DCSR Hadamard product: recursive vs single-phase.

Both kernels are generated from the same expression and differ only in how work is
partitioned. dcsr_mul re-partitions at each sparse intersection level; the
without_recursive_partitioning variant partitions once over the whole loop nest.

The sweep varies two things independently:
  skew       how many of A's active rows are also active in B (the intersection)
  dist_skew  where in A's row-index space those shared rows are drawn from, so the
             surviving work is uniform (0), right-tail biased (>0) or left-tail (<0)

The heatmap is % rows skipped against row skew, coloured by
single-phase time / recursive time, so >1 means recursive partitioning wins.

    python benchmarks/dcsr_lb_comparison.py --device cpu
    python benchmarks/dcsr_lb_comparison.py --device gpu
    python benchmarks/dcsr_lb_comparison.py --replot results.csv --out plot.png
"""

import argparse
import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import nacho

import config

RECURSIVE_CPU = "cpu_dcsr_mul_f32"
SINGLE_PHASE_CPU = "cpu_dcsr_mul_without_recursive_partitioning_f32"
RECURSIVE_GPU = "gpu_dcsr_mul_f32"
SINGLE_PHASE_GPU = "gpu_dcsr_mul_without_recursive_partitioning_f32"


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _build_dcsr_arrays(row_ids, num_rows, num_cols, col_density, dtype, device="cpu"):
    """Sample column indices for each active row and return DCSR arrays.

    The shape vector always stays on the host, whatever device the buffers are on.
    """
    active_rows = row_ids.numel()
    if active_rows == 0:
        return (row_ids,
                torch.zeros(1, dtype=dtype, device=device),
                torch.zeros(0, dtype=dtype, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
                torch.tensor([num_rows, num_cols], dtype=dtype))

    col_mask = torch.rand(active_rows, num_cols, device=device) < col_density
    nnz_per_row = col_mask.sum(dim=1, dtype=torch.int64)
    row_ptr = torch.zeros(active_rows + 1, dtype=torch.int64, device=device)
    row_ptr[1:] = nnz_per_row.cumsum(0)

    nnz = int(row_ptr[-1].item())
    col_idx = col_mask.nonzero(as_tuple=False)[:, 1].to(dtype)
    values = torch.randn(nnz, dtype=torch.float32, device=device)
    shape = torch.tensor([num_rows, num_cols], dtype=dtype)
    return row_ids, row_ptr.to(dtype), col_idx, values, shape


def get_synthetic_dcsr_pair_gpu(num_rows, num_cols, row_density, col_density,
                                skew=None, dist_skew=0.0, dtype=torch.int32):
    """A DCSR pair generated directly on the GPU.

    skew is the fraction of A's active rows that also appear in B. dist_skew biases
    which of A's rows those are: 0 uniform, >0 toward high row indices, <0 toward low.
    """
    device = "cuda"
    if skew is None:
        def _single():
            row_mask = torch.rand(num_rows, device=device) < row_density
            row_ids = row_mask.nonzero(as_tuple=False).squeeze(-1).to(dtype)
            return _build_dcsr_arrays(row_ids, num_rows, num_cols, col_density, dtype, device)
        return _single(), _single()

    row_mask_a = torch.rand(num_rows, device=device) < row_density
    row_ids_a = row_mask_a.nonzero(as_tuple=False).squeeze(-1).to(dtype)
    active_a = row_ids_a.numel()

    total_b = max(1, int(num_rows * row_density))
    shared_count = min(active_a, int(round(skew * min(active_a, total_b))))
    extra_count = max(0, total_b - shared_count)

    shared_rows = torch.zeros(0, dtype=dtype, device=device)
    if shared_count > 0 and active_a > 0:
        if dist_skew == 0.0:
            chosen = torch.randperm(active_a, device=device)[:shared_count]
        else:
            positions = (torch.arange(active_a, dtype=torch.float32, device=device) + 0.5) / active_a
            weights = (positions ** dist_skew if dist_skew > 0
                       else (1.0 - positions) ** (-dist_skew))
            chosen = torch.multinomial(weights, shared_count, replacement=False)
        shared_rows = row_ids_a[chosen]

    extra_rows = torch.zeros(0, dtype=dtype, device=device)
    inactive_rows = (~row_mask_a).nonzero(as_tuple=False).squeeze(-1).to(dtype)
    if extra_count > 0 and inactive_rows.numel() > 0:
        take = min(extra_count, inactive_rows.numel())
        extra_rows = inactive_rows[torch.randperm(inactive_rows.numel(), device=device)[:take]]

    row_ids_b, _ = torch.sort(torch.cat([shared_rows, extra_rows]))
    return (_build_dcsr_arrays(row_ids_a, num_rows, num_cols, col_density, dtype, device),
            _build_dcsr_arrays(row_ids_b, num_rows, num_cols, col_density, dtype, device))


def get_synthetic_dcsr_pair_cpu(num_rows, num_cols, row_density, col_density,
                                skew=None, dist_skew=0.0, dtype=torch.int32):
    """The same pair, generated on the GPU and moved to the host."""
    (ri_a, rp_a, ci_a, v_a, sh_a), (ri_b, rp_b, ci_b, v_b, sh_b) = \
        get_synthetic_dcsr_pair_gpu(num_rows, num_cols, row_density, col_density,
                                    skew=skew, dist_skew=dist_skew, dtype=dtype)
    return ((ri_a.cpu(), rp_a.cpu(), ci_a.cpu(), v_a.cpu(), sh_a),
            (ri_b.cpu(), rp_b.cpu(), ci_b.cpu(), v_b.cpu(), sh_b))


def make_dcsr_cpu(row_ids, row_ptr, col_idx, values, shape):
    return nacho.DCSR_cpu(row_ids, row_ptr, col_idx, values, shape)


def make_dcsr_gpu(row_ids, row_ptr, col_idx, values, shape):
    return nacho.DCSR_gpu(row_ids.contiguous(), row_ptr.contiguous(),
                          col_idx.contiguous(), values.contiguous(), shape)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def row_intersect_pct(row_ids_a, row_ids_b):
    """Percentage of rows active in both A and B, relative to their union."""
    set_a = set(row_ids_a.tolist())
    set_b = set(row_ids_b.tolist())
    union = len(set_a | set_b)
    if union == 0:
        return np.nan
    return 100.0 * len(set_a & set_b) / union


def _trimmed_mean(values):
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    trimmed = sorted_values[config.TRIM: len(sorted_values) - config.TRIM]
    return float(trimmed.mean()) if len(trimmed) > 0 else float(sorted_values.mean())


def _cpu_time_us(fn):
    import time
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1e6


def _gpu_time_us(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3


def flush_gpu_state():
    gc.collect()
    torch.cuda.empty_cache()
    # Push stale data out of L2 with a 40 MB zero-fill.
    torch.empty(int(40 * 1024 ** 2), dtype=torch.int8, device="cuda").zero_()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _dcsr_hadamard_seq(ri_a, rp_a, ci_a, v_a, ri_b, rp_b, ci_b, v_b, num_rows, num_cols):
    """Sequential reference: C[i,j] = A[i,j] * B[i,j], only where both are non-zero."""
    b_rows = {}
    for k in range(ri_b.numel()):
        start, end = int(rp_b[k]), int(rp_b[k + 1])
        b_rows[int(ri_b[k])] = {int(ci_b[idx]): float(v_b[idx]) for idx in range(start, end)}

    dense = torch.zeros(num_rows, num_cols, dtype=torch.float64)
    for k in range(ri_a.numel()):
        row = int(ri_a[k])
        if row not in b_rows:
            continue
        b_row = b_rows[row]
        for idx in range(int(rp_a[k]), int(rp_a[k + 1])):
            col = int(ci_a[idx])
            if col in b_row:
                dense[row, col] = float(v_a[idx]) * b_row[col]
    return dense.float()


def _result_to_dense(result, num_rows, num_cols):
    """Reconstruct a dense tensor from a DCSR result."""
    dense = torch.zeros(num_rows, num_cols, dtype=torch.float32)
    row_ids = result.dim_i_indices.long()
    offsets = result.dim_j_offsets.long()
    cols = result.dim_j_indices.long()
    values = result.values
    for k in range(row_ids.numel()):
        span = slice(int(offsets[k]), int(offsets[k + 1]))
        dense[int(row_ids[k]), cols[span]] = values[span]
    return dense


def run_sanity_check(n_cases=5, tol=1e-4, verbose=True):
    """Check both kernels against a sequential reference and against each other."""
    rule = "=" * 72
    print(f"\n{rule}")
    print("  Sanity check: recursive vs single-phase vs sequential reference")
    print(rule)

    configs = [
        dict(num_rows=50, num_cols=40, row_density=0.8, col_density=0.6),
        dict(num_rows=100, num_cols=80, row_density=0.5, col_density=0.4),
        dict(num_rows=80, num_cols=80, row_density=1.0, col_density=0.5),
        dict(num_rows=120, num_cols=100, row_density=0.3, col_density=0.2),
        dict(num_rows=60, num_cols=50, row_density=0.1, col_density=0.9),
    ]

    all_pass = True
    for case_index, cfg in enumerate(configs[:n_cases]):
        num_rows, num_cols = cfg["num_rows"], cfg["num_cols"]
        torch.manual_seed(case_index * 37 + 7)

        (ri_a, rp_a, ci_a, v_a, sh_a), (ri_b, rp_b, ci_b, v_b, sh_b) = \
            get_synthetic_dcsr_pair_cpu(num_rows, num_cols, cfg["row_density"],
                                        cfg["col_density"], skew=0.7)

        A = make_dcsr_cpu(ri_a, rp_a, ci_a, v_a, sh_a)
        B = make_dcsr_cpu(ri_b, rp_b, ci_b, v_b, sh_b)

        reference = _dcsr_hadamard_seq(ri_a, rp_a, ci_a, v_a,
                                       ri_b, rp_b, ci_b, v_b, num_rows, num_cols)
        recursive = getattr(nacho, RECURSIVE_CPU)(A, B)
        single_phase = getattr(nacho, SINGLE_PHASE_CPU)(A, B)

        recursive_dense = _result_to_dense(recursive, num_rows, num_cols)
        single_dense = _result_to_dense(single_phase, num_rows, num_cols)

        recursive_error = (recursive_dense - reference).abs().max().item()
        single_error = (single_dense - reference).abs().max().item()
        mutual_error = (recursive_dense - single_dense).abs().max().item()
        same_structure = (torch.equal(recursive.dim_j_offsets, single_phase.dim_j_offsets) and
                          torch.equal(recursive.dim_j_indices, single_phase.dim_j_indices))

        passed = max(recursive_error, single_error, mutual_error) < tol
        all_pass = all_pass and passed

        if verbose:
            print(f"  Case {case_index}: {num_rows}x{num_cols}  "
                  f"row_d={cfg['row_density']:.1f}  col_d={cfg['col_density']:.1f}  "
                  f"[{'PASS' if passed else 'FAIL'}]")
            print(f"    recursive    vs seq: max_err={recursive_error:.2e}")
            print(f"    single-phase vs seq: max_err={single_error:.2e}")
            print(f"    recursive vs single: max_err={mutual_error:.2e}  "
                  f"same_structure={same_structure}")

    print(rule)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES - check above'}")
    print(f"{rule}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def _sweep(configs, skew_values, dist_skew_values, n_seeds, on_gpu, verbose):
    """Time both kernels over every (config, skew, dist_skew, seed) combination."""
    recursive_key = "gpu_recursive_runtime_us" if on_gpu else "recursive_runtime_us"
    single_key = "gpu_single_phase_runtime_us" if on_gpu else "single_phase_runtime_us"
    recursive_fn = getattr(nacho, RECURSIVE_GPU if on_gpu else RECURSIVE_CPU)
    single_fn = getattr(nacho, SINGLE_PHASE_GPU if on_gpu else SINGLE_PHASE_CPU)
    make_pair = get_synthetic_dcsr_pair_gpu if on_gpu else get_synthetic_dcsr_pair_cpu
    make_tensor = make_dcsr_gpu if on_gpu else make_dcsr_cpu
    time_us = _gpu_time_us if on_gpu else _cpu_time_us
    label = "[GPU] " if on_gpu else ""

    records = []
    for cfg in configs:
        num_rows, num_cols = cfg["num_rows"], cfg["num_cols"]
        tag_base = cfg.get("tag", f"r{num_rows}")

        for skew in skew_values:
            for dist_skew in dist_skew_values:
                for seed in range(n_seeds):
                    torch.manual_seed(seed * 1000 + int(skew * 1000) + int(dist_skew * 100))
                    try:
                        (ri_a, rp_a, ci_a, v_a, sh_a), (ri_b, rp_b, ci_b, v_b, sh_b) = \
                            make_pair(num_rows, num_cols, cfg["row_density"],
                                      cfg["col_density"], skew=skew, dist_skew=dist_skew)

                        if ri_a.numel() < 2 or ri_b.numel() < 2:
                            continue

                        A = make_tensor(ri_a, rp_a, ci_a, v_a, sh_a)
                        B = make_tensor(ri_b, rp_b, ci_b, v_b, sh_b)
                        pct = row_intersect_pct(ri_a.cpu(), ri_b.cpu())

                        if on_gpu:
                            recursive_fn(A, B)
                            single_fn(A, B)
                            torch.cuda.synchronize()

                        recursive_times, single_times = [], []
                        for _ in range(config.ITER_COUNT):
                            recursive_times.append(time_us(lambda: recursive_fn(A, B)))
                            single_times.append(time_us(lambda: single_fn(A, B)))

                        recursive_us = _trimmed_mean(recursive_times)
                        single_us = _trimmed_mean(single_times)

                        tag = f"{tag_base}_sk{skew:.2f}_ds{dist_skew:.1f}_s{seed}"
                        records.append({
                            "row_intersect_pct": pct,
                            "skew": skew,
                            "dist_skew": dist_skew,
                            "seed": seed,
                            recursive_key: recursive_us,
                            single_key: single_us,
                            "tag": tag,
                        })

                        if verbose:
                            print(f"  {label}{tag_base}  intersect={pct:5.1f}%  "
                                  f"dist_skew={dist_skew:+.1f}  "
                                  f"rt={recursive_us:.0f}/{single_us:.0f} us")

                        if on_gpu:
                            del A, B, ri_a, rp_a, ci_a, v_a, ri_b, rp_b, ci_b, v_b
                            flush_gpu_state()

                    except Exception as error:  # noqa: BLE001 - keep sweeping past a bad cell
                        if verbose:
                            print(f"  {label}SKIPPED {tag_base} skew={skew:.2f} "
                                  f"dist_skew={dist_skew:.1f} seed={seed}: {error}")

    return records


def run_sweep(configs, skew_values, dist_skew_values, n_seeds=3, verbose=True):
    return _sweep(configs, skew_values, dist_skew_values, n_seeds, False, verbose)


def run_sweep_gpu(configs, skew_values, dist_skew_values, n_seeds=3, verbose=True):
    return _sweep(configs, skew_values, dist_skew_values, n_seeds, True, verbose)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_csv(records, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")


def load_csv(csv_path):
    return pd.read_csv(csv_path).to_dict("records")


def print_speedup_summary(records, recursive_col, single_col):
    """Geomean, min and max of single-phase time over recursive time."""
    recursive = np.array([r[recursive_col] for r in records], dtype=np.float64)
    single = np.array([r[single_col] for r in records], dtype=np.float64)

    valid = (recursive > 0) & (single > 0) & np.isfinite(recursive) & np.isfinite(single)
    if not valid.any():
        print("No valid runtime data for speedup summary.")
        return

    speedup = single[valid] / recursive[valid]
    rule = "=" * 72
    print(f"\n{rule}")
    print("  Runtime speedup: recursive vs single-phase partitioning")
    print("  (speedup = single_phase_time / recursive_time;  >1 = recursive faster)")
    print(rule)
    print(f"  Runs:    {valid.sum()}")
    print(f"  Geomean: {float(np.exp(np.log(speedup).mean())):.3f}x")
    print(f"  Min:     {speedup.min():.3f}x")
    print(f"  Max:     {speedup.max():.3f}x")
    print(f"{rule}\n")


def plot_speedup_heatmap(records, out_path, recursive_col, single_col):
    """% rows skipped against row skew, coloured by single-phase / recursive time."""
    FONT_SIZE = 6

    df = pd.DataFrame(records)
    required = {"skew", "dist_skew", recursive_col, single_col, "row_intersect_pct"}
    if not required.issubset(df.columns):
        print(f"Heatmap skipped - missing columns: {required - set(df.columns)}")
        return

    df = df[df[recursive_col].gt(0) & df[single_col].gt(0)].copy()
    df["speedup"] = df[single_col] / df[recursive_col]

    skew_values = sorted(df["skew"].unique())
    dist_skew_values = sorted(df["dist_skew"].unique())

    grid = np.full((len(dist_skew_values), len(skew_values)), np.nan)
    skipped_labels = []
    for j, skew in enumerate(skew_values):
        column = df[df["skew"] == skew]
        mean_skipped = 100.0 - column["row_intersect_pct"].mean()
        skipped_labels.append(f"{int(mean_skipped // 10) * 10 + 5}%")
        for i, dist_skew in enumerate(dist_skew_values):
            cell = column[column["dist_skew"] == dist_skew]["speedup"]
            if len(cell) > 0:
                grid[i, j] = cell.mean()

    if not np.isfinite(grid).any():
        print("No data for heatmap.")
        return

    figure = plt.figure(figsize=(1.5, 1.5))
    ax_left, ax_bottom, ax_width, ax_height = 0.26, 0.26, 0.50, 0.56
    cbar_gap, cbar_width = 0.02, 0.04

    axes = figure.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    cbar_axes = figure.add_axes(
        [ax_left + ax_width + cbar_gap, ax_bottom, cbar_width, ax_height])

    image = axes.imshow(grid, cmap="viridis", aspect="auto", origin="upper")
    colorbar = figure.colorbar(image, cax=cbar_axes)
    colorbar.ax.tick_params(labelsize=FONT_SIZE, pad=1, length=2)
    colorbar.set_label("Speedup", fontsize=FONT_SIZE, labelpad=2)

    axes.set_xticks(range(len(skew_values)))
    axes.set_xticklabels(skipped_labels, fontsize=FONT_SIZE, rotation=45,
                         ha="right", rotation_mode="anchor")
    axes.set_yticks(range(len(dist_skew_values)))
    axes.set_yticklabels([f"{v:.1f}" for v in dist_skew_values], fontsize=FONT_SIZE)

    axes.xaxis.set_ticks_position("bottom")
    axes.tick_params(axis="both", which="major", labelsize=FONT_SIZE, pad=2, length=2)
    axes.set_xlabel("% rows skipped", fontsize=FONT_SIZE, labelpad=2)
    axes.set_ylabel("row skew", fontsize=FONT_SIZE, labelpad=2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    figure.savefig(out_path, dpi=300, bbox_inches="tight")
    if os.path.splitext(out_path)[1].lower() not in (".eps", ".pdf"):
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        figure.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Saved heatmap: {out_path}  {pdf_path}")
    else:
        print(f"Saved heatmap: {out_path}")
    plt.close(figure)


def _columns_for(records):
    """Pick the CPU or GPU runtime columns, whichever the records carry."""
    if records and "gpu_recursive_runtime_us" in records[0]:
        return "gpu_recursive_runtime_us", "gpu_single_phase_runtime_us"
    return "recursive_runtime_us", "single_phase_runtime_us"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(num_rows=100_000, num_cols=100_000, row_density=0.15, col_density=0.04,
         tag="medium"),
]

# Chosen so the intersection percentage lands near the midpoint of each 10% bin of
# "% rows skipped" (= 100 - intersection%). skew = 2t/(1+t).
SKEW_VALUES = [0.10, 0.26, 0.40, 0.52, 0.62]
DIST_SKEW_VALUES = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="which pair of kernels to time (default: cpu)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="random structures per (skew, dist_skew) cell (default: 3)")
    parser.add_argument("--csv", default=None,
                        help="write records here; if the file exists, replot it "
                             "instead of sweeping")
    parser.add_argument("--out", default=None, help="output image path")
    parser.add_argument("--replot", metavar="CSV", default=None,
                        help="replot an existing CSV and exit")
    parser.add_argument("--skip-sanity-check", action="store_true",
                        help="skip the correctness check before the CPU sweep")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.replot:
        records = load_csv(args.replot)
        recursive_col, single_col = _columns_for(records)
        out_path = args.out or os.path.splitext(args.replot)[0] + "_heatmap.png"
        print_speedup_summary(records, recursive_col, single_col)
        plot_speedup_heatmap(records, out_path, recursive_col, single_col)
        return

    on_gpu = args.device == "gpu"
    suffix = "gpu" if on_gpu else "cpu"
    csv_path = args.csv or str(config.RESULTS_DIR / f"dcsr_lb_comparison_{suffix}.csv")
    out_path = args.out or str(config.RESULTS_DIR / f"dcsr_lb_comparison_{suffix}_heatmap.png")

    if args.csv and os.path.exists(args.csv):
        print(f"Loading records from {args.csv} (skipping sweep) ...")
        records = load_csv(args.csv)
    else:
        if not on_gpu and not args.skip_sanity_check:
            run_sanity_check()

        total = len(CONFIGS) * len(SKEW_VALUES) * len(DIST_SKEW_VALUES) * args.seeds
        print(f"Sweeping {len(CONFIGS)} configs x {len(SKEW_VALUES)} intersection points "
              f"x {len(DIST_SKEW_VALUES)} dist_skew x {args.seeds} seeds = {total} runs\n")

        sweep = run_sweep_gpu if on_gpu else run_sweep
        records = sweep(CONFIGS, SKEW_VALUES, DIST_SKEW_VALUES, n_seeds=args.seeds)
        if not records:
            print("No successful runs - nothing to plot.")
            return
        save_csv(records, csv_path)

    recursive_col, single_col = _columns_for(records)
    print_speedup_summary(records, recursive_col, single_col)
    plot_speedup_heatmap(records, out_path, recursive_col, single_col)


if __name__ == "__main__":
    main()
