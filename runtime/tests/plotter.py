"""
Plot benchmark results from CSV files.

Usage:
    python runtime/tests/plotter.py                          # plot all CSVs
    python runtime/tests/plotter.py csr_add_0-1712           # plot specific CSV (no .csv extension)
    python runtime/tests/plotter.py csr_add_0-1712 spgemm_0-1712  # multiple CSVs
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no plt.show() popups
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import glob as glob_mod
import os

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmark_results')
os.makedirs(_RESULTS_DIR, exist_ok=True)


def _path(name, ext):
    return os.path.join(_RESULTS_DIR, f"{name}.{ext}")


def load_and_plot(file_names, plot_name):
    frames = []
    for file in file_names:
        frames.append(pd.read_csv(_path(file, "csv")))
    df = pd.concat(frames, ignore_index=True)
    plot(df["nnz"].tolist(), df["manual_ms"].tolist(),
         df["cusparse_ms"].tolist(), df["pytorch_ms"].tolist(), plot_name)


def plot_3(nnz, lengths, full, partial, no, name):
    pd.DataFrame({
        "nnz": [nnz] * len(lengths), "length": lengths,
        "full_ms": full, "partial_ms": partial, "nofusion_ms": no,
    }).to_csv(_path(name, "csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(lengths / nnz, full, label="LB on (a,b,c)", alpha=0.7, color="blue", marker="o", s=5)
    plt.scatter(lengths / nnz, partial, label="LB on (a,b)", alpha=0.7, color="red", marker="o", s=5)
    plt.scatter(lengths / nnz, no, label="LB on (c)", alpha=0.7, color="green", marker="o", s=5)

    plt.xscale("log")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.xlabel("Size |a|=|b|=|c|")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Fixed %nnz , varying size , |a|=|b|=|c| (%nnz = {nnz})")
    plt.legend()
    plt.grid(True)
    plt.savefig(_path(name, "pdf"), bbox_inches="tight")
    plt.close()


def plot_2(size, lengths, full, partial, no, name):
    pd.DataFrame({
        "size": [size] * len(lengths), "length": lengths,
        "full_ms": full, "partial_ms": partial, "nofusion_ms": no,
    }).to_csv(_path(name, "csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(lengths, full, label="LB on (a,b,c)", alpha=0.7, color="blue", marker="o", s=5)
    plt.scatter(lengths, partial, label="LB on (a,b)", alpha=0.7, color="red", marker="o", s=5)
    plt.scatter(lengths, no, label="LB on (c)", alpha=0.7, color="green", marker="o", s=5)

    plt.xscale("log")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.xlabel("NNZ (|a|=|b|=|c|)")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Fixed Length , varying sparsity , |a|=|b|=|c| (Length = 2*1e9)")
    plt.legend()
    plt.grid(True)
    plt.savefig(_path(name, "pdf"), bbox_inches="tight")
    plt.close()


def _speedup_stats(baseline, other):
    """Compute geomean/min/max speedup of baseline over other (other_time / baseline_time)."""
    b = np.array(baseline, dtype=float)
    o = np.array(other, dtype=float)
    mask = (b > 0) & (o > 0) & np.isfinite(b) & np.isfinite(o)
    if mask.sum() == 0:
        return None
    ratios = o[mask] / b[mask]
    return {
        "geomean": float(np.exp(np.mean(np.log(ratios)))),
        "min": float(ratios.min()),
        "max": float(ratios.max()),
        "n": int(mask.sum()),
    }


def plot(nnz, manual, cusparse, pytorch, name, labels=None, colors=None, markers=None):
    if labels is None:
        labels = ("Manual", "cusparse", "pytorch")
    plt.figure(figsize=(8, 6))
    # Wong's colorblind-safe palette (defaults)
    c_blue   = "#0072B2"
    c_orange = "#D55E00"
    c_teal   = "#009E73"
    if colors is None:
        colors = (c_blue, c_orange, c_teal)
    if markers is None:
        markers = ("o", "^", "s")
    plt.scatter(nnz, manual, label=labels[0], alpha=0.4, color=colors[0], marker=markers[0], s=5)
    if len(cusparse) != 0:
        plt.scatter(nnz, cusparse, label=labels[1], alpha=0.4, color=colors[1], marker=markers[1], s=5)
    if len(pytorch) != 0:
        plt.scatter(nnz, pytorch, label=labels[2], alpha=0.4, color=colors[2], marker=markers[2], s=5)

    # Compute and display speedup stats
    stat_lines = []
    if len(cusparse) != 0:
        s = _speedup_stats(manual, cusparse)
        if s:
            stat_lines.append(f"{labels[0]} vs {labels[1]}: {s['geomean']:.2f}x geomean  [{s['min']:.2f}x, {s['max']:.2f}x]  (n={s['n']})")
    if len(pytorch) != 0:
        s = _speedup_stats(manual, pytorch)
        if s:
            stat_lines.append(f"{labels[0]} vs {labels[2]}: {s['geomean']:.2f}x geomean  [{s['min']:.2f}x, {s['max']:.2f}x]  (n={s['n']})")

    plt.xscale("log")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.xlabel("Total Non-Zeros (nnzA + nnzB)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Non-Zeros")
    plt.legend(fontsize=12, markerscale=3)
    plt.grid(True)

    # Save version without stats
    plt.savefig(_path(name + "_clean", "pdf"), bbox_inches="tight")

    # Save version with stats
    if stat_lines:
        plt.figtext(0.5, -0.02, "\n".join(stat_lines), ha="center", fontsize=7, family="monospace")
    plt.savefig(_path(name, "pdf"), bbox_inches="tight")
    plt.close()

    # Print stats to console
    for line in stat_lines:
        print(f"  {line}")


def plot_bar_graph(size, lengths, full_lb, partial_lb, single_lb, name):
    stage_names = ['Merge-path', 'pre-compute', 'compute']
    color_1 = ['#c6dbef', '#6baed6', '#2171b5']
    color_2 = ['#fdd0a2', '#fd8d3c', '#d94801']
    color_3 = ['#c7e9c0', '#74c476', '#238b45']

    plt.figure(figsize=(8, 6))
    plt.xlabel("NNZ |a|=|b|=|c|")
    plt.ylabel("Runtime (ms)")
    plt.title("Fixed length , varying sparsity , Length = 2*10^9 ")

    legend_patches = []
    legend_patches += [Patch(color=c, label="l.b (a,b,c)  " + s) for c, s in zip(color_1, stage_names)]
    legend_patches += [Patch(color=c, label="l.b (a,b)  " + s) for c, s in zip(color_2, stage_names)]
    legend_patches += [Patch(color=c, label="l.b (c)  " + s) for c, s in zip(color_3, stage_names)]

    plt.xscale("log")
    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.legend(handles=legend_patches, title="Stages")
    plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(_path(name, "pdf"), bbox_inches='tight')
    plt.close()


def load_and_plot2(file_names, plot_name):
    frames = []
    for file in file_names:
        frames.append(pd.read_csv(_path(file, "csv")))
    df = pd.concat(frames, ignore_index=True)

    full_lb = df[["full_mergepath_ms", "full_precompute_ms", "full_compute_ms"]].values.tolist()
    partial_lb = df[["partial_mergepath_ms", "partial_precompute_ms", "partial_compute_ms"]].values.tolist()
    single_lb = df[["nofusion_mergepath_ms", "nofusion_precompute_ms", "nofusion_compute_ms"]].values.tolist()
    lengths = df["total_nnz"].tolist()

    plot_bar_graph_2(0, lengths, full_lb, partial_lb, single_lb, plot_name)


def plot_bar_graph_2(size, lengths, full_lb, partial_lb, single_lb, name):
    # Save CSV
    rows = []
    for i in range(len(lengths)):
        rows.append({
            "total_nnz": lengths[i],
            "full_mergepath_ms": full_lb[i][0],
            "full_precompute_ms": full_lb[i][1],
            "full_compute_ms": full_lb[i][2],
            "partial_mergepath_ms": partial_lb[i][0],
            "partial_precompute_ms": partial_lb[i][1],
            "partial_compute_ms": partial_lb[i][2],
            "nofusion_mergepath_ms": single_lb[i][0],
            "nofusion_precompute_ms": single_lb[i][1],
            "nofusion_compute_ms": single_lb[i][2],
        })
    pd.DataFrame(rows).to_csv(_path(name, "csv"), index=False)

    # Plot
    stage_names = ['Merge-path', 'pre-compute + compute', 'complete']
    color_1 = ['#c6dbef', '#6baed6', '#2171b5']
    color_2 = ['#fdd0a2', '#fd8d3c', '#d94801']
    color_3 = ['#c7e9c0', '#74c476', '#238b45']

    plt.figure(figsize=(8, 6))
    full_lb = np.array(full_lb)
    partial_lb = np.array(partial_lb)
    single_lb = np.array(single_lb)

    plt.scatter(lengths, full_lb[:, 0], label="LB on (a,b,c) - Merge-path", alpha=0.7, color=color_1[0], marker="o", s=2)
    plt.scatter(lengths, full_lb[:, 1] + full_lb[:, 2], label="LB on (a,b,c) - pre-compute + compute", alpha=0.7, color=color_1[1], marker="o", s=2)
    plt.scatter(lengths, full_lb[:, 0] + full_lb[:, 1] + full_lb[:, 2], label="LB on (a,b,c) - complete", alpha=0.7, color=color_1[2], marker="o", s=2)

    plt.scatter(lengths, partial_lb[:, 0], label="LB on (a,b) - Merge-path", alpha=0.7, color=color_2[0], marker="o", s=2)
    plt.scatter(lengths, partial_lb[:, 1] + partial_lb[:, 2], label="LB on (a,b) - pre-compute + compute", alpha=0.7, color=color_2[1], marker="o", s=2)
    plt.scatter(lengths, partial_lb[:, 0] + partial_lb[:, 1] + partial_lb[:, 2], label="LB on (a,b) - complete", alpha=0.7, color=color_2[2], marker="o", s=2)

    plt.scatter(lengths, single_lb[:, 0], label="LB on (c) - Merge-path", alpha=0.7, color=color_3[0], marker="o", s=2)
    plt.scatter(lengths, single_lb[:, 1] + single_lb[:, 2], label="LB on (c) - pre-compute + compute", alpha=0.7, color=color_3[1], marker="o", s=2)
    plt.scatter(lengths, single_lb[:, 0] + single_lb[:, 1] + single_lb[:, 2], label="LB on (c) - complete", alpha=0.7, color=color_3[2], marker="o", s=2)

    plt.xlabel("Total NNZ |a|+|b|+|c|")
    plt.ylabel("Runtime (ms)")
    plt.title("ab+c - varying sparsity")

    legend_patches = []
    legend_patches += [Patch(color=c, label="l.b (a,b,c)  " + s) for c, s in zip(color_1, stage_names)]
    legend_patches += [Patch(color=c, label="l.b (a,b)  " + s) for c, s in zip(color_2, stage_names)]
    legend_patches += [Patch(color=c, label="l.b (c)  " + s) for c, s in zip(color_3, stage_names)]

    plt.yscale("log")
    plt.xscale("log")
    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.legend(handles=legend_patches, title="Stages")
    plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(_path(name, "pdf"), bbox_inches='tight')
    plt.close()


def _plot_csv(name):
    """Auto-detect CSV type and plot it."""
    path = _path(name, "csv")
    if not os.path.isfile(path):
        print(f"Not found: {path}")
        return False
    df = pd.read_csv(path)
    if "full_mergepath_ms" in df.columns:
        load_and_plot2([name], name)
    elif "manual_ms" in df.columns:
        load_and_plot([name], name)
    else:
        print(f"Unknown CSV format: {path}")
        return False
    print(f"Saved: {_path(name, 'pdf')}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV files")
    parser.add_argument("csvs", nargs="*", help="CSV names (without .csv extension). If none, plots all.")
    args = parser.parse_args()

    if args.csvs:
        names = args.csvs
    else:
        names = sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob_mod.glob(os.path.join(_RESULTS_DIR, "*.csv"))
        )
        if not names:
            print(f"No CSV files found in {_RESULTS_DIR}")
            exit(1)
        print(f"Found {len(names)} CSV file(s)")

    for name in names:
        _plot_csv(name)
