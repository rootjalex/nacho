"""Shared plotting for the benchmark scripts.

Every plot also writes an .npz of its raw series next to the figure, so a run can be
re-plotted without re-benchmarking. Output goes to config.RESULTS_DIR.

Figures are typeset with LaTeX to match the paper. Set NACHO_PLOT_USETEX=0 on a machine
without a LaTeX install.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import config

matplotlib.rcParams["text.usetex"] = os.environ.get("NACHO_PLOT_USETEX", "1") == "1"
matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.titlesize": 8,
})

# Colourblind-safe palette, one entry per series that can appear in a scatter.
SERIES_COLORS = {
    "Nacho": "#88CCEE",
    "Nacho Unfused": "#332288",
    "cuSPARSE": "#117733",
    "PyTorch": "#CC6677",
    "Intel MKL": "#DDCC77",
    "Taco": "#AA4499",
}


def _results_path(name, suffix):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(config.RESULTS_DIR / f"{name}{suffix}")


# Vector formats the paper figures are written in: EPS for LaTeX, PDF to view directly.
FIGURE_FORMATS = ("eps", "pdf")


def _save_figure(name, **savefig_kwargs):
    """Write the current figure once per entry in FIGURE_FORMATS."""
    for extension in FIGURE_FORMATS:
        path = _results_path(name, f".{extension}")
        plt.savefig(path, format=extension, **savefig_kwargs)
        print(f"Saved figure: {path}")


def plot(nnz, manual, cusparse, pytorch, name):
    """Runtime against total nnz, on log-log axes."""
    np.savez(_results_path(name, ".npz"), nnz=nnz, manual=manual,
             cusparse=cusparse, pytorch=pytorch)

    plt.figure(figsize=(8, 6))
    plt.scatter(nnz, manual, label="Nacho", alpha=0.7, color="blue", marker="o", s=2)
    if cusparse is not None and len(cusparse) != 0:
        plt.scatter(nnz, cusparse, label="Nacho unfused", alpha=0.7, color="red", marker="o", s=2)
    if pytorch is not None and len(pytorch) != 0:
        plt.scatter(nnz, pytorch, label="PyTorch", alpha=0.7, color="green", marker="o", s=2)

    plt.xscale("log")
    plt.yscale("log")
    for axis in (plt.gca().xaxis, plt.gca().yaxis):
        axis.set_major_locator(ticker.LogLocator(base=10.0))
        axis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    plt.xlabel("Total Non-Zeros (nnzA + nnzB + nnzC)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Non-Zeros")
    plt.legend()
    plt.grid(True)
    plt.savefig(_results_path(name, ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter(filename, x_data, x_label, nacho, cusparse=None, pytorch=None, taco=None,
                 unfused=None, pytorch_as_mkl=False, save=True):
    """Runtime against x_data for each implementation, on log-log axes.

    A None entry in a series marks a run that did not complete (out of memory or timeout);
    those points are dropped rather than plotted. Passing None for a whole series omits it.
    """
    if save:
        np.savez(_results_path(filename, ".npz"), x=x_data, nacho=nacho,
                 cusparse=cusparse, pytorch=pytorch, taco=taco, unfused=unfused)

    plt.figure(figsize=(3.33, 1.15))

    def add_series(data, name):
        if data is None:
            return
        if len(x_data) != len(data):
            raise ValueError(
                f"x_data has {len(x_data)} points but the {name} series has {len(data)}")
        x = np.array(x_data)
        y = np.array(data, dtype=object)
        present = np.array([value is not None for value in y])
        if not np.any(present):
            return
        plt.scatter(x[present], np.maximum(y[present].astype(float), 1e-12),
                    label=name, color=SERIES_COLORS[name], marker=".", s=2, alpha=0.7)

    add_series(taco, "Taco")
    add_series(pytorch, "Intel MKL" if pytorch_as_mkl else "PyTorch")
    add_series(cusparse, "cuSPARSE")
    add_series(unfused, "Nacho Unfused")
    add_series(nacho, "Nacho")

    plt.xscale("log")
    plt.yscale("log")
    axes = plt.gca()
    for axis in (axes.xaxis, axes.yaxis):
        axis.set_major_locator(ticker.LogLocator(base=10.0))
        axis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
        axis.set_minor_locator(ticker.NullLocator())

    plt.xlabel(x_label)
    plt.ylabel("Runtime (ms)")

    # Nacho first in the legend, the baselines after it in plotting order.
    handles, labels = axes.get_legend_handles_labels()
    order = ([labels.index("Nacho")] if "Nacho" in labels else []) + \
            [i for i, label in enumerate(labels) if label != "Nacho"]
    plt.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper left", markerscale=5)
    plt.grid(True)

    _save_figure(filename, bbox_inches="tight", pad_inches=.02)
    plt.close()

    compute_stats("cuSPARSE", cusparse, nacho)
    compute_stats("Intel MKL" if pytorch_as_mkl else "PyTorch", pytorch, nacho)
    compute_stats("TACO", taco, nacho)
    compute_stats("Nacho Unfused", unfused, nacho)


def plot_heatmap(filename, data, exponents, x_label, y_label):
    """Speedup over a grid of two parameters, as a fixed 1.5 x 1.5 inch figure.

    The axes are placed at hardcoded fractions of the figure rather than by a layout
    engine, and the figure is saved without a tight bounding box, so every panel comes
    out the same size whatever the tick labels are.
    """
    np.savez(_results_path(filename, ".npz"), speedups=data, exponents=exponents)

    font_size = 6
    figure = plt.figure(figsize=(1.5, 1.5))

    left, bottom, width, height = 0.26, 0.26, 0.50, 0.56
    gap, bar_width = 0.02, 0.04
    axes = figure.add_axes([left, bottom, width, height])
    bar_axes = figure.add_axes([left + width + gap, bottom, bar_width, height])

    # imshow with aspect="auto" fills the axes box exactly rather than forcing square
    # pixels, which would shrink the map away from the size fixed above.
    image = axes.imshow(data, cmap="viridis", aspect="auto", origin="upper")

    bar = figure.colorbar(image, cax=bar_axes)
    bar.ax.tick_params(labelsize=font_size, pad=1, length=2)
    bar.set_label("Speedup", fontsize=font_size, labelpad=2)

    axes.set_xticks(np.arange(len(exponents)))
    axes.set_yticks(np.arange(len(exponents)))
    axes.set_xticklabels([f"{e:.1f}" for e in exponents], fontsize=font_size,
                         rotation=45, ha="right", rotation_mode="anchor")
    axes.set_yticklabels([f"{e:.1f}" for e in exponents], fontsize=font_size)
    axes.xaxis.set_ticks_position("bottom")
    axes.tick_params(axis="both", which="major", labelsize=font_size, pad=2, length=2)
    axes.set_xlabel(x_label, fontsize=font_size, labelpad=2)
    axes.set_ylabel(y_label, fontsize=font_size, labelpad=2)

    _save_figure(filename)
    plt.close(figure)


def compute_stats(name, baseline, nacho):
    """Print the speedup distribution of nacho over one baseline series."""
    if baseline is None:
        return

    nacho_arr = np.array(nacho, dtype=object)
    base_arr = np.array(baseline, dtype=object)

    nacho_timeouts = np.sum([n is None for n in nacho_arr])
    base_timeouts = np.sum([b is None for b in base_arr])

    comparable = np.array([
        (n is not None) and (b is not None) and (n > 0) and (b > 0)
        for n, b in zip(nacho_arr, base_arr)
    ])
    n_valid = int(np.sum(comparable))
    n_total = len(nacho_arr)

    print(f"\n=== Nacho vs {name} ===")
    if n_valid == 0:
        print("No valid comparisons.")
        print(f"Nacho timeouts: {nacho_timeouts}")
        print(f"{name} timeouts: {base_timeouts}")
        return

    speedups = base_arr[comparable].astype(float) / nacho_arr[comparable].astype(float)

    print(f"Total benchmarks: {n_total}")
    print(f"Valid comparisons: {n_valid}")
    print(f"\nTimeouts:\n  Nacho   : {nacho_timeouts}\n  {name:<8}: {base_timeouts}")
    print("\nSpeedup (baseline / nacho):")
    print(f"  Geomean: {np.exp(np.mean(np.log(speedups))):.3f}x")
    print(f"  Median : {np.median(speedups):.3f}x")
    print(f"  Min    : {np.min(speedups):.3f}x")
    print(f"  Max    : {np.max(speedups):.3f}x")
    print("\nDistribution:")
    print(f"  p25 / p75: {np.percentile(speedups, 25):.3f}x / {np.percentile(speedups, 75):.3f}x")
    print(f"  >=2x speedup: {np.sum(speedups >= 2.0) / n_valid * 100:.1f}%")
    print(f"  <1x (losses): {np.sum(speedups < 1.0) / n_valid * 100:.1f}%")
    print("\nWin/Loss:")
    print(f"  Wins  : {int(np.sum(speedups > 1.0))}")
    print(f"  Losses: {int(np.sum(speedups < 1.0))}")
    print(f"  Ties  : {int(np.sum(np.isclose(speedups, 1.0)))}")
