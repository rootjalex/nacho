"""Sort and deduplicate a FROSTT tensor into the form the benchmarks require.

The generated kernels co-iterate their operands in coordinate order and assume each
coordinate appears once, which the FROSTT files as published do not guarantee. This
rewrites one into `<name>_sorted.tns`: coordinates in lexicographic order, values of
duplicate coordinates summed.

    python benchmarks/datasets/sort_frostt.py nell-2
    python benchmarks/datasets/sort_frostt.py --all

Indices stay 1-based, as `.tns` files are. Everything is held in memory at once, so peak
usage is roughly 40 bytes per non-zero — around 3 GB for nell-2, and considerably more
for the larger tensors.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config

# Tensors whose modes are permuted on the way out, as the new order of the original modes.
# fb-m is (23344784, 23344784, 166): with the short mode innermost, a level tree spends two
# levels of 23M coordinates before reaching anything that fans out. Leading with it gives
# 166 outer coordinates instead.
MODE_ORDER = {
    "fb-m": (2, 0, 1),
}


def permutation_tag(modes):
    """The mode order as letters, e.g. (2, 0, 1) -> 'kij', for the output file name."""
    return "".join("ijk"[mode] for mode in modes)


def read_tns(path):
    """(coordinates, values) from a whitespace-separated `i j k value` file."""
    print(f"  reading {path}", flush=True)
    frame = pd.read_csv(path, sep=r"\s+", header=None,
                        names=["i", "j", "k", "value"],
                        dtype={"i": np.int64, "j": np.int64, "k": np.int64,
                               "value": np.float64})
    coordinates = frame[["i", "j", "k"]].to_numpy()
    values = frame["value"].to_numpy()
    print(f"  {len(values):,} non-zeros", flush=True)
    return coordinates, values


def sort_and_coalesce(coordinates, values):
    """Lexicographically ordered coordinates, with duplicates summed into one entry."""
    print("  sorting", flush=True)
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    coordinates, values = coordinates[order], values[order]
    del order

    # Sorting puts equal coordinates next to each other, so a run is one output entry.
    starts = np.ones(len(coordinates), dtype=bool)
    starts[1:] = np.any(coordinates[1:] != coordinates[:-1], axis=1)
    run_starts = np.flatnonzero(starts)

    duplicates = len(coordinates) - len(run_starts)
    if duplicates:
        print(f"  summing {duplicates:,} duplicate coordinates "
              f"({len(coordinates):,} -> {len(run_starts):,} non-zeros)", flush=True)
        return coordinates[run_starts], np.add.reduceat(values, run_starts)
    print("  no duplicate coordinates", flush=True)
    return coordinates, values


def write_tns(path, coordinates, values):
    print(f"  writing {path}", flush=True)
    np.savetxt(path, np.column_stack((coordinates, values)), fmt="%d %d %d %g")


def process(directory, stem):
    source = directory / f"{stem}.tns"
    if not source.is_file():
        print(f"{stem}: {source} not found, skipped")
        return False

    modes = MODE_ORDER.get(stem)
    suffix = "_sorted" if modes is None else f"_sorted_{permutation_tag(modes)}"
    target = directory / f"{stem}{suffix}.tns"
    print(f"{stem}:")
    coordinates, values = read_tns(source)
    if modes is not None:
        # Permute before sorting, so the lexicographic order below is over the new modes.
        print(f"  reordering modes to {permutation_tag(modes)}", flush=True)
        coordinates = coordinates[:, list(modes)]
    coordinates, values = sort_and_coalesce(coordinates, values)
    write_tns(target, coordinates, values)
    if modes is not None:
        print(f"  dimensions are permuted: FROSTT_TENSORS must list them in "
              f"{permutation_tag(modes)} order", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tensors", nargs="*",
                        help="file stems to process, without the .tns suffix")
    parser.add_argument("--all", action="store_true",
                        help="process every .tns under the directory that is not already sorted")
    parser.add_argument("--dir", type=Path, default=config.FROSTT_DIR,
                        help="directory holding the .tns files (default: NACHO_FROSTT_DIR)")
    args = parser.parse_args()

    config.require_dataset_dir(args.dir, "FROSTT", "NACHO_FROSTT_DIR")

    stems = args.tensors
    if args.all:
        stems = sorted(path.stem for path in args.dir.glob("*.tns")
                       if "_sorted" not in path.stem)
    if not stems:
        parser.error("name at least one tensor stem, or pass --all")

    processed = sum(process(args.dir, stem) for stem in stems)
    print(f"\nSorted {processed} of {len(stems)} tensors.")
    print("Add each one to FROSTT_TENSORS in benchmarks/config.py to benchmark it.")
    return 0 if processed == len(stems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
