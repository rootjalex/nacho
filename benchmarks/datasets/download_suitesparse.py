"""Download the SuiteSparse matrices the 2-D benchmarks read.

The matrices, and the order the sweeps walk them in, come from
`benchmarks/suitesparse_stats.csv`. That file lists names only, so the collection's own
index is fetched first to find the group each matrix belongs to, which is what its
download URL is keyed on.

Each matrix arrives as a tarball unpacking to `<name>/<name>.mtx`, the layout
`common/parser.py` expects. Matrices already on disk are skipped, so an interrupted run
can be repeated.

    python benchmarks/datasets/download_suitesparse.py --limit 400

The full collection is several hundred GB. `--limit` takes the N smallest by non-zero
count, which is the prefix the `--start`/`--end` sweeps index into.
"""

import argparse
import io
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config

# Maps a matrix name to the group its URL is keyed on. Two header lines precede the rows.
INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
MATRIX_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"


def wanted_matrices(limit):
    """The matrix names the benchmarks use, ascending by non-zero count."""
    df = pd.read_csv(config.STATS_CSV, header=None,
                     names=["name", "nnz", "percent_nnz", "total_elements", "rows", "columns"])
    df = df.sort_values(by="nnz", ascending=True)
    names = [name.removesuffix(".mtx") for name in df["name"]]
    return names[:limit] if limit else names


def matrix_groups():
    """name -> group, from the collection's index."""
    print(f"Fetching the collection index from {INDEX_URL}")
    with urllib.request.urlopen(INDEX_URL) as response:
        text = response.read().decode()

    groups = {}
    for line in text.splitlines()[2:]:
        fields = line.split(",")
        if len(fields) >= 2:
            groups[fields[1]] = fields[0]
    print(f"  {len(groups)} matrices in the collection")
    return groups


def already_present(destination, name):
    """Whether the matrix can already be read from destination."""
    return (destination / name / f"{name}.mtx").is_file() or (destination / f"{name}.mtx").is_file()


def download(destination, name, group):
    """Fetch one matrix and unpack it under destination."""
    url = MATRIX_URL.format(group=group, name=name)
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=config.SUITESPARSE_DIR,
                        help="where to unpack the matrices (default: NACHO_SUITESPARSE_DIR)")
    parser.add_argument("--limit", type=int, default=0,
                        help="download only the N smallest matrices by nnz (default: all)")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    names = wanted_matrices(args.limit)
    groups = matrix_groups()

    print(f"\n{len(names)} matrices requested into {args.dest}")
    downloaded, skipped, failed = 0, 0, []
    for index, name in enumerate(names, start=1):
        if already_present(args.dest, name):
            skipped += 1
            continue
        if name not in groups:
            print(f"  [{index}/{len(names)}] {name}: not in the collection index, skipped")
            failed.append(name)
            continue

        print(f"  [{index}/{len(names)}] {groups[name]}/{name}", flush=True)
        try:
            download(args.dest, name, groups[name])
            downloaded += 1
        except (urllib.error.URLError, tarfile.TarError, OSError) as error:
            print(f"      failed: {error}")
            failed.append(name)

    print(f"\nDownloaded {downloaded}, already present {skipped}, failed {len(failed)}")
    if failed:
        print("Failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
