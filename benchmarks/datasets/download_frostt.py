"""Download FROSTT tensors for the 3-D benchmarks.

Files arrive gzipped and are decompressed in place, leaving `<name>.tns` under
FROSTT_DIR. They are not usable yet: `sort_frostt.py` has to run over each one before the
benchmarks can read it.

    python benchmarks/datasets/download_frostt.py nell-2
    python benchmarks/datasets/download_frostt.py --all

Sizes run from a few hundred MB to tens of GB decompressed, so tensors are named
individually rather than fetched as a set by default.
"""

import argparse
import gzip
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Tensor name -> the file stem it lands under and where FROSTT serves it from. The stem is
# what config.FROSTT_TENSORS names, before sort_frostt.py appends its own suffix.
FROSTT_BASE = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data"
TENSORS = {
    "nell-2":  ("nell-2",          f"{FROSTT_BASE}/nell/nell-2.tns.gz"),
    "nell-1":  ("nell-1",          f"{FROSTT_BASE}/nell/nell-1.tns.gz"),
    "darpa":   ("1998DARPA",       f"{FROSTT_BASE}/1998DARPA/1998DARPA.tns.gz"),
    "amazon":  ("amazon-reviews",  f"{FROSTT_BASE}/amazon/amazon-reviews.tns.gz"),
    "patents": ("patents",         f"{FROSTT_BASE}/patents/patents.tns.gz"),
    "reddit":  ("reddit-2015",     f"{FROSTT_BASE}/reddit-2015/reddit-2015.tns.gz"),
}


def download(destination, name):
    """Fetch one tensor and decompress it under destination."""
    stem, url = TENSORS[name]
    target = destination / f"{stem}.tns"
    if target.is_file():
        print(f"  {name}: {target} already present, skipped")
        return True

    archive = destination / f"{stem}.tns.gz"
    print(f"  {name}: {url}", flush=True)
    try:
        with urllib.request.urlopen(url) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        print(f"      decompressing to {target}", flush=True)
        with gzip.open(archive, "rb") as compressed, open(target, "wb") as out:
            shutil.copyfileobj(compressed, out)
    except (urllib.error.URLError, OSError, gzip.BadGzipFile) as error:
        print(f"      failed: {error}")
        target.unlink(missing_ok=True)
        return False
    finally:
        archive.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tensors", nargs="*", metavar="TENSOR",
                        help=f"which tensors to download, from: {', '.join(sorted(TENSORS))}")
    parser.add_argument("--all", action="store_true", help="download every known tensor")
    parser.add_argument("--dest", type=Path, default=config.FROSTT_DIR,
                        help="where to write the .tns files (default: NACHO_FROSTT_DIR)")
    args = parser.parse_args()

    names = sorted(TENSORS) if args.all else args.tensors
    if not names:
        parser.error("name at least one tensor, or pass --all. "
                     f"Known: {', '.join(sorted(TENSORS))}")
    unknown = [name for name in names if name not in TENSORS]
    if unknown:
        parser.error(f"unknown tensor(s) {', '.join(unknown)}. "
                     f"Known: {', '.join(sorted(TENSORS))}")

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"{len(names)} tensors into {args.dest}")
    failed = [name for name in names if not download(args.dest, name)]

    print(f"\nDone. Failed: {len(failed)}{' — ' + ', '.join(failed) if failed else ''}")
    print("Run sort_frostt.py over each .tns before benchmarking.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
