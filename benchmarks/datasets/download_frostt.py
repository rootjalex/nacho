"""Download FROSTT tensors for the 3-D benchmarks.

Files arrive gzipped and are decompressed in place, leaving `<name>.tns` under
config.FROSTT_DIR. They are not usable yet: `sort_frostt.py` has to run over each one
before the benchmarks can read it.

    python benchmarks/datasets/download_frostt.py            # what config asks for
    python benchmarks/datasets/download_frostt.py nell-2     # one named tensor

With no names, the set comes from config.FROSTT_TENSORS, so the download follows whatever
the benchmarks are configured to run.
"""

import argparse
import gzip
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Tensor name -> the file stem it lands under and where FROSTT serves it from. The stem is
# what config.FROSTT_TENSORS names, before sort_frostt.py appends its own suffix.
FROSTT_BASE = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data"
# frostt.io serves a few tensors from a second bucket, under their own capitalisation.
FROSTT_ALT = "https://frostt-tensors.s3.us-east-2.amazonaws.com"
TENSORS = {
    "nell-2":  ("nell-2",          f"{FROSTT_BASE}/nell/nell-2.tns.gz"),
    "nell-1":  ("nell-1",          f"{FROSTT_BASE}/nell/nell-1.tns.gz"),
    "darpa":   ("1998DARPA",       f"{FROSTT_ALT}/1998DARPA/1998darpa.tns.gz"),
    "fb-m":    ("fb-m",            f"{FROSTT_ALT}/FB-M/fb-m.tns.gz"),
    "amazon":  ("amazon-reviews",  f"{FROSTT_BASE}/amazon/amazon-reviews.tns.gz"),
    "patents": ("patents",         f"{FROSTT_BASE}/patents/patents.tns.gz"),
    "reddit":  ("reddit-2015",     f"{FROSTT_BASE}/reddit-2015/reddit-2015.tns.gz"),
}


def configured_names():
    """The tensors config.FROSTT_TENSORS asks for, in the order it lists them."""
    return [name for name, _stem, _dimensions in config.FROSTT_TENSORS]


def place_tensor(decompressed, target):
    """Move a decompressed download into place as target.

    Part of the collection is served as a gzipped tar rather than a bare tensor, so what
    comes out of gzip is sometimes an archive holding the .tns alongside the AppleDouble
    sidecars of whoever packed it.
    """
    if not tarfile.is_tarfile(decompressed):
        decompressed.replace(target)
        return

    with tarfile.open(decompressed) as archive:
        members = [member for member in archive.getmembers()
                   if member.isfile() and member.name.endswith(".tns")
                   and not Path(member.name).name.startswith("._")]
        if len(members) != 1:
            raise OSError(f"expected one .tns in the archive, found "
                          f"{[member.name for member in members]}")
        with archive.extractfile(members[0]) as source, open(target, "wb") as out:
            shutil.copyfileobj(source, out)


def download(destination, name):
    """Fetch one tensor and decompress it under destination."""
    stem, url = TENSORS[name]
    target = destination / f"{stem}.tns"
    if target.is_file():
        print(f"  {name}: {target} already present, skipped")
        return True

    archive = destination / f"{stem}.tns.gz"
    staged = destination / f"{stem}.tns.part"
    print(f"  {name}: {url}", flush=True)
    try:
        with urllib.request.urlopen(url) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        print(f"      decompressing to {target}", flush=True)
        with gzip.open(archive, "rb") as compressed, open(staged, "wb") as out:
            shutil.copyfileobj(compressed, out)
        place_tensor(staged, target)
    except (urllib.error.URLError, OSError, gzip.BadGzipFile, tarfile.TarError) as error:
        print(f"      failed: {error}")
        target.unlink(missing_ok=True)
        return False
    finally:
        archive.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tensors", nargs="*", metavar="TENSOR",
                        help=f"which tensors to download, from: {', '.join(sorted(TENSORS))}")
    parser.add_argument("--all", action="store_true",
                        help="download everything config.FROSTT_TENSORS lists (the default)")
    parser.add_argument("--dest", type=Path, default=config.FROSTT_DIR,
                        help="where to write the .tns files (default: config.FROSTT_DIR)")
    args = parser.parse_args()

    names = args.tensors or configured_names()
    unknown = [name for name in names if name not in TENSORS]
    if unknown:
        parser.error(
            f"no download URL for {', '.join(unknown)}. FROSTT serves: "
            f"{', '.join(sorted(TENSORS))}. Tensors named in config.FROSTT_TENSORS that "
            f"FROSTT does not host have to be fetched by hand.")

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"{len(names)} tensors into {args.dest}")
    failed = [name for name in names if not download(args.dest, name)]

    print(f"\nDone. Failed: {len(failed)}{' — ' + ', '.join(failed) if failed else ''}")
    print("Run sort_frostt.py over each .tns before benchmarking.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
