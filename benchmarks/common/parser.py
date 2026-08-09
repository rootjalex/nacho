"""Loading SuiteSparse matrices as torch sparse tensors.

Conversion into nacho tensor classes lives in the nacho package itself
(nacho.to_csr and friends).
"""

import os

import pandas as pd
import torch

import nacho

import config


def _candidate_paths(name):
    """The places `name` can sit under SUITESPARSE_DIR.

    A SuiteSparse tarball unpacks into a directory named after the matrix, so
    `1138_bus.mtx` arrives as `1138_bus/1138_bus.mtx`. Files placed directly in
    SUITESPARSE_DIR are accepted too.
    """
    return (config.SUITESPARSE_DIR / name,
            config.SUITESPARSE_DIR / name.removesuffix(".mtx") / name)


def matrix_path(name):
    """Where `name` can be read from, or None if it is unusable on this machine."""
    for candidate in _candidate_paths(name):
        if candidate.is_file() and os.access(candidate, os.R_OK):
            return candidate
    return None


def parsable(path):
    """Whether parse2D can read the file, judged from its Matrix Market banner.

    The banner is `%%MatrixMarket <object> <format> <field> <symmetry>`. parse2D reads a
    coordinate list of real values, and expects `rows cols nnz` on the dimension line --
    an `array` file gives only `rows cols`, so the missing count is read as garbage.
    """
    with open(path, errors="replace") as banner:
        fields = banner.readline().split()
    return (len(fields) > 4
            and fields[2] == "coordinate"
            and fields[3] == "real"
            and fields[4] in ("general", "symmetric", "skew-symmetric"))


def matrix_list():
    """SuiteSparse matrices readable on this machine, ascending by nnz."""
    df = pd.read_csv(config.STATS_CSV)
    df.columns = ["name", "nnz", "percent_nnz", "total_elements", "rows", "columns"]

    df["nnz"] = df["nnz"].astype(int)
    df["total_elements"] = df["total_elements"].astype(int)
    df["rows"] = df["rows"].astype(int)
    df["columns"] = df["columns"].astype(int)
    df["percent_nnz"] = df["percent_nnz"].astype(float)

    config.require_dataset_dir(config.SUITESPARSE_DIR, "SuiteSparse", "NACHO_SUITESPARSE_DIR")

    usable = df["name"].apply(lambda f: matrix_path(f) is not None)
    on_disk = df["name"].apply(lambda f: any(c.is_file() for c in _candidate_paths(f)))
    unreadable = int((on_disk & ~usable).sum())
    if unreadable:
        print(f"  {unreadable} matrices under {config.SUITESPARSE_DIR} are not readable "
              f"and were skipped")

    df = df[usable]

    supported = df["name"].apply(lambda f: parsable(matrix_path(f)))
    unsupported = int((~supported).sum())
    if unsupported:
        print(f"  {unsupported} matrices are not real-valued coordinate Matrix Market "
              f"files and were skipped")
    df = df[supported]

    if df.empty:
        raise FileNotFoundError(
            f"No matrices from {config.STATS_CSV.name} found in {config.SUITESPARSE_DIR}.\n"
            f"Set NACHO_SUITESPARSE_DIR to the directory holding the SuiteSparse .mtx files.")
    return df.sort_values(by=["nnz"], ascending=[True])


def parse_matrix(matrix, return_coo=False, device="cuda"):
    """Load a .mtx as a torch sparse tensor, CSR unless return_coo."""
    coo = nacho.parse2D(str(matrix_path(matrix)))

    row = coo.row.to(dtype=torch.long, device=device)
    col = coo.col.to(dtype=torch.long, device=device)
    indices = torch.stack([row, col], dim=0)
    values = coo.data.to(dtype=torch.float32, device=device)
    tensor = torch.sparse_coo_tensor(indices, values, (coo.N, coo.M))

    if return_coo:
        return tensor
    return tensor.to_sparse_csr()


def parse_vector(matrix):
    """Load a .mtx flattened column-major into a sorted (indices, values, length)."""
    coo = nacho.parse2D(str(matrix_path(matrix)))

    row = coo.row.to(dtype=torch.long, device="cuda")
    col = coo.col.to(dtype=torch.long, device="cuda")
    indices = col * coo.N + row
    indices, _ = torch.sort(indices)
    values = coo.data.to(dtype=torch.float32, device="cuda")
    return indices, values, coo.N * coo.M
