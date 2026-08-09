"""Loading SuiteSparse matrices as torch sparse tensors.

Conversion into nacho tensor classes lives in the nacho package itself
(nacho.to_csr and friends).
"""

import os

import pandas as pd
import torch

import nacho

import config


def matrix_list():
    """SuiteSparse matrices present on this machine, ascending by nnz."""
    df = pd.read_csv(config.STATS_CSV)
    df.columns = ["name", "nnz", "percent_nnz", "total_elements", "rows", "columns"]

    df["nnz"] = df["nnz"].astype(int)
    df["total_elements"] = df["total_elements"].astype(int)
    df["rows"] = df["rows"].astype(int)
    df["columns"] = df["columns"].astype(int)
    df["percent_nnz"] = df["percent_nnz"].astype(float)

    config.require_dataset_dir(config.SUITESPARSE_DIR, "SuiteSparse", "NACHO_SUITESPARSE_DIR")
    df = df[df["name"].apply(
        lambda f: os.path.exists(os.path.join(config.SUITESPARSE_DIR, f)))]
    if df.empty:
        raise FileNotFoundError(
            f"No matrices from {config.STATS_CSV.name} found in {config.SUITESPARSE_DIR}.\n"
            f"Set NACHO_SUITESPARSE_DIR to the directory holding the SuiteSparse .mtx files.")
    return df.sort_values(by=["nnz"], ascending=[True])


def parse_matrix(matrix, return_coo=False, device="cuda"):
    """Load a .mtx as a torch sparse tensor, CSR unless return_coo."""
    coo = nacho.parse2D(str(config.SUITESPARSE_DIR / matrix))

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
    coo = nacho.parse2D(str(config.SUITESPARSE_DIR / matrix))

    row = coo.row.to(dtype=torch.long, device="cuda")
    col = coo.col.to(dtype=torch.long, device="cuda")
    indices = col * coo.N + row
    indices, _ = torch.sort(indices)
    values = coo.data.to(dtype=torch.float32, device="cuda")
    return indices, values, coo.N * coo.M
