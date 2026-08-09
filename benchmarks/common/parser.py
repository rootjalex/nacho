"""Loading SuiteSparse matrices into torch tensors and into nacho's tensor classes."""

import os

import numpy as np
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

    df = df[df["name"].apply(lambda f: os.path.exists(os.path.join(config.MATRIX_DIR, f)))]
    return df.sort_values(by=["nnz"], ascending=[True])


def parse_matrix(matrix, return_coo=False, device="cuda"):
    """Load a .mtx as a torch sparse tensor, CSR unless return_coo."""
    coo = nacho.parse2D(str(config.MATRIX_DIR / matrix))

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
    coo = nacho.parse2D(str(config.MATRIX_DIR / matrix))

    row = coo.row.to(dtype=torch.long, device="cuda")
    col = coo.col.to(dtype=torch.long, device="cuda")
    indices = col * coo.N + row
    indices, _ = torch.sort(indices)
    values = coo.data.to(dtype=torch.float32, device="cuda")
    return indices, values, coo.N * coo.M


# --------------------------------------------------------------------------------------
# torch sparse -> nacho tensor classes.
#
# Argument order follows the tensor's level order: for each level its offsets and
# coordinates, then the values, then the dimension sizes.
# --------------------------------------------------------------------------------------

def _shape(rows, cols):
    return torch.tensor([rows, cols], dtype=torch.int32)


def to_csr(csr, device="cpu"):
    """torch CSR -> nacho.CSR_cpu / CSR_gpu."""
    cls = nacho.CSR_cpu if device == "cpu" else nacho.CSR_gpu
    return cls(
        csr.crow_indices().to(dtype=torch.int32, device=device),
        csr.col_indices().to(dtype=torch.int32, device=device),
        csr.values().to(dtype=torch.float32, device=device),
        _shape(csr.shape[0], csr.shape[1]),
    )


def to_dcsr(csr, device="cpu"):
    """torch CSR -> nacho.DCSR_cpu / DCSR_gpu.

    DCSR compresses the row dimension too, so empty rows are dropped and the surviving
    row indices are stored explicitly.
    """
    cls = nacho.DCSR_cpu if device == "cpu" else nacho.DCSR_gpu
    indptr = csr.crow_indices().to(dtype=torch.int64)
    nnz_per_row = indptr[1:] - indptr[:-1]
    present = (nnz_per_row > 0).nonzero(as_tuple=False).squeeze(-1)

    compressed_indptr = torch.zeros(present.numel() + 1, dtype=torch.int64)
    compressed_indptr[1:] = nnz_per_row[present].cumsum(0)

    return cls(
        present.to(dtype=torch.int32, device=device),
        compressed_indptr.to(dtype=torch.int32, device=device),
        csr.col_indices().to(dtype=torch.int32, device=device),
        csr.values().to(dtype=torch.float32, device=device),
        _shape(csr.shape[0], csr.shape[1]),
    )


def to_coo(coo, device="cpu"):
    """torch COO -> nacho.COO_cpu / COO_gpu. Coordinates must be sorted, so coalesce."""
    cls = nacho.COO_cpu if device == "cpu" else nacho.COO_gpu
    coo = coo.coalesce()
    indices = coo.indices()
    return cls(
        indices[0].to(dtype=torch.int32, device=device),
        indices[1].to(dtype=torch.int32, device=device),
        coo.values().to(dtype=torch.float32, device=device),
        _shape(coo.shape[0], coo.shape[1]),
    )


def to_csf3(coordinates, values, shape, device="cpu"):
    """Lexicographically sorted, deduplicated 3D (i, j, k) coordinates -> nacho.CSF3_*.

    Every level is compressed, so each level stores only the coordinates present under
    its parent, plus the offsets delimiting each parent's run.
    """
    cls = nacho.CSF3_cpu if device == "cpu" else nacho.CSF3_gpu

    def runs(keys):
        """Start of each distinct run in `keys`, in order, and the keys themselves."""
        distinct, first = np.unique(keys, axis=0, return_index=True)
        order = np.argsort(first)
        return distinct[order], first[order]

    ij, ij_starts = runs(coordinates[:, :2])
    i, i_starts = runs(ij[:, 0])

    def tensor(array, dtype):
        return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=dtype, device=device)

    return cls(
        tensor(i, torch.int32),
        tensor(np.append(i_starts, len(ij)), torch.int32),
        tensor(ij[:, 1], torch.int32),
        tensor(np.append(ij_starts, len(coordinates)), torch.int32),
        tensor(coordinates[:, 2], torch.int32),
        tensor(values, torch.float32),
        torch.tensor(shape, dtype=torch.int32),
    )
