"""Building nacho tensors from torch tensors.

Each generated tensor class takes its buffers in level order: for each level its offsets
and coordinates, then the values, then the dimension sizes. These wrap a torch tensor's
existing buffers in that order, casting index arrays to int32 and values to float32,
which is what the generated kernels expect.

    import nacho
    A = nacho.to_csr(torch_csr_tensor, device="cuda")
"""

import numpy as np
import torch

from . import nacho_ext as _ext

__all__ = ["to_csr", "to_dcsr", "to_coo", "to_coo3", "to_csf3"]

def _shape(rows, cols):
    return torch.tensor([rows, cols], dtype=torch.int32)


def to_csr(csr, device="cpu"):
    """torch CSR -> _ext.CSR_cpu / CSR_gpu."""
    cls = _ext.CSR_cpu if device == "cpu" else _ext.CSR_gpu
    return cls(
        csr.crow_indices().to(dtype=torch.int32, device=device),
        csr.col_indices().to(dtype=torch.int32, device=device),
        csr.values().to(dtype=torch.float32, device=device),
        _shape(csr.shape[0], csr.shape[1]),
    )


def to_dcsr(csr, device="cpu"):
    """torch CSR -> _ext.DCSR_cpu / DCSR_gpu.

    DCSR compresses the row dimension too, so empty rows are dropped and the surviving
    row indices are stored explicitly.
    """
    cls = _ext.DCSR_cpu if device == "cpu" else _ext.DCSR_gpu
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
    """torch COO -> _ext.COO_cpu / COO_gpu. Coordinates must be sorted, so coalesce."""
    cls = _ext.COO_cpu if device == "cpu" else _ext.COO_gpu
    coo = coo.coalesce()
    indices = coo.indices()
    return cls(
        indices[0].to(dtype=torch.int32, device=device),
        indices[1].to(dtype=torch.int32, device=device),
        coo.values().to(dtype=torch.float32, device=device),
        _shape(coo.shape[0], coo.shape[1]),
    )


def to_coo3(coordinates, values, shape, device="cpu"):
    """Lexicographically sorted, deduplicated 3D (i, j, k) coordinates -> _ext.COO3_*.

    The three coordinate arrays run in parallel, one entry per non-zero, unlike CSF3
    where each level stores only the coordinates present under its parent.
    """
    cls = _ext.COO3_cpu if device == "cpu" else _ext.COO3_gpu

    def tensor(array, dtype):
        return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=dtype, device=device)

    return cls(
        tensor(coordinates[:, 0], torch.int32),
        tensor(coordinates[:, 1], torch.int32),
        tensor(coordinates[:, 2], torch.int32),
        tensor(values, torch.float32),
        torch.tensor(shape, dtype=torch.int32),
    )


def to_csf3(coordinates, values, shape, device="cpu"):
    """Lexicographically sorted, deduplicated 3D (i, j, k) coordinates -> _ext.CSF3_*.

    Every level is compressed, so each level stores only the coordinates present under
    its parent, plus the offsets delimiting each parent's run.
    """
    cls = _ext.CSF3_cpu if device == "cpu" else _ext.CSF3_gpu

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
