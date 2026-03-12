"""Shared fixtures and helpers for runtime tests."""
import torch
import pytest


def make_random_csr(rows, cols, nnz_per_row_range=(1, 5), seed=None):
    """Generate a random CSR matrix on GPU.

    Returns a torch sparse_csr_tensor on CUDA.
    """
    import random
    rng = random.Random(seed)
    indptr = [0]
    indices = []
    values = []
    for _ in range(rows):
        nnz = rng.randint(*nnz_per_row_range)
        nnz = min(nnz, cols)
        cols_in_row = sorted(rng.sample(range(cols), nnz))
        indices.extend(cols_in_row)
        values.extend(rng.uniform(0.1, 10.0) for _ in range(nnz))
        indptr.append(indptr[-1] + nnz)

    return torch.sparse_csr_tensor(
        torch.tensor(indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(indices, dtype=torch.int32, device="cuda"),
        torch.tensor(values, dtype=torch.float32, device="cuda"),
        size=(rows, cols),
    )


def make_random_coo(rows, cols, nnz, seed=None):
    """Generate a random COO matrix on GPU.

    Returns a coalesced torch sparse_coo_tensor on CUDA.
    """
    import random
    rng = random.Random(seed)
    total_elements = rows * cols
    nnz = min(nnz, total_elements)
    flat = sorted(rng.sample(range(total_elements), nnz))
    row_indices = [f // cols for f in flat]
    col_indices = [f % cols for f in flat]
    values = [rng.uniform(0.1, 10.0) for _ in range(nnz)]

    return torch.sparse_coo_tensor(
        torch.tensor([row_indices, col_indices], dtype=torch.long, device="cuda"),
        torch.tensor(values, dtype=torch.float32, device="cuda"),
        size=(rows, cols),
    ).coalesce()
