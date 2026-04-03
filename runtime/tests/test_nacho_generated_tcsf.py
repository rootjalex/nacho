"""Correctness tests for nacho-generated higher-order sparse operations."""
import random

import pytest
import torch

from nacho_runtime import TCSF, nacho_tcsf_add


def make_tcsf(i_indices, j_offsets, j_indices, k_offsets, k_indices, values, shape):
    """Create a TCSF object from Python lists."""
    i = torch.tensor(i_indices, dtype=torch.int32, device="cuda")
    jo = torch.tensor(j_offsets, dtype=torch.int32, device="cuda")
    j = torch.tensor(j_indices, dtype=torch.int32, device="cuda")
    ko = torch.tensor(k_offsets, dtype=torch.int32, device="cuda")
    k = torch.tensor(k_indices, dtype=torch.int32, device="cuda")
    v = torch.tensor(values, dtype=torch.float32, device="cuda")
    return TCSF(i, jo, j, ko, k, v, shape[0], shape[1], shape[2])


def dense_to_tcsf(dense):
    """Convert a dense 3D CUDA tensor to TCSF."""
    assert dense.dim() == 3
    dim_i_size, dim_j_size, dim_k_size = dense.shape
    nz = (dense != 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return make_tcsf([], [0], [], [0], [], [], dense.shape)

    # Enforce lexicographic order by (i, j, k).
    key = (
        nz[:, 0].to(torch.int64) * (dim_j_size * dim_k_size)
        + nz[:, 1].to(torch.int64) * dim_k_size
        + nz[:, 2].to(torch.int64)
    )
    nz = nz[torch.argsort(key)]
    vals = dense[nz[:, 0], nz[:, 1], nz[:, 2]]

    coords = nz.cpu().tolist()
    data = [float(v) for v in vals.cpu().tolist()]

    i_indices = []
    j_offsets = [0]
    j_indices = []
    k_offsets = [0]
    k_indices = []
    out_values = []

    p = 0
    while p < len(coords):
        i = coords[p][0]
        i_indices.append(i)
        while p < len(coords) and coords[p][0] == i:
            j = coords[p][1]
            j_indices.append(j)
            while p < len(coords) and coords[p][0] == i and coords[p][1] == j:
                k_indices.append(coords[p][2])
                out_values.append(data[p])
                p += 1
            k_offsets.append(len(k_indices))
        j_offsets.append(len(j_indices))

    return make_tcsf(i_indices, j_offsets, j_indices, k_offsets, k_indices,
                     out_values, dense.shape)


def tcsf_to_dense(tensor):
    """Convert TCSF back to dense CUDA tensor."""
    dense = torch.zeros(
        (tensor.dim_i_size, tensor.dim_j_size, tensor.dim_k_size),
        dtype=torch.float32,
        device="cuda",
    )
    for i_pos in range(tensor.i_indices.shape[0]):
        i = int(tensor.i_indices[i_pos].item())
        j_start = int(tensor.j_offsets[i_pos].item())
        j_end = int(tensor.j_offsets[i_pos + 1].item())
        for j_pos in range(j_start, j_end):
            j = int(tensor.j_indices[j_pos].item())
            k_start = int(tensor.k_offsets[j_pos].item())
            k_end = int(tensor.k_offsets[j_pos + 1].item())
            if k_end > k_start:
                dense[i, j, tensor.k_indices[k_start:k_end].to(torch.int64)] = (
                    tensor.data[k_start:k_end]
                )
    return dense


def make_random_sparse_dense(shape, nnz, seed, avoid=None):
    """Generate a random dense tensor with nnz sparse entries."""
    rng = random.Random(seed)
    total = shape[0] * shape[1] * shape[2]
    if avoid is None:
        avoid = set()
    candidates = [idx for idx in range(total) if idx not in avoid]
    nnz = min(nnz, len(candidates))
    flat_indices = rng.sample(candidates, nnz)

    dense = torch.zeros(shape, dtype=torch.float32, device="cuda")
    plane = shape[1] * shape[2]
    for flat in flat_indices:
        i = flat // plane
        rem = flat % plane
        j = rem // shape[2]
        k = rem % shape[2]
        dense[i, j, k] = rng.uniform(0.1, 10.0)
    return dense, set(flat_indices)


class TestNachoTcsfAdd:
    """PyTorch correctness checks for generated TCSF add."""

    def test_basic_correctness(self):
        a_dense = torch.zeros((3, 4, 5), dtype=torch.float32, device="cuda")
        b_dense = torch.zeros((3, 4, 5), dtype=torch.float32, device="cuda")

        a_dense[0, 1, 2] = 2.0
        a_dense[2, 0, 4] = 3.5
        b_dense[0, 1, 2] = 4.0
        b_dense[1, 3, 1] = 6.0

        result = nacho_tcsf_add(dense_to_tcsf(a_dense), dense_to_tcsf(b_dense))
        torch.cuda.synchronize()

        got = tcsf_to_dense(result)
        expected = a_dense + b_dense
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)
        assert result.j_offsets.shape[0] == result.i_indices.shape[0] + 1
        assert result.k_offsets.shape[0] == result.j_indices.shape[0] + 1

    def test_outer_sparse_structure_correctness(self):
        a_dense = torch.zeros((4, 7, 4), dtype=torch.float32, device="cuda")
        b_dense = torch.zeros((4, 7, 4), dtype=torch.float32, device="cuda")

        for i, j, k, v in [
            (0, 0, 0, 1.1),
            (0, 5, 3, 2.2),
            (1, 1, 1, 3.3),
            (2, 6, 0, 4.4),
            (3, 3, 2, 5.5),
        ]:
            a_dense[i, j, k] = v

        for i, j, k, v in [
            (0, 2, 2, 6.6),
            (1, 1, 1, 7.7),
            (1, 4, 0, 8.8),
            (2, 0, 3, 9.9),
            (3, 3, 1, 10.1),
        ]:
            b_dense[i, j, k] = v

        result = nacho_tcsf_add(dense_to_tcsf(a_dense), dense_to_tcsf(b_dense))
        torch.cuda.synchronize()

        got = tcsf_to_dense(result)
        expected = a_dense + b_dense
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_mixed_overlap_correctness(self):
        a_dense = torch.zeros((5, 5, 5), dtype=torch.float32, device="cuda")
        b_dense = torch.zeros((5, 5, 5), dtype=torch.float32, device="cuda")

        for i, j, k, v in [
            (0, 0, 0, 1.0),
            (0, 0, 4, 2.0),
            (0, 2, 2, 3.0),
            (1, 4, 1, 4.0),
            (3, 1, 3, 5.0),
            (4, 4, 4, 6.0),
        ]:
            a_dense[i, j, k] = v

        for i, j, k, v in [
            (0, 0, 4, 7.0),
            (0, 3, 1, 8.0),
            (1, 4, 2, 9.0),
            (2, 2, 2, 10.0),
            (3, 1, 3, 11.0),
            (4, 0, 4, 12.0),
        ]:
            b_dense[i, j, k] = v

        result = nacho_tcsf_add(dense_to_tcsf(a_dense), dense_to_tcsf(b_dense))
        torch.cuda.synchronize()

        got = tcsf_to_dense(result)
        expected = a_dense + b_dense
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_correctness_vs_pytorch_overlap(self, seed):
        """Random overlap-heavy correctness against PyTorch dense add."""
        rng = random.Random(seed)
        shape = (
            rng.randint(4, 12),
            rng.randint(4, 12),
            rng.randint(4, 12),
        )
        total = shape[0] * shape[1] * shape[2]
        nnz_a = rng.randint(1, max(1, total // 6))
        nnz_b = rng.randint(1, max(1, total // 6))
        a_dense, _ = make_random_sparse_dense(shape, nnz_a, seed * 1000)
        b_dense, _ = make_random_sparse_dense(shape, nnz_b, seed * 1000 + 1)

        result = nacho_tcsf_add(dense_to_tcsf(a_dense), dense_to_tcsf(b_dense))
        torch.cuda.synchronize()

        got = tcsf_to_dense(result)
        expected = a_dense + b_dense
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_correctness_vs_pytorch_disjoint(self, seed):
        """Random disjoint-support correctness against PyTorch dense add."""
        rng = random.Random(seed + 1000)
        shape = (
            rng.randint(4, 12),
            rng.randint(4, 12),
            rng.randint(4, 12),
        )
        total = shape[0] * shape[1] * shape[2]
        nnz_a = rng.randint(1, max(1, total // 6))
        nnz_b = rng.randint(1, max(1, total // 6))
        a_dense, a_support = make_random_sparse_dense(shape, nnz_a, seed * 2000)
        b_dense, _ = make_random_sparse_dense(
            shape, nnz_b, seed * 2000 + 1, avoid=a_support
        )

        result = nacho_tcsf_add(dense_to_tcsf(a_dense), dense_to_tcsf(b_dense))
        torch.cuda.synchronize()

        got = tcsf_to_dense(result)
        expected = a_dense + b_dense
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)
