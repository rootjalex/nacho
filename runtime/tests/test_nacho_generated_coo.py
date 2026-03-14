"""Correctness tests for nacho-generated COO (Coordinate) add operations (2D-4D)."""
import random

import pytest
import torch

from nacho_runtime import (
    NachoCOO2D, NachoCOO3D, NachoCOO4D,
    nacho_coo2_add, nacho_coo3_add, nacho_coo4_add,
)


# ---------------------------------------------------------------------------
# 2D COO helpers
# ---------------------------------------------------------------------------

def make_coo2d(dim_i_indices, dim_j_indices, values, shape):
    """Create a NachoCOO2D from Python lists."""
    i = torch.tensor(dim_i_indices, dtype=torch.int32, device="cuda")
    j = torch.tensor(dim_j_indices, dtype=torch.int32, device="cuda")
    v = torch.tensor(values, dtype=torch.float32, device="cuda")
    return NachoCOO2D(i, j, v, shape[0], shape[1])


def dense_to_coo2d(dense):
    """Convert a dense 2D CUDA tensor to NachoCOO2D."""
    assert dense.dim() == 2
    nz = (dense != 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return make_coo2d([], [], [], dense.shape)
    dim_i_size, dim_j_size = dense.shape
    key = nz[:, 0].to(torch.int64) * dim_j_size + nz[:, 1].to(torch.int64)
    nz = nz[torch.argsort(key)]
    vals = dense[nz[:, 0], nz[:, 1]]
    return make_coo2d(
        nz[:, 0].to(torch.int32).tolist(),
        nz[:, 1].to(torch.int32).tolist(),
        vals.tolist(),
        dense.shape,
    )


def coo2d_to_dense(tensor):
    """Convert NachoCOO2D back to dense CUDA tensor."""
    dense = torch.zeros(
        (tensor.dim_i_size, tensor.dim_j_size),
        dtype=torch.float32, device="cuda",
    )
    nnz = tensor.data.shape[0]
    for p in range(nnz):
        i = int(tensor.dim_i_indices[p].item())
        j = int(tensor.dim_j_indices[p].item())
        dense[i, j] = tensor.data[p].item()
    return dense


# ---------------------------------------------------------------------------
# 3D COO helpers
# ---------------------------------------------------------------------------

def make_coo3d(dim_i_indices, dim_j_indices, dim_k_indices, values, shape):
    """Create a NachoCOO3D from Python lists."""
    i = torch.tensor(dim_i_indices, dtype=torch.int32, device="cuda")
    j = torch.tensor(dim_j_indices, dtype=torch.int32, device="cuda")
    k = torch.tensor(dim_k_indices, dtype=torch.int32, device="cuda")
    v = torch.tensor(values, dtype=torch.float32, device="cuda")
    return NachoCOO3D(i, j, k, v, shape[0], shape[1], shape[2])


def dense_to_coo3d(dense):
    """Convert a dense 3D CUDA tensor to NachoCOO3D."""
    assert dense.dim() == 3
    nz = (dense != 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return make_coo3d([], [], [], [], dense.shape)
    s0, s1, s2 = dense.shape
    key = (nz[:, 0].to(torch.int64) * s1 * s2
           + nz[:, 1].to(torch.int64) * s2
           + nz[:, 2].to(torch.int64))
    nz = nz[torch.argsort(key)]
    vals = dense[nz[:, 0], nz[:, 1], nz[:, 2]]
    return make_coo3d(
        nz[:, 0].to(torch.int32).tolist(),
        nz[:, 1].to(torch.int32).tolist(),
        nz[:, 2].to(torch.int32).tolist(),
        vals.tolist(),
        dense.shape,
    )


def coo3d_to_dense(tensor):
    """Convert NachoCOO3D back to dense CUDA tensor."""
    dense = torch.zeros(
        (tensor.dim_i_size, tensor.dim_j_size, tensor.dim_k_size),
        dtype=torch.float32, device="cuda",
    )
    nnz = tensor.data.shape[0]
    for p in range(nnz):
        i = int(tensor.dim_i_indices[p].item())
        j = int(tensor.dim_j_indices[p].item())
        k = int(tensor.dim_k_indices[p].item())
        dense[i, j, k] = tensor.data[p].item()
    return dense


# ---------------------------------------------------------------------------
# 4D COO helpers
# ---------------------------------------------------------------------------

def make_coo4d(dim_i_indices, dim_j_indices, dim_k_indices, dim_l_indices,
               values, shape):
    """Create a NachoCOO4D from Python lists."""
    i = torch.tensor(dim_i_indices, dtype=torch.int32, device="cuda")
    j = torch.tensor(dim_j_indices, dtype=torch.int32, device="cuda")
    k = torch.tensor(dim_k_indices, dtype=torch.int32, device="cuda")
    l = torch.tensor(dim_l_indices, dtype=torch.int32, device="cuda")
    v = torch.tensor(values, dtype=torch.float32, device="cuda")
    return NachoCOO4D(i, j, k, l, v, shape[0], shape[1], shape[2], shape[3])


def dense_to_coo4d(dense):
    """Convert a dense 4D CUDA tensor to NachoCOO4D."""
    assert dense.dim() == 4
    nz = (dense != 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return make_coo4d([], [], [], [], [], dense.shape)
    s0, s1, s2, s3 = dense.shape
    key = (nz[:, 0].to(torch.int64) * s1 * s2 * s3
           + nz[:, 1].to(torch.int64) * s2 * s3
           + nz[:, 2].to(torch.int64) * s3
           + nz[:, 3].to(torch.int64))
    nz = nz[torch.argsort(key)]
    vals = dense[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]]
    return make_coo4d(
        nz[:, 0].to(torch.int32).tolist(),
        nz[:, 1].to(torch.int32).tolist(),
        nz[:, 2].to(torch.int32).tolist(),
        nz[:, 3].to(torch.int32).tolist(),
        vals.tolist(),
        dense.shape,
    )


def coo4d_to_dense(tensor):
    """Convert NachoCOO4D back to dense CUDA tensor."""
    dense = torch.zeros(
        (tensor.dim_i_size, tensor.dim_j_size,
         tensor.dim_k_size, tensor.dim_l_size),
        dtype=torch.float32, device="cuda",
    )
    nnz = tensor.data.shape[0]
    for p in range(nnz):
        i = int(tensor.dim_i_indices[p].item())
        j = int(tensor.dim_j_indices[p].item())
        k = int(tensor.dim_k_indices[p].item())
        l = int(tensor.dim_l_indices[p].item())
        dense[i, j, k, l] = tensor.data[p].item()
    return dense


# ---------------------------------------------------------------------------
# Generic random dense tensor generation (n-dimensional)
# ---------------------------------------------------------------------------

def make_random_sparse_dense(shape, nnz, seed, avoid=None):
    """Generate a random dense tensor with nnz nonzero entries."""
    rng = random.Random(seed)
    total = 1
    for s in shape:
        total *= s
    if avoid is None:
        avoid = set()
    candidates = [idx for idx in range(total) if idx not in avoid]
    nnz = min(nnz, len(candidates))
    flat_indices = rng.sample(candidates, nnz)

    dense = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for flat in flat_indices:
        coords = []
        rem = flat
        for d in reversed(shape):
            coords.append(rem % d)
            rem //= d
        coords.reverse()
        dense[tuple(coords)] = rng.uniform(0.1, 10.0)
    return dense, set(flat_indices)


# ===========================================================================
# 2D COO Add tests
# ===========================================================================

class TestNachoCoo2dAdd:
    """PyTorch correctness checks for generated COO 2D add."""

    def test_basic(self):
        a = torch.zeros((4, 5), dtype=torch.float32, device="cuda")
        b = torch.zeros((4, 5), dtype=torch.float32, device="cuda")
        a[0, 1] = 2.0; a[2, 3] = 3.5
        b[0, 1] = 4.0; b[1, 4] = 6.0

        result = nacho_coo2_add(dense_to_coo2d(a), dense_to_coo2d(b))
        torch.cuda.synchronize()
        got = coo2d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    def test_disjoint(self):
        a = torch.zeros((3, 3), dtype=torch.float32, device="cuda")
        b = torch.zeros((3, 3), dtype=torch.float32, device="cuda")
        a[0, 0] = 1.0; a[1, 1] = 2.0; a[2, 2] = 3.0
        b[0, 2] = 4.0; b[1, 0] = 5.0; b[2, 1] = 6.0

        result = nacho_coo2_add(dense_to_coo2d(a), dense_to_coo2d(b))
        torch.cuda.synchronize()
        got = coo2d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    def test_empty_a(self):
        a = torch.zeros((3, 4), dtype=torch.float32, device="cuda")
        b = torch.zeros((3, 4), dtype=torch.float32, device="cuda")
        b[1, 2] = 7.0

        result = nacho_coo2_add(dense_to_coo2d(a), dense_to_coo2d(b))
        torch.cuda.synchronize()
        got = coo2d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    def test_empty_b(self):
        a = torch.zeros((3, 4), dtype=torch.float32, device="cuda")
        b = torch.zeros((3, 4), dtype=torch.float32, device="cuda")
        a[2, 0] = 5.0

        result = nacho_coo2_add(dense_to_coo2d(a), dense_to_coo2d(b))
        torch.cuda.synchronize()
        got = coo2d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_overlap(self, seed):
        rng = random.Random(seed)
        shape = (rng.randint(3, 16), rng.randint(3, 16))
        total = shape[0] * shape[1]
        nnz_a = rng.randint(1, max(1, total // 4))
        nnz_b = rng.randint(1, max(1, total // 4))
        a, _ = make_random_sparse_dense(shape, nnz_a, seed * 1000)
        b, _ = make_random_sparse_dense(shape, nnz_b, seed * 1000 + 1)

        result = nacho_coo2_add(dense_to_coo2d(a), dense_to_coo2d(b))
        torch.cuda.synchronize()
        got = coo2d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_disjoint(self, seed):
        rng = random.Random(seed + 500)
        shape = (rng.randint(3, 16), rng.randint(3, 16))
        total = shape[0] * shape[1]
        nnz_a = rng.randint(1, max(1, total // 4))
        nnz_b = rng.randint(1, max(1, total // 4))
        a, a_sup = make_random_sparse_dense(shape, nnz_a, seed * 2000)
        b, _ = make_random_sparse_dense(shape, nnz_b, seed * 2000 + 1, avoid=a_sup)

        result = nacho_coo2_add(dense_to_coo2d(a), dense_to_coo2d(b))
        torch.cuda.synchronize()
        got = coo2d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)


# ===========================================================================
# 3D COO Add tests
# ===========================================================================

class TestNachoCoo3dAdd:
    """PyTorch correctness checks for generated COO 3D add."""

    def test_basic(self):
        a = torch.zeros((3, 4, 5), dtype=torch.float32, device="cuda")
        b = torch.zeros((3, 4, 5), dtype=torch.float32, device="cuda")
        a[0, 1, 2] = 2.0; a[2, 0, 4] = 3.5
        b[0, 1, 2] = 4.0; b[1, 3, 1] = 6.0

        result = nacho_coo3_add(dense_to_coo3d(a), dense_to_coo3d(b))
        torch.cuda.synchronize()
        got = coo3d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    def test_disjoint(self):
        a = torch.zeros((4, 4, 4), dtype=torch.float32, device="cuda")
        b = torch.zeros((4, 4, 4), dtype=torch.float32, device="cuda")
        a[0, 0, 0] = 1.0; a[1, 2, 3] = 2.0; a[3, 3, 3] = 3.0
        b[0, 1, 0] = 4.0; b[2, 0, 2] = 5.0; b[3, 1, 1] = 6.0

        result = nacho_coo3_add(dense_to_coo3d(a), dense_to_coo3d(b))
        torch.cuda.synchronize()
        got = coo3d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_overlap(self, seed):
        rng = random.Random(seed)
        shape = (rng.randint(3, 10), rng.randint(3, 10), rng.randint(3, 10))
        total = shape[0] * shape[1] * shape[2]
        nnz_a = rng.randint(1, max(1, total // 6))
        nnz_b = rng.randint(1, max(1, total // 6))
        a, _ = make_random_sparse_dense(shape, nnz_a, seed * 1000)
        b, _ = make_random_sparse_dense(shape, nnz_b, seed * 1000 + 1)

        result = nacho_coo3_add(dense_to_coo3d(a), dense_to_coo3d(b))
        torch.cuda.synchronize()
        got = coo3d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_disjoint(self, seed):
        rng = random.Random(seed + 500)
        shape = (rng.randint(3, 10), rng.randint(3, 10), rng.randint(3, 10))
        total = shape[0] * shape[1] * shape[2]
        nnz_a = rng.randint(1, max(1, total // 6))
        nnz_b = rng.randint(1, max(1, total // 6))
        a, a_sup = make_random_sparse_dense(shape, nnz_a, seed * 2000)
        b, _ = make_random_sparse_dense(shape, nnz_b, seed * 2000 + 1, avoid=a_sup)

        result = nacho_coo3_add(dense_to_coo3d(a), dense_to_coo3d(b))
        torch.cuda.synchronize()
        got = coo3d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)


# ===========================================================================
# 4D COO Add tests
# ===========================================================================

class TestNachoCoo4dAdd:
    """PyTorch correctness checks for generated COO 4D add."""

    def test_basic(self):
        a = torch.zeros((3, 3, 3, 3), dtype=torch.float32, device="cuda")
        b = torch.zeros((3, 3, 3, 3), dtype=torch.float32, device="cuda")
        a[0, 1, 2, 0] = 2.0; a[2, 0, 1, 2] = 3.5
        b[0, 1, 2, 0] = 4.0; b[1, 2, 0, 1] = 6.0

        result = nacho_coo4_add(dense_to_coo4d(a), dense_to_coo4d(b))
        torch.cuda.synchronize()
        got = coo4d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    def test_disjoint(self):
        a = torch.zeros((3, 3, 3, 3), dtype=torch.float32, device="cuda")
        b = torch.zeros((3, 3, 3, 3), dtype=torch.float32, device="cuda")
        a[0, 0, 0, 0] = 1.0; a[1, 1, 1, 1] = 2.0; a[2, 2, 2, 2] = 3.0
        b[0, 1, 2, 0] = 4.0; b[1, 0, 2, 1] = 5.0; b[2, 1, 0, 2] = 6.0

        result = nacho_coo4_add(dense_to_coo4d(a), dense_to_coo4d(b))
        torch.cuda.synchronize()
        got = coo4d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_overlap(self, seed):
        rng = random.Random(seed)
        shape = (rng.randint(2, 6), rng.randint(2, 6),
                 rng.randint(2, 6), rng.randint(2, 6))
        total = shape[0] * shape[1] * shape[2] * shape[3]
        nnz_a = rng.randint(1, max(1, total // 8))
        nnz_b = rng.randint(1, max(1, total // 8))
        a, _ = make_random_sparse_dense(shape, nnz_a, seed * 1000)
        b, _ = make_random_sparse_dense(shape, nnz_b, seed * 1000 + 1)

        result = nacho_coo4_add(dense_to_coo4d(a), dense_to_coo4d(b))
        torch.cuda.synchronize()
        got = coo4d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("seed", range(25))
    def test_random_disjoint(self, seed):
        rng = random.Random(seed + 500)
        shape = (rng.randint(2, 6), rng.randint(2, 6),
                 rng.randint(2, 6), rng.randint(2, 6))
        total = shape[0] * shape[1] * shape[2] * shape[3]
        nnz_a = rng.randint(1, max(1, total // 8))
        nnz_b = rng.randint(1, max(1, total // 8))
        a, a_sup = make_random_sparse_dense(shape, nnz_a, seed * 2000)
        b, _ = make_random_sparse_dense(shape, nnz_b, seed * 2000 + 1, avoid=a_sup)

        result = nacho_coo4_add(dense_to_coo4d(a), dense_to_coo4d(b))
        torch.cuda.synchronize()
        got = coo4d_to_dense(result)
        assert torch.allclose(got, a + b, rtol=1e-4, atol=1e-5)
