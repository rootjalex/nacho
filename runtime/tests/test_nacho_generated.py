"""Tests for nacho-generated operations."""
import pytest
import random
import torch
from nanobind_cuda_example import CVector, nacho_sparse_vec_mul


def make_cvector(indices, values, size):
    """Create a CVector from Python lists."""
    idx = torch.tensor(indices, dtype=torch.int32, device="cuda")
    val = torch.tensor(values, dtype=torch.float32, device="cuda")
    return CVector(idx, val, size)


def reference_sparse_vec_mul(a_indices, a_values, b_indices, b_values):
    """CPU reference implementation for sparse vector element-wise multiply."""
    a_dict = dict(zip(a_indices, a_values))
    b_dict = dict(zip(b_indices, b_values))
    common = sorted(set(a_indices) & set(b_indices))
    out_values = [a_dict[k] * b_dict[k] for k in common]
    return common, out_values


def random_sparse_vec(size, nnz, seed=None):
    """Generate random sorted sparse vector data."""
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(size), nnz))
    values = [rng.uniform(0.1, 10.0) for _ in range(nnz)]
    return indices, values


class TestNachoSparseVecMul:
    """Tests for the nacho-generated sparse_vec_mul operation."""

    def test_basic_intersection(self):
        """Vectors with known intersection at index 3."""
        A = make_cvector([1, 3, 4], [2.0, 4.0, 5.0], 5)
        B = make_cvector([0, 2, 3], [1.0, 3.0, 4.0], 5)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.shape[0] == 1
        assert result.indices.cpu().tolist() == [3]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [16.0]

    def test_full_overlap(self):
        """All indices overlap."""
        A = make_cvector([0, 1, 2], [2.0, 3.0, 4.0], 3)
        B = make_cvector([0, 1, 2], [5.0, 6.0, 7.0], 3)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.shape[0] == 3
        assert result.indices.cpu().tolist() == [0, 1, 2]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [10.0, 18.0, 28.0]

    def test_no_overlap(self):
        """No indices overlap — result should be empty."""
        A = make_cvector([0, 1, 2], [1.0, 2.0, 3.0], 10)
        B = make_cvector([3, 4, 5], [1.0, 2.0, 3.0], 10)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.shape[0] == 0

    def test_single_element(self):
        """Single-element vectors that intersect."""
        A = make_cvector([5], [3.0], 10)
        B = make_cvector([5], [7.0], 10)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.shape[0] == 1
        assert result.indices.cpu().tolist() == [5]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [21.0]

    def test_result_size_preserved(self):
        """Result vector should have same size as inputs."""
        A = make_cvector([1, 3], [2.0, 4.0], 100)
        B = make_cvector([3, 5], [3.0, 6.0], 100)
        result = nacho_sparse_vec_mul(A, B)
        assert result.size == 100

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        """Randomized correctness test against CPU reference."""
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))

        a_indices, a_values = random_sparse_vec(size, nnz_a, seed=seed * 1000)
        b_indices, b_values = random_sparse_vec(size, nnz_b, seed=seed * 1000 + 1)

        A = make_cvector(a_indices, a_values, size)
        B = make_cvector(b_indices, b_values, size)

        result = nacho_sparse_vec_mul(A, B)
        torch.cuda.synchronize()

        ref_indices, ref_values = reference_sparse_vec_mul(
            a_indices, a_values, b_indices, b_values
        )

        out_nnz = result.indices.shape[0]
        assert out_nnz == len(ref_indices), (
            f"nnz mismatch: got {out_nnz}, expected {len(ref_indices)}"
        )

        if out_nnz > 0:
            got_indices = result.indices.cpu().tolist()
            got_values = result.data.cpu().tolist()
            assert got_indices == ref_indices
            assert pytest.approx(got_values, rel=1e-4) == ref_values
