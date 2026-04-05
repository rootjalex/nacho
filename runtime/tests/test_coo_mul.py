"""Tests for nacho-generated COO sparse matrix element-wise multiplication."""
import pytest
import torch
import nacho_runtime

from conftest import make_random_coo


def _to_nb_coo(t):
    """Convert a coalesced torch sparse_coo_tensor to nanobind COO."""
    t = t.coalesce()
    M, N = t.shape
    return nacho_runtime.COO(
        t.indices()[0].to(torch.int32),
        t.indices()[1].to(torch.int32),
        t.values().to(torch.float32),
        torch.tensor([M, N], dtype=torch.int32),
    )


def _coo_mul_reference(A, B):
    """Compute element-wise multiplication of two COO tensors via dense."""
    A_dense = A.to_dense()
    B_dense = B.to_dense()
    C_dense = A_dense * B_dense
    return C_dense


def _run_coo_mul(A_torch, B_torch):
    """Run nacho COO mul and return result + dense reference."""
    A_nb = _to_nb_coo(A_torch)
    B_nb = _to_nb_coo(B_torch)
    C_nacho = nacho_runtime.nacho_coo_mul(A_nb, B_nb)
    torch.cuda.synchronize()
    return C_nacho


def _coo_result_to_dense(coo_result, shape):
    """Convert a nanobind COO result to a dense torch tensor."""
    rows, cols = shape
    row = coo_result.row.to(torch.long)
    col = coo_result.col.to(torch.long)
    data = coo_result.data
    if row.numel() == 0:
        return torch.zeros(rows, cols, dtype=torch.float32, device="cuda")
    t = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        data,
        size=(rows, cols),
    ).coalesce()
    return t.to_dense()


class TestCOOMul:
    """Correctness tests for COO element-wise multiplication."""

    def test_disjoint(self):
        """Non-overlapping entries -> all zeros."""
        A = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device="cuda"),
            torch.tensor([1.0, 2.0], device="cuda"),
            size=(3, 3),
        ).coalesce()
        B = torch.sparse_coo_tensor(
            torch.tensor([[2], [2]], dtype=torch.long, device="cuda"),
            torch.tensor([3.0], device="cuda"),
            size=(3, 3),
        ).coalesce()

        C = _run_coo_mul(A, B)
        C_dense = _coo_result_to_dense(C, (3, 3))
        expected = torch.zeros(3, 3, dtype=torch.float32, device="cuda")
        assert torch.allclose(C_dense, expected)

    def test_full_overlap(self):
        """All entries overlap -> values multiplied."""
        A = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device="cuda"),
            torch.tensor([2.0, 3.0], device="cuda"),
            size=(2, 2),
        ).coalesce()
        B = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device="cuda"),
            torch.tensor([10.0, 20.0], device="cuda"),
            size=(2, 2),
        ).coalesce()

        C = _run_coo_mul(A, B)
        C_dense = _coo_result_to_dense(C, (2, 2))
        expected = _coo_mul_reference(A, B)
        assert torch.allclose(C_dense, expected)

    def test_partial_overlap(self):
        """Some entries overlap, some don't."""
        A = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long, device="cuda"),
            torch.tensor([2.0, 3.0, 4.0], device="cuda"),
            size=(2, 2),
        ).coalesce()
        B = torch.sparse_coo_tensor(
            torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.long, device="cuda"),
            torch.tensor([5.0, 6.0, 7.0], device="cuda"),
            size=(2, 2),
        ).coalesce()

        C = _run_coo_mul(A, B)
        C_dense = _coo_result_to_dense(C, (2, 2))
        expected = _coo_mul_reference(A, B)
        assert torch.allclose(C_dense, expected)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        """Randomized: nacho COO mul vs dense element-wise product."""
        import random
        rng = random.Random(seed)
        rows = rng.randint(5, 50)
        cols = rng.randint(5, 50)
        nnz_a = rng.randint(1, min(rows * cols, 50))
        nnz_b = rng.randint(1, min(rows * cols, 50))

        A = make_random_coo(rows, cols, nnz_a, seed=seed * 1000)
        B = make_random_coo(rows, cols, nnz_b, seed=seed * 1000 + 1)

        C = _run_coo_mul(A, B)
        C_dense = _coo_result_to_dense(C, (rows, cols))
        expected = _coo_mul_reference(A, B)
        assert torch.allclose(C_dense, expected, rtol=1e-5, atol=1e-5), \
            f"mismatch for {rows}x{cols} with nnz_a={nnz_a}, nnz_b={nnz_b}"
