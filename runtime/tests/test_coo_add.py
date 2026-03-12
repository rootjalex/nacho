"""Tests for COO sparse matrix addition: manual kernel vs PyTorch."""
import pytest
import torch
import nanobind_cuda_example

from conftest import make_random_coo


def _to_nb_coo(t):
    """Convert a coalesced torch sparse_coo_tensor to nanobind COO."""
    t = t.coalesce()
    M, N = t.shape
    return nanobind_cuda_example.COO(
        t.indices()[0].to(torch.int32),
        t.indices()[1].to(torch.int32),
        t.values().to(torch.float32),
        torch.tensor([M, N], dtype=torch.int32),
    )


def _run_coo_add(A_torch, B_torch):
    """Run manual COO add and PyTorch reference."""
    A_nb = _to_nb_coo(A_torch)
    B_nb = _to_nb_coo(B_torch)
    C_manual = nanobind_cuda_example.gpu_coo_add_f32(A_nb, B_nb)
    C_pytorch = (A_torch + B_torch).coalesce()
    torch.cuda.synchronize()
    return C_manual, C_pytorch


class TestCOOAdd:
    """Correctness tests for COO matrix addition."""

    def test_disjoint(self):
        """Non-overlapping entries."""
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

        C_manual, C_pytorch = _run_coo_add(A, B)
        assert torch.equal(C_pytorch.indices()[0].to(torch.int32), C_manual.row)
        assert torch.equal(C_pytorch.indices()[1].to(torch.int32), C_manual.col)
        assert torch.equal(C_pytorch.values(), C_manual.data)

    def test_full_overlap(self):
        """All entries overlap — values should sum."""
        A = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device="cuda"),
            torch.tensor([1.0, 2.0], device="cuda"),
            size=(2, 2),
        ).coalesce()
        B = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device="cuda"),
            torch.tensor([10.0, 20.0], device="cuda"),
            size=(2, 2),
        ).coalesce()

        C_manual, C_pytorch = _run_coo_add(A, B)
        assert torch.equal(C_pytorch.indices()[0].to(torch.int32), C_manual.row)
        assert torch.equal(C_pytorch.indices()[1].to(torch.int32), C_manual.col)
        assert torch.equal(C_pytorch.values(), C_manual.data)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        """Randomized: manual COO add vs PyTorch on random matrices."""
        import random
        rng = random.Random(seed)
        rows = rng.randint(5, 50)
        cols = rng.randint(5, 50)
        nnz_a = rng.randint(1, min(rows * cols, 50))
        nnz_b = rng.randint(1, min(rows * cols, 50))

        A = make_random_coo(rows, cols, nnz_a, seed=seed * 1000)
        B = make_random_coo(rows, cols, nnz_b, seed=seed * 1000 + 1)

        C_manual, C_pytorch = _run_coo_add(A, B)

        assert torch.equal(C_pytorch.indices()[0].to(torch.int32), C_manual.row), "row mismatch"
        assert torch.equal(C_pytorch.indices()[1].to(torch.int32), C_manual.col), "col mismatch"
        assert torch.equal(C_pytorch.values(), C_manual.data), "data mismatch"
