"""Tests for CSR sparse matrix addition: manual kernel vs cuSPARSE vs PyTorch."""
import pytest
import torch
import nanobind_cuda_example

from conftest import make_random_csr


def _to_nb_csr(t):
    """Convert a torch sparse_csr_tensor to nanobind CSR."""
    M, N = t.shape
    return nanobind_cuda_example.CSR(
        t.crow_indices().to(torch.int32),
        t.col_indices().to(torch.int32),
        t.values().to(torch.float32),
        torch.tensor([M, N], dtype=torch.int32),
    )


def _run_csr_add(A_torch, B_torch):
    """Run manual and cuSPARSE CSR add, return results + PyTorch reference."""
    A_nb = _to_nb_csr(A_torch)
    B_nb = _to_nb_csr(B_torch)
    C_manual = nanobind_cuda_example.gpu_csr_add_f32(A_nb, B_nb, False)
    C_cusparse = nanobind_cuda_example.gpu_csr_add_f32(A_nb, B_nb, True)
    C_pytorch = A_torch + B_torch
    torch.cuda.synchronize()
    return C_manual, C_cusparse, C_pytorch


class TestCSRAdd:
    """Correctness tests for CSR matrix addition."""

    def test_identity_add(self):
        """Adding a matrix to a zero matrix should return the original."""
        A = torch.sparse_csr_tensor(
            torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 2, 1], dtype=torch.int32, device="cuda"),
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"),
            size=(2, 3),
        )
        B = torch.sparse_csr_tensor(
            torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda"),
            torch.tensor([], dtype=torch.int32, device="cuda"),
            torch.tensor([], dtype=torch.float32, device="cuda"),
            size=(2, 3),
        )
        C_manual, C_cusparse, C_pytorch = _run_csr_add(A, B)
        assert torch.equal(C_manual.indptr, C_cusparse.indptr)
        assert torch.equal(C_manual.indices, C_cusparse.indices)
        assert torch.equal(C_manual.data, C_cusparse.data)
        assert torch.equal(C_pytorch.crow_indices().to(torch.int32), C_manual.indptr)
        assert torch.equal(C_pytorch.col_indices().to(torch.int32), C_manual.indices)
        assert torch.equal(C_pytorch.values(), C_manual.data)

    def test_disjoint_columns(self):
        """Matrices with non-overlapping column indices."""
        A = torch.sparse_csr_tensor(
            torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 0], dtype=torch.int32, device="cuda"),
            torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda"),
            size=(2, 4),
        )
        B = torch.sparse_csr_tensor(
            torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([3, 3], dtype=torch.int32, device="cuda"),
            torch.tensor([5.0, 6.0], dtype=torch.float32, device="cuda"),
            size=(2, 4),
        )
        C_manual, C_cusparse, C_pytorch = _run_csr_add(A, B)
        assert torch.equal(C_manual.indptr, C_cusparse.indptr)
        assert torch.equal(C_manual.indices, C_cusparse.indices)
        assert torch.equal(C_manual.data, C_cusparse.data)

    def test_full_overlap(self):
        """All nonzeros in the same positions — values should sum."""
        A = torch.sparse_csr_tensor(
            torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda"),
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"),
            size=(2, 2),
        )
        B = torch.sparse_csr_tensor(
            torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda"),
            torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device="cuda"),
            size=(2, 2),
        )
        C_manual, C_cusparse, C_pytorch = _run_csr_add(A, B)
        assert torch.equal(C_manual.data, C_cusparse.data)
        expected_vals = torch.tensor([11.0, 22.0, 33.0, 44.0], device="cuda")
        assert torch.equal(C_manual.data, expected_vals)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        """Randomized: manual vs cuSPARSE vs PyTorch on random CSR matrices."""
        import random
        rng = random.Random(seed)
        rows = rng.randint(5, 100)
        cols = rng.randint(5, 100)

        A = make_random_csr(rows, cols, nnz_per_row_range=(0, min(5, cols)), seed=seed * 1000)
        B = make_random_csr(rows, cols, nnz_per_row_range=(0, min(5, cols)), seed=seed * 1000 + 1)

        C_manual, C_cusparse, C_pytorch = _run_csr_add(A, B)

        # Manual == cuSPARSE (structural)
        assert torch.equal(C_manual.indptr, C_cusparse.indptr), "indptr mismatch: manual vs cuSPARSE"
        assert torch.equal(C_manual.indices, C_cusparse.indices), "indices mismatch: manual vs cuSPARSE"
        assert torch.equal(C_manual.data, C_cusparse.data), "data mismatch: manual vs cuSPARSE"

        # Manual == PyTorch
        assert torch.equal(C_pytorch.crow_indices().to(torch.int32), C_manual.indptr), "indptr mismatch: manual vs PyTorch"
        assert torch.equal(C_pytorch.col_indices().to(torch.int32), C_manual.indices), "indices mismatch: manual vs PyTorch"
        assert torch.equal(C_pytorch.values(), C_manual.data), "data mismatch: manual vs PyTorch"
