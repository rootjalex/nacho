"""Tests for SpGEMM: manual kernel vs cuSPARSE."""
import pytest
import torch
import nacho_runtime

from conftest import make_random_csr


def _to_nb_csr(t):
    """Convert a torch sparse_csr_tensor to nanobind CSR."""
    M, N = t.shape
    return nacho_runtime.CSR(
        t.crow_indices().to(torch.int32),
        t.col_indices().to(torch.int32),
        t.values().to(torch.float32),
        torch.tensor([M, N], dtype=torch.int32),
    )


def _run_spgemm(A_torch, B_torch):
    """Run manual and cuSPARSE SpGEMM, return both results."""
    A_nb = _to_nb_csr(A_torch)
    B_nb = _to_nb_csr(B_torch)
    C_manual = nacho_runtime.spgemm(A_nb, B_nb, False)
    C_cusparse = nacho_runtime.spgemm(A_nb, B_nb, True)
    torch.cuda.synchronize()
    return C_manual, C_cusparse


class TestSpGEMM:
    """Correctness tests for sparse matrix-matrix multiply."""

    def test_identity_multiply(self):
        """Multiplying by a sparse identity should return the original structure."""
        # 3x3 identity
        I = torch.sparse_csr_tensor(
            torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"),
            size=(3, 3),
        )
        A = torch.sparse_csr_tensor(
            torch.tensor([0, 2, 3, 5], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 2, 1, 0, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda"),
            size=(3, 3),
        )
        C_manual, C_cusparse = _run_spgemm(A, I)
        assert torch.equal(C_manual.indptr, C_cusparse.indptr)
        assert torch.equal(C_manual.indices, C_cusparse.indices)

    def test_square_self_multiply(self):
        """A * A for a small square matrix."""
        A = torch.sparse_csr_tensor(
            torch.tensor([0, 2, 4, 5], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 1, 0, 2, 1], dtype=torch.int32, device="cuda"),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda"),
            size=(3, 3),
        )
        C_manual, C_cusparse = _run_spgemm(A, A)
        assert torch.equal(C_manual.indptr, C_cusparse.indptr)
        assert torch.equal(C_manual.indices, C_cusparse.indices)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_AxA(self, seed):
        """Randomized A*A on random square CSR matrices."""
        import random
        rng = random.Random(seed)
        n = rng.randint(5, 50)
        A = make_random_csr(n, n, nnz_per_row_range=(0, min(5, n)), seed=seed)

        C_manual, C_cusparse = _run_spgemm(A, A)
        assert torch.equal(C_manual.indptr, C_cusparse.indptr), "indptr mismatch (A*A)"
        assert torch.equal(C_manual.indices, C_cusparse.indices), "indices mismatch (A*A)"

    @pytest.mark.parametrize("seed", range(20))
    def test_random_AxB(self, seed):
        """Randomized A*B on compatible random CSR matrices."""
        import random
        rng = random.Random(seed)
        M = rng.randint(5, 30)
        K = rng.randint(5, 30)
        N = rng.randint(5, 30)

        A = make_random_csr(M, K, nnz_per_row_range=(0, min(5, K)), seed=seed * 1000)
        B = make_random_csr(K, N, nnz_per_row_range=(0, min(5, N)), seed=seed * 1000 + 1)

        C_manual, C_cusparse = _run_spgemm(A, B)
        assert torch.equal(C_manual.indptr, C_cusparse.indptr), "indptr mismatch (A*B)"
        assert torch.equal(C_manual.indices, C_cusparse.indices), "indices mismatch (A*B)"
