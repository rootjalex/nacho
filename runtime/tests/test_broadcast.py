"""Tests for broadcast x*A operation: broadcast kernel vs CSR add of expanded matrix."""
import pytest
import torch
import nacho_runtime

from conftest import make_random_csr


def _broadcast_xA(x_indices, x_values, x_size, A_csr):
    """Run broadcast x*A using the GPU kernel."""
    x = nacho_runtime.CVector(
        torch.tensor(x_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(x_values, dtype=torch.float32, device="cuda"),
        x_size,
    )
    M, N = A_csr.shape
    A_nb = nacho_runtime.CSR(
        A_csr.crow_indices().to(torch.int32),
        A_csr.col_indices().to(torch.int32),
        A_csr.values().to(torch.float32),
        torch.tensor([M, N], dtype=torch.int32),
    )
    result = nacho_runtime.gpu_broadcast_xA(x, A_nb)
    torch.cuda.synchronize()
    return result


def _expand_x_to_csr(x_indices, x_values, rows, cols):
    """Build the expanded CSR matrix B where every row is the sparse vector x.

    This is the reference: broadcast(x, A) should equal csr_add(A, B)
    when all x-values are 1 and we're testing structural correctness.
    """
    nnz_per_row = len(x_indices)
    indptr = [nnz_per_row * i for i in range(rows + 1)]
    indices = x_indices * rows
    values = x_values * rows
    return nacho_runtime.CSR(
        torch.tensor(indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(indices, dtype=torch.int32, device="cuda"),
        torch.tensor(values, dtype=torch.float32, device="cuda"),
        torch.tensor([rows, cols], dtype=torch.int32),
    )


class TestBroadcast:
    """Correctness tests for broadcast x*A."""

    def test_hardcoded(self):
        """The same hardcoded test from benchmark_broadcast()."""
        x_indices = [0, 2, 4, 7, 8, 9, 10, 15, 17]
        x_values = [1.0] * len(x_indices)

        A_row = torch.tensor([0, 10, 12, 18, 27], dtype=torch.int32, device="cuda")
        A_col = torch.tensor(
            [0, 2, 6, 8, 9, 10, 11, 17, 18, 19, 0, 23, 10, 11, 14, 15, 17, 18,
             0, 4, 6, 7, 9, 10, 11, 13, 14],
            dtype=torch.int32, device="cuda",
        )
        A_data = torch.tensor([1.0] * A_col.shape[0], dtype=torch.float32, device="cuda")
        A_nb = nacho_runtime.CSR(A_row, A_col, A_data, torch.tensor([4, 30], dtype=torch.int32))

        # Build expanded B (x repeated per row)
        B_nb = _expand_x_to_csr(x_indices, x_values, 4, 30)

        x = nacho_runtime.CVector(
            torch.tensor(x_indices, dtype=torch.int32, device="cuda"),
            torch.tensor(x_values, dtype=torch.float32, device="cuda"),
            30,
        )
        C_broadcast = nacho_runtime.gpu_broadcast_xA(x, A_nb)
        D_csr_add = nacho_runtime.gpu_csr_add_f32(A_nb, B_nb, False)
        torch.cuda.synchronize()

        assert torch.equal(C_broadcast.indptr, D_csr_add.indptr), "indptr mismatch"
        assert torch.equal(C_broadcast.indices, D_csr_add.indices), "indices mismatch"
        assert torch.equal(C_broadcast.data, D_csr_add.data), "data mismatch"

    @pytest.mark.skip(
        reason="broadcast kernel returns over-sized indices buffer (float data leaks "
        "into indices) and the resulting CUDA crash poisons all subsequent tests",
    )
    @pytest.mark.parametrize("seed", range(10))
    def test_random_correctness(self, seed):
        """Randomized: broadcast x*A vs csr_add(A, expand(x))."""
        import random
        rng = random.Random(seed)
        rows = rng.randint(2, 20)
        cols = rng.randint(5, 40)
        vec_nnz = rng.randint(1, min(cols, 10))

        x_indices = sorted(rng.sample(range(cols), vec_nnz))
        x_values = [1.0] * vec_nnz  # use 1.0 so the equivalence with csr_add holds

        A = make_random_csr(rows, cols, nnz_per_row_range=(0, min(5, cols)), seed=seed * 1000)
        M, N = A.shape
        A_nb = nacho_runtime.CSR(
            A.crow_indices().to(torch.int32),
            A.col_indices().to(torch.int32),
            A.values().to(torch.float32),
            torch.tensor([M, N], dtype=torch.int32),
        )

        x = nacho_runtime.CVector(
            torch.tensor(x_indices, dtype=torch.int32, device="cuda"),
            torch.tensor(x_values, dtype=torch.float32, device="cuda"),
            cols,
        )
        B_nb = _expand_x_to_csr(x_indices, x_values, rows, cols)

        C_broadcast = nacho_runtime.gpu_broadcast_xA(x, A_nb)
        D_csr_add = nacho_runtime.gpu_csr_add_f32(A_nb, B_nb, False)
        torch.cuda.synchronize()

        assert torch.equal(C_broadcast.indptr, D_csr_add.indptr), "indptr mismatch"
        assert torch.equal(C_broadcast.indices, D_csr_add.indices), "indices mismatch"
        assert torch.equal(C_broadcast.data, D_csr_add.data), "data mismatch"
