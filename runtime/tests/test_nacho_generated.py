"""Tests for nacho-generated operations."""
import pytest
import random
import torch
from conftest import make_random_csr
from nacho_runtime import CVector, CSR, DCSR
from nacho_runtime import (
    nacho_sparse_vec_mul,
    nacho_sparse_vec_add,
    nacho_sparse_vec_apb_c,
    nacho_sparse_vec_ab_pc,
    nacho_csr_add,
    nacho_dcsr_mul,
    nacho_dcsr_add,
)


# ===== Helpers =====

def make_cvector(indices, values, size):
    """Create a CVector from Python lists."""
    idx = torch.tensor(indices, dtype=torch.int32, device="cuda")
    val = torch.tensor(values, dtype=torch.float32, device="cuda")
    return CVector(idx, val, size)


def make_dcsr(row_indices, row_offsets, col_indices, values, nrows, ncols):
    """Create a DCSR from Python lists."""
    ri = torch.tensor(row_indices, dtype=torch.int32, device="cuda")
    ro = torch.tensor(row_offsets, dtype=torch.int32, device="cuda")
    ci = torch.tensor(col_indices, dtype=torch.int32, device="cuda")
    v = torch.tensor(values, dtype=torch.float32, device="cuda")
    return DCSR(ri, ro, ci, v, nrows, ncols)


def make_csr(row_offsets, col_indices, values, nrows, ncols):
    """Create a CSR from Python lists."""
    ro = torch.tensor(row_offsets, dtype=torch.int32, device="cuda")
    ci = torch.tensor(col_indices, dtype=torch.int32, device="cuda")
    v = torch.tensor(values, dtype=torch.float32, device="cuda")
    shape = torch.tensor([nrows, ncols], dtype=torch.int32)
    return CSR(ro, ci, v, shape)


def torch_csr_to_nb(t):
    """Convert torch sparse CSR tensor to runtime CSR."""
    return CSR(
        t.crow_indices().to(torch.int32),
        t.col_indices().to(torch.int32),
        t.values().to(torch.float32),
        torch.tensor([t.shape[0], t.shape[1]], dtype=torch.int32),
    )


def cvector_to_dense(vec):
    """Convert CVector to dense CUDA tensor."""
    dense = torch.zeros(vec.size, dtype=torch.float32, device="cuda")
    if vec.indices.numel() > 0:
        dense[vec.indices.to(torch.int64)] = vec.data
    return dense


def dcsr_to_dense(mat):
    """Convert DCSR to dense CUDA tensor."""
    dense = torch.zeros((mat.nrows, mat.ncols), dtype=torch.float32, device="cuda")
    ri = mat.row_indices
    ro = mat.row_offsets
    ci = mat.col_indices
    vals = mat.data
    for i in range(ri.shape[0]):
        row = int(ri[i].item())
        start = int(ro[i].item())
        end = int(ro[i + 1].item())
        if end > start:
            dense[row, ci[start:end].to(torch.int64)] = vals[start:end]
    return dense


def csr_to_dense(mat):
    """Convert CSR to dense CUDA tensor."""
    nrows = int(mat.shape[0].item())
    ncols = int(mat.shape[1].item())
    return torch.sparse_csr_tensor(
        mat.indptr.to(torch.int64),
        mat.indices.to(torch.int64),
        mat.data,
        size=(nrows, ncols),
    ).to_dense()


def random_sparse_vec(size, nnz, seed=None):
    """Generate random sorted sparse vector data."""
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(size), nnz))
    values = [rng.uniform(0.1, 10.0) for _ in range(nnz)]
    return indices, values


def reference_sparse_vec_mul(a_indices, a_values, b_indices, b_values):
    """CPU reference: element-wise multiply (intersection)."""
    a_dict = dict(zip(a_indices, a_values))
    b_dict = dict(zip(b_indices, b_values))
    common = sorted(set(a_indices) & set(b_indices))
    out_values = [a_dict[k] * b_dict[k] for k in common]
    return common, out_values


def reference_sparse_vec_add(a_indices, a_values, b_indices, b_values):
    """CPU reference: element-wise add (union)."""
    result = {}
    for i, v in zip(a_indices, a_values):
        result[i] = result.get(i, 0.0) + v
    for i, v in zip(b_indices, b_values):
        result[i] = result.get(i, 0.0) + v
    indices = sorted(result.keys())
    values = [result[k] for k in indices]
    return indices, values


def reference_sparse_vec_apb_c(a_idx, a_val, b_idx, b_val, c_idx, c_val):
    """CPU reference: (a + b) * c."""
    ab_idx, ab_val = reference_sparse_vec_add(a_idx, a_val, b_idx, b_val)
    return reference_sparse_vec_mul(ab_idx, ab_val, c_idx, c_val)


def reference_sparse_vec_ab_pc(a_idx, a_val, b_idx, b_val, c_idx, c_val):
    """CPU reference: (a * b) + c."""
    ab_idx, ab_val = reference_sparse_vec_mul(a_idx, a_val, b_idx, b_val)
    return reference_sparse_vec_add(ab_idx, ab_val, c_idx, c_val)


def random_dcsr(nrows, ncols, density, seed=None):
    """Generate random DCSR data.

    Returns (row_indices, row_offsets, col_indices, values).
    """
    rng = random.Random(seed)
    max_nnz_per_row = max(1, int(ncols * density))

    row_indices = []
    row_offsets = [0]
    col_indices = []
    values = []

    for r in range(nrows):
        if rng.random() < density * 3:  # include this row
            nnz_row = rng.randint(1, min(max_nnz_per_row, ncols))
            cols = sorted(rng.sample(range(ncols), nnz_row))
            row_indices.append(r)
            col_indices.extend(cols)
            values.extend([rng.uniform(0.1, 10.0) for _ in range(nnz_row)])
            row_offsets.append(len(col_indices))

    return row_indices, row_offsets, col_indices, values


def reference_dcsr_mul(a_ri, a_ro, a_ci, a_val, b_ri, b_ro, b_ci, b_val):
    """CPU reference: DCSR element-wise multiply (intersection)."""
    # Build row -> {col -> val} maps
    def to_dict(ri, ro, ci, val):
        d = {}
        for i, r in enumerate(ri):
            start, end = ro[i], ro[i + 1]
            d[r] = dict(zip(ci[start:end], val[start:end]))
        return d

    a_dict = to_dict(a_ri, a_ro, a_ci, a_val)
    b_dict = to_dict(b_ri, b_ro, b_ci, b_val)

    common_rows = sorted(set(a_dict.keys()) & set(b_dict.keys()))
    out_ri, out_ro, out_ci, out_val = [], [0], [], []
    for r in common_rows:
        common_cols = sorted(set(a_dict[r].keys()) & set(b_dict[r].keys()))
        if common_cols:
            out_ri.append(r)
            for c in common_cols:
                out_ci.append(c)
                out_val.append(a_dict[r][c] * b_dict[r][c])
            out_ro.append(len(out_ci))

    return out_ri, out_ro, out_ci, out_val


def reference_dcsr_add(a_ri, a_ro, a_ci, a_val, b_ri, b_ro, b_ci, b_val):
    """CPU reference: DCSR element-wise add (union)."""
    def to_dict(ri, ro, ci, val):
        d = {}
        for i, r in enumerate(ri):
            start, end = ro[i], ro[i + 1]
            d[r] = dict(zip(ci[start:end], val[start:end]))
        return d

    a_dict = to_dict(a_ri, a_ro, a_ci, a_val)
    b_dict = to_dict(b_ri, b_ro, b_ci, b_val)

    all_rows = sorted(set(a_dict.keys()) | set(b_dict.keys()))
    out_ri, out_ro, out_ci, out_val = [], [0], [], []
    for r in all_rows:
        merged = {}
        if r in a_dict:
            for c, v in a_dict[r].items():
                merged[c] = merged.get(c, 0.0) + v
        if r in b_dict:
            for c, v in b_dict[r].items():
                merged[c] = merged.get(c, 0.0) + v
        if merged:
            out_ri.append(r)
            for c in sorted(merged.keys()):
                out_ci.append(c)
                out_val.append(merged[c])
            out_ro.append(len(out_ci))

    return out_ri, out_ro, out_ci, out_val


# ===== sparse_vec_mul tests =====

class TestNachoSparseVecMul:
    """Tests for the nacho-generated sparse_vec_mul operation."""

    def test_basic_intersection(self):
        A = make_cvector([1, 3, 4], [2.0, 4.0, 5.0], 5)
        B = make_cvector([0, 2, 3], [1.0, 3.0, 4.0], 5)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.shape[0] == 1
        assert result.indices.cpu().tolist() == [3]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [16.0]

    def test_full_overlap(self):
        A = make_cvector([0, 1, 2], [2.0, 3.0, 4.0], 3)
        B = make_cvector([0, 1, 2], [5.0, 6.0, 7.0], 3)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.shape[0] == 3
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [10.0, 18.0, 28.0]

    def test_no_overlap(self):
        A = make_cvector([0, 1, 2], [1.0, 2.0, 3.0], 10)
        B = make_cvector([3, 4, 5], [1.0, 2.0, 3.0], 10)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.shape[0] == 0

    def test_single_element(self):
        A = make_cvector([5], [3.0], 10)
        B = make_cvector([5], [7.0], 10)
        result = nacho_sparse_vec_mul(A, B)
        assert result.indices.cpu().tolist() == [5]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [21.0]

    def test_result_size_preserved(self):
        A = make_cvector([1, 3], [2.0, 4.0], 100)
        B = make_cvector([3, 5], [3.0, 6.0], 100)
        result = nacho_sparse_vec_mul(A, B)
        assert result.size == 100

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed=seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed=seed * 1000 + 1)
        A = make_cvector(a_idx, a_val, size)
        B = make_cvector(b_idx, b_val, size)
        result = nacho_sparse_vec_mul(A, B)
        torch.cuda.synchronize()
        got_dense = cvector_to_dense(result)
        expected_dense = cvector_to_dense(A) * cvector_to_dense(B)
        assert torch.allclose(got_dense, expected_dense, rtol=1e-4, atol=1e-5)


# ===== sparse_vec_add tests =====

class TestNachoSparseVecAdd:
    """Tests for the nacho-generated sparse_vec_add operation."""

    def test_basic_union(self):
        A = make_cvector([1, 3], [2.0, 4.0], 5)
        B = make_cvector([2, 3], [3.0, 5.0], 5)
        result = nacho_sparse_vec_add(A, B)
        assert result.indices.cpu().tolist() == [1, 2, 3]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [2.0, 3.0, 9.0]

    def test_no_overlap(self):
        A = make_cvector([0, 1], [1.0, 2.0], 10)
        B = make_cvector([3, 4], [3.0, 4.0], 10)
        result = nacho_sparse_vec_add(A, B)
        assert result.indices.shape[0] == 4
        assert result.indices.cpu().tolist() == [0, 1, 3, 4]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [1.0, 2.0, 3.0, 4.0]

    def test_full_overlap(self):
        A = make_cvector([0, 1, 2], [1.0, 2.0, 3.0], 3)
        B = make_cvector([0, 1, 2], [4.0, 5.0, 6.0], 3)
        result = nacho_sparse_vec_add(A, B)
        assert result.indices.shape[0] == 3
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [5.0, 7.0, 9.0]

    def test_empty_result(self):
        """Both inputs empty."""
        A = make_cvector([], [], 10)
        B = make_cvector([], [], 10)
        result = nacho_sparse_vec_add(A, B)
        assert result.indices.shape[0] == 0

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed=seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed=seed * 1000 + 1)
        A = make_cvector(a_idx, a_val, size)
        B = make_cvector(b_idx, b_val, size)
        result = nacho_sparse_vec_add(A, B)
        torch.cuda.synchronize()
        got_dense = cvector_to_dense(result)
        expected_dense = cvector_to_dense(A) + cvector_to_dense(B)
        assert torch.allclose(got_dense, expected_dense, rtol=1e-4, atol=1e-5)


# ===== sparse_vec_apb_c tests =====

class TestNachoSparseVecApbC:
    """Tests for the nacho-generated sparse_vec_apb_c: (a + b) * c."""

    def test_basic(self):
        A = make_cvector([1, 3], [2.0, 4.0], 5)
        B = make_cvector([2, 3], [3.0, 5.0], 5)
        C = make_cvector([1, 2, 3], [10.0, 10.0, 10.0], 5)
        result = nacho_sparse_vec_apb_c(A, B, C)
        # (a+b) at index 1: 2.0, at 2: 3.0, at 3: 9.0
        # intersection with c: {1, 2, 3}
        # result: {1: 20.0, 2: 30.0, 3: 90.0}
        assert result.indices.cpu().tolist() == [1, 2, 3]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [20.0, 30.0, 90.0]

    def test_no_overlap_with_c(self):
        A = make_cvector([0, 1], [2.0, 3.0], 10)
        B = make_cvector([1, 2], [4.0, 5.0], 10)
        C = make_cvector([5, 6], [1.0, 1.0], 10)
        result = nacho_sparse_vec_apb_c(A, B, C)
        assert result.indices.shape[0] == 0

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        nnz_c = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed=seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed=seed * 1000 + 1)
        c_idx, c_val = random_sparse_vec(size, nnz_c, seed=seed * 1000 + 2)
        A = make_cvector(a_idx, a_val, size)
        B = make_cvector(b_idx, b_val, size)
        C = make_cvector(c_idx, c_val, size)
        result = nacho_sparse_vec_apb_c(A, B, C)
        torch.cuda.synchronize()
        got_dense = cvector_to_dense(result)
        expected_dense = (cvector_to_dense(A) + cvector_to_dense(B)) * cvector_to_dense(C)
        assert torch.allclose(got_dense, expected_dense, rtol=1e-4, atol=1e-5)


# ===== sparse_vec_ab_pc tests =====

class TestNachoSparseVecAbPc:
    """Tests for the nacho-generated sparse_vec_ab_pc: (a * b) + c."""

    def test_basic(self):
        A = make_cvector([1, 3], [2.0, 4.0], 5)
        B = make_cvector([1, 3], [5.0, 6.0], 5)
        C = make_cvector([0, 1], [100.0, 7.0], 5)
        result = nacho_sparse_vec_ab_pc(A, B, C)
        # a*b: {1: 10.0, 3: 24.0}
        # + c: {0: 100.0, 1: 7.0}
        # result: {0: 100.0, 1: 17.0, 3: 24.0}
        assert result.indices.cpu().tolist() == [0, 1, 3]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [100.0, 17.0, 24.0]

    def test_ab_empty(self):
        A = make_cvector([0, 1], [1.0, 2.0], 10)
        B = make_cvector([5, 6], [1.0, 2.0], 10)
        C = make_cvector([3], [42.0], 10)
        result = nacho_sparse_vec_ab_pc(A, B, C)
        # a*b is empty, result = c
        assert result.indices.cpu().tolist() == [3]
        assert pytest.approx(result.data.cpu().tolist(), rel=1e-5) == [42.0]

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        nnz_c = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed=seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed=seed * 1000 + 1)
        c_idx, c_val = random_sparse_vec(size, nnz_c, seed=seed * 1000 + 2)
        A = make_cvector(a_idx, a_val, size)
        B = make_cvector(b_idx, b_val, size)
        C = make_cvector(c_idx, c_val, size)
        result = nacho_sparse_vec_ab_pc(A, B, C)
        torch.cuda.synchronize()
        got_dense = cvector_to_dense(result)
        expected_dense = (cvector_to_dense(A) * cvector_to_dense(B)) + cvector_to_dense(C)
        assert torch.allclose(got_dense, expected_dense, rtol=1e-4, atol=1e-5)


# ===== dcsr_mul tests =====

# ===== csr_add tests =====

class TestNachoCsrAdd:
    """Tests for the nacho-generated csr_add operation."""

    def test_basic_union(self):
        A = make_csr([0, 1, 2], [0, 1], [1.0, 2.0], 2, 3)
        B = make_csr([0, 1, 2], [1, 2], [3.0, 4.0], 2, 3)
        result = nacho_csr_add(A, B)
        expected = torch.tensor(
            [[1.0, 3.0, 0.0],
             [0.0, 2.0, 4.0]],
            dtype=torch.float32,
            device="cuda",
        )
        assert torch.allclose(csr_to_dense(result), expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        nrows = rng.randint(5, 100)
        ncols = rng.randint(5, 100)
        max_nnz = max(0, min(5, ncols))
        A_torch = make_random_csr(
            nrows, ncols, nnz_per_row_range=(0, max_nnz), seed=seed * 1000
        )
        B_torch = make_random_csr(
            nrows, ncols, nnz_per_row_range=(0, max_nnz), seed=seed * 1000 + 1
        )
        A = torch_csr_to_nb(A_torch)
        B = torch_csr_to_nb(B_torch)
        result = nacho_csr_add(A, B)
        torch.cuda.synchronize()
        got_dense = csr_to_dense(result)
        expected_dense = (A_torch + B_torch).to_dense()
        assert torch.allclose(got_dense, expected_dense, rtol=1e-4, atol=1e-5)


# ===== dcsr_mul tests =====

class TestNachoDcsrMul:
    """Tests for the nacho-generated dcsr_mul operation."""

    def test_basic_intersection(self):
        # A: row 0: {0: 2.0, 1: 3.0}, row 2: {1: 4.0}
        A = make_dcsr([0, 2], [0, 2, 3], [0, 1, 1], [2.0, 3.0, 4.0], 3, 3)
        # B: row 0: {1: 5.0}, row 1: {0: 6.0}
        B = make_dcsr([0, 1], [0, 1, 2], [1, 0], [5.0, 6.0], 3, 3)
        result = nacho_dcsr_mul(A, B)
        # Intersection: row 0 col 1 -> 3.0 * 5.0 = 15.0
        ri = result.row_indices.cpu().tolist()
        ro = result.row_offsets.cpu().tolist()
        ci = result.col_indices.cpu().tolist()
        vals = result.data.cpu().tolist()
        assert ri == [0]
        assert ro == [0, 1]
        assert ci == [1]
        assert pytest.approx(vals, rel=1e-5) == [15.0]

    def test_no_overlap(self):
        A = make_dcsr([0], [0, 1], [0], [1.0], 5, 5)
        B = make_dcsr([1], [0, 1], [1], [2.0], 5, 5)
        result = nacho_dcsr_mul(A, B)
        assert result.col_indices.shape[0] == 0  # nnz = 0

    def test_full_overlap(self):
        A = make_dcsr([0, 1], [0, 2, 4], [0, 1, 0, 1], [1.0, 2.0, 3.0, 4.0], 2, 2)
        B = make_dcsr([0, 1], [0, 2, 4], [0, 1, 0, 1], [5.0, 6.0, 7.0, 8.0], 2, 2)
        result = nacho_dcsr_mul(A, B)
        ri = result.row_indices.cpu().tolist()
        ci = result.col_indices.cpu().tolist()
        vals = result.data.cpu().tolist()
        assert ri == [0, 1]
        assert ci == [0, 1, 0, 1]
        assert pytest.approx(vals, rel=1e-5) == [5.0, 12.0, 21.0, 32.0]

    @pytest.mark.parametrize("seed", range(10))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        nrows = rng.randint(10, 100)
        ncols = rng.randint(10, 100)
        density = rng.uniform(0.05, 0.3)
        a_ri, a_ro, a_ci, a_val = random_dcsr(nrows, ncols, density, seed=seed * 1000)
        b_ri, b_ro, b_ci, b_val = random_dcsr(nrows, ncols, density, seed=seed * 1000 + 1)

        if not a_ri or not b_ri:
            return  # skip if random gen produced empty matrices

        A = make_dcsr(a_ri, a_ro, a_ci, a_val, nrows, ncols)
        B = make_dcsr(b_ri, b_ro, b_ci, b_val, nrows, ncols)
        result = nacho_dcsr_mul(A, B)
        torch.cuda.synchronize()
        got_dense = dcsr_to_dense(result)
        expected_dense = dcsr_to_dense(A) * dcsr_to_dense(B)
        assert torch.allclose(got_dense, expected_dense, rtol=1e-4, atol=1e-5)


# ===== dcsr_add tests =====

class TestNachoDcsrAdd:
    """Tests for the nacho-generated dcsr_add operation."""

    def test_basic_union(self):
        A = make_dcsr([0], [0, 1], [0], [1.0], 3, 3)
        B = make_dcsr([0], [0, 1], [1], [2.0], 3, 3)
        result = nacho_dcsr_add(A, B)
        ri = result.row_indices.cpu().tolist()
        ci = result.col_indices.cpu().tolist()
        vals = result.data.cpu().tolist()
        assert ri == [0]
        assert sorted(ci) == [0, 1]
        assert pytest.approx(sorted(vals), rel=1e-5) == [1.0, 2.0]

    def test_overlapping_entries(self):
        A = make_dcsr([0, 1], [0, 2, 3], [0, 1, 0], [1.0, 2.0, 3.0], 2, 2)
        B = make_dcsr([0, 1], [0, 2, 3], [0, 1, 0], [4.0, 5.0, 6.0], 2, 2)
        result = nacho_dcsr_add(A, B)
        ri = result.row_indices.cpu().tolist()
        ci = result.col_indices.cpu().tolist()
        vals = result.data.cpu().tolist()
        assert ri == [0, 1]
        assert ci == [0, 1, 0]
        assert pytest.approx(vals, rel=1e-5) == [5.0, 7.0, 9.0]

    def test_no_overlap_rows(self):
        A = make_dcsr([0], [0, 1], [0], [1.0], 5, 5)
        B = make_dcsr([2], [0, 1], [1], [2.0], 5, 5)
        result = nacho_dcsr_add(A, B)
        ri = result.row_indices.cpu().tolist()
        assert ri == [0, 2]

    @pytest.mark.parametrize("seed", range(10))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        nrows = rng.randint(10, 100)
        ncols = rng.randint(10, 100)
        density = rng.uniform(0.05, 0.3)
        a_ri, a_ro, a_ci, a_val = random_dcsr(nrows, ncols, density, seed=seed * 1000)
        b_ri, b_ro, b_ci, b_val = random_dcsr(nrows, ncols, density, seed=seed * 1000 + 1)

        if not a_ri or not b_ri:
            return

        A = make_dcsr(a_ri, a_ro, a_ci, a_val, nrows, ncols)
        B = make_dcsr(b_ri, b_ro, b_ci, b_val, nrows, ncols)
        result = nacho_dcsr_add(A, B)
        torch.cuda.synchronize()
        got_dense = dcsr_to_dense(result)
        expected_dense = dcsr_to_dense(A) + dcsr_to_dense(B)
        assert torch.allclose(got_dense, expected_dense, rtol=1e-4, atol=1e-5)
