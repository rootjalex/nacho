"""Tests for nacho-generated CPU operations via ctypes."""
import ctypes
import os
import random

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

BUILD_DIR = os.environ.get(
    "NACHO_BUILD_DIR",
    os.path.join(os.path.dirname(__file__), "..", "build"),
)
_lib_path = os.path.join(BUILD_DIR, "libcpu_codegen.so")
_lib = ctypes.CDLL(_lib_path)

# ---------------------------------------------------------------------------
# ctypes type aliases
# ---------------------------------------------------------------------------

c_int = ctypes.c_int
c_int_p = ctypes.POINTER(ctypes.c_int)
c_int_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
c_float = ctypes.c_float
c_float_p = ctypes.POINTER(ctypes.c_float)
c_float_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))


def _int_array(lst):
    arr = (ctypes.c_int * len(lst))(*lst)
    return arr


def _float_array(lst):
    arr = (ctypes.c_float * len(lst))(*lst)
    return arr


def _copy_int_buf(ptr, n):
    return [ptr[i] for i in range(n)]


def _copy_float_buf(ptr, n):
    return [ptr[i] for i in range(n)]


def _free(ptr):
    _lib.cpu_free(ctypes.cast(ptr, ctypes.c_void_p))


# ---------------------------------------------------------------------------
# Dense conversion helpers
# ---------------------------------------------------------------------------


def cvector_to_dense(indices, values, size):
    dense = np.zeros(size, dtype=np.float32)
    for i, v in zip(indices, values):
        dense[i] = v
    return dense


def csr_to_dense(offsets, indices, values, nrows, ncols):
    dense = np.zeros((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        for p in range(offsets[i], offsets[i + 1]):
            dense[i, indices[p]] = values[p]
    return dense


def dcsr_to_dense(row_indices, row_offsets, col_indices, values, nrows, ncols):
    dense = np.zeros((nrows, ncols), dtype=np.float32)
    for idx in range(len(row_indices)):
        row = row_indices[idx]
        start = row_offsets[idx]
        end = row_offsets[idx + 1]
        for p in range(start, end):
            dense[row, col_indices[p]] = values[p]
    return dense


def coo_to_dense(row_indices, col_indices, values, nrows, ncols):
    dense = np.zeros((nrows, ncols), dtype=np.float32)
    for r, c, v in zip(row_indices, col_indices, values):
        dense[r, c] += v
    return dense


def tcsf_to_dense(i_indices, j_offsets, j_indices, k_offsets, k_indices,
                  values, shape):
    dense = np.zeros(shape, dtype=np.float32)
    for ii in range(len(i_indices)):
        i = i_indices[ii]
        j_start = j_offsets[ii]
        j_end = j_offsets[ii + 1]
        for jj in range(j_start, j_end):
            j = j_indices[jj]
            k_start = k_offsets[jj]
            k_end = k_offsets[jj + 1]
            for kk in range(k_start, k_end):
                k = k_indices[kk]
                dense[i, j, k] = values[kk]
    return dense


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------


def random_sparse_vec(size, nnz, seed):
    rng = random.Random(seed)
    nnz = min(nnz, size)
    indices = sorted(rng.sample(range(size), nnz))
    values = [rng.uniform(0.1, 10.0) for _ in range(nnz)]
    return indices, values


def random_csr(nrows, ncols, nnz_per_row_range, seed):
    rng = random.Random(seed)
    lo, hi = nnz_per_row_range
    offsets = [0]
    indices = []
    values = []
    for _ in range(nrows):
        nnz_row = rng.randint(lo, min(hi, ncols))
        cols = sorted(rng.sample(range(ncols), nnz_row))
        indices.extend(cols)
        values.extend([rng.uniform(0.1, 10.0) for _ in range(nnz_row)])
        offsets.append(len(indices))
    return offsets, indices, values


def random_dcsr(nrows, ncols, density, seed):
    rng = random.Random(seed)
    max_nnz_per_row = max(1, int(ncols * density))
    row_indices = []
    row_offsets = [0]
    col_indices = []
    values = []
    for r in range(nrows):
        if rng.random() < density * 3:
            nnz_row = rng.randint(1, min(max_nnz_per_row, ncols))
            cols = sorted(rng.sample(range(ncols), nnz_row))
            row_indices.append(r)
            col_indices.extend(cols)
            values.extend([rng.uniform(0.1, 10.0) for _ in range(nnz_row)])
            row_offsets.append(len(col_indices))
    return row_indices, row_offsets, col_indices, values


def random_tcsf(shape, density, seed):
    rng = random.Random(seed)
    ni, nj, nk = shape
    i_indices = []
    j_offsets = [0]
    j_indices = []
    k_offsets = [0]
    k_indices = []
    values = []
    for i in range(ni):
        if rng.random() < density * 3:
            i_indices.append(i)
            j_count = 0
            for j in range(nj):
                if rng.random() < density * 3:
                    j_indices.append(j)
                    j_count += 1
                    k_count = 0
                    for k in range(nk):
                        if rng.random() < density:
                            k_indices.append(k)
                            values.append(rng.uniform(0.1, 10.0))
                            k_count += 1
                    k_offsets.append(len(k_indices))
                    if k_count == 0:
                        # Remove empty j entry
                        j_indices.pop()
                        k_offsets.pop()
                        j_count -= 1
            j_offsets.append(j_offsets[-1] + j_count)
            if j_count == 0:
                # Remove empty i entry
                i_indices.pop()
                j_offsets.pop()
    return i_indices, j_offsets, j_indices, k_offsets, k_indices, values


def random_coo(nrows, ncols, density, seed):
    rng = random.Random(seed)
    row_indices = []
    col_indices = []
    values = []
    for r in range(nrows):
        for c in range(ncols):
            if rng.random() < density:
                row_indices.append(r)
                col_indices.append(c)
                values.append(rng.uniform(0.1, 10.0))
    return row_indices, col_indices, values


# ---------------------------------------------------------------------------
# ctypes call wrappers
# ---------------------------------------------------------------------------


def call_sparse_vec_mul(a_idx, a_val, size, b_idx, b_val):
    out_nnz = c_int(0)
    out_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_sparse_vec_mul(
        c_int(size), c_int(len(a_idx)), _int_array(a_idx), _float_array(a_val),
        c_int(len(a_idx)),
        c_int(size), c_int(len(b_idx)), _int_array(b_idx), _float_array(b_val),
        c_int(len(b_idx)),
        c_int(size),
        ctypes.byref(out_nnz), ctypes.byref(out_indices),
        ctypes.byref(out_values),
    )
    n = out_nnz.value
    r_idx = _copy_int_buf(out_indices, n)
    r_val = _copy_float_buf(out_values, n)
    if n > 0:
        _free(out_indices)
        _free(out_values)
    return r_idx, r_val


def call_sparse_vec_add(a_idx, a_val, size, b_idx, b_val):
    out_nnz = c_int(0)
    out_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_sparse_vec_add(
        c_int(size), c_int(len(a_idx)), _int_array(a_idx), _float_array(a_val),
        c_int(len(a_idx)),
        c_int(size), c_int(len(b_idx)), _int_array(b_idx), _float_array(b_val),
        c_int(len(b_idx)),
        c_int(size),
        ctypes.byref(out_nnz), ctypes.byref(out_indices),
        ctypes.byref(out_values),
    )
    n = out_nnz.value
    r_idx = _copy_int_buf(out_indices, n)
    r_val = _copy_float_buf(out_values, n)
    if n > 0:
        _free(out_indices)
        _free(out_values)
    return r_idx, r_val


def call_sparse_vec_apb_c(a_idx, a_val, size, b_idx, b_val, c_idx, c_val):
    out_nnz = c_int(0)
    out_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_sparse_vec_apb_c(
        c_int(size), c_int(len(a_idx)), _int_array(a_idx), _float_array(a_val),
        c_int(len(a_idx)),
        c_int(size), c_int(len(b_idx)), _int_array(b_idx), _float_array(b_val),
        c_int(len(b_idx)),
        c_int(size), c_int(len(c_idx)), _int_array(c_idx), _float_array(c_val),
        c_int(len(c_idx)),
        c_int(size),
        ctypes.byref(out_nnz), ctypes.byref(out_indices),
        ctypes.byref(out_values),
    )
    n = out_nnz.value
    r_idx = _copy_int_buf(out_indices, n)
    r_val = _copy_float_buf(out_values, n)
    if n > 0:
        _free(out_indices)
        _free(out_values)
    return r_idx, r_val


def call_sparse_vec_ab_pc(a_idx, a_val, size, b_idx, b_val, c_idx, c_val):
    out_nnz = c_int(0)
    out_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_sparse_vec_ab_pc(
        c_int(size), c_int(len(a_idx)), _int_array(a_idx), _float_array(a_val),
        c_int(len(a_idx)),
        c_int(size), c_int(len(b_idx)), _int_array(b_idx), _float_array(b_val),
        c_int(len(b_idx)),
        c_int(size), c_int(len(c_idx)), _int_array(c_idx), _float_array(c_val),
        c_int(len(c_idx)),
        c_int(size),
        ctypes.byref(out_nnz), ctypes.byref(out_indices),
        ctypes.byref(out_values),
    )
    n = out_nnz.value
    r_idx = _copy_int_buf(out_indices, n)
    r_val = _copy_float_buf(out_values, n)
    if n > 0:
        _free(out_indices)
        _free(out_values)
    return r_idx, r_val


def call_csr_add(a_offsets, a_indices, a_values, nrows, ncols,
                 b_offsets, b_indices, b_values):
    out_nnz = c_int(0)
    out_indices = c_int_p()
    out_offsets = c_int_p()
    out_values = c_float_p()
    _lib.cpu_csr_add(
        c_int(nrows), c_int(ncols),
        _int_array(a_offsets), c_int(len(a_indices)),
        _int_array(a_indices), _float_array(a_values), c_int(len(a_indices)),
        c_int(nrows), c_int(ncols),
        _int_array(b_offsets), c_int(len(b_indices)),
        _int_array(b_indices), _float_array(b_values), c_int(len(b_indices)),
        c_int(nrows), c_int(ncols),
        ctypes.byref(out_nnz), ctypes.byref(out_indices),
        ctypes.byref(out_offsets), ctypes.byref(out_values),
    )
    n = out_nnz.value
    r_offsets = _copy_int_buf(out_offsets, nrows + 1)
    r_indices = _copy_int_buf(out_indices, n)
    r_values = _copy_float_buf(out_values, n)
    _free(out_offsets)
    if n > 0:
        _free(out_indices)
        _free(out_values)
    return r_offsets, r_indices, r_values


def call_dcsr_mul(a_ri, a_ro, a_ci, a_val, nrows, ncols,
                  b_ri, b_ro, b_ci, b_val):
    out_nnz = c_int(0)
    out_dim_i_length = c_int(0)
    out_j_indices = c_int_p()
    out_j_offsets = c_int_p()
    out_i_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_dcsr_mul(
        c_int(nrows), c_int(ncols),
        c_int(len(a_ri)), _int_array(a_ri), _int_array(a_ro),
        c_int(len(a_ci)), _int_array(a_ci), _float_array(a_val),
        c_int(len(a_ci)),
        c_int(nrows), c_int(ncols),
        c_int(len(b_ri)), _int_array(b_ri), _int_array(b_ro),
        c_int(len(b_ci)), _int_array(b_ci), _float_array(b_val),
        c_int(len(b_ci)),
        c_int(nrows), c_int(ncols),
        ctypes.byref(out_nnz), ctypes.byref(out_dim_i_length),
        ctypes.byref(out_j_indices), ctypes.byref(out_j_offsets),
        ctypes.byref(out_i_indices), ctypes.byref(out_values),
    )
    nnz = out_nnz.value
    if nnz == 0:
        return [], [0], [], []
    n_rows_out = out_dim_i_length.value
    r_ri = _copy_int_buf(out_i_indices, n_rows_out)
    r_ro = _copy_int_buf(out_j_offsets, n_rows_out + 1)
    r_ci = _copy_int_buf(out_j_indices, nnz)
    r_val = _copy_float_buf(out_values, nnz)
    _free(out_i_indices)
    _free(out_j_offsets)
    _free(out_j_indices)
    _free(out_values)
    return r_ri, r_ro, r_ci, r_val


def call_dcsr_add(a_ri, a_ro, a_ci, a_val, nrows, ncols,
                  b_ri, b_ro, b_ci, b_val):
    out_nnz = c_int(0)
    out_dim_i_length = c_int(0)
    out_j_indices = c_int_p()
    out_j_offsets = c_int_p()
    out_i_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_dcsr_add(
        c_int(nrows), c_int(ncols),
        c_int(len(a_ri)), _int_array(a_ri), _int_array(a_ro),
        c_int(len(a_ci)), _int_array(a_ci), _float_array(a_val),
        c_int(len(a_ci)),
        c_int(nrows), c_int(ncols),
        c_int(len(b_ri)), _int_array(b_ri), _int_array(b_ro),
        c_int(len(b_ci)), _int_array(b_ci), _float_array(b_val),
        c_int(len(b_ci)),
        c_int(nrows), c_int(ncols),
        ctypes.byref(out_nnz), ctypes.byref(out_dim_i_length),
        ctypes.byref(out_j_indices), ctypes.byref(out_j_offsets),
        ctypes.byref(out_i_indices), ctypes.byref(out_values),
    )
    nnz = out_nnz.value
    if nnz == 0:
        return [], [0], [], []
    n_rows_out = out_dim_i_length.value
    r_ri = _copy_int_buf(out_i_indices, n_rows_out)
    r_ro = _copy_int_buf(out_j_offsets, n_rows_out + 1)
    r_ci = _copy_int_buf(out_j_indices, nnz)
    r_val = _copy_float_buf(out_values, nnz)
    _free(out_i_indices)
    _free(out_j_offsets)
    _free(out_j_indices)
    _free(out_values)
    return r_ri, r_ro, r_ci, r_val


def call_tcsf_add(a_ii, a_jo, a_ji, a_ko, a_ki, a_val, shape,
                  b_ii, b_jo, b_ji, b_ko, b_ki, b_val):
    ni, nj, nk = shape
    out_nnz = c_int(0)
    out_dim_i_length = c_int(0)
    out_dim_j_length = c_int(0)
    out_k_indices = c_int_p()
    out_k_offsets = c_int_p()
    out_j_indices = c_int_p()
    out_j_offsets = c_int_p()
    out_i_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_tcsf_add(
        c_int(ni), c_int(nj), c_int(nk),
        c_int(len(a_ii)), _int_array(a_ii),
        _int_array(a_jo), c_int(len(a_ji)), _int_array(a_ji),
        _int_array(a_ko), c_int(len(a_ki)), _int_array(a_ki),
        _float_array(a_val), c_int(len(a_ki)),
        c_int(ni), c_int(nj), c_int(nk),
        c_int(len(b_ii)), _int_array(b_ii),
        _int_array(b_jo), c_int(len(b_ji)), _int_array(b_ji),
        _int_array(b_ko), c_int(len(b_ki)), _int_array(b_ki),
        _float_array(b_val), c_int(len(b_ki)),
        c_int(ni), c_int(nj), c_int(nk),
        ctypes.byref(out_nnz), ctypes.byref(out_dim_i_length),
        ctypes.byref(out_dim_j_length),
        ctypes.byref(out_k_indices), ctypes.byref(out_k_offsets),
        ctypes.byref(out_j_indices), ctypes.byref(out_j_offsets),
        ctypes.byref(out_i_indices), ctypes.byref(out_values),
    )
    nnz = out_nnz.value
    if nnz == 0:
        return [], [0], [], [0], [], []
    n_i = out_dim_i_length.value
    n_j = out_dim_j_length.value
    r_ii = _copy_int_buf(out_i_indices, n_i)
    r_jo = _copy_int_buf(out_j_offsets, n_i + 1)
    r_ji = _copy_int_buf(out_j_indices, n_j)
    r_ko = _copy_int_buf(out_k_offsets, n_j + 1)
    r_ki = _copy_int_buf(out_k_indices, nnz)
    r_val = _copy_float_buf(out_values, nnz)
    _free(out_i_indices)
    _free(out_j_offsets)
    _free(out_j_indices)
    _free(out_k_offsets)
    _free(out_k_indices)
    _free(out_values)
    return r_ii, r_jo, r_ji, r_ko, r_ki, r_val


def call_coo_add(a_ri, a_ci, a_val, nrows, ncols,
                 b_ri, b_ci, b_val):
    out_nnz = c_int(0)
    out_dim_i_length = c_int(0)
    out_j_indices = c_int_p()
    out_i_indices = c_int_p()
    out_values = c_float_p()
    _lib.cpu_coo_add(
        c_int(nrows), c_int(ncols),
        c_int(len(a_ri)), _int_array(a_ri),
        c_int(len(a_ci)), _int_array(a_ci),
        _float_array(a_val), c_int(len(a_val)),
        c_int(nrows), c_int(ncols),
        c_int(len(b_ri)), _int_array(b_ri),
        c_int(len(b_ci)), _int_array(b_ci),
        _float_array(b_val), c_int(len(b_val)),
        c_int(nrows), c_int(ncols),
        ctypes.byref(out_nnz), ctypes.byref(out_dim_i_length),
        ctypes.byref(out_j_indices), ctypes.byref(out_i_indices),
        ctypes.byref(out_values),
    )
    nnz = out_nnz.value
    if nnz == 0:
        return [], [], []
    n_i = out_dim_i_length.value
    r_ri = _copy_int_buf(out_i_indices, n_i)
    r_ci = _copy_int_buf(out_j_indices, nnz)
    r_val = _copy_float_buf(out_values, nnz)
    _free(out_i_indices)
    _free(out_j_indices)
    _free(out_values)
    return r_ri, r_ci, r_val


# ===========================================================================
# Test classes
# ===========================================================================


class TestSparseVecMul:
    def test_basic_intersection(self):
        r_idx, r_val = call_sparse_vec_mul(
            [1, 3, 4], [2.0, 4.0, 5.0], 5,
            [0, 2, 3], [1.0, 3.0, 4.0],
        )
        assert r_idx == [3]
        assert pytest.approx(r_val, rel=1e-5) == [16.0]

    def test_full_overlap(self):
        r_idx, r_val = call_sparse_vec_mul(
            [0, 1, 2], [2.0, 3.0, 4.0], 3,
            [0, 1, 2], [5.0, 6.0, 7.0],
        )
        assert r_idx == [0, 1, 2]
        assert pytest.approx(r_val, rel=1e-5) == [10.0, 18.0, 28.0]

    def test_no_overlap(self):
        r_idx, r_val = call_sparse_vec_mul(
            [0, 1, 2], [1.0, 2.0, 3.0], 10,
            [3, 4, 5], [1.0, 2.0, 3.0],
        )
        assert r_idx == []

    def test_single_element(self):
        r_idx, r_val = call_sparse_vec_mul(
            [5], [3.0], 10,
            [5], [7.0],
        )
        assert r_idx == [5]
        assert pytest.approx(r_val, rel=1e-5) == [21.0]

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed * 1000 + 1)
        r_idx, r_val = call_sparse_vec_mul(a_idx, a_val, size, b_idx, b_val)
        got = cvector_to_dense(r_idx, r_val, size)
        a_d = cvector_to_dense(a_idx, a_val, size)
        b_d = cvector_to_dense(b_idx, b_val, size)
        expected = a_d * b_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


class TestSparseVecAdd:
    def test_basic_union(self):
        r_idx, r_val = call_sparse_vec_add(
            [1, 3], [2.0, 4.0], 5,
            [0, 2, 3], [1.0, 3.0, 4.0],
        )
        assert r_idx == [0, 1, 2, 3]
        assert pytest.approx(r_val, rel=1e-5) == [1.0, 2.0, 3.0, 8.0]

    def test_no_overlap(self):
        r_idx, r_val = call_sparse_vec_add(
            [0, 1], [1.0, 2.0], 10,
            [3, 4], [3.0, 4.0],
        )
        assert r_idx == [0, 1, 3, 4]
        assert pytest.approx(r_val, rel=1e-5) == [1.0, 2.0, 3.0, 4.0]

    def test_full_overlap(self):
        r_idx, r_val = call_sparse_vec_add(
            [0, 1, 2], [1.0, 2.0, 3.0], 3,
            [0, 1, 2], [4.0, 5.0, 6.0],
        )
        assert r_idx == [0, 1, 2]
        assert pytest.approx(r_val, rel=1e-5) == [5.0, 7.0, 9.0]

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed * 1000 + 1)
        r_idx, r_val = call_sparse_vec_add(a_idx, a_val, size, b_idx, b_val)
        got = cvector_to_dense(r_idx, r_val, size)
        a_d = cvector_to_dense(a_idx, a_val, size)
        b_d = cvector_to_dense(b_idx, b_val, size)
        expected = a_d + b_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


class TestSparseVecApbC:
    def test_basic(self):
        r_idx, r_val = call_sparse_vec_apb_c(
            [1, 3], [2.0, 4.0], 5,
            [0, 2, 3], [1.0, 3.0, 4.0],
            [2, 3], [5.0, 6.0],
        )
        assert r_idx == [2, 3]
        assert pytest.approx(r_val, rel=1e-5) == [15.0, 48.0]

    def test_no_overlap_with_c(self):
        r_idx, r_val = call_sparse_vec_apb_c(
            [0, 1], [2.0, 3.0], 10,
            [1, 2], [4.0, 5.0], [5, 6], [1.0, 1.0],
        )
        assert r_idx == []

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        nnz_c = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed * 1000 + 1)
        c_idx, c_val = random_sparse_vec(size, nnz_c, seed * 1000 + 2)
        r_idx, r_val = call_sparse_vec_apb_c(
            a_idx, a_val, size, b_idx, b_val, c_idx, c_val,
        )
        got = cvector_to_dense(r_idx, r_val, size)
        a_d = cvector_to_dense(a_idx, a_val, size)
        b_d = cvector_to_dense(b_idx, b_val, size)
        c_d = cvector_to_dense(c_idx, c_val, size)
        expected = (a_d + b_d) * c_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


class TestSparseVecAbPc:
    def test_basic(self):
        r_idx, r_val = call_sparse_vec_ab_pc(
            [1, 3], [2.0, 4.0], 5,
            [0, 2, 3], [1.0, 3.0, 4.0],
            [1, 4], [10.0, 7.0],
        )
        assert r_idx == [1, 3, 4]
        assert pytest.approx(r_val, rel=1e-5) == [10.0, 16.0, 7.0]

    def test_ab_empty(self):
        r_idx, r_val = call_sparse_vec_ab_pc(
            [0, 1], [1.0, 2.0], 10,
            [5, 6], [1.0, 2.0],
            [3], [42.0],
        )
        assert r_idx == [3]
        assert pytest.approx(r_val, rel=1e-5) == [42.0]

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        size = rng.randint(100, 5000)
        nnz_a = rng.randint(10, min(size, 200))
        nnz_b = rng.randint(10, min(size, 200))
        nnz_c = rng.randint(10, min(size, 200))
        a_idx, a_val = random_sparse_vec(size, nnz_a, seed * 1000)
        b_idx, b_val = random_sparse_vec(size, nnz_b, seed * 1000 + 1)
        c_idx, c_val = random_sparse_vec(size, nnz_c, seed * 1000 + 2)
        r_idx, r_val = call_sparse_vec_ab_pc(
            a_idx, a_val, size, b_idx, b_val, c_idx, c_val,
        )
        got = cvector_to_dense(r_idx, r_val, size)
        a_d = cvector_to_dense(a_idx, a_val, size)
        b_d = cvector_to_dense(b_idx, b_val, size)
        c_d = cvector_to_dense(c_idx, c_val, size)
        expected = (a_d * b_d) + c_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


class TestCsrAdd:
    def test_basic(self):
        r_off, r_idx, r_val = call_csr_add(
            [0, 2, 4, 5], [1, 3, 0, 2, 1], [1.0, 2.0, 3.0, 4.0, 5.0], 3, 4,
            [0, 1, 3, 5], [1, 0, 1, 1, 3], [10.0, 20.0, 30.0, 40.0, 50.0],
        )
        got = csr_to_dense(r_off, r_idx, r_val, 3, 4)
        expected = np.array([
            [0, 11, 0, 2],
            [23, 30, 4, 0],
            [0, 45, 0, 50],
        ], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_no_overlap(self):
        r_off, r_idx, r_val = call_csr_add(
            [0, 1, 1], [0], [1.0], 2, 3,
            [0, 0, 1], [2], [2.0],
        )
        got = csr_to_dense(r_off, r_idx, r_val, 2, 3)
        expected = np.array([[1, 0, 0], [0, 0, 2]], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        nrows = rng.randint(5, 100)
        ncols = rng.randint(5, 100)
        max_nnz = max(0, min(5, ncols))
        a_off, a_idx, a_val = random_csr(nrows, ncols, (0, max_nnz), seed * 1000)
        b_off, b_idx, b_val = random_csr(nrows, ncols, (0, max_nnz), seed * 1000 + 1)
        r_off, r_idx, r_val = call_csr_add(
            a_off, a_idx, a_val, nrows, ncols,
            b_off, b_idx, b_val,
        )
        got = csr_to_dense(r_off, r_idx, r_val, nrows, ncols)
        a_d = csr_to_dense(a_off, a_idx, a_val, nrows, ncols)
        b_d = csr_to_dense(b_off, b_idx, b_val, nrows, ncols)
        expected = a_d + b_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


_xfail_codegen = pytest.mark.xfail(
    reason="Generated CPU code has merge-path thread boundary bug for "
           "multi-dim sparse operations (dcsr_mul/add, tcsf_add)",
    strict=False,
)


class TestDcsrMul:
    def test_basic_intersection(self):
        r_ri, r_ro, r_ci, r_val = call_dcsr_mul(
            [0, 2], [0, 2, 3], [0, 1, 1], [2.0, 3.0, 4.0], 3, 3,
            [0, 1], [0, 1, 2], [1, 0], [5.0, 6.0],
        )
        got = dcsr_to_dense(r_ri, r_ro, r_ci, r_val, 3, 3)
        expected = np.zeros((3, 3), dtype=np.float32)
        expected[0, 1] = 15.0  # 3.0 * 5.0
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_no_overlap(self):
        r_ri, r_ro, r_ci, r_val = call_dcsr_mul(
            [0], [0, 1], [0], [1.0], 5, 5,
            [1], [0, 1], [1], [2.0],
        )
        assert r_ci == []

    def test_full_overlap(self):
        r_ri, r_ro, r_ci, r_val = call_dcsr_mul(
            [0, 1], [0, 2, 4], [0, 1, 0, 1], [1.0, 2.0, 3.0, 4.0], 2, 2,
            [0, 1], [0, 2, 4], [0, 1, 0, 1], [5.0, 6.0, 7.0, 8.0],
        )
        assert r_ri == [0, 1]
        assert r_ci == [0, 1, 0, 1]
        assert pytest.approx(r_val, rel=1e-5) == [5.0, 12.0, 21.0, 32.0]

    @_xfail_codegen
    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        nrows = rng.randint(10, 100)
        ncols = rng.randint(10, 100)
        density = rng.uniform(0.05, 0.3)
        a_ri, a_ro, a_ci, a_val = random_dcsr(nrows, ncols, density, seed * 1000)
        b_ri, b_ro, b_ci, b_val = random_dcsr(nrows, ncols, density, seed * 1000 + 1)
        if not a_ri or not b_ri:
            return
        r_ri, r_ro, r_ci, r_val = call_dcsr_mul(
            a_ri, a_ro, a_ci, a_val, nrows, ncols,
            b_ri, b_ro, b_ci, b_val,
        )
        got = dcsr_to_dense(r_ri, r_ro, r_ci, r_val, nrows, ncols)
        a_d = dcsr_to_dense(a_ri, a_ro, a_ci, a_val, nrows, ncols)
        b_d = dcsr_to_dense(b_ri, b_ro, b_ci, b_val, nrows, ncols)
        expected = a_d * b_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


class TestDcsrAdd:
    def test_basic_union(self):
        r_ri, r_ro, r_ci, r_val = call_dcsr_add(
            [0], [0, 1], [0], [1.0], 3, 3,
            [0], [0, 1], [1], [2.0],
        )
        got = dcsr_to_dense(r_ri, r_ro, r_ci, r_val, 3, 3)
        expected = np.zeros((3, 3), dtype=np.float32)
        expected[0, 0] = 1.0
        expected[0, 1] = 2.0
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_overlapping_entries(self):
        r_ri, r_ro, r_ci, r_val = call_dcsr_add(
            [0, 1], [0, 2, 3], [0, 1, 0], [1.0, 2.0, 3.0], 2, 2,
            [0, 1], [0, 2, 3], [0, 1, 0], [4.0, 5.0, 6.0],
        )
        got = dcsr_to_dense(r_ri, r_ro, r_ci, r_val, 2, 2)
        expected = np.array([[5.0, 7.0], [9.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_no_overlap_rows(self):
        r_ri, r_ro, r_ci, r_val = call_dcsr_add(
            [0], [0, 1], [0], [1.0], 5, 5,
            [2], [0, 1], [1], [2.0],
        )
        got = dcsr_to_dense(r_ri, r_ro, r_ci, r_val, 5, 5)
        expected = np.zeros((5, 5), dtype=np.float32)
        expected[0, 0] = 1.0
        expected[2, 1] = 2.0
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        nrows = rng.randint(10, 100)
        ncols = rng.randint(10, 100)
        density = rng.uniform(0.05, 0.3)
        a_ri, a_ro, a_ci, a_val = random_dcsr(nrows, ncols, density, seed * 1000)
        b_ri, b_ro, b_ci, b_val = random_dcsr(nrows, ncols, density, seed * 1000 + 1)
        if not a_ri or not b_ri:
            return
        r_ri, r_ro, r_ci, r_val = call_dcsr_add(
            a_ri, a_ro, a_ci, a_val, nrows, ncols,
            b_ri, b_ro, b_ci, b_val,
        )
        got = dcsr_to_dense(r_ri, r_ro, r_ci, r_val, nrows, ncols)
        a_d = dcsr_to_dense(a_ri, a_ro, a_ci, a_val, nrows, ncols)
        b_d = dcsr_to_dense(b_ri, b_ro, b_ci, b_val, nrows, ncols)
        expected = a_d + b_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


class TestTcsfAdd:
    def test_basic(self):
        # A: tensor[0,0,0]=1.0, tensor[0,0,1]=2.0
        # B: tensor[0,0,1]=3.0, tensor[0,1,0]=4.0
        r = call_tcsf_add(
            [0], [0, 1], [0], [0, 2], [0, 1], [1.0, 2.0], (2, 2, 2),
            [0], [0, 2], [0, 1], [0, 1, 2], [1, 0], [3.0, 4.0],
        )
        r_ii, r_jo, r_ji, r_ko, r_ki, r_val = r
        got = tcsf_to_dense(r_ii, r_jo, r_ji, r_ko, r_ki, r_val, (2, 2, 2))
        expected = np.zeros((2, 2, 2), dtype=np.float32)
        expected[0, 0, 0] = 1.0
        expected[0, 0, 1] = 5.0  # 2.0 + 3.0
        expected[0, 1, 0] = 4.0
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_no_overlap(self):
        r = call_tcsf_add(
            [0], [0, 1], [0], [0, 1], [0], [1.0], (3, 3, 3),
            [1], [0, 1], [1], [0, 1], [2], [2.0],
        )
        r_ii, r_jo, r_ji, r_ko, r_ki, r_val = r
        got = tcsf_to_dense(r_ii, r_jo, r_ji, r_ko, r_ki, r_val, (3, 3, 3))
        expected = np.zeros((3, 3, 3), dtype=np.float32)
        expected[0, 0, 0] = 1.0
        expected[1, 1, 2] = 2.0
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        ni = rng.randint(3, 15)
        nj = rng.randint(3, 15)
        nk = rng.randint(3, 15)
        shape = (ni, nj, nk)
        density = rng.uniform(0.1, 0.4)
        a_ii, a_jo, a_ji, a_ko, a_ki, a_val = random_tcsf(shape, density, seed * 1000)
        b_ii, b_jo, b_ji, b_ko, b_ki, b_val = random_tcsf(shape, density, seed * 1000 + 1)
        if not a_ii or not b_ii:
            return
        r = call_tcsf_add(
            a_ii, a_jo, a_ji, a_ko, a_ki, a_val, shape,
            b_ii, b_jo, b_ji, b_ko, b_ki, b_val,
        )
        r_ii, r_jo, r_ji, r_ko, r_ki, r_val = r
        got = tcsf_to_dense(r_ii, r_jo, r_ji, r_ko, r_ki, r_val, shape)
        a_d = tcsf_to_dense(a_ii, a_jo, a_ji, a_ko, a_ki, a_val, shape)
        b_d = tcsf_to_dense(b_ii, b_jo, b_ji, b_ko, b_ki, b_val, shape)
        expected = a_d + b_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


class TestCooAdd:
    def test_basic_union(self):
        # A: (0,0)=1, (0,1)=2, (1,0)=3
        # B: (0,1)=4, (1,0)=5, (1,1)=6
        r_ri, r_ci, r_val = call_coo_add(
            [0, 0, 1], [0, 1, 0], [1.0, 2.0, 3.0], 2, 2,
            [0, 1, 1], [1, 0, 1], [4.0, 5.0, 6.0],
        )
        got = coo_to_dense(r_ri, r_ci, r_val, 2, 2)
        expected = np.array([[1.0, 6.0], [8.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_no_overlap(self):
        r_ri, r_ci, r_val = call_coo_add(
            [0], [0], [1.0], 2, 2,
            [1], [1], [2.0],
        )
        got = coo_to_dense(r_ri, r_ci, r_val, 2, 2)
        expected = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_full_overlap(self):
        r_ri, r_ci, r_val = call_coo_add(
            [0, 1], [0, 1], [1.0, 2.0], 2, 2,
            [0, 1], [0, 1], [3.0, 4.0],
        )
        got = coo_to_dense(r_ri, r_ci, r_val, 2, 2)
        expected = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_correctness(self, seed):
        rng = random.Random(seed)
        nrows = rng.randint(3, 20)
        ncols = rng.randint(3, 20)
        density = rng.uniform(0.1, 0.4)
        a_ri, a_ci, a_val = random_coo(nrows, ncols, density, seed * 1000)
        b_ri, b_ci, b_val = random_coo(nrows, ncols, density, seed * 1000 + 1)
        if not a_ri or not b_ri:
            return
        r_ri, r_ci, r_val = call_coo_add(
            a_ri, a_ci, a_val, nrows, ncols,
            b_ri, b_ci, b_val,
        )
        got = coo_to_dense(r_ri, r_ci, r_val, nrows, ncols)
        a_d = coo_to_dense(a_ri, a_ci, a_val, nrows, ncols)
        b_d = coo_to_dense(b_ri, b_ci, b_val, nrows, ncols)
        expected = a_d + b_d
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
