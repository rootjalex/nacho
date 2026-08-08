"""Correctness check for the generated kernels against scipy on small random inputs.

Run this after rebuilding the extension and before any timing work:

    python benchmarks/smoke.py

Kernels absent from the build are reported as skipped rather than failing, so this works
whatever subset `./build/compiler --kernels ...` last emitted.
"""

import sys

import numpy as np
import scipy.sparse as sp
import torch

import nacho

import config
import parser

RNG = np.random.default_rng(0)
ROWS, COLS, DENSITY = 200, 180, 0.05
SHAPE_3D, NNZ_3D = (40, 30, 25), 900
TOLERANCE = 1e-4


def random_csr():
    matrix = sp.random(ROWS, COLS, density=DENSITY, format="csr", dtype=np.float32,
                       random_state=RNG.integers(2**31))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def random_coo_3d():
    """Sorted, deduplicated (coordinates, values) for a 3D tensor."""
    coordinates = np.unique(
        np.stack([RNG.integers(0, extent, NNZ_3D) for extent in SHAPE_3D], axis=1), axis=0)
    return coordinates, RNG.random(coordinates.shape[0]).astype(np.float32)


def as_torch_csr(matrix):
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int32)),
        torch.from_numpy(matrix.indices.astype(np.int32)),
        torch.from_numpy(matrix.data.astype(np.float32)),
        size=matrix.shape,
    )


def buffer(array):
    """A kernel result buffer as a numpy array, whichever device it was produced on."""
    if array.is_cuda:
        torch.cuda.synchronize()
    return array.cpu().numpy()


def from_csr_result(result):
    """nacho CSR_cpu / CSR_gpu -> scipy CSR."""
    return sp.csr_matrix(
        (buffer(result.values), buffer(result.dim_j_indices), buffer(result.dim_j_offsets)),
        shape=(int(result.shape[0]), int(result.shape[1])),
    )


def from_dcsr_result(result, shape):
    """nacho DCSR_cpu / DCSR_gpu -> scipy CSR, re-expanding the compressed row dimension."""
    row_ids = buffer(result.dim_i_indices)
    offsets = buffer(result.dim_j_offsets)
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    indptr[row_ids + 1] = offsets[1:] - offsets[:-1]
    return sp.csr_matrix(
        (buffer(result.values), buffer(result.dim_j_indices), np.cumsum(indptr)),
        shape=shape,
    )


def from_coo_result(result):
    """nacho COO_cpu / COO_gpu -> scipy CSR."""
    return sp.coo_matrix(
        (buffer(result.values), (buffer(result.dim_i_indices), buffer(result.dim_j_indices))),
        shape=(int(result.shape[0]), int(result.shape[1])),
    ).tocsr()


def from_csf3_result(result):
    """nacho CSF3_cpu / CSF3_gpu -> {(i, j, k): value}, since scipy has no 3D sparse type."""
    dim_i_indices = buffer(result.dim_i_indices)
    dim_j_offsets, dim_j_indices = buffer(result.dim_j_offsets), buffer(result.dim_j_indices)
    dim_k_offsets, dim_k_indices = buffer(result.dim_k_offsets), buffer(result.dim_k_indices)
    values = buffer(result.values)

    entries = {}
    for i_p, i in enumerate(dim_i_indices):
        for j_p in range(dim_j_offsets[i_p], dim_j_offsets[i_p + 1]):
            for k_p in range(dim_k_offsets[j_p], dim_k_offsets[j_p + 1]):
                entries[(int(i), int(dim_j_indices[j_p]), int(dim_k_indices[k_p]))] = \
                    float(values[k_p])
    return entries


def matches(actual, expected):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return False
        return all(abs(actual[key] - expected[key]) < TOLERANCE for key in expected)
    actual = actual.tocsr()
    actual.sum_duplicates()
    actual.sort_indices()
    expected = expected.tocsr()
    expected.eliminate_zeros()
    expected.sum_duplicates()
    expected.sort_indices()
    difference = abs(actual - expected)
    return difference.nnz == 0 or difference.max() < TOLERANCE


# Each case takes the device its kernel runs on and returns (actual, expected).
def case_csr_add(device):
    a, b = random_csr(), random_csr()
    result = kernel("csr_add", device)(
        parser.to_csr(as_torch_csr(a), device), parser.to_csr(as_torch_csr(b), device))
    return from_csr_result(result), a + b


def case_csr_add_3(device):
    a, b, c = random_csr(), random_csr(), random_csr()
    result = kernel("csr_add_3", device)(
        parser.to_csr(as_torch_csr(a), device), parser.to_csr(as_torch_csr(b), device),
        parser.to_csr(as_torch_csr(c), device))
    return from_csr_result(result), a + b + c


def case_csr_mul(device):
    a, b = random_csr(), random_csr()
    result = kernel("csr_mul", device)(
        parser.to_csr(as_torch_csr(a), device), parser.to_csr(as_torch_csr(b), device))
    return from_csr_result(result), a.multiply(b)


def case_dcsr_add(device):
    a, b = random_csr(), random_csr()
    result = kernel("dcsr_add", device)(
        parser.to_dcsr(as_torch_csr(a), device), parser.to_dcsr(as_torch_csr(b), device))
    return from_dcsr_result(result, a.shape), a + b


def case_dcsr_mul(device):
    a, b = random_csr(), random_csr()
    result = kernel("dcsr_mul", device)(
        parser.to_dcsr(as_torch_csr(a), device), parser.to_dcsr(as_torch_csr(b), device))
    return from_dcsr_result(result, a.shape), a.multiply(b)


def case_coo_add(device):
    a, b = random_csr(), random_csr()
    to_coo = lambda m: parser.to_coo(as_torch_csr(m).to_sparse_coo(), device)
    result = kernel("coo_add", device)(to_coo(a), to_coo(b))
    return from_coo_result(result), a + b


def case_coo_mul(device):
    a, b = random_csr(), random_csr()
    to_coo = lambda m: parser.to_coo(as_torch_csr(m).to_sparse_coo(), device)
    result = kernel("coo_mul", device)(to_coo(a), to_coo(b))
    return from_coo_result(result), a.multiply(b)


def case_csf_add(device):
    a_coordinates, a_values = random_coo_3d()
    b_coordinates, b_values = random_coo_3d()
    result = kernel("csf_add", device)(
        parser.to_csf3(a_coordinates, a_values, SHAPE_3D, device),
        parser.to_csf3(b_coordinates, b_values, SHAPE_3D, device))

    expected = {tuple(int(x) for x in c): float(v) for c, v in zip(a_coordinates, a_values)}
    for coordinate, value in zip(b_coordinates, b_values):
        key = tuple(int(x) for x in coordinate)
        expected[key] = expected.get(key, 0.0) + float(value)
    return from_csf3_result(result), expected


CASES = [
    ("csr_add", case_csr_add),
    ("csr_add_3", case_csr_add_3),
    ("csr_mul", case_csr_mul),
    ("dcsr_add", case_dcsr_add),
    ("dcsr_mul", case_dcsr_mul),
    ("coo_add", case_coo_add),
    ("coo_mul", case_coo_mul),
    ("csf_add", case_csf_add),
]

DEVICES = ["cpu", "cuda"]


def entry_point(name, device):
    """Name of the generated wrapper for a kernel on a device."""
    return f"{'cpu' if device == 'cpu' else 'gpu'}_{name}_f32"


def kernel(name, device):
    return getattr(nacho, entry_point(name, device))


def main():
    failures = 0
    for device in DEVICES:
        for name, run in CASES:
            label = entry_point(name, device)
            if not hasattr(nacho, label):
                print(f"  SKIP  {label} (not in this build)")
                continue
            try:
                actual, expected = run(device)
            except Exception as error:  # noqa: BLE001 - report and keep going
                print(f"  ERROR {label}: {type(error).__name__}: {error}")
                failures += 1
                continue
            if matches(actual, expected):
                print(f"  ok    {label}")
            else:
                print(f"  FAIL  {label}: result differs from the reference")
                failures += 1

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
