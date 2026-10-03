#pragma once

// Comparison point for the generated spmv kernel. Takes and returns raw
// pointers; the caller owns the operands and takes ownership of the result,
// which is allocated with cudaMallocAsync so Python can adopt it with the same
// deallocator the generated kernels use.

#include <cstdint>

// y = A * x, with A being m x k and x being k x 1 (dense vector).
void gpu_spmv_cusparse_f32(int m, int k, int *colOffsA, int *rowIndsA,
                           float *ValsA, int64_t nnzA, float *x, float *&y);

// Load-balanced search: one thread per nonzero, binary search for the owning
// column, atomic add into y.
void gpu_spmv_lbs_f32(int m, int k, int *colOffsA, int *rowIndsA, float *ValsA,
                      int64_t nnzA, float *x, float *&y);
