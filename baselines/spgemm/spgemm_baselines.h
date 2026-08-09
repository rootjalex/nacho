#pragma once

// Comparison point for the generated spgemm kernel, and for the unfused route sssmm is
// measured against. Takes and returns raw pointers; the caller owns the operands and
// takes ownership of the result, which is allocated with cudaMallocAsync so Python can
// adopt it with the same deallocator the generated kernels use.

#include <cstdint>

// C = A * B, with A being m x k and B being k x n.
void gpu_spgemm_cusparse_f32(int m, int k, int n,
        int* rowOffsA, int* colIndsA, float* ValsA, int64_t nnzA,
        int* rowOffsB, int* colIndsB, float* ValsB, int64_t nnzB,
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* nnzC);
