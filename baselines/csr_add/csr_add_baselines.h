#pragma once

// Comparison points for the generated csr_add kernel. Both take and return raw pointers;
// the caller owns the operands and takes ownership of the results.
//
// GPU results are allocated with cudaMallocAsync, CPU results with malloc, matching what
// the generated kernels do so Python can adopt them with the same deallocators.

#include <cstdint>

// One thread per row, no merge-path partitioning of work between threads.
void gpu_csr_add_taco_f32(int* shape,
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA,
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB,
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* nnzC);

void gpu_csr_add_cusparse_f32(int* shape,
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA,
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB,
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* nnzC);
