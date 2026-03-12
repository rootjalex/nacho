#pragma once
#include <cstdint>

void gpu_cusparse_csr_add_f32(int* shape, 
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA, 
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB, 
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* &nnzC);

void gpu_manual_csr_add_f32(int* shape, 
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA, 
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB, 
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* &nnzC);


void print_cuda(int * & ptr, int size);