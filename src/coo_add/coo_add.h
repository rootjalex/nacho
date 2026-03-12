#pragma once
#include <cstdint>


void gpu_manual_coo_add_f32(int* shape, 
        int* rowsA, int* colsA, float* ValsA, uint64_t nnzA, 
        int* rowsB, int* colsB, float* ValsB, uint64_t nnzB, 
        int* &rowsC, int* &colsC, float* &ValsC, int* &nnzC);


