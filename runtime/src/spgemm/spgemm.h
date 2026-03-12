#pragma once
#include <cstdint>
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"




void spgemm_cusparse_impl(
            CSRMatrix<int, float> A, CSRMatrix<int, float> B,
            int * & C_row_offsets, int * & C_indices, float * & C_values, int * & C_nnz
        );

void spgemm_impl(CSRMatrix<int, float> A, CSRMatrix<int, float> B,
    int * & C_row_offsets, int * & C_indices, float * & C_values, int * & C_nnz, float * & D_times);



