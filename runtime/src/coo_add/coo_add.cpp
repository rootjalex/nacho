#include "coo_add.hpp"

COO<int32_t, float> nb_gpu_coo_add_f32(COO<int32_t, float> A, COO<int32_t, float> B) {

    int32_t * result_row, * result_col; 
    float * result_values;
    int * result_nnz;
    result_nnz = (int*)malloc(sizeof(int));
    
    
    gpu_manual_coo_add_f32(A.shape.data(), 
        A.row.data(), A.col.data(), A.data.data(), A.row.shape(0), 
        B.row.data(), B.col.data(), B.data.data(), B.row.shape(0),
        result_row, result_col, result_values, result_nnz);

    COO<int32_t, float> C = COO<int32_t, float>(result_row, result_col, result_values, A.shape(0), A.shape(1), *result_nnz);

    return C;    
}
