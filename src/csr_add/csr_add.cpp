#include "csr_add.hpp"

CSR<int32_t, float> nb_gpu_csr_add_f32(CSR<int32_t, float> A, CSR<int32_t, float> B, bool useCusparse) {

    int32_t * result_row_offsets, * result_col_indices; 
    float * result_values;
    int * result_nnz;
    result_nnz = (int*)malloc(sizeof(int));
    
    if(useCusparse)
        gpu_cusparse_csr_add_f32(A.shape.data(), 
            A.indptr.data(), A.indices.data(), A.data.data(), A.indices.shape(0), 
            B.indptr.data(), B.indices.data(), B.data.data(), B.indices.shape(0),
            result_row_offsets, result_col_indices, result_values, result_nnz);
    else 
        gpu_manual_csr_add_f32(A.shape.data(), 
            A.indptr.data(), A.indices.data(), A.data.data(), A.indices.shape(0), 
            B.indptr.data(), B.indices.data(), B.data.data(), B.indices.shape(0),
            result_row_offsets, result_col_indices, result_values, result_nnz);

    
    CSR<int32_t, float> C = CSR<int32_t, float>(result_row_offsets, result_col_indices, result_values, A.shape(0), A.shape(1), *result_nnz);

    return C;    
}
