#include "nacho_ops.h"
#include "nacho_ops.hpp"
#include "../cuda_utils/cuda_utils.h"

// Helper: allocate minimal buffers for empty CVector results
static void alloc_empty_cvector(int *&indices, float *&values) {
    CHECK_CUDA(cudaMalloc((void **)&indices, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&values, sizeof(float)));
}

// Helper: allocate minimal buffers for empty DCSR results
static void alloc_empty_dcsr(int *&row_indices, int *&row_offsets,
                              int *&col_indices, float *&values) {
    CHECK_CUDA(cudaMalloc((void **)&row_indices, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&row_offsets, sizeof(int)));
    CHECK_CUDA(cudaMemset(row_offsets, 0, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&col_indices, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&values, sizeof(float)));
}

// ===== Sparse vector 2-operand: a * b =====

CVector<int, int, float> nacho_sparse_vec_mul_nb(CVector<int, int, float> A,
                                                  CVector<int, int, float> B) {
    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_mul<int, float>(
        A.size, (int)A.indices.shape(0), (int *)A.indices.data(),
        (float *)A.data.data(), (int)A.indices.shape(0),
        B.size, (int)B.indices.shape(0), (int *)B.indices.data(),
        (float *)B.data.data(), (int)B.indices.shape(0),
        A.size,
        out_nnz, out_indices, out_values);

    if (out_nnz == 0) {
        alloc_empty_cvector(out_indices, out_values);
    }

    return CVector<int, int, float>(out_indices, out_values, A.size, out_nnz);
}

// ===== Sparse vector 2-operand: a + b =====

CVector<int, int, float> nacho_sparse_vec_add_nb(CVector<int, int, float> A,
                                                  CVector<int, int, float> B) {
    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_add<int, float>(
        A.size, (int)A.indices.shape(0), (int *)A.indices.data(),
        (float *)A.data.data(), (int)A.indices.shape(0),
        B.size, (int)B.indices.shape(0), (int *)B.indices.data(),
        (float *)B.data.data(), (int)B.indices.shape(0),
        A.size,
        out_nnz, out_indices, out_values);

    if (out_nnz == 0) {
        alloc_empty_cvector(out_indices, out_values);
    }

    return CVector<int, int, float>(out_indices, out_values, A.size, out_nnz);
}

// ===== Sparse vector 3-operand: (a + b) * c =====

CVector<int, int, float> nacho_sparse_vec_apb_c_nb(CVector<int, int, float> A,
                                                     CVector<int, int, float> B,
                                                     CVector<int, int, float> C) {
    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_apb_c<int, float>(
        A.size, (int)A.indices.shape(0), (int *)A.indices.data(),
        (float *)A.data.data(), (int)A.indices.shape(0),
        B.size, (int)B.indices.shape(0), (int *)B.indices.data(),
        (float *)B.data.data(), (int)B.indices.shape(0),
        C.size, (int)C.indices.shape(0), (int *)C.indices.data(),
        (float *)C.data.data(), (int)C.indices.shape(0),
        A.size,
        out_nnz, out_indices, out_values);

    if (out_nnz == 0) {
        alloc_empty_cvector(out_indices, out_values);
    }

    return CVector<int, int, float>(out_indices, out_values, A.size, out_nnz);
}

// ===== Sparse vector 3-operand: (a * b) + c =====

CVector<int, int, float> nacho_sparse_vec_ab_pc_nb(CVector<int, int, float> A,
                                                     CVector<int, int, float> B,
                                                     CVector<int, int, float> C) {
    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_ab_pc<int, float>(
        A.size, (int)A.indices.shape(0), (int *)A.indices.data(),
        (float *)A.data.data(), (int)A.indices.shape(0),
        B.size, (int)B.indices.shape(0), (int *)B.indices.data(),
        (float *)B.data.data(), (int)B.indices.shape(0),
        C.size, (int)C.indices.shape(0), (int *)C.indices.data(),
        (float *)C.data.data(), (int)C.indices.shape(0),
        A.size,
        out_nnz, out_indices, out_values);

    if (out_nnz == 0) {
        alloc_empty_cvector(out_indices, out_values);
    }

    return CVector<int, int, float>(out_indices, out_values, A.size, out_nnz);
}

// ===== DCSR 2-operand: A * B =====

DCSR<int, float> nacho_dcsr_mul_nb(DCSR<int, float> A, DCSR<int, float> B) {
    int out_nnz;
    int out_dim_i_length;
    int *out_dim_i_indices;
    int *out_dim_j_offsets;
    int *out_dim_j_indices;
    float *out_values;

    dcsr_mul<int, float>(
        A.nrows, A.ncols,
        (int)A.row_indices.shape(0), (int *)A.row_indices.data(),
        (int *)A.row_offsets.data(), (int)A.col_indices.shape(0),
        (int *)A.col_indices.data(), (float *)A.data.data(),
        (int)A.col_indices.shape(0),
        B.nrows, B.ncols,
        (int)B.row_indices.shape(0), (int *)B.row_indices.data(),
        (int *)B.row_offsets.data(), (int)B.col_indices.shape(0),
        (int *)B.col_indices.data(), (float *)B.data.data(),
        (int)B.col_indices.shape(0),
        A.nrows, A.ncols,
        out_nnz, out_dim_i_length,
        out_dim_j_indices, out_dim_j_offsets,
        out_dim_i_indices, out_values);

    if (out_nnz == 0) {
        alloc_empty_dcsr(out_dim_i_indices, out_dim_j_offsets,
                         out_dim_j_indices, out_values);
        out_dim_i_length = 0;
    }

    return DCSR<int, float>(out_dim_i_indices, out_dim_j_offsets,
                             out_dim_j_indices, out_values,
                             A.nrows, A.ncols,
                             out_dim_i_length, out_nnz);
}

// ===== DCSR 2-operand: A + B =====

DCSR<int, float> nacho_dcsr_add_nb(DCSR<int, float> A, DCSR<int, float> B) {
    int out_nnz;
    int out_dim_i_length;
    int *out_dim_i_indices;
    int *out_dim_j_offsets;
    int *out_dim_j_indices;
    float *out_values;

    dcsr_add<int, float>(
        A.nrows, A.ncols,
        (int)A.row_indices.shape(0), (int *)A.row_indices.data(),
        (int *)A.row_offsets.data(), (int)A.col_indices.shape(0),
        (int *)A.col_indices.data(), (float *)A.data.data(),
        (int)A.col_indices.shape(0),
        B.nrows, B.ncols,
        (int)B.row_indices.shape(0), (int *)B.row_indices.data(),
        (int *)B.row_offsets.data(), (int)B.col_indices.shape(0),
        (int *)B.col_indices.data(), (float *)B.data.data(),
        (int)B.col_indices.shape(0),
        A.nrows, A.ncols,
        out_nnz, out_dim_i_length,
        out_dim_j_indices, out_dim_j_offsets,
        out_dim_i_indices, out_values);

    if (out_nnz == 0) {
        alloc_empty_dcsr(out_dim_i_indices, out_dim_j_offsets,
                         out_dim_j_indices, out_values);
        out_dim_i_length = 0;
    }

    return DCSR<int, float>(out_dim_i_indices, out_dim_j_offsets,
                             out_dim_j_indices, out_values,
                             A.nrows, A.ncols,
                             out_dim_i_length, out_nnz);
}
