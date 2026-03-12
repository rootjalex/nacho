#include "nacho_ops.h"
#include "nacho_ops.hpp"
#include "../cuda_utils/cuda_utils.h"

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
        // Allocate minimal buffers for empty result to avoid null pointers
        CHECK_CUDA(cudaMalloc((void **)&out_indices, sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&out_values, sizeof(float)));
    }

    return CVector<int, int, float>(out_indices, out_values, A.size, out_nnz);
}
