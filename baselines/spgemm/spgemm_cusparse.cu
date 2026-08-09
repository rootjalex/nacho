// cuSPARSE sparse-sparse matrix product, the vendor-library comparison point.

#include "spgemm/spgemm_baselines.h"
#include "cuda_utils/cuda_utils.h"

#include <cuda_runtime.h>
#include <cusparse.h>

static cusparseHandle_t handle = nullptr;

void gpu_spgemm_cusparse_f32(int m, int k, int n,
        int* rowOffsA, int* colIndsA, float* ValsA, int64_t nnzA,
        int* rowOffsB, int* colIndsB, float* ValsB, int64_t nnzB,
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* nnzC) {

    const cudaStream_t stream = 0;

    if (handle == nullptr) {
        cusparseCreate(&handle);
        cusparseSetStream(handle, stream);
    }

    const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cudaDataType compute_type = CUDA_R_32F;
    const float alpha = 1.0f, beta = 0.0f;

    cusparseSpMatDescr_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, k, nnzA,
                                     (void*)rowOffsA, (void*)colIndsA, (void*)ValsA,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, k, n, nnzB,
                                     (void*)rowOffsB, (void*)colIndsB, (void*)ValsB,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // The row offsets are the only part of C whose size is known before the product is
    // symbolically computed; the rest is allocated once cuSPARSE reports the count.
    CHECK_CUDA(cudaMallocAsync(&rowOffsC, sizeof(int) * (m + 1), stream));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, m, n, 0,
                                     rowOffsC, nullptr, nullptr,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpGEMMDescr_t spgemm_desc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemm_desc));

    // Both phases are asked for their scratch size with a null buffer, then run with it.
    void *buffer1 = nullptr, *buffer2 = nullptr;
    size_t buffer1_bytes = 0, buffer2_bytes = 0;

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC,
                                                 compute_type, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemm_desc, &buffer1_bytes, nullptr));
    CHECK_CUDA(cudaMallocAsync(&buffer1, buffer1_bytes, stream));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC,
                                                 compute_type, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemm_desc, &buffer1_bytes, buffer1));

    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC,
                                          compute_type, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemm_desc, &buffer2_bytes, nullptr));
    CHECK_CUDA(cudaMallocAsync(&buffer2, buffer2_bytes, stream));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC,
                                          compute_type, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemm_desc, &buffer2_bytes, buffer2));

    int64_t rowsC = 0, colsC = 0, count = 0;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &rowsC, &colsC, &count));
    *nnzC = static_cast<int>(count);

    CHECK_CUDA(cudaMallocAsync(&colIndsC, sizeof(int) * (*nnzC), stream));
    CHECK_CUDA(cudaMallocAsync(&ValsC, sizeof(float) * (*nnzC), stream));
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, rowOffsC, colIndsC, ValsC));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC,
                                       compute_type, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc));

    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemm_desc));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    CHECK_CUDA(cudaFreeAsync(buffer1, stream));
    CHECK_CUDA(cudaFreeAsync(buffer2, stream));
}
