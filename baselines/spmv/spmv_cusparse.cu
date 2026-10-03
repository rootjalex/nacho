// cuSPARSE sparse matrix vector product, the vendor-library comparison point.

#include "cuda_utils/cuda_utils.h"
#include "spmv/spmv_baselines.h"

#include <cuda_runtime.h>
#include <cusparse.h>

// connection to cuSPARSE library, created on first use and reused for all
// subsequent calls
static cusparseHandle_t handle = nullptr;

void gpu_spmv_cusparse_f32(int m, int k, int *colOffsA, int *rowIndsA,
                           float *ValsA, int64_t nnzA, float *x, float *&y) {

    const cudaStream_t stream = 0;
    // const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // initialize cuSPARSE handle if not already done
    if (handle == nullptr) {
        cusparseCreate(&handle);
        cusparseSetStream(handle, stream);
    }

    const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cudaDataType compute_type = CUDA_R_32F;
    const float alpha = 1.0f, beta = 0.0f;

    // describe CSC A
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsc(
        &matA,              // Output: cuSPARSE descriptor for sparse matrix A
        m,                  // Number of rows in A
        k,                  // Number of columns in A
        nnzA,               // Number of nonzero elements in A
        (void *)colOffsA,   // CSC column offsets, length k + 1
        (void *)rowIndsA,   // Row index of each nonzero, length nnzA
        (void *)ValsA,      // Value of each nonzero, length nnzA
        CUSPARSE_INDEX_32I, // Data type of column offsets: 32-bit integer
        CUSPARSE_INDEX_32I, // Data type of row indices: 32-bit integer
        CUSPARSE_INDEX_BASE_ZERO, // Indices are zero-based (0, 1, 2, ...)
        compute_type              // Matrix values are 32-bit floats
        ));

    // describe dense x
    cusparseDnVecDescr_t vecX;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX,       // descriptor to create
                                       k,           // number of elements in x
                                       (void *)x,   // GPU pointer to x's data
                                       compute_type // x contains 32-bit floats
                                       ));

    // describe dense y
    CHECK_CUDA(cudaMallocAsync(&y, sizeof(float) * m, stream));
    cusparseDnVecDescr_t vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY,       // descriptor to create
                                       m,           // number of elements in y
                                       (void *)y,   // GPU pointer to y's data
                                       compute_type // y contains 32-bit floats
                                       ));

    // ask for scratch space size
    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle,                    // cuSPARSE handle
        opA,                       // operation on A (NON_TRANSPOSE for CSC(A))
        &alpha,                    // scalar multiplier for A*x
        matA,                      // sparse matrix A descriptor
        vecX,                      // dense input vector x descriptor
        &beta,                     // scalar multiplier for existing y
        vecY,                      // dense output vector y descriptor
        compute_type,              // computation performed in FP32
        CUSPARSE_SPMV_ALG_DEFAULT, // let cuSPARSE choose/default SpMV algorithm
        &buffer_size               // output: required temporary memory in bytes
        ));

    // allocate scratch space
    void *buffer = nullptr;
    if (buffer_size > 0) {
        CHECK_CUDA(cudaMallocAsync(
            &buffer,     // where to store the allocated GPU pointer
            buffer_size, // number of bytes requested by cuSPARSE
            stream       // allocation ordered on this CUDA stream
            ));
    }

    // perform the sparse matrix-vector product
    CHECK_CUSPARSE(cusparseSpMV(
        handle,       // cuSPARSE library handle
        opA,          // Operation on A (e.g. NON_TRANSPOSE => use A as-is)
        &alpha,       // Scalar alpha in y = alpha * A * x + beta * y
        matA,         // Sparse matrix A descriptor (CSC in our case)
        vecX,         // Dense input vector x descriptor
        &beta,        // Scalar beta in y = alpha * A * x + beta * y
        vecY,         // Dense output vector y descriptor
        compute_type, // Compute using 32-bit floating-point arithmetic
        CUSPARSE_SPMV_ALG_DEFAULT, // Use cuSPARSE's default SpMV algorithm
        buffer                     // Temporary GPU workspace allocated earlier
        ));

    // free scratch space
    if (buffer != nullptr) {
        CHECK_CUDA(cudaFreeAsync(buffer, stream));
    }

    // destroy cuSPARSE descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
}
