// cuSPARSE CSR addition, the vendor-library comparison point.

#include "csr_add/csr_add_baselines.h"
#include "cuda_utils/cuda_utils.h"

#include <cuda_runtime.h>
#include <cusparse.h>

static cusparseHandle_t handle = nullptr;

void gpu_csr_add_cusparse_f32(int* shape,
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA, 
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB, 
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* nnzC) {
    
    const cudaStream_t stream = 0;

    int m = shape[0];
    int n = shape[1];
    //printf("nnzA %d nnzB %d\n", nnzA, nnzB);
    
    if(handle == nullptr) {
        cusparseCreate(&handle);
        cusparseSetStream(handle, stream);
    }

    // Create matrix descriptors
    cusparseMatDescr_t descrA, descrB, descrC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));

    // Set matrix types (general sparse matrix with 0-based indexing)
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));

    // Get buffer size for the operation
    size_t bufferSize;
      // C = alpha*A + beta*B

    float alpha_h = 1.0f;     // host value
    

    float beta_h = 1.0f;     // host value
    
    CHECK_CUDA(cudaMallocAsync((void**)&rowOffsC, sizeof(int)*(m+1), stream));

    //printf("Calculating buffer size for csrgeam2\n");
    
    cusparseScsrgeam2_bufferSizeExt(handle, m, n,
        &alpha_h,
        descrA, nnzA, ValsA, rowOffsA, colIndsA,
        &beta_h,
        descrB, nnzB, ValsB, rowOffsB, colIndsB,
        descrC,
        ValsC, rowOffsC, colIndsC,
        &bufferSize);

    
    // Allocate workspace buffer
    void* buffer = nullptr;
    CHECK_CUDA(cudaMallocAsync(&buffer, sizeof(char)*bufferSize, stream));
    // Get number of non-zero elements in result matrix

    //printf("nnzA %ld nnZB %ld buffer size = %ld\n", nnzA, nnzB, sizeof(char)*bufferSize);
    
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, m, n,
        descrA, nnzA, rowOffsA, colIndsA,
        descrB, nnzB, rowOffsB, colIndsB,
        descrC, rowOffsC, nnzC,
        buffer));
    
    //printf("nnzC %d\n", *nnzC);
    
    // CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA(cudaMallocAsync((void**)&colIndsC, sizeof(int)*(*nnzC), stream));
    CHECK_CUDA(cudaMallocAsync((void**)&ValsC, sizeof(float)*(*nnzC), stream));
    // Perform the actual matrix addition C = alpha*A + beta*B
    CHECK_CUSPARSE(cusparseScsrgeam2(handle, m, n,
        &alpha_h,
        descrA, nnzA, ValsA, rowOffsA, colIndsA,
        &beta_h,
        descrB, nnzB, ValsB, rowOffsB, colIndsB,
        descrC,
        ValsC, rowOffsC, colIndsC,
        buffer));

    //printf("Finished csrgeam2\n");
    // Clean up
    CHECK_CUDA(cudaFreeAsync(buffer, stream));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrB));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
   //CHECK_CUSPARSE(cusparseDestroy(handle));
    // CHECK_CUDA( cudaGetLastError() );
}



