// Load-balanced search (LBS) SpMV on CSC(A): one thread per nonzero. SpMV is
// not coiterative, so each nonzero's contribution is independent and the whole
// product is a single kernel. Each thread recovers the column that owns its
// nonzero by binary searching the column offsets, then scatters into y with an
// atomic add. Work per thread is constant regardless of column length skew.

#include "cuda_utils/cuda_utils.h"
#include "spmv/spmv_baselines.h"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;

// Largest j in [0, k) with colOffs[j] <= p. Empty columns share an offset with
// the next column, so taking the largest such j lands on the column whose range
// [colOffs[j], colOffs[j + 1]) actually contains p.
__device__ __forceinline__ int owning_column(const int *__restrict__ colOffs,
                                             int k, int64_t p) {
    int lo = 0, hi = k - 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        if (colOffs[mid] <= p) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

__global__ void spmv_lbs_kernel(int k, const int *__restrict__ colOffs,
                                const int *__restrict__ rowInds,
                                const float *__restrict__ vals, int64_t nnz,
                                const float *__restrict__ x,
                                float *__restrict__ y) {
    const int64_t p =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= nnz)
        return;

    const int j = owning_column(colOffs, k, p);
    const int i = rowInds[p];
    atomicAdd(&y[i], vals[p] * x[j]);
}

} // namespace

void gpu_spmv_lbs_f32(int m, int k, int *colOffsA, int *rowIndsA, float *ValsA,
                      int64_t nnzA, float *x, float *&y) {

    const cudaStream_t stream = 0;

    // y is accumulated into with atomics, so it must start at zero.
    CHECK_CUDA(cudaMallocAsync(&y, sizeof(float) * m, stream));
    CHECK_CUDA(cudaMemsetAsync(y, 0, sizeof(float) * m, stream));

    if (nnzA == 0)
        return;

    const int64_t blocks = (nnzA + kBlockSize - 1) / kBlockSize;
    spmv_lbs_kernel<<<blocks, kBlockSize, 0, stream>>>(k, colOffsA, rowIndsA,
                                                       ValsA, nnzA, x, y);
    CHECK_CUDA(cudaGetLastError());
}
