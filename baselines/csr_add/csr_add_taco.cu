// TACO-equivalent CSR addition on the GPU: one thread per row, no merge-path
// partitioning of the work between threads. This is the comparison point for what
// nacho's load balancing buys on skewed inputs.

#include "csr_add/csr_add_baselines.h"
#include "cuda_utils/cuda_utils.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace {

template<typename index_t>
__device__ __inline__ 
index_t binary_search_ub(const index_t* __restrict__ arr, const index_t target_value, index_t start_index, index_t end_index) {
  index_t mid = start_index + (((end_index - start_index) + 1) / 2);
  while (start_index < end_index) {
    mid = start_index + (((end_index - start_index) + 1) / 2);
    if (arr[mid] <= target_value) {
      start_index = mid;
    } else {
      end_index = mid - 1;
    }
  }
  mid = start_index + (((end_index - start_index) + 1) / 2);
  return mid;
}
template<typename index_t>
__device__ __inline__ 
index_t binary_search_lb(const index_t* __restrict__ arr, const index_t target_value, index_t start_index, index_t end_index) {
  index_t mid = start_index + (((end_index - start_index) + 1) / 2);
  while (start_index < end_index) {
    mid = start_index + (((end_index - start_index) + 1) / 2);
    if (arr[mid] < target_value) {
      start_index = mid;
    } else {
      if (target_value < arr[mid]) {
        end_index = mid - 1;
      } else {
        if (start_index == (mid - 1)) {
          return mid;
        }
        end_index = mid;
      }
    }
  }
  mid = start_index + (((end_index - start_index) + 1) / 2);
  return mid;
}
template<typename index_t, typename value_t>
struct a_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  index_t* dim_j_offsets;
  index_t dim_j_length;
  index_t* dim_j_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct b_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  index_t* dim_j_offsets;
  index_t dim_j_length;
  index_t* dim_j_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct Z_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  index_t* dim_j_offsets;
  index_t dim_j_length;
  index_t* dim_j_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t>
struct result_per_thread_count {
  index_t* dim_j_count;
};
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_a_dim_i(const a_tensor_format<index_t, value_t> a, const index_t i) {
  index_t end = a.dim_j_offsets[(0 + i) + 1];
  index_t start = a.dim_j_offsets[0 + 0];
  index_t count = end - start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_b_dim_i(const b_tensor_format<index_t, value_t> b, const index_t i) {
  index_t end = b.dim_j_offsets[(0 + i) + 1];
  index_t start = b.dim_j_offsets[0 + 0];
  index_t count = end - start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_a_dim_j(const a_tensor_format<index_t, value_t> a, const index_t i, const index_t j_p) {
  index_t end = j_p + 1;
  index_t start = a.dim_j_offsets[0 + i];
  index_t count = end - start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_b_dim_j(const b_tensor_format<index_t, value_t> b, const index_t i, const index_t j_p) {
  index_t end = j_p + 1;
  index_t start = b.dim_j_offsets[0 + i];
  index_t count = end - start;
  return count;
}
template<typename index_t>
struct partition_ij {
  index_t* i;
  index_t* a_j_p;
  index_t* b_j_p;
};



template<typename index_t, typename value_t>
__global__
void assembly(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, result_per_thread_count<index_t> count_offsets) {
  int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id >= a.dim_i_size) return;
  index_t jpA = a.dim_j_offsets[row_id];
  const index_t jpA_end = a.dim_j_offsets[row_id + 1];
  index_t jpB = b.dim_j_offsets[row_id];
  const index_t jpB_end = b.dim_j_offsets[row_id + 1];

  index_t jpZ = 0;

  while (jpA < jpA_end && jpB < jpB_end) {
    index_t jA = a.dim_j_indices[jpA];
    index_t jB = b.dim_j_indices[jpB];
    index_t j = min(jA, jB);
    if (j == jA && j == jB) {
      jpZ++;
    } else if (j == jA) {
      jpZ++;
    } else { // j == jB
      jpZ++;
    }

    jpA += (jA == j);
    jpB += (jB == j);
  }

  while (jpA < jpA_end) {
    jpZ++;
    jpA++;
  }

  while (jpB < jpB_end) {
    jpZ++;
    jpB++;
  }

  count_offsets.dim_j_count[row_id] = jpZ;
}

template<typename index_t, typename value_t>
__global__
void compute(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, const result_per_thread_count<index_t> count_offsets, Z_tensor_format<index_t, value_t> Z) {
  int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id >= a.dim_i_size) return;
  index_t jpA = a.dim_j_offsets[row_id];
  const index_t jpA_end = a.dim_j_offsets[row_id + 1];
  index_t jpB = b.dim_j_offsets[row_id];
  const index_t jpB_end = b.dim_j_offsets[row_id + 1];

  index_t jpZ = count_offsets.dim_j_count[row_id];

  if (row_id == 0) {
    Z.dim_j_offsets[0] = 0;
  }

  while (jpA < jpA_end && jpB < jpB_end) {
    index_t jA = a.dim_j_indices[jpA];
    index_t jB = b.dim_j_indices[jpB];
    index_t j = min(jA, jB);
    if (j == jA && j == jB) {
      Z.dim_j_indices[jpZ] = j;
      Z.values[jpZ] = a.values[jpA] + b.values[jpB];
      jpZ++;
    } else if (j == jA) {
      Z.dim_j_indices[jpZ] = j;
      Z.values[jpZ] = a.values[jpA];
      jpZ++;
    } else { // j == jB
      Z.dim_j_indices[jpZ] = j;
      Z.values[jpZ] = b.values[jpB];
      jpZ++;
    }

    jpA += (jA == j);
    jpB += (jB == j);
  }

  while (jpA < jpA_end) {
    Z.dim_j_indices[jpZ] = a.dim_j_indices[jpA];
    Z.values[jpZ] = a.values[jpA];
    jpZ++;
    jpA++;
  }

  while (jpB < jpB_end) {
    Z.dim_j_indices[jpZ] = b.dim_j_indices[jpB];
    Z.values[jpZ] = b.values[jpB];
    jpZ++;
    jpB++;
  }

  Z.dim_j_offsets[row_id + 1] = jpZ;
}


} // namespace

void gpu_csr_add_taco_f32(int* shape,
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA,
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB,
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* nnzC) {

    const cudaStream_t stream = 0;

    a_tensor_format<int,float> A;
    b_tensor_format<int,float> B;
    Z_tensor_format<int,float> C;

    A.dim_i_size = shape[0]; A.dim_j_size = shape[1];
    A.dim_j_offsets = rowOffsA; A.dim_j_indices = colIndsA;
    A.dim_j_length = nnzA; A.nnz = nnzA; A.values = ValsA;

    B.dim_i_size = shape[0]; B.dim_j_size = shape[1];
    B.dim_j_offsets = rowOffsB; B.dim_j_indices = colIndsB;
    B.dim_j_length = nnzB; B.nnz = nnzB; B.values = ValsB;

    const int n_rows = shape[0];
    const int blockSize = 256;
    const int numBlocks = (n_rows + blockSize - 1) / blockSize;

    // Allocate n_rows+1 counts: [0..n_rows-1] filled by assembly, [n_rows]=0.
    // After ExclusiveSum over n_rows+1 elements, d_count[n_rows] = total nnzC.
    result_per_thread_count<int> count_offsets;
    size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes,
        (int*)nullptr, (int*)nullptr, n_rows + 1, stream);

    void* temp_buffer = nullptr;
    size_t total_temp_bytes = sizeof(int) * (n_rows + 1) + cub_bytes;
    CHECK_CUDA(cudaMallocAsync(&temp_buffer, total_temp_bytes, stream));

    count_offsets.dim_j_count = (int*)temp_buffer;
    void* d_temp_storage = count_offsets.dim_j_count + (n_rows + 1);

    // Zero the sentinel element so ExclusiveSum[n_rows] = total nnzC.
    CHECK_CUDA(cudaMemsetAsync(count_offsets.dim_j_count + n_rows, 0, sizeof(int), stream));

    // Phase 1: count nnz per row.
    assembly<int,float><<<numBlocks, blockSize, 0, stream>>>(A, B, count_offsets);

    // Phase 2: exclusive prefix sum over n_rows+1 elements.
    // After scan: count[row] = start offset for row; count[n_rows] = total nnzC.
    cub::DeviceScan::ExclusiveSum(d_temp_storage, cub_bytes,
        count_offsets.dim_j_count, count_offsets.dim_j_count, n_rows + 1, stream);
    CHECK_CUDA(cudaMemcpyAsync(nnzC, count_offsets.dim_j_count + n_rows,
        sizeof(int), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaMallocAsync((void**)&rowOffsC, sizeof(int) * (shape[0] + 1), stream));
    CHECK_CUDA(cudaMemsetAsync(rowOffsC, 0, sizeof(int) * (shape[0] + 1), stream));
    CHECK_CUDA(cudaMallocAsync((void**)&colIndsC, sizeof(int) * (*nnzC), stream));
    CHECK_CUDA(cudaMallocAsync((void**)&ValsC, sizeof(float) * (*nnzC), stream));

    C.dim_i_size = shape[0]; C.dim_j_size = shape[1];
    C.dim_j_offsets = rowOffsC; C.dim_j_indices = colIndsC; C.values = ValsC;

    // Phase 3: write values; compute also fills rowOffsC[row+1].
    compute<int,float><<<numBlocks, blockSize, 0, stream>>>(A, B, count_offsets, C);

    cudaFreeAsync(temp_buffer, stream);
}

