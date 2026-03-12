#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace sparse_vec_apb_c_ns {

template<typename index_t>
__device__ __inline__ 
index_t binary_search(const index_t* __restrict__ arr, const index_t target_value, int32_t start_index, int32_t end_index) {
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
template<typename index_t, typename value_t>
struct a_tensor_format {
  index_t dim_i_size;
  index_t dim_i_length;
  index_t* dim_i_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct b_tensor_format {
  index_t dim_i_size;
  index_t dim_i_length;
  index_t* dim_i_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct c_tensor_format {
  index_t dim_i_size;
  index_t dim_i_length;
  index_t* dim_i_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct Z_tensor_format {
  index_t dim_i_size;
  index_t dim_i_length;
  index_t* dim_i_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t>
struct result_per_thread_count {
  index_t* dim_i_count;
};
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_i_a_dim_i(const a_tensor_format<index_t, value_t> a, const index_t i_p) {
  index_t count_end = i_p + 1;
  index_t count_start = 0;
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_i_b_dim_i(const b_tensor_format<index_t, value_t> b, const index_t i_p) {
  index_t count_end = i_p + 1;
  index_t count_start = 0;
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_i_c_dim_i(const c_tensor_format<index_t, value_t> c, const index_t i_p) {
  index_t count_end = i_p + 1;
  index_t count_start = 0;
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t>
struct partition_i {
  index_t* a_i_p;
  index_t* b_i_p;
  index_t* c_i_p;
};
template<typename index_t, typename value_t>
__global__ 
void partition_i_kernel(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, const c_tensor_format<index_t, value_t> c, const Z_tensor_format<index_t, value_t> Z, partition_i<index_t> partitions, const index_t per_thread_work, const index_t total_work) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  index_t count = thread_id * per_thread_work;
  if (count == 0) {
    partitions.a_i_p[thread_id] = 0 - 1;
    partitions.b_i_p[thread_id] = 0 - 1;
    partitions.c_i_p[thread_id] = 0 - 1;
    return;
  }
  if (total_work <= count) {
    partitions.a_i_p[thread_id] = a.dim_i_length - 1;
    partitions.b_i_p[thread_id] = b.dim_i_length - 1;
    partitions.c_i_p[thread_id] = c.dim_i_length - 1;
    return;
  }
  index_t rem_count = count;
  index_t work = 0;
  bool is_a_partitioned = 0;
  bool is_b_partitioned = 0;
  bool is_c_partitioned = 0;
  index_t start_i = -1;
  index_t end_i = a.dim_i_size - 1;
  index_t i = start_i + ((end_i - start_i) / 2);
  index_t start_a_i_p = -1;
  index_t end_a_i_p = a.dim_i_length - 1;
  index_t a_i_p = end_a_i_p;
  index_t start_b_i_p = -1;
  index_t end_b_i_p = b.dim_i_length - 1;
  index_t b_i_p = end_b_i_p;
  index_t start_c_i_p = -1;
  index_t end_c_i_p = c.dim_i_length - 1;
  index_t c_i_p = end_c_i_p;
  while (1) {
    i = start_i + ((end_i - start_i) / 2);
    a_i_p = binary_search(a.dim_i_indices, i, start_a_i_p, end_a_i_p);
    b_i_p = binary_search(b.dim_i_indices, i, start_b_i_p, end_b_i_p);
    c_i_p = binary_search(c.dim_i_indices, i, start_c_i_p, end_c_i_p);
    work = 0;
    work = work + work_i_a_dim_i(a, a_i_p);
    work = work + work_i_b_dim_i(b, b_i_p);
    work = work + work_i_c_dim_i(c, c_i_p);
    if (end_i <= start_i) {
      partitions.a_i_p[thread_id] = a_i_p;
      partitions.b_i_p[thread_id] = b_i_p;
      partitions.c_i_p[thread_id] = c_i_p;
      rem_count = rem_count - work;
      break;
    }
    if (work < rem_count) {
      start_i = i + 1;
      if (!is_a_partitioned) {
        start_a_i_p = a_i_p;
      }
      if (!is_b_partitioned) {
        start_b_i_p = b_i_p;
      }
      if (!is_c_partitioned) {
        start_c_i_p = c_i_p;
      }
    } else {
      end_i = i;
      if (!is_a_partitioned) {
        end_a_i_p = a_i_p;
      }
      if (!is_b_partitioned) {
        end_b_i_p = b_i_p;
      }
      if (!is_c_partitioned) {
        end_c_i_p = c_i_p;
      }
    }
  }
  return;
}
template<typename index_t, typename value_t>
__global__ 
void precompute_i_kernel(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, const c_tensor_format<index_t, value_t> c, const partition_i<index_t> partitions, result_per_thread_count<index_t> count_offsets, const index_t per_thread_work) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t max_thread_id = (gridDim.x * blockDim.x) - 1;
  index_t start_a_i_p = partitions.a_i_p[thread_id];
  index_t end_a_i_p = (thread_id < max_thread_id) ? partitions.a_i_p[thread_id + 1] : (a.dim_i_length - 1);
  index_t start_b_i_p = partitions.b_i_p[thread_id];
  index_t end_b_i_p = (thread_id < max_thread_id) ? partitions.b_i_p[thread_id + 1] : (b.dim_i_length - 1);
  index_t start_c_i_p = partitions.c_i_p[thread_id];
  index_t end_c_i_p = (thread_id < max_thread_id) ? partitions.c_i_p[thread_id + 1] : (c.dim_i_length - 1);
  start_a_i_p++;
  start_b_i_p++;
  start_c_i_p++;
  index_t count = thread_id * per_thread_work;
  index_t count_i = 0;
  index_t iter_a_i_p = start_a_i_p;
  index_t stop_a_i_p = end_a_i_p;
  index_t iter_b_i_p = start_b_i_p;
  index_t stop_b_i_p = end_b_i_p;
  index_t iter_c_i_p = start_c_i_p;
  index_t stop_c_i_p = end_c_i_p;
  while (((iter_a_i_p <= stop_a_i_p) && (iter_b_i_p <= stop_b_i_p)) && (iter_c_i_p <= stop_c_i_p)) {
    index_t a_i = a.dim_i_indices[iter_a_i_p];
    index_t b_i = b.dim_i_indices[iter_b_i_p];
    index_t c_i = c.dim_i_indices[iter_c_i_p];
    index_t i = min(c_i, min(b_i, a_i));
    if (((i == a_i) && (i == b_i)) && (i == c_i)) {
      count_i++;
    } else {
      if ((i == a_i) && (i == c_i)) {
        count_i++;
      } else {
        if ((i == b_i) && (i == c_i)) {
          count_i++;
        }
      }
    }
    iter_a_i_p += (a_i == i);
    iter_b_i_p += (b_i == i);
    iter_c_i_p += (c_i == i);
  }
  while ((iter_a_i_p <= stop_a_i_p) && (iter_c_i_p <= stop_c_i_p)) {
    index_t a_i = a.dim_i_indices[iter_a_i_p];
    index_t c_i = c.dim_i_indices[iter_c_i_p];
    index_t i = min(c_i, a_i);
    if ((i == a_i) && (i == c_i)) {
      count_i++;
    }
    iter_a_i_p += (a_i == i);
    iter_c_i_p += (c_i == i);
  }
  while ((iter_b_i_p <= stop_b_i_p) && (iter_c_i_p <= stop_c_i_p)) {
    index_t b_i = b.dim_i_indices[iter_b_i_p];
    index_t c_i = c.dim_i_indices[iter_c_i_p];
    index_t i = min(c_i, b_i);
    if ((i == b_i) && (i == c_i)) {
      count_i++;
    }
    iter_b_i_p += (b_i == i);
    iter_c_i_p += (c_i == i);
  }
  count_offsets.dim_i_count[thread_id] = count_i;
  return;
}
template<typename index_t, typename value_t>
__global__ 
void compute_i_kernel(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, const c_tensor_format<index_t, value_t> c, const partition_i<index_t> partitions, const result_per_thread_count<index_t> count_offsets, const index_t per_thread_work, Z_tensor_format<index_t, value_t> Z) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t max_thread_id = (gridDim.x * blockDim.x) - 1;
  index_t start_a_i_p = partitions.a_i_p[thread_id];
  index_t end_a_i_p = (thread_id < max_thread_id) ? partitions.a_i_p[thread_id + 1] : (a.dim_i_length - 1);
  index_t start_b_i_p = partitions.b_i_p[thread_id];
  index_t end_b_i_p = (thread_id < max_thread_id) ? partitions.b_i_p[thread_id + 1] : (b.dim_i_length - 1);
  index_t start_c_i_p = partitions.c_i_p[thread_id];
  index_t end_c_i_p = (thread_id < max_thread_id) ? partitions.c_i_p[thread_id + 1] : (c.dim_i_length - 1);
  start_a_i_p++;
  start_b_i_p++;
  start_c_i_p++;
  index_t offset_i = count_offsets.dim_i_count[thread_id];
  index_t iter_a_i_p = start_a_i_p;
  index_t stop_a_i_p = end_a_i_p;
  index_t iter_b_i_p = start_b_i_p;
  index_t stop_b_i_p = end_b_i_p;
  index_t iter_c_i_p = start_c_i_p;
  index_t stop_c_i_p = end_c_i_p;
  while (((iter_a_i_p <= stop_a_i_p) && (iter_b_i_p <= stop_b_i_p)) && (iter_c_i_p <= stop_c_i_p)) {
    index_t a_i = a.dim_i_indices[iter_a_i_p];
    index_t b_i = b.dim_i_indices[iter_b_i_p];
    index_t c_i = c.dim_i_indices[iter_c_i_p];
    index_t i = min(c_i, min(b_i, a_i));
    if (((i == a_i) && (i == b_i)) && (i == c_i)) {
      Z.dim_i_indices[offset_i] = i;
      Z.values[offset_i] = (a.values[iter_a_i_p] + b.values[iter_b_i_p]) * c.values[iter_c_i_p];
      offset_i++;
    } else {
      if ((i == a_i) && (i == c_i)) {
        Z.dim_i_indices[offset_i] = i;
        Z.values[offset_i] = a.values[iter_a_i_p] * c.values[iter_c_i_p];
        offset_i++;
      } else {
        if ((i == b_i) && (i == c_i)) {
          Z.dim_i_indices[offset_i] = i;
          Z.values[offset_i] = b.values[iter_b_i_p] * c.values[iter_c_i_p];
          offset_i++;
        }
      }
    }
    iter_a_i_p += (a_i == i);
    iter_b_i_p += (b_i == i);
    iter_c_i_p += (c_i == i);
  }
  while ((iter_a_i_p <= stop_a_i_p) && (iter_c_i_p <= stop_c_i_p)) {
    index_t a_i = a.dim_i_indices[iter_a_i_p];
    index_t c_i = c.dim_i_indices[iter_c_i_p];
    index_t i = min(c_i, a_i);
    if ((i == a_i) && (i == c_i)) {
      Z.dim_i_indices[offset_i] = i;
      Z.values[offset_i] = a.values[iter_a_i_p] * c.values[iter_c_i_p];
      offset_i++;
    }
    iter_a_i_p += (a_i == i);
    iter_c_i_p += (c_i == i);
  }
  while ((iter_b_i_p <= stop_b_i_p) && (iter_c_i_p <= stop_c_i_p)) {
    index_t b_i = b.dim_i_indices[iter_b_i_p];
    index_t c_i = c.dim_i_indices[iter_c_i_p];
    index_t i = min(c_i, b_i);
    if ((i == b_i) && (i == c_i)) {
      Z.dim_i_indices[offset_i] = i;
      Z.values[offset_i] = b.values[iter_b_i_p] * c.values[iter_c_i_p];
      offset_i++;
    }
    iter_b_i_p += (b_i == i);
    iter_c_i_p += (c_i == i);
  }
  return;
}
template<typename index_t, typename value_t>
__host__ 
void Z_compute(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, const c_tensor_format<index_t, value_t> c, Z_tensor_format<index_t, value_t>& Z) {
  index_t num_blocks = 256;
  index_t threads_per_block = 256;
  index_t num_threads = num_blocks * threads_per_block;
  // ========== Phase 0 ==========
  index_t total_work_0 = a.dim_i_length + b.dim_i_length + c.dim_i_length;
  index_t per_thread_work_0 = total_work_0 / num_threads + 1;
  partition_i<index_t> partitions_0;
  cudaMalloc((void**)&partitions_0.a_i_p, num_threads * sizeof(index_t));
  cudaMalloc((void**)&partitions_0.b_i_p, num_threads * sizeof(index_t));
  cudaMalloc((void**)&partitions_0.c_i_p, num_threads * sizeof(index_t));
  partition_i_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(a, b, c, Z, partitions_0, per_thread_work_0, total_work_0);
  result_per_thread_count<index_t> count_offsets_0;
  cudaMalloc((void**)&count_offsets_0.dim_i_count, num_threads * sizeof(index_t));
  precompute_i_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(a, b, c, partitions_0, count_offsets_0, per_thread_work_0);
  index_t* count_offsets_0_dim_i_count_prefix;
  cudaMalloc((void**)&count_offsets_0_dim_i_count_prefix, (num_threads + 1) * sizeof(index_t));
  cudaMemset(count_offsets_0_dim_i_count_prefix, 0, (num_threads + 1) * sizeof(index_t));
  void* d_temp_storage_0_dim_i_count = nullptr;
  size_t temp_storage_bytes_0_dim_i_count = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage_0_dim_i_count, temp_storage_bytes_0_dim_i_count, count_offsets_0.dim_i_count, count_offsets_0_dim_i_count_prefix + 1, num_threads);
  cudaMalloc(&d_temp_storage_0_dim_i_count, temp_storage_bytes_0_dim_i_count);
  cub::DeviceScan::InclusiveSum(d_temp_storage_0_dim_i_count, temp_storage_bytes_0_dim_i_count, count_offsets_0.dim_i_count, count_offsets_0_dim_i_count_prefix + 1, num_threads);
  cudaFree(d_temp_storage_0_dim_i_count);
  cudaFree(count_offsets_0.dim_i_count);
  count_offsets_0.dim_i_count = count_offsets_0_dim_i_count_prefix;
  index_t nnz_i_0;
  cudaMemcpy(&nnz_i_0, count_offsets_0_dim_i_count_prefix + num_threads, sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaMalloc((void**)&Z.dim_i_indices, nnz_i_0 * sizeof(index_t));
  cudaMalloc((void**)&Z.values, nnz_i_0 * sizeof(value_t));
  Z.dim_i_length = nnz_i_0;
  Z.nnz = nnz_i_0;
  compute_i_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(a, b, c, partitions_0, count_offsets_0, per_thread_work_0, Z);
  cudaFree(partitions_0.a_i_p);
  cudaFree(partitions_0.b_i_p);
  cudaFree(partitions_0.c_i_p);
  cudaFree(count_offsets_0.dim_i_count);
}

} // namespace sparse_vec_apb_c_ns

using namespace sparse_vec_apb_c_ns;

template<typename index_t, typename value_t>
__host__ 
void sparse_vec_apb_c(index_t a_dim_i_size, index_t a_dim_i_length, index_t* a_dim_i_indices, value_t* a_values, index_t a_nnz, index_t b_dim_i_size, index_t b_dim_i_length, index_t* b_dim_i_indices, value_t* b_values, index_t b_nnz, index_t c_dim_i_size, index_t c_dim_i_length, index_t* c_dim_i_indices, value_t* c_values, index_t c_nnz, index_t result_dim_i_size, index_t& out_nnz, index_t*& out_dim_i_indices, value_t*& out_values) {
  a_tensor_format<index_t, value_t> a;
  a.dim_i_size = a_dim_i_size;
  a.dim_i_length = a_dim_i_length;
  a.dim_i_indices = a_dim_i_indices;
  a.values = a_values;
  a.nnz = a_nnz;
  b_tensor_format<index_t, value_t> b;
  b.dim_i_size = b_dim_i_size;
  b.dim_i_length = b_dim_i_length;
  b.dim_i_indices = b_dim_i_indices;
  b.values = b_values;
  b.nnz = b_nnz;
  c_tensor_format<index_t, value_t> c;
  c.dim_i_size = c_dim_i_size;
  c.dim_i_length = c_dim_i_length;
  c.dim_i_indices = c_dim_i_indices;
  c.values = c_values;
  c.nnz = c_nnz;
  Z_tensor_format<index_t, value_t> Z;
  Z.dim_i_size = result_dim_i_size;
  Z_compute<index_t, value_t>(a, b, c, Z);
  out_nnz = Z.nnz;
  out_dim_i_indices = Z.dim_i_indices;
  out_values = Z.values;
}

// Explicit template instantiation
template void sparse_vec_apb_c<int, float>(int, int, int*, float*, int, int, int, int*, float*, int, int, int, int*, float*, int, int, int&, int*&, float*&);
