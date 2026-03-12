#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace dcsr_mul_ns {

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
struct A_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  index_t dim_i_length;
  index_t* dim_i_indices;
  index_t* dim_j_offsets;
  index_t dim_j_length;
  index_t* dim_j_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct B_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  index_t dim_i_length;
  index_t* dim_i_indices;
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
  index_t dim_i_length;
  index_t* dim_i_indices;
  index_t* dim_j_offsets;
  index_t dim_j_length;
  index_t* dim_j_indices;
  value_t* values;
  index_t nnz;
};
template<typename index_t>
struct result_per_thread_count {
  index_t* dim_i_count;
  index_t* dim_j_count;
};
template<typename index_t>
struct result_to_operand_pos_map {
  index_t* A_i_p;
  index_t* B_i_p;
};
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_i_A_dim_i(const A_tensor_format<index_t, value_t> A, const index_t i_p) {
  index_t count_end = i_p + 1;
  index_t count_start = 0;
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_i_B_dim_i(const B_tensor_format<index_t, value_t> B, const index_t i_p) {
  index_t count_end = i_p + 1;
  index_t count_start = 0;
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t>
struct partition_i {
  index_t* A_i_p;
  index_t* B_i_p;
};
template<typename index_t, typename value_t>
__global__ 
void partition_i_kernel(const A_tensor_format<index_t, value_t> A, const B_tensor_format<index_t, value_t> B, const Z_tensor_format<index_t, value_t> Z, partition_i<index_t> partitions, const index_t per_thread_work, const index_t total_work) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  index_t count = thread_id * per_thread_work;
  if (count == 0) {
    partitions.A_i_p[thread_id] = 0 - 1;
    partitions.B_i_p[thread_id] = 0 - 1;
    return;
  }
  if (total_work <= count) {
    partitions.A_i_p[thread_id] = A.dim_i_length - 1;
    partitions.B_i_p[thread_id] = B.dim_i_length - 1;
    return;
  }
  index_t rem_count = count;
  index_t work = 0;
  bool is_A_partitioned = 0;
  bool is_B_partitioned = 0;
  index_t start_A_i_p = -1;
  index_t end_A_i_p = A.dim_i_length - 1;
  index_t start_B_i_p = -1;
  index_t end_B_i_p = B.dim_i_length - 1;
  index_t start_i = start_A_i_p;
  index_t end_i = end_A_i_p;
  index_t A_i_p = start_i + (((end_i - start_i) + 1) / 2);
  index_t B_i_p = start_B_i_p + (rem_count - (A_i_p - start_A_i_p));
  while (start_i < end_i) {
    A_i_p = start_i + (((end_i - start_i) + 1) / 2);
    B_i_p = start_B_i_p + (rem_count - (A_i_p - start_A_i_p));
    if ((start_B_i_p <= B_i_p) && (B_i_p <= end_B_i_p)) {
      if (((start_B_i_p < B_i_p) && ((A_i_p + 1) <= end_A_i_p)) && (A.dim_i_indices[A_i_p + 1] < B.dim_i_indices[B_i_p])) {
        start_i = A_i_p + 1;
      } else {
        if (((start_A_i_p < A_i_p) && ((B_i_p + 1) <= end_B_i_p)) && (B.dim_i_indices[B_i_p + 1] < A.dim_i_indices[A_i_p])) {
          end_i = A_i_p - 1;
        } else {
          start_i = A_i_p;
          end_i = A_i_p;
        }
      }
    } else {
      if (end_B_i_p < B_i_p) {
        start_i = A_i_p + 1;
      } else {
        end_i = A_i_p - 1;
      }
    }
  }
  A_i_p = start_i + (((end_i - start_i) + 1) / 2);
  B_i_p = start_B_i_p + (rem_count - (A_i_p - start_A_i_p));
  if (((start_B_i_p < B_i_p) && ((A_i_p + 1) <= end_A_i_p)) && (A.dim_i_indices[A_i_p + 1] == B.dim_i_indices[B_i_p])) {
    A_i_p = A_i_p + 1;
  } else {
    if (((start_A_i_p < A_i_p) && ((B_i_p + 1) <= end_B_i_p)) && (B.dim_i_indices[B_i_p + 1] == A.dim_i_indices[A_i_p])) {
      B_i_p = B_i_p + 1;
    }
  }
  partitions.A_i_p[thread_id] = A_i_p;
  partitions.B_i_p[thread_id] = B_i_p;
  return;
}
template<typename index_t, typename value_t>
__global__ 
void precompute_i_kernel(const A_tensor_format<index_t, value_t> A, const B_tensor_format<index_t, value_t> B, const partition_i<index_t> partitions, result_per_thread_count<index_t> count_offsets, const index_t per_thread_work) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t max_thread_id = (gridDim.x * blockDim.x) - 1;
  index_t start_A_i_p = partitions.A_i_p[thread_id];
  index_t end_A_i_p = (thread_id < max_thread_id) ? partitions.A_i_p[thread_id + 1] : (A.dim_i_length - 1);
  index_t start_B_i_p = partitions.B_i_p[thread_id];
  index_t end_B_i_p = (thread_id < max_thread_id) ? partitions.B_i_p[thread_id + 1] : (B.dim_i_length - 1);
  start_A_i_p++;
  start_B_i_p++;
  index_t count = thread_id * per_thread_work;
  index_t count_i = 0;
  index_t iter_A_i_p = start_A_i_p;
  index_t stop_A_i_p = end_A_i_p;
  index_t iter_B_i_p = start_B_i_p;
  index_t stop_B_i_p = end_B_i_p;
  while ((iter_A_i_p <= stop_A_i_p) && (iter_B_i_p <= stop_B_i_p)) {
    index_t A_i = A.dim_i_indices[iter_A_i_p];
    index_t B_i = B.dim_i_indices[iter_B_i_p];
    index_t i = min(B_i, A_i);
    if ((i == A_i) && (i == B_i)) {
      count_i++;
    }
    iter_A_i_p += (A_i == i);
    iter_B_i_p += (B_i == i);
  }
  count_offsets.dim_i_count[thread_id] = count_i;
  return;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_A_dim_i(const A_tensor_format<index_t, value_t> A, const index_t i_p) {
  index_t j_p_end = A.dim_j_offsets[i_p + 1];
  index_t j_p_start = A.dim_j_offsets[i_p];
  index_t count_end = j_p_end;
  index_t count_start = j_p_start;
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_B_dim_i(const B_tensor_format<index_t, value_t> B, const index_t i_p) {
  index_t j_p_end = B.dim_j_offsets[i_p + 1];
  index_t j_p_start = B.dim_j_offsets[i_p];
  index_t count_end = j_p_end;
  index_t count_start = j_p_start;
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t, typename value_t>
__global__ 
void compute_i_kernel(const A_tensor_format<index_t, value_t> A, const B_tensor_format<index_t, value_t> B, const partition_i<index_t> partitions, const result_per_thread_count<index_t> count_offsets, const index_t per_thread_work, Z_tensor_format<index_t, value_t> Z, index_t* T_work_offsets, result_to_operand_pos_map<index_t> Z_pos_map) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t max_thread_id = (gridDim.x * blockDim.x) - 1;
  index_t start_A_i_p = partitions.A_i_p[thread_id];
  index_t end_A_i_p = (thread_id < max_thread_id) ? partitions.A_i_p[thread_id + 1] : (A.dim_i_length - 1);
  index_t start_B_i_p = partitions.B_i_p[thread_id];
  index_t end_B_i_p = (thread_id < max_thread_id) ? partitions.B_i_p[thread_id + 1] : (B.dim_i_length - 1);
  start_A_i_p++;
  start_B_i_p++;
  index_t offset_i = count_offsets.dim_i_count[thread_id];
  index_t iter_A_i_p = start_A_i_p;
  index_t stop_A_i_p = end_A_i_p;
  index_t iter_B_i_p = start_B_i_p;
  index_t stop_B_i_p = end_B_i_p;
  while ((iter_A_i_p <= stop_A_i_p) && (iter_B_i_p <= stop_B_i_p)) {
    index_t A_i = A.dim_i_indices[iter_A_i_p];
    index_t B_i = B.dim_i_indices[iter_B_i_p];
    index_t i = min(B_i, A_i);
    if ((i == A_i) && (i == B_i)) {
      Z.dim_i_indices[offset_i] = i;
      Z_pos_map.A_i_p[offset_i] = iter_A_i_p;
      Z_pos_map.B_i_p[offset_i] = iter_B_i_p;
      T_work_offsets[offset_i] = (0 + work_ij_A_dim_i(A, iter_A_i_p)) + work_ij_B_dim_i(B, iter_B_i_p);
      offset_i++;
    }
    iter_A_i_p += (A_i == i);
    iter_B_i_p += (B_i == i);
  }
  return;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_i_Z_dim_i(const Z_tensor_format<index_t, value_t> Z, const index_t i_p, const index_t* __restrict__ T_work_offsets) {
  index_t count_end = i_p + 1;
  index_t count_start = 0;
  index_t count = T_work_offsets[count_end] - T_work_offsets[count_start];
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_A_dim_j(const A_tensor_format<index_t, value_t> A, const index_t i_p, const index_t j_p) {
  index_t count_end = j_p + 1;
  index_t count_start = A.dim_j_offsets[i_p];
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t, typename value_t>
__device__ __inline__ 
index_t work_ij_B_dim_j(const B_tensor_format<index_t, value_t> B, const index_t i_p, const index_t j_p) {
  index_t count_end = j_p + 1;
  index_t count_start = B.dim_j_offsets[i_p];
  index_t count = count_end - count_start;
  return count;
}
template<typename index_t>
struct partition_ij {
  index_t* Z_i_p;
  index_t* A_j_p;
  index_t* B_j_p;
};
template<typename index_t, typename value_t>
__global__ 
void partition_ij_kernel(const A_tensor_format<index_t, value_t> A, const B_tensor_format<index_t, value_t> B, const Z_tensor_format<index_t, value_t> Z, partition_ij<index_t> partitions, const index_t per_thread_work, const index_t total_work, const index_t* __restrict__ T_work_offsets, const result_to_operand_pos_map<index_t> Z_pos_map) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  index_t count = thread_id * per_thread_work;
  if (count == 0) {
    partitions.Z_i_p[thread_id] = 0;
    partitions.A_j_p[thread_id] = (Z_pos_map.A_i_p[0] != -1) ? (A.dim_j_offsets[Z_pos_map.A_i_p[0]] - 1) : (A.dim_j_length - 1);
    partitions.B_j_p[thread_id] = (Z_pos_map.B_i_p[0] != -1) ? (B.dim_j_offsets[Z_pos_map.B_i_p[0]] - 1) : (B.dim_j_length - 1);
    return;
  }
  if (total_work <= count) {
    partitions.Z_i_p[thread_id] = Z.dim_i_length - 1;
    partitions.A_j_p[thread_id] = (Z_pos_map.A_i_p[Z.dim_i_length - 1] != -1) ? (A.dim_j_offsets[Z_pos_map.A_i_p[Z.dim_i_length - 1] + 1] - 1) : 0;
    partitions.B_j_p[thread_id] = (Z_pos_map.B_i_p[Z.dim_i_length - 1] != -1) ? (B.dim_j_offsets[Z_pos_map.B_i_p[Z.dim_i_length - 1] + 1] - 1) : 0;
    return;
  }
  index_t rem_count = count;
  index_t work = 0;
  bool is_A_partitioned = 0;
  bool is_B_partitioned = 0;
  index_t start_Z_i_p = -1;
  index_t end_Z_i_p = Z.dim_i_length - 1;
  index_t Z_i_p = start_Z_i_p + (((end_Z_i_p - start_Z_i_p) + 1) / 2);
  index_t A_i_p;
  index_t B_i_p;
  while (1) {
    Z_i_p = start_Z_i_p + (((end_Z_i_p - start_Z_i_p) + 1) / 2);
    work = work_i_Z_dim_i(Z, Z_i_p, T_work_offsets);
    if (work < rem_count) {
      start_Z_i_p = Z_i_p;
    } else {
      end_Z_i_p = Z_i_p - 1;
    }
    if (end_Z_i_p <= start_Z_i_p) {
      Z_i_p = start_Z_i_p + (((end_Z_i_p - start_Z_i_p) + 1) / 2);
      work = work_i_Z_dim_i(Z, Z_i_p, T_work_offsets);
      rem_count = rem_count - work;
      partitions.Z_i_p[thread_id] = Z_i_p + 1;
      A_i_p = Z_pos_map.A_i_p[Z_i_p + 1];
      is_A_partitioned = is_A_partitioned || (A_i_p == -1);
      B_i_p = Z_pos_map.B_i_p[Z_i_p + 1];
      is_B_partitioned = is_B_partitioned || (B_i_p == -1);
      break;
    }
  }
  index_t i = Z.dim_i_indices[Z_i_p + 1];
  index_t start_j = -1;
  index_t end_j = A.dim_j_size - 1;
  index_t j = start_j + ((end_j - start_j) / 2);
  index_t start_A_j_p = A.dim_j_offsets[A_i_p] - 1;
  index_t end_A_j_p = A.dim_j_offsets[A_i_p + 1] - 1;
  index_t A_j_p = end_A_j_p;
  index_t start_B_j_p = B.dim_j_offsets[B_i_p] - 1;
  index_t end_B_j_p = B.dim_j_offsets[B_i_p + 1] - 1;
  index_t B_j_p = end_B_j_p;
  while (1) {
    j = start_j + ((end_j - start_j) / 2);
    if (!is_A_partitioned) {
      A_j_p = binary_search(A.dim_j_indices, j, start_A_j_p, end_A_j_p);
    }
    if (!is_B_partitioned) {
      B_j_p = binary_search(B.dim_j_indices, j, start_B_j_p, end_B_j_p);
    }
    work = 0;
    work = work + (is_A_partitioned ? 0 : work_ij_A_dim_j(A, A_i_p, A_j_p));
    work = work + (is_B_partitioned ? 0 : work_ij_B_dim_j(B, B_i_p, B_j_p));
    if (end_j <= start_j) {
      partitions.A_j_p[thread_id] = A_j_p;
      partitions.B_j_p[thread_id] = B_j_p;
      rem_count = rem_count - work;
      break;
    }
    if (work < rem_count) {
      start_j = j + 1;
      if (!is_A_partitioned) {
        start_A_j_p = A_j_p;
      }
      if (!is_B_partitioned) {
        start_B_j_p = B_j_p;
      }
    } else {
      end_j = j;
      if (!is_A_partitioned) {
        end_A_j_p = A_j_p;
      }
      if (!is_B_partitioned) {
        end_B_j_p = B_j_p;
      }
    }
  }
  return;
}
template<typename index_t, typename value_t>
__global__ 
void precompute_ij_kernel(const A_tensor_format<index_t, value_t> A, const B_tensor_format<index_t, value_t> B, const partition_ij<index_t> partitions, result_per_thread_count<index_t> count_offsets, const index_t per_thread_work, const Z_tensor_format<index_t, value_t> Z, result_to_operand_pos_map<index_t> Z_pos_map) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t max_thread_id = (gridDim.x * blockDim.x) - 1;
  index_t start_Z_i_p = partitions.Z_i_p[thread_id];
  index_t end_Z_i_p = (thread_id < max_thread_id) ? partitions.Z_i_p[thread_id + 1] : (Z.dim_i_length - 1);
  index_t start_A_j_p = partitions.A_j_p[thread_id];
  index_t end_A_j_p = (thread_id < max_thread_id) ? partitions.A_j_p[thread_id + 1] : ((Z_pos_map.A_i_p[Z.dim_i_length - 1] != -1) ? (A.dim_j_offsets[Z_pos_map.A_i_p[Z.dim_i_length - 1] + 1] - 1) : 0);
  index_t start_B_j_p = partitions.B_j_p[thread_id];
  index_t end_B_j_p = (thread_id < max_thread_id) ? partitions.B_j_p[thread_id + 1] : ((Z_pos_map.B_i_p[Z.dim_i_length - 1] != -1) ? (B.dim_j_offsets[Z_pos_map.B_i_p[Z.dim_i_length - 1] + 1] - 1) : 0);
  start_A_j_p++;
  start_B_j_p++;
  index_t count = thread_id * per_thread_work;
  index_t count_i = 0;
  index_t count_j = 0;
  index_t iter_Z_i_p = start_Z_i_p;
  index_t stop_Z_i_p = end_Z_i_p;
  for (; iter_Z_i_p <= stop_Z_i_p; iter_Z_i_p += 1) {
    index_t iter_A_i_p = (Z_pos_map.A_i_p[iter_Z_i_p] != -1) ? Z_pos_map.A_i_p[iter_Z_i_p] : A.dim_i_size;
    index_t iter_B_i_p = (Z_pos_map.B_i_p[iter_Z_i_p] != -1) ? Z_pos_map.B_i_p[iter_Z_i_p] : B.dim_i_size;
    index_t A_i = A.dim_i_indices[iter_A_i_p];
    index_t B_i = B.dim_i_indices[iter_B_i_p];
    index_t i = min(B_i, A_i);
    index_t iter_A_j_p = (iter_Z_i_p == start_Z_i_p) ? start_A_j_p : A.dim_j_offsets[iter_A_i_p];
    index_t stop_A_j_p = (iter_Z_i_p == end_Z_i_p) ? end_A_j_p : (A.dim_j_offsets[(iter_A_i_p + 1)] - 1);
    index_t iter_B_j_p = (iter_Z_i_p == start_Z_i_p) ? start_B_j_p : B.dim_j_offsets[iter_B_i_p];
    index_t stop_B_j_p = (iter_Z_i_p == end_Z_i_p) ? end_B_j_p : (B.dim_j_offsets[(iter_B_i_p + 1)] - 1);
    while ((iter_A_j_p <= stop_A_j_p) && (iter_B_j_p <= stop_B_j_p)) {
      index_t A_j = A.dim_j_indices[iter_A_j_p];
      index_t B_j = B.dim_j_indices[iter_B_j_p];
      index_t j = min(B_j, A_j);
      if ((j == A_j) && (j == B_j)) {
        count_j++;
      }
      iter_A_j_p += (A_j == j);
      iter_B_j_p += (B_j == j);
    }
    if ((iter_Z_i_p != stop_Z_i_p) || (thread_id == max_thread_id)) {
      count_i++;
    }
  }
  count_offsets.dim_i_count[thread_id] = count_i;
  count_offsets.dim_j_count[thread_id] = count_j;
  return;
}
template<typename index_t, typename value_t>
__global__ 
void compute_ij_kernel(const A_tensor_format<index_t, value_t> A, const B_tensor_format<index_t, value_t> B, const partition_ij<index_t> partitions, const result_per_thread_count<index_t> count_offsets, const index_t per_thread_work, Z_tensor_format<index_t, value_t> Z, result_to_operand_pos_map<index_t> Z_pos_map) {
  int32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t max_thread_id = (gridDim.x * blockDim.x) - 1;
  index_t start_Z_i_p = partitions.Z_i_p[thread_id];
  index_t end_Z_i_p = (thread_id < max_thread_id) ? partitions.Z_i_p[thread_id + 1] : (Z.dim_i_length - 1);
  index_t start_A_j_p = partitions.A_j_p[thread_id];
  index_t end_A_j_p = (thread_id < max_thread_id) ? partitions.A_j_p[thread_id + 1] : ((Z_pos_map.A_i_p[Z.dim_i_length - 1] != -1) ? (A.dim_j_offsets[Z_pos_map.A_i_p[Z.dim_i_length - 1] + 1] - 1) : 0);
  index_t start_B_j_p = partitions.B_j_p[thread_id];
  index_t end_B_j_p = (thread_id < max_thread_id) ? partitions.B_j_p[thread_id + 1] : ((Z_pos_map.B_i_p[Z.dim_i_length - 1] != -1) ? (B.dim_j_offsets[Z_pos_map.B_i_p[Z.dim_i_length - 1] + 1] - 1) : 0);
  start_A_j_p++;
  start_B_j_p++;
  index_t offset_i = count_offsets.dim_i_count[thread_id];
  index_t offset_j = count_offsets.dim_j_count[thread_id];
  index_t iter_Z_i_p = start_Z_i_p;
  index_t stop_Z_i_p = end_Z_i_p;
  for (; iter_Z_i_p <= stop_Z_i_p; iter_Z_i_p += 1) {
    index_t iter_A_i_p = (Z_pos_map.A_i_p[iter_Z_i_p] != -1) ? Z_pos_map.A_i_p[iter_Z_i_p] : A.dim_i_size;
    index_t iter_B_i_p = (Z_pos_map.B_i_p[iter_Z_i_p] != -1) ? Z_pos_map.B_i_p[iter_Z_i_p] : B.dim_i_size;
    index_t A_i = A.dim_i_indices[iter_A_i_p];
    index_t B_i = B.dim_i_indices[iter_B_i_p];
    index_t i = min(B_i, A_i);
    index_t iter_A_j_p = (iter_Z_i_p == start_Z_i_p) ? start_A_j_p : A.dim_j_offsets[iter_A_i_p];
    index_t stop_A_j_p = (iter_Z_i_p == end_Z_i_p) ? end_A_j_p : (A.dim_j_offsets[(iter_A_i_p + 1)] - 1);
    index_t iter_B_j_p = (iter_Z_i_p == start_Z_i_p) ? start_B_j_p : B.dim_j_offsets[iter_B_i_p];
    index_t stop_B_j_p = (iter_Z_i_p == end_Z_i_p) ? end_B_j_p : (B.dim_j_offsets[(iter_B_i_p + 1)] - 1);
    while ((iter_A_j_p <= stop_A_j_p) && (iter_B_j_p <= stop_B_j_p)) {
      index_t A_j = A.dim_j_indices[iter_A_j_p];
      index_t B_j = B.dim_j_indices[iter_B_j_p];
      index_t j = min(B_j, A_j);
      if ((j == A_j) && (j == B_j)) {
        Z.dim_j_indices[offset_j] = j;
        Z.values[offset_j] = A.values[iter_A_j_p] * B.values[iter_B_j_p];
        offset_j++;
      }
      iter_A_j_p += (A_j == j);
      iter_B_j_p += (B_j == j);
    }
    if ((iter_Z_i_p != stop_Z_i_p) || (thread_id == max_thread_id)) {
      Z.dim_j_offsets[offset_i + 1] = offset_j;
      offset_i++;
    }
  }
  return;
}
template<typename index_t, typename value_t>
__host__ 
void Z_compute(const A_tensor_format<index_t, value_t> A, const B_tensor_format<index_t, value_t> B, Z_tensor_format<index_t, value_t>& Z) {
  index_t num_blocks = 256;
  index_t threads_per_block = 256;
  index_t num_threads = num_blocks * threads_per_block;
  // ========== Phase 0 ==========
  index_t total_work_0 = A.dim_i_length + B.dim_i_length;
  index_t per_thread_work_0 = total_work_0 / num_threads + 1;
  partition_i<index_t> partitions_0;
  cudaMalloc((void**)&partitions_0.A_i_p, num_threads * sizeof(index_t));
  cudaMalloc((void**)&partitions_0.B_i_p, num_threads * sizeof(index_t));
  partition_i_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(A, B, Z, partitions_0, per_thread_work_0, total_work_0);
  result_per_thread_count<index_t> count_offsets_0;
  cudaMalloc((void**)&count_offsets_0.dim_i_count, num_threads * sizeof(index_t));
  cudaMalloc((void**)&count_offsets_0.dim_j_count, num_threads * sizeof(index_t));
  precompute_i_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(A, B, partitions_0, count_offsets_0, per_thread_work_0);
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
  index_t* count_offsets_0_dim_j_count_prefix;
  cudaMalloc((void**)&count_offsets_0_dim_j_count_prefix, (num_threads + 1) * sizeof(index_t));
  cudaMemset(count_offsets_0_dim_j_count_prefix, 0, (num_threads + 1) * sizeof(index_t));
  void* d_temp_storage_0_dim_j_count = nullptr;
  size_t temp_storage_bytes_0_dim_j_count = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage_0_dim_j_count, temp_storage_bytes_0_dim_j_count, count_offsets_0.dim_j_count, count_offsets_0_dim_j_count_prefix + 1, num_threads);
  cudaMalloc(&d_temp_storage_0_dim_j_count, temp_storage_bytes_0_dim_j_count);
  cub::DeviceScan::InclusiveSum(d_temp_storage_0_dim_j_count, temp_storage_bytes_0_dim_j_count, count_offsets_0.dim_j_count, count_offsets_0_dim_j_count_prefix + 1, num_threads);
  cudaFree(d_temp_storage_0_dim_j_count);
  cudaFree(count_offsets_0.dim_j_count);
  count_offsets_0.dim_j_count = count_offsets_0_dim_j_count_prefix;
  index_t nnz_i_0;
  cudaMemcpy(&nnz_i_0, count_offsets_0_dim_i_count_prefix + num_threads, sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaMalloc((void**)&Z.dim_i_indices, nnz_i_0 * sizeof(index_t));
  Z.dim_i_length = nnz_i_0;
  if (nnz_i_0 == 0) {
    Z.dim_j_length = 0;
    Z.nnz = 0;
    cudaFree(partitions_0.A_i_p);
    cudaFree(partitions_0.B_i_p);
    cudaFree(count_offsets_0.dim_i_count);
    cudaFree(count_offsets_0.dim_j_count);
    return;
  }
  result_to_operand_pos_map<index_t> Z_pos_map;
  cudaMalloc((void**)&Z_pos_map.A_i_p, nnz_i_0 * sizeof(index_t));
  cudaMalloc((void**)&Z_pos_map.B_i_p, nnz_i_0 * sizeof(index_t));
  index_t* T_work_offsets;
  cudaMalloc((void**)&T_work_offsets, nnz_i_0 * sizeof(index_t));
  compute_i_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(A, B, partitions_0, count_offsets_0, per_thread_work_0, Z, T_work_offsets, Z_pos_map);
  index_t* T_work_offsets_prefix;
  cudaMalloc((void**)&T_work_offsets_prefix, (nnz_i_0 + 1) * sizeof(index_t));
  cudaMemset(T_work_offsets_prefix, 0, (nnz_i_0 + 1) * sizeof(index_t));
  void* d_temp_tw_0 = nullptr;
  size_t temp_bytes_tw_0 = 0;
  cub::DeviceScan::InclusiveSum(d_temp_tw_0, temp_bytes_tw_0, T_work_offsets, T_work_offsets_prefix + 1, nnz_i_0);
  cudaMalloc(&d_temp_tw_0, temp_bytes_tw_0);
  cub::DeviceScan::InclusiveSum(d_temp_tw_0, temp_bytes_tw_0, T_work_offsets, T_work_offsets_prefix + 1, nnz_i_0);
  cudaFree(d_temp_tw_0);
  cudaFree(T_work_offsets);
  T_work_offsets = T_work_offsets_prefix;
  cudaFree(partitions_0.A_i_p);
  cudaFree(partitions_0.B_i_p);
  // ========== Phase 1 ==========
  index_t total_work_1;
  cudaMemcpy(&total_work_1, &T_work_offsets_prefix[nnz_i_0], sizeof(index_t), cudaMemcpyDeviceToHost);
  index_t per_thread_work_1 = total_work_1 / num_threads + 1;
  partition_ij<index_t> partitions_1;
  cudaMalloc((void**)&partitions_1.Z_i_p, num_threads * sizeof(index_t));
  cudaMalloc((void**)&partitions_1.A_j_p, num_threads * sizeof(index_t));
  cudaMalloc((void**)&partitions_1.B_j_p, num_threads * sizeof(index_t));
  partition_ij_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(A, B, Z, partitions_1, per_thread_work_1, total_work_1, T_work_offsets, Z_pos_map);
  result_per_thread_count<index_t> count_offsets_1;
  cudaMalloc((void**)&count_offsets_1.dim_i_count, num_threads * sizeof(index_t));
  cudaMalloc((void**)&count_offsets_1.dim_j_count, num_threads * sizeof(index_t));
  precompute_ij_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(A, B, partitions_1, count_offsets_1, per_thread_work_1, Z, Z_pos_map);
  index_t* count_offsets_1_dim_i_count_prefix;
  cudaMalloc((void**)&count_offsets_1_dim_i_count_prefix, (num_threads + 1) * sizeof(index_t));
  cudaMemset(count_offsets_1_dim_i_count_prefix, 0, (num_threads + 1) * sizeof(index_t));
  void* d_temp_storage_1_dim_i_count = nullptr;
  size_t temp_storage_bytes_1_dim_i_count = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage_1_dim_i_count, temp_storage_bytes_1_dim_i_count, count_offsets_1.dim_i_count, count_offsets_1_dim_i_count_prefix + 1, num_threads);
  cudaMalloc(&d_temp_storage_1_dim_i_count, temp_storage_bytes_1_dim_i_count);
  cub::DeviceScan::InclusiveSum(d_temp_storage_1_dim_i_count, temp_storage_bytes_1_dim_i_count, count_offsets_1.dim_i_count, count_offsets_1_dim_i_count_prefix + 1, num_threads);
  cudaFree(d_temp_storage_1_dim_i_count);
  cudaFree(count_offsets_1.dim_i_count);
  count_offsets_1.dim_i_count = count_offsets_1_dim_i_count_prefix;
  index_t* count_offsets_1_dim_j_count_prefix;
  cudaMalloc((void**)&count_offsets_1_dim_j_count_prefix, (num_threads + 1) * sizeof(index_t));
  cudaMemset(count_offsets_1_dim_j_count_prefix, 0, (num_threads + 1) * sizeof(index_t));
  void* d_temp_storage_1_dim_j_count = nullptr;
  size_t temp_storage_bytes_1_dim_j_count = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage_1_dim_j_count, temp_storage_bytes_1_dim_j_count, count_offsets_1.dim_j_count, count_offsets_1_dim_j_count_prefix + 1, num_threads);
  cudaMalloc(&d_temp_storage_1_dim_j_count, temp_storage_bytes_1_dim_j_count);
  cub::DeviceScan::InclusiveSum(d_temp_storage_1_dim_j_count, temp_storage_bytes_1_dim_j_count, count_offsets_1.dim_j_count, count_offsets_1_dim_j_count_prefix + 1, num_threads);
  cudaFree(d_temp_storage_1_dim_j_count);
  cudaFree(count_offsets_1.dim_j_count);
  count_offsets_1.dim_j_count = count_offsets_1_dim_j_count_prefix;
  index_t nnz_i_1;
  cudaMemcpy(&nnz_i_1, count_offsets_1_dim_i_count_prefix + num_threads, sizeof(index_t), cudaMemcpyDeviceToHost);
  index_t nnz_j_1;
  cudaMemcpy(&nnz_j_1, count_offsets_1_dim_j_count_prefix + num_threads, sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaMalloc((void**)&Z.dim_j_indices, nnz_j_1 * sizeof(index_t));
  cudaMalloc((void**)&Z.values, nnz_j_1 * sizeof(value_t));
  cudaMalloc((void**)&Z.dim_j_offsets, (nnz_i_1 + 1) * sizeof(index_t));
  cudaMemset(Z.dim_j_offsets, 0, (nnz_i_1 + 1) * sizeof(index_t));
  Z.dim_j_length = nnz_j_1;
  Z.nnz = nnz_j_1;
  compute_ij_kernel<index_t, value_t><<<num_blocks, threads_per_block>>>(A, B, partitions_1, count_offsets_1, per_thread_work_1, Z, Z_pos_map);
  cudaFree(partitions_1.Z_i_p);
  cudaFree(partitions_1.A_j_p);
  cudaFree(partitions_1.B_j_p);
  cudaFree(count_offsets_1.dim_i_count);
  cudaFree(count_offsets_1.dim_j_count);
}

} // namespace dcsr_mul_ns

using namespace dcsr_mul_ns;

template<typename index_t, typename value_t>
__host__ 
void dcsr_mul(index_t A_dim_i_size, index_t A_dim_j_size, index_t A_dim_i_length, index_t* A_dim_i_indices, index_t* A_dim_j_offsets, index_t A_dim_j_length, index_t* A_dim_j_indices, value_t* A_values, index_t A_nnz, index_t B_dim_i_size, index_t B_dim_j_size, index_t B_dim_i_length, index_t* B_dim_i_indices, index_t* B_dim_j_offsets, index_t B_dim_j_length, index_t* B_dim_j_indices, value_t* B_values, index_t B_nnz, index_t result_dim_i_size, index_t result_dim_j_size, index_t& out_nnz, index_t& out_dim_i_length, index_t*& out_dim_j_indices, index_t*& out_dim_j_offsets, index_t*& out_dim_i_indices, value_t*& out_values) {
  A_tensor_format<index_t, value_t> A;
  A.dim_i_size = A_dim_i_size;
  A.dim_j_size = A_dim_j_size;
  A.dim_i_length = A_dim_i_length;
  A.dim_i_indices = A_dim_i_indices;
  A.dim_j_offsets = A_dim_j_offsets;
  A.dim_j_length = A_dim_j_length;
  A.dim_j_indices = A_dim_j_indices;
  A.values = A_values;
  A.nnz = A_nnz;
  B_tensor_format<index_t, value_t> B;
  B.dim_i_size = B_dim_i_size;
  B.dim_j_size = B_dim_j_size;
  B.dim_i_length = B_dim_i_length;
  B.dim_i_indices = B_dim_i_indices;
  B.dim_j_offsets = B_dim_j_offsets;
  B.dim_j_length = B_dim_j_length;
  B.dim_j_indices = B_dim_j_indices;
  B.values = B_values;
  B.nnz = B_nnz;
  Z_tensor_format<index_t, value_t> Z;
  Z.dim_i_size = result_dim_i_size;
  Z.dim_j_size = result_dim_j_size;
  Z_compute<index_t, value_t>(A, B, Z);
  out_nnz = Z.nnz;
  out_dim_i_length = Z.dim_i_length;
  out_dim_j_indices = Z.dim_j_indices;
  out_dim_j_offsets = Z.dim_j_offsets;
  out_dim_i_indices = Z.dim_i_indices;
  out_values = Z.values;
}

// Explicit template instantiation
template void dcsr_mul<int, float>(int, int, int, int*, int*, int, int*, float*, int, int, int, int, int*, int*, int, int*, float*, int, int, int, int&, int&, int*&, int*&, int*&, float*&);
