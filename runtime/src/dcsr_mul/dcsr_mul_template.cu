#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace dcsr_mul_ns {

template <typename index_t>
__device__ __forceinline__ index_t lower_bound_device(const index_t *arr,
                                                      index_t size,
                                                      index_t target) {
  index_t lo = 0;
  index_t hi = size;
  while (lo < hi) {
    index_t mid = lo + (hi - lo) / 2;
    if (arr[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename index_t>
__host__ __forceinline__ int num_blocks_for(index_t n, int threads_per_block) {
  if (n <= 0) {
    return 1;
  }
  int blocks = static_cast<int>((n + threads_per_block - 1) / threads_per_block);
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > 65535) {
    blocks = 65535;
  }
  return blocks;
}

template <typename index_t>
__global__ void mark_row_matches_kernel(const index_t *A_dim_i_indices,
                                        index_t A_dim_i_length,
                                        const index_t *B_dim_i_indices,
                                        index_t B_dim_i_length,
                                        index_t *row_match_flags,
                                        index_t *row_match_b_pos) {
  index_t tid = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);

  for (index_t a_row_pos = tid; a_row_pos < A_dim_i_length; a_row_pos += stride) {
    index_t row = A_dim_i_indices[a_row_pos];
    index_t b_row_pos = lower_bound_device(B_dim_i_indices, B_dim_i_length, row);
    bool matched = (b_row_pos < B_dim_i_length && B_dim_i_indices[b_row_pos] == row);
    row_match_flags[a_row_pos] = matched ? 1 : 0;
    row_match_b_pos[a_row_pos] = matched ? b_row_pos : -1;
  }
}

template <typename index_t>
__global__ void scatter_row_matches_kernel(const index_t *row_match_flags,
                                           const index_t *row_match_prefix,
                                           const index_t *row_match_b_pos,
                                           index_t A_dim_i_length,
                                           index_t *matched_a_row_pos,
                                           index_t *matched_b_row_pos) {
  index_t tid = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);

  for (index_t a_row_pos = tid; a_row_pos < A_dim_i_length; a_row_pos += stride) {
    if (row_match_flags[a_row_pos] == 0) {
      continue;
    }
    index_t out_pos = row_match_prefix[a_row_pos];
    matched_a_row_pos[out_pos] = a_row_pos;
    matched_b_row_pos[out_pos] = row_match_b_pos[a_row_pos];
  }
}

template <typename index_t>
__global__ void count_col_intersections_kernel(
    const index_t *A_dim_j_offsets, const index_t *A_dim_j_indices,
    const index_t *B_dim_j_offsets, const index_t *B_dim_j_indices,
    const index_t *matched_a_row_pos, const index_t *matched_b_row_pos,
    index_t matched_row_count, index_t *row_col_counts, index_t *row_col_flags) {
  index_t tid = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);

  for (index_t row_pos = tid; row_pos < matched_row_count; row_pos += stride) {
    index_t a_row_pos = matched_a_row_pos[row_pos];
    index_t b_row_pos = matched_b_row_pos[row_pos];

    index_t a_ptr = A_dim_j_offsets[a_row_pos];
    index_t a_end = A_dim_j_offsets[a_row_pos + 1];
    index_t b_ptr = B_dim_j_offsets[b_row_pos];
    index_t b_end = B_dim_j_offsets[b_row_pos + 1];

    index_t count = 0;
    while (a_ptr < a_end && b_ptr < b_end) {
      index_t a_col = A_dim_j_indices[a_ptr];
      index_t b_col = B_dim_j_indices[b_ptr];
      if (a_col == b_col) {
        ++count;
        ++a_ptr;
        ++b_ptr;
      } else if (a_col < b_col) {
        ++a_ptr;
      } else {
        ++b_ptr;
      }
    }

    row_col_counts[row_pos] = count;
    row_col_flags[row_pos] = (count > 0) ? 1 : 0;
  }
}

template <typename index_t>
__global__ void scatter_output_rows_kernel(
    const index_t *A_dim_i_indices, const index_t *matched_a_row_pos,
    const index_t *row_col_flags, const index_t *row_col_prefix,
    const index_t *nnz_prefix, index_t matched_row_count,
    index_t *out_dim_i_indices, index_t *out_dim_j_offsets) {
  index_t tid = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);

  for (index_t row_pos = tid; row_pos < matched_row_count; row_pos += stride) {
    if (row_col_flags[row_pos] == 0) {
      continue;
    }
    index_t out_row_pos = row_col_prefix[row_pos];
    index_t a_row_pos = matched_a_row_pos[row_pos];
    out_dim_i_indices[out_row_pos] = A_dim_i_indices[a_row_pos];
    out_dim_j_offsets[out_row_pos] = nnz_prefix[row_pos];
  }
}

template <typename index_t, typename value_t>
__global__ void fill_col_intersections_kernel(
    const index_t *A_dim_j_offsets, const index_t *A_dim_j_indices,
    const value_t *A_values, const index_t *B_dim_j_offsets,
    const index_t *B_dim_j_indices, const value_t *B_values,
    const index_t *matched_a_row_pos, const index_t *matched_b_row_pos,
    const index_t *row_col_flags, const index_t *row_col_prefix,
    const index_t *out_dim_j_offsets, index_t matched_row_count,
    index_t *out_dim_j_indices, value_t *out_values) {
  index_t tid = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);

  for (index_t row_pos = tid; row_pos < matched_row_count; row_pos += stride) {
    if (row_col_flags[row_pos] == 0) {
      continue;
    }

    index_t a_row_pos = matched_a_row_pos[row_pos];
    index_t b_row_pos = matched_b_row_pos[row_pos];
    index_t out_row_pos = row_col_prefix[row_pos];
    index_t out_ptr = out_dim_j_offsets[out_row_pos];

    index_t a_ptr = A_dim_j_offsets[a_row_pos];
    index_t a_end = A_dim_j_offsets[a_row_pos + 1];
    index_t b_ptr = B_dim_j_offsets[b_row_pos];
    index_t b_end = B_dim_j_offsets[b_row_pos + 1];

    while (a_ptr < a_end && b_ptr < b_end) {
      index_t a_col = A_dim_j_indices[a_ptr];
      index_t b_col = B_dim_j_indices[b_ptr];
      if (a_col == b_col) {
        out_dim_j_indices[out_ptr] = a_col;
        out_values[out_ptr] = A_values[a_ptr] * B_values[b_ptr];
        ++out_ptr;
        ++a_ptr;
        ++b_ptr;
      } else if (a_col < b_col) {
        ++a_ptr;
      } else {
        ++b_ptr;
      }
    }
  }
}

template <typename index_t, typename value_t>
__host__ void dcsr_mul_impl(
    index_t A_dim_i_size, index_t A_dim_j_size, index_t A_dim_i_length,
    index_t *A_dim_i_indices, index_t *A_dim_j_offsets, index_t A_dim_j_length,
    index_t *A_dim_j_indices, value_t *A_values, index_t A_nnz,
    index_t B_dim_i_size, index_t B_dim_j_size, index_t B_dim_i_length,
    index_t *B_dim_i_indices, index_t *B_dim_j_offsets, index_t B_dim_j_length,
    index_t *B_dim_j_indices, value_t *B_values, index_t B_nnz,
    index_t result_dim_i_size, index_t result_dim_j_size, index_t &out_nnz,
    index_t &out_dim_i_length, index_t *&out_dim_j_indices,
    index_t *&out_dim_j_offsets, index_t *&out_dim_i_indices,
    value_t *&out_values) {
  (void)A_dim_i_size;
  (void)A_dim_j_size;
  (void)A_dim_j_length;
  (void)A_nnz;
  (void)B_dim_i_size;
  (void)B_dim_j_size;
  (void)B_dim_j_length;
  (void)B_nnz;
  (void)result_dim_i_size;
  (void)result_dim_j_size;

  out_nnz = 0;
  out_dim_i_length = 0;
  out_dim_j_indices = nullptr;
  out_dim_j_offsets = nullptr;
  out_dim_i_indices = nullptr;
  out_values = nullptr;

  if (A_dim_i_length <= 0 || B_dim_i_length <= 0) {
    return;
  }

  constexpr int threads_per_block = 256;
  const int row_scan_blocks = num_blocks_for(A_dim_i_length, threads_per_block);

  index_t *d_row_match_flags = nullptr;
  index_t *d_row_match_b_pos = nullptr;
  index_t *d_row_match_prefix = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_row_match_flags),
             A_dim_i_length * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&d_row_match_b_pos),
             A_dim_i_length * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&d_row_match_prefix),
             (A_dim_i_length + 1) * sizeof(index_t));
  cudaMemset(d_row_match_prefix, 0, (A_dim_i_length + 1) * sizeof(index_t));

  mark_row_matches_kernel<index_t><<<row_scan_blocks, threads_per_block>>>(
      A_dim_i_indices, A_dim_i_length, B_dim_i_indices, B_dim_i_length,
      d_row_match_flags, d_row_match_b_pos);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_row_match_flags, d_row_match_prefix + 1,
                                A_dim_i_length);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_row_match_flags, d_row_match_prefix + 1,
                                A_dim_i_length);
  cudaFree(d_temp_storage);

  index_t matched_row_count = 0;
  cudaMemcpy(&matched_row_count, d_row_match_prefix + A_dim_i_length,
             sizeof(index_t), cudaMemcpyDeviceToHost);
  if (matched_row_count <= 0) {
    cudaFree(d_row_match_flags);
    cudaFree(d_row_match_b_pos);
    cudaFree(d_row_match_prefix);
    return;
  }

  index_t *d_matched_a_row_pos = nullptr;
  index_t *d_matched_b_row_pos = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_matched_a_row_pos),
             matched_row_count * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&d_matched_b_row_pos),
             matched_row_count * sizeof(index_t));

  scatter_row_matches_kernel<index_t><<<row_scan_blocks, threads_per_block>>>(
      d_row_match_flags, d_row_match_prefix, d_row_match_b_pos, A_dim_i_length,
      d_matched_a_row_pos, d_matched_b_row_pos);

  cudaFree(d_row_match_flags);
  cudaFree(d_row_match_b_pos);
  cudaFree(d_row_match_prefix);

  const int matched_row_blocks =
      num_blocks_for(matched_row_count, threads_per_block);

  index_t *d_row_col_counts = nullptr;
  index_t *d_row_col_flags = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_row_col_counts),
             matched_row_count * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&d_row_col_flags),
             matched_row_count * sizeof(index_t));

  count_col_intersections_kernel<index_t><<<matched_row_blocks,
                                            threads_per_block>>>(
      A_dim_j_offsets, A_dim_j_indices, B_dim_j_offsets, B_dim_j_indices,
      d_matched_a_row_pos, d_matched_b_row_pos, matched_row_count,
      d_row_col_counts, d_row_col_flags);

  index_t *d_row_col_prefix = nullptr;
  index_t *d_nnz_prefix = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_row_col_prefix),
             (matched_row_count + 1) * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&d_nnz_prefix),
             (matched_row_count + 1) * sizeof(index_t));
  cudaMemset(d_row_col_prefix, 0, (matched_row_count + 1) * sizeof(index_t));
  cudaMemset(d_nnz_prefix, 0, (matched_row_count + 1) * sizeof(index_t));

  d_temp_storage = nullptr;
  temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_row_col_flags, d_row_col_prefix + 1,
                                matched_row_count);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_row_col_flags, d_row_col_prefix + 1,
                                matched_row_count);
  cudaFree(d_temp_storage);

  d_temp_storage = nullptr;
  temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_row_col_counts, d_nnz_prefix + 1,
                                matched_row_count);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_row_col_counts, d_nnz_prefix + 1,
                                matched_row_count);
  cudaFree(d_temp_storage);

  cudaMemcpy(&out_dim_i_length, d_row_col_prefix + matched_row_count,
             sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&out_nnz, d_nnz_prefix + matched_row_count, sizeof(index_t),
             cudaMemcpyDeviceToHost);

  if (out_dim_i_length <= 0 || out_nnz <= 0) {
    out_dim_i_length = 0;
    out_nnz = 0;
    cudaFree(d_matched_a_row_pos);
    cudaFree(d_matched_b_row_pos);
    cudaFree(d_row_col_counts);
    cudaFree(d_row_col_flags);
    cudaFree(d_row_col_prefix);
    cudaFree(d_nnz_prefix);
    return;
  }

  cudaMalloc(reinterpret_cast<void **>(&out_dim_i_indices),
             out_dim_i_length * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&out_dim_j_offsets),
             (out_dim_i_length + 1) * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&out_dim_j_indices),
             out_nnz * sizeof(index_t));
  cudaMalloc(reinterpret_cast<void **>(&out_values), out_nnz * sizeof(value_t));

  scatter_output_rows_kernel<index_t><<<matched_row_blocks, threads_per_block>>>(
      A_dim_i_indices, d_matched_a_row_pos, d_row_col_flags, d_row_col_prefix,
      d_nnz_prefix, matched_row_count, out_dim_i_indices, out_dim_j_offsets);

  cudaMemcpy(out_dim_j_offsets + out_dim_i_length, &out_nnz, sizeof(index_t),
             cudaMemcpyHostToDevice);

  fill_col_intersections_kernel<index_t, value_t>
      <<<matched_row_blocks, threads_per_block>>>(
          A_dim_j_offsets, A_dim_j_indices, A_values, B_dim_j_offsets,
          B_dim_j_indices, B_values, d_matched_a_row_pos, d_matched_b_row_pos,
          d_row_col_flags, d_row_col_prefix, out_dim_j_offsets,
          matched_row_count, out_dim_j_indices, out_values);

  cudaFree(d_matched_a_row_pos);
  cudaFree(d_matched_b_row_pos);
  cudaFree(d_row_col_counts);
  cudaFree(d_row_col_flags);
  cudaFree(d_row_col_prefix);
  cudaFree(d_nnz_prefix);
}

} // namespace dcsr_mul_ns

using namespace dcsr_mul_ns;

template <typename index_t, typename value_t>
__host__ void dcsr_mul(
    index_t A_dim_i_size, index_t A_dim_j_size, index_t A_dim_i_length,
    index_t *A_dim_i_indices, index_t *A_dim_j_offsets, index_t A_dim_j_length,
    index_t *A_dim_j_indices, value_t *A_values, index_t A_nnz,
    index_t B_dim_i_size, index_t B_dim_j_size, index_t B_dim_i_length,
    index_t *B_dim_i_indices, index_t *B_dim_j_offsets, index_t B_dim_j_length,
    index_t *B_dim_j_indices, value_t *B_values, index_t B_nnz,
    index_t result_dim_i_size, index_t result_dim_j_size, index_t &out_nnz,
    index_t &out_dim_i_length, index_t *&out_dim_j_indices,
    index_t *&out_dim_j_offsets, index_t *&out_dim_i_indices,
    value_t *&out_values) {
  dcsr_mul_impl<index_t, value_t>(
      A_dim_i_size, A_dim_j_size, A_dim_i_length, A_dim_i_indices,
      A_dim_j_offsets, A_dim_j_length, A_dim_j_indices, A_values, A_nnz,
      B_dim_i_size, B_dim_j_size, B_dim_i_length, B_dim_i_indices,
      B_dim_j_offsets, B_dim_j_length, B_dim_j_indices, B_values, B_nnz,
      result_dim_i_size, result_dim_j_size, out_nnz, out_dim_i_length,
      out_dim_j_indices, out_dim_j_offsets, out_dim_i_indices, out_values);
}

// Explicit template instantiation
template void dcsr_mul<int, float>(
    int, int, int, int *, int *, int, int *, float *, int, int, int, int,
    int *, int *, int, int *, float *, int, int, int, int &, int &, int *&,
    int *&, int *&, float *&);
