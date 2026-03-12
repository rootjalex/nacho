#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace dcsr_add_ns {

// ============================================================
// Hand-written DCSR element-wise addition: Z = A + B
//
// Strategy:
//   Phase 1: Row-union merge (one thread block).
//            Counts result rows and, for each result row,
//            counts the column-union size. Writes row_indices
//            and per-row col counts.
//   Phase 2: Prefix-sum the per-row col counts → row_offsets.
//   Phase 3: Column-union merge (one thread per result row).
//            Fills col_indices, values.
// ============================================================

// Phase 1: Merge row indices from A and B (union).
// One thread does the whole merge (row counts are typically small for DCSR).
// Writes: Z_row_indices[0..nrows_z-1], col_counts[0..nrows_z-1], *nrows_z_out.
template<typename index_t, typename value_t>
__global__
void row_union_kernel(
    const index_t* A_row_indices, const index_t* A_row_offsets, index_t A_nrows,
    const index_t* A_col_indices,
    const index_t* B_row_indices, const index_t* B_row_offsets, index_t B_nrows,
    const index_t* B_col_indices,
    index_t* Z_row_indices, index_t* col_counts, index_t* nrows_z_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    index_t ai = 0, bi = 0, zi = 0;
    while (ai < A_nrows && bi < B_nrows) {
        index_t a_row = A_row_indices[ai];
        index_t b_row = B_row_indices[bi];
        if (a_row == b_row) {
            Z_row_indices[zi] = a_row;
            // Count column union for this row
            index_t a_start = A_row_offsets[ai], a_end = A_row_offsets[ai + 1];
            index_t b_start = B_row_offsets[bi], b_end = B_row_offsets[bi + 1];
            index_t count = 0;
            index_t aj = a_start, bj = b_start;
            while (aj < a_end && bj < b_end) {
                index_t ac = A_col_indices[aj], bc = B_col_indices[bj];
                count++;
                aj += (ac <= bc);
                bj += (bc <= ac);
            }
            count += (a_end - aj) + (b_end - bj);
            col_counts[zi] = count;
            ai++; bi++; zi++;
        } else if (a_row < b_row) {
            Z_row_indices[zi] = a_row;
            col_counts[zi] = A_row_offsets[ai + 1] - A_row_offsets[ai];
            ai++; zi++;
        } else {
            Z_row_indices[zi] = b_row;
            col_counts[zi] = B_row_offsets[bi + 1] - B_row_offsets[bi];
            bi++; zi++;
        }
    }
    for (; ai < A_nrows; ai++, zi++) {
        Z_row_indices[zi] = A_row_indices[ai];
        col_counts[zi] = A_row_offsets[ai + 1] - A_row_offsets[ai];
    }
    for (; bi < B_nrows; bi++, zi++) {
        Z_row_indices[zi] = B_row_indices[bi];
        col_counts[zi] = B_row_offsets[bi + 1] - B_row_offsets[bi];
    }
    *nrows_z_out = zi;
}

// Phase 3: For each result row, merge columns from A and/or B.
// One thread per result row.
template<typename index_t, typename value_t>
__global__
void col_merge_kernel(
    const index_t* A_row_indices, const index_t* A_row_offsets, index_t A_nrows,
    const index_t* A_col_indices, const value_t* A_values,
    const index_t* B_row_indices, const index_t* B_row_offsets, index_t B_nrows,
    const index_t* B_col_indices, const value_t* B_values,
    const index_t* Z_row_indices, const index_t* Z_row_offsets, index_t Z_nrows,
    index_t* Z_col_indices, value_t* Z_values)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Z_nrows) return;

    index_t z_row = Z_row_indices[tid];
    index_t out_start = Z_row_offsets[tid];
    index_t out_pos = out_start;

    // Binary search for z_row in A_row_indices
    index_t a_lo = 0, a_hi = A_nrows;
    while (a_lo < a_hi) {
        index_t mid = a_lo + (a_hi - a_lo) / 2;
        if (A_row_indices[mid] < z_row) a_lo = mid + 1;
        else a_hi = mid;
    }
    bool a_has = (a_lo < A_nrows && A_row_indices[a_lo] == z_row);
    index_t a_start = a_has ? A_row_offsets[a_lo] : 0;
    index_t a_end   = a_has ? A_row_offsets[a_lo + 1] : 0;

    // Binary search for z_row in B_row_indices
    index_t b_lo = 0, b_hi = B_nrows;
    while (b_lo < b_hi) {
        index_t mid = b_lo + (b_hi - b_lo) / 2;
        if (B_row_indices[mid] < z_row) b_lo = mid + 1;
        else b_hi = mid;
    }
    bool b_has = (b_lo < B_nrows && B_row_indices[b_lo] == z_row);
    index_t b_start = b_has ? B_row_offsets[b_lo] : 0;
    index_t b_end   = b_has ? B_row_offsets[b_lo + 1] : 0;

    // Merge columns (union)
    index_t aj = a_start, bj = b_start;
    while (aj < a_end && bj < b_end) {
        index_t ac = A_col_indices[aj], bc = B_col_indices[bj];
        if (ac == bc) {
            Z_col_indices[out_pos] = ac;
            Z_values[out_pos] = A_values[aj] + B_values[bj];
            out_pos++; aj++; bj++;
        } else if (ac < bc) {
            Z_col_indices[out_pos] = ac;
            Z_values[out_pos] = A_values[aj];
            out_pos++; aj++;
        } else {
            Z_col_indices[out_pos] = bc;
            Z_values[out_pos] = B_values[bj];
            out_pos++; bj++;
        }
    }
    for (; aj < a_end; aj++) {
        Z_col_indices[out_pos] = A_col_indices[aj];
        Z_values[out_pos] = A_values[aj];
        out_pos++;
    }
    for (; bj < b_end; bj++) {
        Z_col_indices[out_pos] = B_col_indices[bj];
        Z_values[out_pos] = B_values[bj];
        out_pos++;
    }
}

template<typename index_t, typename value_t>
__host__
void Z_compute(
    index_t A_nrows, const index_t* A_row_indices, const index_t* A_row_offsets,
    const index_t* A_col_indices, const value_t* A_values,
    index_t B_nrows, const index_t* B_row_indices, const index_t* B_row_offsets,
    const index_t* B_col_indices, const value_t* B_values,
    index_t& out_nrows, index_t& out_nnz,
    index_t*& out_row_indices, index_t*& out_row_offsets,
    index_t*& out_col_indices, value_t*& out_values)
{
    index_t max_rows = A_nrows + B_nrows;

    // Allocate temporary arrays for Phase 1
    index_t* d_row_indices;
    index_t* d_col_counts;
    index_t* d_nrows_z;
    cudaMalloc(&d_row_indices, max_rows * sizeof(index_t));
    cudaMalloc(&d_col_counts, max_rows * sizeof(index_t));
    cudaMalloc(&d_nrows_z, sizeof(index_t));

    // Phase 1: Row union + column counting
    row_union_kernel<index_t, value_t><<<1, 1>>>(
        A_row_indices, A_row_offsets, A_nrows, A_col_indices,
        B_row_indices, B_row_offsets, B_nrows, B_col_indices,
        d_row_indices, d_col_counts, d_nrows_z);

    index_t h_nrows_z;
    cudaMemcpy(&h_nrows_z, d_nrows_z, sizeof(index_t), cudaMemcpyDeviceToHost);
    cudaFree(d_nrows_z);

    if (h_nrows_z == 0) {
        cudaFree(d_row_indices);
        cudaFree(d_col_counts);
        out_nrows = 0;
        out_nnz = 0;
        out_row_indices = nullptr;
        out_row_offsets = nullptr;
        out_col_indices = nullptr;
        out_values = nullptr;
        return;
    }

    // Phase 2: Exclusive prefix sum of col_counts → row_offsets
    index_t* d_row_offsets;
    cudaMalloc(&d_row_offsets, (h_nrows_z + 1) * sizeof(index_t));
    cudaMemset(d_row_offsets, 0, sizeof(index_t));  // offsets[0] = 0

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes,
        d_col_counts, d_row_offsets + 1, h_nrows_z);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes,
        d_col_counts, d_row_offsets + 1, h_nrows_z);
    cudaFree(d_temp);
    cudaFree(d_col_counts);

    // Read total nnz
    index_t h_nnz;
    cudaMemcpy(&h_nnz, d_row_offsets + h_nrows_z, sizeof(index_t), cudaMemcpyDeviceToHost);

    // Allocate output arrays
    index_t* d_col_indices;
    value_t* d_values;
    if (h_nnz > 0) {
        cudaMalloc(&d_col_indices, h_nnz * sizeof(index_t));
        cudaMalloc(&d_values, h_nnz * sizeof(value_t));
    } else {
        d_col_indices = nullptr;
        d_values = nullptr;
    }

    // Phase 3: Column merge (one thread per result row)
    if (h_nrows_z > 0 && h_nnz > 0) {
        int threads = 256;
        int blocks = (h_nrows_z + threads - 1) / threads;
        col_merge_kernel<index_t, value_t><<<blocks, threads>>>(
            A_row_indices, A_row_offsets, A_nrows, A_col_indices, A_values,
            B_row_indices, B_row_offsets, B_nrows, B_col_indices, B_values,
            d_row_indices, d_row_offsets, h_nrows_z,
            d_col_indices, d_values);
    }

    out_nrows = h_nrows_z;
    out_nnz = h_nnz;
    out_row_indices = d_row_indices;
    out_row_offsets = d_row_offsets;
    out_col_indices = d_col_indices;
    out_values = d_values;
}

} // namespace dcsr_add_ns

using namespace dcsr_add_ns;

template<typename index_t, typename value_t>
__host__
void dcsr_add(
    index_t A_dim_i_size, index_t A_dim_j_size,
    index_t A_dim_i_length, index_t* A_dim_i_indices,
    index_t* A_dim_j_offsets, index_t A_dim_j_length,
    index_t* A_dim_j_indices, value_t* A_values, index_t A_nnz,
    index_t B_dim_i_size, index_t B_dim_j_size,
    index_t B_dim_i_length, index_t* B_dim_i_indices,
    index_t* B_dim_j_offsets, index_t B_dim_j_length,
    index_t* B_dim_j_indices, value_t* B_values, index_t B_nnz,
    index_t result_dim_i_size, index_t result_dim_j_size,
    index_t& out_nnz, index_t& out_dim_i_length,
    index_t*& out_dim_j_indices, index_t*& out_dim_j_offsets,
    index_t*& out_dim_i_indices, value_t*& out_values)
{
    index_t nrows_z;
    index_t nnz_z;
    index_t* row_indices_z;
    index_t* row_offsets_z;
    index_t* col_indices_z;
    value_t* values_z;

    Z_compute<index_t, value_t>(
        A_dim_i_length, A_dim_i_indices, A_dim_j_offsets,
        A_dim_j_indices, A_values,
        B_dim_i_length, B_dim_i_indices, B_dim_j_offsets,
        B_dim_j_indices, B_values,
        nrows_z, nnz_z,
        row_indices_z, row_offsets_z, col_indices_z, values_z);

    out_nnz = nnz_z;
    out_dim_i_length = nrows_z;
    out_dim_i_indices = row_indices_z;
    out_dim_j_offsets = row_offsets_z;
    out_dim_j_indices = col_indices_z;
    out_values = values_z;
}

// Explicit template instantiation
template void dcsr_add<int, float>(int, int, int, int*, int*, int, int*, float*, int, int, int, int, int*, int*, int, int*, float*, int, int, int, int&, int&, int*&, int*&, int*&, float*&);
