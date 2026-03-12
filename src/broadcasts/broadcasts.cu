#include "broadcasts.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>




template<typename index_t, typename coord_t, typename value_t>
void broadcast_xA_impl(SparseVector<index_t, coord_t, value_t> x, CSRMatrix<index_t, value_t> A,
    index_t * & D_row_offsets, index_t * & D_indices, value_t * & D_values, index_t * & D_nnz) {

    int num_blocks = 256;
    int threads_per_block = 256;

    index_t total_size = x.nnz * A.rows + A.nnz;
    index_t per_thread_work = (total_size)/(num_blocks*threads_per_block) + 18;

    index_t * mergepath_partition_vector, *mergepath_partition_matrix_row, *mergepath_partition_matrix_col;

    SparseVector<index_t, coord_t, value_t>* g_vector;
    CSRMatrix<index_t, value_t> * g_matrix;


    CHECK_CUDA(cudaMalloc(&g_vector, sizeof(SparseVector<index_t, coord_t, value_t>)));
    CHECK_CUDA(cudaMalloc(&g_matrix, sizeof(CSRMatrix<index_t, value_t>)));

    CHECK_CUDA(cudaMemcpy(g_vector, &x, sizeof(SparseVector<index_t, coord_t, value_t>), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g_matrix, &A, sizeof(CSRMatrix<index_t, value_t>), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&mergepath_partition_vector, num_blocks* threads_per_block * sizeof(index_t)));
    CHECK_CUDA(cudaMalloc(&mergepath_partition_matrix_row, num_blocks* threads_per_block * sizeof(index_t)));
    CHECK_CUDA(cudaMalloc(&mergepath_partition_matrix_col, num_blocks* threads_per_block * sizeof(index_t)));

    index_t * nnz_count;


    CHECK_CUDA(cudaMalloc(&nnz_count, num_blocks * threads_per_block * sizeof(index_t)));
    CHECK_CUDA(cudaMemset(nnz_count, 0, num_blocks * threads_per_block * sizeof(index_t)));


    kern_mergepath_partition_vector_matrix<<<num_blocks, threads_per_block>>>(
        mergepath_partition_matrix_row,
        mergepath_partition_matrix_col,
        mergepath_partition_vector,
        total_size,
        g_vector,
        g_matrix,
        per_thread_work
    );

    //CHECK_CUDA(cudaDeviceSynchronize());

    // printf(" row : ");print_cuda(mergepath_partition_matrix_row, 20);
    // printf(" col : ");print_cuda(mergepath_partition_matrix_col, 20);
    // printf(" vector : ");print_cuda(mergepath_partition_vector, 20);
    
    kern_broadcast_xA_precompute<<<num_blocks, threads_per_block>>>(
        g_vector,
        g_matrix,
        mergepath_partition_vector,
        mergepath_partition_matrix_row,
        mergepath_partition_matrix_col,
        per_thread_work,
        nnz_count
    );

    index_t * nnz_prefix;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cudaMalloc((void**)&nnz_prefix, sizeof(index_t)*((num_blocks * threads_per_block) + 1)));
    CHECK_CUDA(cudaMemset(nnz_prefix, 0, sizeof(index_t)*((num_blocks * threads_per_block) + 1)));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnz_prefix+1, num_blocks * threads_per_block));

    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnz_prefix+1, num_blocks * threads_per_block));
    CHECK_CUDA( cudaGetLastError() );
    
    cudaFree(d_temp_storage);

    CHECK_CUDA(cudaMemcpy(D_nnz, nnz_prefix + num_blocks * threads_per_block, sizeof(index_t), cudaMemcpyDeviceToHost));

    //printf("Computed nnz: %d\n", *D_nnz);
    //print_cuda(nnz_prefix, 20);

    if(*D_nnz == 0) {
        throw std::runtime_error("Resultant broadcasted sparse matrix addition has zero non-zeros.");

    } else{
        //printf("Total nnz in output: %d\n", *D_nnz);
        CHECK_CUDA(cudaMalloc(&D_row_offsets, (A.rows + 1) * sizeof(index_t)));
        CHECK_CUDA(cudaMalloc(&D_indices, (*D_nnz) * sizeof(coord_t)));
        CHECK_CUDA(cudaMalloc(&D_values, (*D_nnz) * sizeof(value_t)));

        kern_broadcast_xA_compute<<<num_blocks, threads_per_block>>>(
            g_vector,
            g_matrix,
            mergepath_partition_vector,
            mergepath_partition_matrix_row,
            mergepath_partition_matrix_col,
            per_thread_work,
            D_row_offsets, D_indices, D_values,
            nnz_prefix
        );
    }

    CHECK_CUDA(cudaFree(g_vector));
    CHECK_CUDA(cudaFree(g_matrix));
    CHECK_CUDA(cudaFree(mergepath_partition_vector));
    CHECK_CUDA(cudaFree(mergepath_partition_matrix_row));
    CHECK_CUDA(cudaFree(mergepath_partition_matrix_col));
};



template<typename index_t, typename coord_t, typename value_t>
__global__ void kern_broadcast_xA_precompute(
    SparseVector<index_t, coord_t, value_t>* x,
    CSRMatrix<index_t, value_t>* A,
    index_t * mergepath_partition_vector,
    index_t * mergepath_partition_matrix_row,
    index_t * mergepath_partition_matrix_col,
    index_t per_thread_work,
    index_t * nnz_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_row, end_row;
    
    start_row = mergepath_partition_matrix_row[tid];

    if(tid < gridDim.x * blockDim.x) {
        end_row = mergepath_partition_matrix_row[tid + 1];
    } else {
        end_row = A->rows-1;
    }

    index_t count = 0;
    
    for(int row = start_row; row <= end_row; row++) {
        index_t idx_a = A->row_offsets[row]-1, end_a = A->row_offsets[row+1]-1;
        index_t idx_b = -1, end_b = x->nnz - 1;


        if(row == start_row){
            idx_a = mergepath_partition_matrix_col[tid];
            idx_b = mergepath_partition_vector[tid];
        }

        if(row == end_row){
            if( tid < gridDim.x * blockDim.x) {
                end_a = mergepath_partition_matrix_col[tid + 1];
                end_b = mergepath_partition_vector[tid + 1];
            } else {
                end_a = A->cols-1;
                end_b = x->nnz-1;
            }
        }


        idx_a++;
        idx_b++;

        while(idx_a <= end_a && idx_b <= end_b) {
            coord_t crd_a = A->col_indices[idx_a];
            coord_t crd_b = x->indices[idx_b];
            index_t crd = min(crd_a, crd_b);

            if(crd == crd_a && crd == crd_b) {
                count++;
            } else if(crd == crd_a) {
                count++;
            } else if(crd == crd_b) {
                count++;
            }
            
            idx_a += (index_t)(crd_a == crd);
            idx_b += (index_t)(crd_b == crd);
            
        }

        while(idx_a <= end_a) {

            count++;
            idx_a++;
        }

        while(idx_b <= end_b) {

            count++;
            idx_b++;
        }
    }

    nnz_count[tid] = count;
};



template<typename index_t, typename coord_t, typename value_t>
__global__ void kern_broadcast_xA_compute(
    SparseVector<index_t, coord_t, value_t>* x,
    CSRMatrix<index_t, value_t>* A,
    index_t * mergepath_partition_vector,
    index_t * mergepath_partition_matrix_row,
    index_t * mergepath_partition_matrix_col,
    index_t per_thread_work,
    index_t * D_row_offsets, index_t * D_indices, value_t * D_values,
    index_t * nnz_prefix
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_row, end_row;
    
    start_row = mergepath_partition_matrix_row[tid];

    if(tid < gridDim.x * blockDim.x) {
        end_row = mergepath_partition_matrix_row[tid + 1];
    } else {
        end_row = A->rows-1;
    }

    index_t count = 0;
    index_t idx_output = nnz_prefix[tid];

    if(tid==0){
        D_row_offsets[0] = 0;
    }

    for(int row = start_row; row <= end_row; row++) {
        index_t idx_a = A->row_offsets[row]-1, end_a = A->row_offsets[row+1]-1;
        index_t idx_b = -1, end_b = x->nnz - 1;


        if(row == start_row){
            idx_a = mergepath_partition_matrix_col[tid];
            idx_b = mergepath_partition_vector[tid];
        }

        if(row == end_row){
            if( tid < gridDim.x * blockDim.x) {
                end_a = mergepath_partition_matrix_col[tid + 1];
                end_b = mergepath_partition_vector[tid + 1];
            } else {
                end_a = A->cols-1;
                end_b = x->nnz-1;
            }
        }


        idx_a++;
        idx_b++;

        while(idx_a <= end_a && idx_b <= end_b) {
            coord_t crd_a = A->col_indices[idx_a];
            coord_t crd_b = x->indices[idx_b];
            index_t crd = min(crd_a, crd_b);

            if(crd == crd_a && crd == crd_b) {
                D_indices[idx_output] = crd;
                D_values[idx_output] = A->values[idx_a] + x->values[idx_b];
                idx_output++;
            } else if(crd == crd_a) {
                D_indices[idx_output] = crd;
                D_values[idx_output] = A->values[idx_a];
                idx_output++;
            } else if(crd == crd_b) {
                D_indices[idx_output] = crd;
                D_values[idx_output] = x->values[idx_b];
                idx_output++;
            }
            
            idx_a += (index_t)(crd_a == crd);
            idx_b += (index_t)(crd_b == crd);
            
        }

        while(idx_a <= end_a) {
            D_indices[idx_output] = A->col_indices[idx_a];
            D_values[idx_output] = A->values[idx_a];
            idx_output++;
            idx_a++;
        }

        while(idx_b <= end_b) {
            D_indices[idx_output] = x->indices[idx_b];
            D_values[idx_output] = x->values[idx_b];
            idx_output++;
            idx_b++;
        }


        if( idx_a == A->row_offsets[row+1] && idx_b == x->nnz) {
            D_row_offsets[row+1] = idx_output;
        }
    }

};


template void broadcast_xA_impl(SparseVector<int, int, float> x, CSRMatrix<int, float> A,
    int * & D_row_offsets, int * & D_indices, float * & D_values, int * & D_nnz);
