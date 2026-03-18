#pragma once
#include "../cuda_utils/cuda_utils.h"
#include <cstdint>
#include <iostream>
#include <sstream>
// mergepath_partition_find finds the balanced merge path partition boundaries for each thread
// 
// params:
//   num_tensors : number of input tensors
//   total_size : total number of non-zeros across all tensors
//   indices : indices[i] has indices array for the ith tensor
//   sizes : sizes[i] has number of non-zeros for each tensor
//   mergepath_boundary : (Must be preallocated) mergepath_boundary[i][t] has the output boundary in i^th tensor for the j^th thread
//                        boundary is inclusive, i.e. the last index that should be processed by that thread , -1 indicates no work for that tensor 
//   per_thread_work :  number of non-zeros each thread should process (this could be +-1 due to balancing)
template<typename index_t, typename coord_t, typename value_t>
void sparse_vectors_balanced_mergepath(
    const uint num_vectors,
    index_t * mergepath_partitions,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vectors,
    const uint num_blocks,
    const uint threads_per_block,
    const index_t per_thread_work,
    bool old
);


template<typename index_t, typename coord_t, typename value_t>
void vector_matrix_mergepath_partition(
    index_t * mergepath_partitions_matrix_row,
    index_t * mergepath_partitions_matrix_col,
    index_t * mergepath_partitions_vector,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vector,
    const CSRMatrix<index_t, value_t> * matrix,
    const uint num_blocks,
    const uint threads_per_block,
    const index_t per_thread_work
);



template<typename index_t, typename coord_t, typename value_t>
__global__ void kern_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work, bool old);


template<typename index_t, typename coord_t, typename value_t>
__global__ void kern_mergepath_partition_vector_matrix(
    index_t * mergepath_partitions_matrix_row,
    index_t * mergepath_partitions_matrix_col,
    index_t * mergepath_partitions_vector,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vector,
    const CSRMatrix<index_t, value_t> * matrix,
    const index_t per_thread_work
) ;