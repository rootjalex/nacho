#pragma once
#include <cstdint>
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"

template<typename index_t, typename coord_t, typename value_t>
void sparse_vector_fusion_test(const SparseVector<index_t, coord_t, value_t> A, const SparseVector<index_t, coord_t, value_t> B, const SparseVector<index_t, coord_t, value_t> C, 
    coord_t * & D_indices, value_t * & D_values, index_t * & D_nnz, int num_fused, value_t * & D_times);