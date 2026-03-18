#pragma once
#include <cstdint>
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"

template<typename index_t, typename coord_t, typename value_t>
void broadcast_xA_impl(SparseVector<index_t, coord_t, value_t> x, CSRMatrix<index_t, value_t> A,
    index_t * & D_row_offsets, index_t * & D_indices, value_t * & D_values, index_t * & D_nnz);