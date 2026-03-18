#pragma once

#include "sparse_vector.h"
#include "sp_ab_c.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include "../nb_utils.hpp"

template<typename index_t, typename coord_t, typename value_t>
CVector<index_t, coord_t, value_t> nb_3dmergepath_test(CVector<index_t, coord_t, value_t> A, CVector<index_t, coord_t, value_t> B, CVector<index_t, coord_t, value_t> C, int num_fused,  bool expr);


