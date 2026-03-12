#pragma once

#include "broadcasts.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include "../nb_utils.hpp"

template<typename index_t, typename coord_t, typename value_t>
CSR<index_t, value_t> broadcast_xA(CVector<index_t, coord_t, value_t> x, CSR<index_t, value_t> A);


