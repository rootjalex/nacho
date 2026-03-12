#pragma once

#include "spgemm.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include "../nb_utils.hpp"

template<typename index_t, typename value_t>
CSR<index_t, value_t> spgemm(CSR<index_t, value_t> A, CSR<index_t, value_t> B, bool use_cusparse);


