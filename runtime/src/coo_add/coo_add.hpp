#pragma once

#include "coo_add.h"
#include "../nb_utils.hpp"

COO<int32_t, float> nb_gpu_coo_add_f32(COO<int32_t, float> A, COO<int32_t, float> B);