#pragma once

#include "csr_add.h"
#include "../nb_utils.hpp"

CSR<int32_t, float> nb_gpu_csr_add_f32(CSR<int32_t, float> A, CSR<int32_t, float> B, bool useCusparse);