#pragma once

// Tensor classes for the hand-written comparison kernels.

#include "nacho_nb.h"

namespace nacho {
namespace baselines {

// Compressed sparse rows: per-row offsets into the column indices, then the values.
template <template <typename> class Array>
struct CSR {
    Array<int32_t> indptr;
    Array<int32_t> indices;
    Array<float> values;
    ShapeTuple<2> shape;
};

using CSRCpu = CSR<ArrayCPU>;
using CSRGpu = CSR<ArrayGPU>;

void register_baseline_types(nb::module_ &m);

} // namespace baselines
} // namespace nacho
