#pragma once

#include "Format.h"

namespace nacho {

enum class dType {
    Float32,
    Float64,
};

struct TensorType {
    Format format;
    dType dtype;
};

} // namespace nacho
