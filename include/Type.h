#pragma once

#include "Format.h"

namespace nacho {

enum class dType {
    Float32,
    Float64,
    TemplateT,
};

struct TensorType {
    Format format;
    dType dtype;

    TensorType(Format format, dType dtype)
        : format(std::move(format)), dtype(std::move(dtype)) {}

    TensorType() : dtype(dType::TemplateT) {}
};

} // namespace nacho
