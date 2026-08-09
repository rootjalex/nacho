#pragma once

#include "nacho_nb.h"

namespace nacho {
namespace runtime {

// Registers everything hand-written in the module: the matrix/tensor file readers and
// the comparison baselines. The generated module file calls this before registering the
// generated tensor classes and kernels.
void register_runtime(nb::module_ &m);

} // namespace runtime
} // namespace nacho
