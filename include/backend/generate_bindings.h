#pragma once

#include "backend/tensor.h"

#include <map>
#include <string>

namespace nacho {
namespace backend {

// Everything the binding generator needs about one generated kernel variant.
struct BindingSpec {
    std::string kernel_name;   // e.g. "csr_mul" — also the generated namespace
    std::string target_tag;    // "cpu" or "gpu"
    std::string python_name;   // e.g. "cpu_csr_mul_f32"
    bool is_cpu;
    std::map<std::string, TensorLowerer> operand_tensors;
    TensorLowerer result_tensor;
};

// Writes "<kernel>_<target>_nb.cpp": a wrapper taking/returning the generated Python
// tensor classes, plus a registration function for it. Records the tensor classes it
// needs so finalize_bindings() can emit them once.
void generate_bindings(const BindingSpec &spec);

// Writes the files that depend on the whole set of kernels generated this run:
// "nacho_types.h"/".cpp" (one Python class per distinct format and device) and
// "nacho_ext.cpp" (the module, registering every kernel). Call once, after the last
// generate_bindings(). A no-op if no bindings were generated.
void finalize_bindings();

} // namespace backend
} // namespace nacho
