#pragma once

#include "Frontend.h"

#include <string>
#include <vector>

namespace nacho {

enum class Target {
    CPU,
    GPU,
};

// "cpu" / "gpu" — the infix used in generated file names, entry points and Python names.
const char *target_tag(Target target);

// Declares one kernel: an expression, the targets to generate it for, and the
// Python names to expose it under.
//
//     Kernel("csr_mul")
//         .expr(a_csr_ij * b_csr_ij)
//         .targets({Target::CPU, Target::GPU})
//         .emit();
//
// Everything but the name and expression has a default, so the common case is
// three lines. Operand order in both the C++ signature and the Python wrapper is
// lexicographic by tensor name, so name operands a, b, c to match expression order.
class Kernel {
public:
    explicit Kernel(std::string name);

    Kernel &expr(Expr expression);

    // Defaults to {Target::CPU, Target::GPU}.
    Kernel &targets(std::vector<Target> targets);

    // Name of the result tensor, and so of its generated struct. Defaults to "Z".
    Kernel &result_name(std::string name);

    // Overrides the Python function name for one target.
    // Defaults to "<cpu|gpu>_<kernel name>_f32".
    Kernel &python_name(Target target, std::string name);

    // Whether to generate nanobind bindings alongside the kernel. Defaults to true.
    Kernel &bindings(bool enabled);

    // Generates the kernel for each target. A no-op if this kernel was excluded by
    // set_selected_kernels().
    void emit();

    std::string python_name_for(Target target) const;

private:
    std::string name_;
    Expr expr_;
    std::vector<Target> targets_{Target::CPU, Target::GPU};
    std::string result_name_ = "Z";
    std::vector<std::pair<Target, std::string>> python_names_;
    bool bindings_ = true;
};

// Restricts which kernels emit(). An empty list means "all", which is the default.
void set_selected_kernels(std::vector<std::string> names);
bool kernel_is_selected(const std::string &name);

// Handles the compiler's command line:
//   --kernels csr_mul,dcsr_mul   generate only these (default: all)
//   --output-dir DIR             where to write generated sources (default: generated)
// Returns false if the arguments were malformed, after printing usage.
bool parse_command_line(int argc, char **argv);

} // namespace nacho
