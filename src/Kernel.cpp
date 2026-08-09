#include "Kernel.h"

#include "Compile.h"
#include "Error.h"
#include "backend/compile.h"
#include "backend/generate_bindings.h"
#include "backend/output.h"

#include <algorithm>
#include <iostream>
#include <set>

namespace nacho {

namespace {

std::set<std::string> &selected_kernels() {
    static std::set<std::string> selected;
    return selected;
}

std::vector<std::string> split(const std::string &text, char separator) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find(separator, start);
        if (end == std::string::npos) {
            end = text.size();
        }
        if (end > start) {
            parts.emplace_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
    return parts;
}

} // namespace

const char *target_tag(Target target) {
    return target == Target::CPU ? "cpu" : "gpu";
}

void set_selected_kernels(std::vector<std::string> names) {
    auto &selected = selected_kernels();
    selected.clear();
    selected.insert(names.begin(), names.end());
}

bool kernel_is_selected(const std::string &name) {
    const auto &selected = selected_kernels();
    return selected.empty() || selected.count(name) > 0;
}

Kernel::Kernel(std::string name) : name_(std::move(name)) {
    internal_assert(!name_.empty()) << "Kernel name cannot be empty";
}

Kernel &Kernel::expr(Expr expression) {
    expr_ = std::move(expression);
    return *this;
}

Kernel &Kernel::targets(std::vector<Target> targets) {
    internal_assert(!targets.empty()) << "Kernel '" << name_ << "' needs at least one target";
    targets_ = std::move(targets);
    return *this;
}

Kernel &Kernel::result_name(std::string name) {
    internal_assert(!name.empty()) << "Kernel '" << name_ << "' result name cannot be empty";
    result_name_ = std::move(name);
    return *this;
}

Kernel &Kernel::python_name(Target target, std::string name) {
    python_names_.emplace_back(target, std::move(name));
    return *this;
}

Kernel &Kernel::bindings(bool enabled) {
    bindings_ = enabled;
    return *this;
}

Kernel &Kernel::recursive_partitioning(bool enabled) {
    recursive_partitioning_ = enabled;
    return *this;
}

std::string Kernel::python_name_for(Target target) const {
    for (const auto &[override_target, override_name] : python_names_) {
        if (override_target == target) {
            return override_name;
        }
    }
    return std::string(target_tag(target)) + "_" + name_ + "_f32";
}

void Kernel::emit() {
    if (!kernel_is_selected(name_)) {
        return;
    }
    internal_assert(expr_.defined()) << "Kernel '" << name_ << "' has no expression; call .expr(...)";

    for (Target target : targets_) {
        const bool is_cpu = target == Target::CPU;
        backend::CINLowerer lowerer(compile_to_cin(expr_, result_name_), name_, is_cpu);
        lowerer.recursive_partitioning = recursive_partitioning_;
        lowerer.lower_cin();

        if (bindings_) {
            backend::generate_bindings(backend::BindingSpec{
                .kernel_name = name_,
                .target_tag = target_tag(target),
                .python_name = python_name_for(target),
                .is_cpu = is_cpu,
                .operand_tensors = lowerer.operand_tensors,
                .result_tensor = lowerer.result_tensor,
            });
        }
    }
}

bool parse_command_line(int argc, char **argv) {
    auto usage = [&]() {
        std::cerr << "usage: " << (argc > 0 ? argv[0] : "compiler")
                  << " [--kernels name[,name...]] [--output-dir DIR]\n";
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool needs_value = (arg == "--kernels" || arg == "--output-dir");
        if (needs_value && i + 1 >= argc) {
            std::cerr << "error: " << arg << " requires a value\n";
            usage();
            return false;
        }

        if (arg == "--kernels") {
            set_selected_kernels(split(argv[++i], ','));
        } else if (arg == "--output-dir") {
            backend::set_output_directory(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            usage();
            return false;
        } else {
            std::cerr << "error: unknown argument '" << arg << "'\n";
            usage();
            return false;
        }
    }
    return true;
}

} // namespace nacho
