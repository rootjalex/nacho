#include "backend/generate_bindings.h"

#include "Error.h"
#include "Format.h"
#include "backend/output.h"

#include <fstream>
#include <map>
#include <vector>

namespace nacho {
namespace backend {

namespace {

// One buffer of the tensor, exposed to Python as an ndarray.
struct ArrayField {
    std::string name;       // e.g. "dim_j_offsets"
    std::string element;    // "int32_t" or "float"
    // Number of elements, as a C++ expression over a tensor struct named `t`. Used when
    // wrapping a kernel result, whose buffer sizes are only known from its own fields.
    std::string extent;
    // Struct field this array's length is recorded in, if any. Filled from the ndarray's
    // extent when unwrapping a Python tensor.
    std::string length_field;
};

// A Python class standing for one (format, device) pair, e.g. CSR_cpu.
struct TensorClass {
    std::string class_name;
    std::string target_tag;
    std::vector<std::string> shape_fields;  // dimension sizes, in level order
    std::vector<ArrayField> arrays;
    bool has_nnz = false;
};

struct KernelRegistration {
    std::string header;         // generated header holding the entry point
    std::string register_fn;    // registration function defined in the kernel's wrapper TU
};

std::map<std::string, TensorClass> &tensor_classes() {
    static std::map<std::string, TensorClass> classes;
    return classes;
}

std::vector<KernelRegistration> &kernel_registrations() {
    static std::vector<KernelRegistration> registrations;
    return registrations;
}

std::string value_type_name(dType dtype) {
    switch (dtype) {
        case dType::Float32: return "float";
        case dType::Float64: return "double";
        case dType::TemplateT: break;
    }
    internal_assert(false) << "Cannot bind tensors of dtype " << static_cast<int>(dtype);
    return "";
}

// Entry points are generated as <kernel>_<target>_i32_f32, so index_t is fixed here too.
const char *kIndexType = "int32_t";

std::string field_of(const std::string &field) {
    return "t." + field;
}

// Mirrors TensorLowerer::lower_tensor_struct_definition() and the result allocation
// sizes in StitchAndGenerate::generate_result_tensor_allocations(), so the Python class
// always matches the struct the kernel actually reads and writes.
TensorClass describe(const TensorLowerer &tensor, const std::string &target_tag) {
    const Format &format = tensor.type().format;
    internal_assert(format.bc_levels.empty())
        << "Cannot bind tensor '" << tensor.tensor_name << "': broadcast levels are not supported";

    TensorClass described;
    described.class_name = format_name(format) + "_" + target_tag;
    described.target_tag = target_tag;
    described.has_nnz = true;

    const std::string value_type = value_type_name(tensor.type().dtype);

    // Tracks the element count of the values buffer as levels are walked: a sparse level
    // resets it to that level's length, a dense level multiplies in its size.
    std::string values_extent = "1";

    for (size_t level = 0; level < format.levels.size(); ++level) {
        const Level &current = format.levels[level];
        const std::vector<std::string> &indices = current.index.indices;

        for (const std::string &index_name : indices) {
            described.shape_fields.push_back(tensor.get_size_field_name(TensorIndex(index_name)));
        }

        if (!is_sparse_format(current.format)) {
            values_extent = values_extent + " * " + field_of(tensor.get_size_field_name(current.index));
            continue;
        }

        // A compressed level is indexed through per-parent offsets. The outermost level has
        // no parent, so it carries coordinates only.
        if (is_compressed_format(current.format) && level > 0) {
            const Level &parent = format.levels[level - 1];
            const std::string parent_extent =
                is_sparse_format(parent.format) ? tensor.get_length_field_name(parent.index)
                                                : tensor.get_size_field_name(parent.index);
            described.arrays.push_back(ArrayField{
                .name = tensor.get_offsets_field_name(current.index),
                .element = kIndexType,
                .extent = field_of(parent_extent) + " + 1",
                .length_field = "",
            });
        }

        for (const std::string &index_name : indices) {
            const TensorIndex index{index_name};
            described.arrays.push_back(ArrayField{
                .name = tensor.get_indices_field_name(index),
                .element = kIndexType,
                .extent = field_of(tensor.get_length_field_name(index)),
                .length_field = tensor.get_length_field_name(index),
            });
            values_extent = field_of(tensor.get_length_field_name(index));
        }
    }

    described.arrays.push_back(ArrayField{
        .name = tensor.get_values_field_name(),
        .element = value_type,
        .extent = values_extent,
        .length_field = described.has_nnz ? "nnz" : "",
    });

    return described;
}

void record(const TensorClass &described) {
    auto [entry, inserted] = tensor_classes().emplace(described.class_name, described);
    if (inserted) {
        return;
    }
    // Two formats sharing a name would collide in the module; catch it here rather than
    // at C++ compile time.
    internal_assert(entry->second.arrays.size() == described.arrays.size() &&
                    entry->second.shape_fields.size() == described.shape_fields.size())
        << "Two different layouts both map to the Python class '" << described.class_name
        << "'. Give them distinct names via Format::named().";
}

std::string array_alias(const TensorClass &described, const std::string &element) {
    const std::string container = described.target_tag == "cpu" ? "ArrayCPU" : "ArrayGPU";
    return "nacho::" + container + "<" + element + ">";
}

void write_tensor_class(std::ostream &os, const TensorClass &described) {
    os << "struct " << described.class_name << " {\n";
    for (const ArrayField &array : described.arrays) {
        os << "    " << array_alias(described, array.element) << " " << array.name << ";\n";
    }
    os << "    nacho::ShapeTuple<" << described.shape_fields.size() << "> shape;\n";
    os << "};\n\n";

    // Templated on the kernel's own struct: every kernel names its tensor structs after
    // its operands (a_tensor_format, Z_tensor_format), but the layout is the same.
    os << "template <typename TensorStruct>\n";
    os << "TensorStruct unwrap_" << described.class_name << "(const " << described.class_name << " &v) {\n";
    os << "    TensorStruct t = {};\n";
    for (size_t i = 0; i < described.shape_fields.size(); ++i) {
        os << "    t." << described.shape_fields[i] << " = v.shape.data()[" << i << "];\n";
    }
    for (const ArrayField &array : described.arrays) {
        os << "    t." << array.name << " = nacho::borrow<" << array.element << ">(v." << array.name << ");\n";
        if (!array.length_field.empty()) {
            os << "    t." << array.length_field << " = (" << kIndexType << ")v." << array.name << ".shape(0);\n";
        }
    }
    os << "    return t;\n";
    os << "}\n\n";

    const std::string adopt = described.target_tag == "cpu" ? "nacho::adopt_cpu" : "nacho::adopt_gpu";
    os << "template <typename TensorStruct>\n";
    os << described.class_name << " wrap_" << described.class_name << "(const TensorStruct &t) {\n";
    os << "    return " << described.class_name << "{\n";
    for (const ArrayField &array : described.arrays) {
        os << "        " << adopt << "<" << array.element << ">(t." << array.name
           << ", (size_t)(" << array.extent << ")),\n";
    }
    os << "        nacho::make_shape<" << described.shape_fields.size() << ">({";
    for (size_t i = 0; i < described.shape_fields.size(); ++i) {
        os << (i ? ", " : "") << "t." << described.shape_fields[i];
    }
    os << "}),\n";
    os << "    };\n";
    os << "}\n\n";
}

void write_tensor_class_registration(std::ostream &os, const TensorClass &described) {
    os << "    nb::class_<" << described.class_name << ">(m, \"" << described.class_name << "\")\n";
    os << "        .def(nb::init<";
    for (const ArrayField &array : described.arrays) {
        os << "const " << array_alias(described, array.element) << " &, ";
    }
    os << "const nacho::ShapeTuple<" << described.shape_fields.size() << "> &>())\n";
    for (const ArrayField &array : described.arrays) {
        os << "        .def_ro(\"" << array.name << "\", &" << described.class_name << "::" << array.name
           << ", nb::rv_policy::reference)\n";
    }
    os << "        .def_ro(\"shape\", &" << described.class_name << "::shape, nb::rv_policy::reference);\n\n";
}

std::ofstream open_generated(const std::string &filename) {
    std::ofstream file(output_directory() + "/" + filename);
    internal_assert(file.is_open())
        << "Could not open '" << output_directory() << "/" << filename << "' for writing";
    return file;
}

} // namespace

void generate_bindings(const BindingSpec &spec) {
    const std::string header = spec.kernel_name + "_" + spec.target_tag + ".h";
    const std::string entry_point = spec.kernel_name + "::" + spec.kernel_name + "_" + spec.target_tag + "_i32_f32";
    const std::string register_fn = "register_" + spec.kernel_name + "_" + spec.target_tag;

    const TensorClass result_class = describe(spec.result_tensor, spec.target_tag);
    record(result_class);

    std::vector<std::pair<std::string, TensorClass>> operands;  // parameter name -> class
    for (const auto &[operand_name, operand] : spec.operand_tensors) {
        TensorClass operand_class = describe(operand, spec.target_tag);
        record(operand_class);
        operands.emplace_back(operand_name, std::move(operand_class));
    }

    // The wrapper includes the kernel header, so a GPU wrapper carries __global__ and
    // __device__ declarations and has to be compiled as CUDA.
    const std::string wrapper_suffix = spec.is_cpu ? "_nb.cpp" : "_nb.cu";
    std::ofstream out = open_generated(spec.kernel_name + "_" + spec.target_tag + wrapper_suffix);
    out << "#include \"nacho_types.h\"\n";
    out << "#include \"" << header << "\"\n\n";
    out << "namespace {\n\n";

    out << result_class.class_name << " " << spec.python_name << "(";
    for (size_t i = 0; i < operands.size(); ++i) {
        out << (i ? ", " : "") << "const " << operands[i].second.class_name << " &" << operands[i].first;
    }
    if (!spec.is_cpu) {
        out << (operands.empty() ? "" : ", ") << "int32_t num_blocks, int32_t threads_per_block";
    }
    out << ") {\n";

    out << "    auto result = " << entry_point << "(\n";
    for (size_t i = 0; i < operands.size(); ++i) {
        const auto &[operand_name, operand_class] = operands[i];
        const TensorLowerer &operand = spec.operand_tensors.at(operand_name);
        out << "        unwrap_" << operand_class.class_name << "<" << spec.kernel_name
            << "::" << operand.get_struct_name() << "<int32_t, "
            << value_type_name(operand.type().dtype) << ">>(" << operand_name << ")";
        const bool last = (i + 1 == operands.size()) && spec.is_cpu;
        out << (last ? "" : ",") << "\n";
    }
    if (!spec.is_cpu) {
        out << "        num_blocks,\n        threads_per_block\n";
    }
    out << "    );\n";
    out << "    return wrap_" << result_class.class_name << "(result);\n";
    out << "}\n\n";
    out << "} // namespace\n\n";

    out << "void " << register_fn << "(nb::module_ &m) {\n";
    out << "    m.def(\"" << spec.python_name << "\", &" << spec.python_name;
    if (!spec.is_cpu) {
        out << ",\n          nb::arg(\"" ;
        for (size_t i = 0; i < operands.size(); ++i) {
            out << (i ? "\"), nb::arg(\"" : "") << operands[i].first;
        }
        out << "\"), nb::arg(\"num_blocks\") = 1024, nb::arg(\"threads_per_block\") = 256";
    }
    out << ");\n";
    out << "}\n";
    out.close();

    kernel_registrations().push_back(KernelRegistration{header, register_fn});
}

void finalize_bindings() {
    if (kernel_registrations().empty()) {
        return;
    }

    std::ofstream types = open_generated("nacho_types.h");
    types << "#pragma once\n\n";
    types << "#include \"nacho_nb.h\"\n\n";
    for (const auto &[class_name, described] : tensor_classes()) {
        write_tensor_class(types, described);
    }
    types << "void register_nacho_types(nb::module_ &m);\n";
    types.close();

    std::ofstream types_impl = open_generated("nacho_types.cpp");
    types_impl << "#include \"nacho_types.h\"\n\n";
    types_impl << "void register_nacho_types(nb::module_ &m) {\n";
    for (const auto &[class_name, described] : tensor_classes()) {
        write_tensor_class_registration(types_impl, described);
    }
    types_impl << "}\n";
    types_impl.close();

    std::ofstream module_file = open_generated("nacho_ext.cpp");
    module_file << "#include \"nacho_types.h\"\n";
    module_file << "#include \"register_runtime.h\"\n\n";
    for (const KernelRegistration &registration : kernel_registrations()) {
        module_file << "void " << registration.register_fn << "(nb::module_ &m);\n";
    }
    module_file << "\nNB_MODULE(nacho_ext, m) {\n";
    module_file << "    nacho::runtime::register_runtime(m);\n";
    module_file << "    register_nacho_types(m);\n";
    for (const KernelRegistration &registration : kernel_registrations()) {
        module_file << "    " << registration.register_fn << "(m);\n";
    }
    module_file << "}\n";
    module_file.close();
}

} // namespace backend
} // namespace nacho
