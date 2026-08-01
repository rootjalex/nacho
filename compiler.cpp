#include "Nacho.h"

// Temporary, for make_binary_search example
#include "GeneratePartition.h"
#include "Lattice.h"
#include "Simplify.h"
#include "llir/Function.h"
#include "llir/LLIR.h"

#include "Printer.h"

#include "backend/compile.h"
#include "backend/tensor.h"

#include <chrono>
#include <fstream>
#include <iostream>

struct ScopeTimer {
    using Clock = std::chrono::high_resolution_clock;

    std::string name;
    Clock::time_point start;

    explicit ScopeTimer(std::string label = "")
        : name(std::move(label)), start(Clock::now()) {}

    ~ScopeTimer() {
        auto end = Clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        if (!name.empty()) {
            std::cerr << "[" << name << "] ";
        }
        std::cerr << "Elapsed: " << duration << " ms\n";
    }
};

using namespace nacho;

// Do a hacky thing and look-up the field names of each struct in a global
// array.
static std::map<std::string, std::string> nacho_names;

std::string get_hacky_field_names(const llir::lType &struct_type) {
    const llir::Struct_t *as_struct_t = struct_type.as<llir::Struct_t>();
    internal_assert(as_struct_t)
        << struct_type << " " << struct_type.as<llir::Generic_t>();
    std::string ret = "";
    for (const auto &name : as_struct_t->fields) {
        ret += name.first;
    }
    return ret;
}

struct GetArgs : nacho::Visitor {
    std::vector<std::pair<std::string, TensorType>> args;
    void visit(const Tensor *node) override {
        args.push_back({node->name, node->type});
    }
};

void compile_func(const std::string &name, Expr expr) {
    const auto header_file = "generated/" + name + "_utils.h";
    std::ofstream out(header_file);
    out << "#include \"utils.h\"\n\n";
    out << "namespace " << name << " {\n\n";

    // Define '__runnable__' based on compilation target.
    out << "#if defined(__CUDA_ARCH__)\n";
    out << "  #define __runnable__ __device__ inline\n";
    out << "#else\n";
    out << "#define __runnable__ inline\n";
    out << "#endif\n\n";
    const auto cin = compile_to_cin(expr);
    auto kernels = nacho::backend::CINLowerer(cin, out).lower_cin();
    out << "\n#undef __runnable__\n";
    out << "} // namespace " << name << std::endl;

    const auto gpu_src = "generated/" + name + "_cuda.cu";
    std::ofstream gpu_src_f(gpu_src);

    gpu_src_f << "#include \"" << name << "_cuda.h\"\n\n";
    gpu_src_f << "#include \"" << name << "_utils.h\"\n\n";

    // TODO: other headers?

    auto print_gpu_func = [](std::ofstream &os, const std::string &prefix,
                             const llir::Function *func) {
        Printer printer(os);
        internal_assert(!func->generics.empty());
        os << "template<";
        bool first = true;
        for (const auto &g : func->generics) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << "typename " << g;
        }
        os << ">\n";

        os << "__global__ void " << prefix << "_" << func->name << "(";

        bool has_thread_id = false;
        bool has_max_thread_id = false;

        for (size_t i = 0, e = func->args.size(); i < e; i++) {
            // TODO: if either of these are at i = 0, this prints an extra
            // comma, oops.
            if (func->args[i].name == "thread_id") {
                has_thread_id = true;
                continue;
            }
            if (func->args[i].name == "max_thread_id") {
                has_max_thread_id = true;
                continue;
            }
            if (i != 0) {
                os << ", ";
            }
            const auto &arg = func->args[i];
            if (!arg.mutating) {
                os << "const ";
            }
            arg.type.accept(&printer);
            // if (!arg.mutating && arg.type.is<llir::Ptr_t>()) {
            //     os << " __restrict__";
            // }
            os << " " << arg.name;
        }

        os << ") {\n";

        if (has_thread_id) {
            os << "  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;\n";
        }
        if (has_max_thread_id) {
            os << "  int max_thread_id = (gridDim.x * blockDim.x) - 1;\n";
        }
        os << "  " << prefix << "::" << func->name << "(";

        for (size_t i = 0, e = func->args.size(); i < e; i++) {
            if (i != 0) {
                os << ", ";
            }
            os << func->args[i].name;
        }

        os << ");\n}\n";
    };

    for (const auto &kernel : kernels) {
        const llir::Function *func = kernel.as<llir::Function>();
        internal_assert(func) << kernel;

        // const std::string &func_name = func->name;

        print_gpu_func(gpu_src_f, name, func);
    }

    // Get the arguments from the expression.
    GetArgs getter;
    expr.accept(&getter);
    auto args = std::move(getter.args);
    // Output is always "Z".
    args.push_back({"Z", expr.type()});

    // We now have the structs we need to pass as arguments.

    // Do a hacky thing and look-up the field names of each struct in a global
    // array.

    std::vector<std::pair<std::string, std::string>> nacho_arg_types;

    std::cout << "name: " << name << std::endl;
    for (const auto &arg : args) {
        auto hack = get_hacky_field_names(
            backend::TensorLowerer("", arg.second, {}, false)
                .lower_tensor_struct_definition());
        internal_assert(nacho_names.contains(hack)) << hack;
        nacho_arg_types.push_back({arg.first, nacho_names.at(hack)});
        // std::cout << " " << arg.first << " -> " <<
        // nacho_arg_types.back().second
        //           << std::endl;
    }

    auto print_args = [&](std::ofstream &os, std::string index_t,
                          std::string value_t, bool cpu) {
        bool first = true;
        for (const auto &arg : nacho_arg_types) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << arg.second;
            if (cpu) {
                os << "_cpu";
            }
            os << "<" << index_t << ", " << value_t << "> " << arg.first;
        }
    };

    auto print_arg_names = [&](std::ofstream &os) {
        bool first = true;
        for (const auto &arg : nacho_arg_types) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << arg.first;
        }
    };

    // Print exposed functions to the *_cuda.cu file.
    // TODO: generate stitching code!!
    gpu_src_f << "template<typename index_t, typename value_t>\n";
    gpu_src_f << "void " << "cuda_" << name << "(";
    print_args(gpu_src_f, "index_t", "value_t", true);
    gpu_src_f << ") { TODO: GENERATE GPU STITCH CODE }\n";

    gpu_src_f << "void " << "cuda_" << name << "_i32_f32(";
    print_args(gpu_src_f, "int32_t", "float", false);
    gpu_src_f << ") {\n    cuda_" << name << "(";
    print_arg_names(gpu_src_f);
    gpu_src_f << ");\n}\n";

    gpu_src_f << "void " << "cuda_" << name << "_i64_f32(";
    print_args(gpu_src_f, "int64_t", "float", false);
    gpu_src_f << ") {\n    cuda_" << name << "(";
    print_arg_names(gpu_src_f);
    gpu_src_f << ");\n}\n";

    // Declare in the *_cuda.h file.
    const auto gpu_header = "generated/" + name + "_cuda.h";
    std::ofstream gpu_header_f(gpu_header);
    gpu_header_f << "#pragma once\n\n";

    // TODO: need to generate the CPP data structures.
    gpu_header_f << "#include \"cpp_data_structures.h\"\n\n";
    gpu_header_f << "void " << "cuda_" << name << "_i32_f32(";
    print_args(gpu_header_f, "int32_t", "float", false);
    gpu_header_f << ");\n";

    gpu_header_f << "void " << "cuda_" << name << "_i64_f32(";
    print_args(gpu_header_f, "int64_t", "float", false);
    gpu_header_f << ");\n";

    // Now generate the CPU code.
    const auto cpu_header = "generated/" + name + "_cpu.h";
    std::ofstream cpu_header_f(cpu_header);
    cpu_header_f << "#pragma once\n\n";

    // TODO: need to generate the CPP data structures.
    cpu_header_f << "#include \"cpp_data_structures.h\"\n\n";
    cpu_header_f << "void " << "cpu_" << name << "_i32_f32(";
    print_args(cpu_header_f, "int32_t", "float", true);
    cpu_header_f << ");\n";

    cpu_header_f << "void " << "cpu_" << name << "_i64_f32(";
    print_args(cpu_header_f, "int64_t", "float", true);
    cpu_header_f << ");\n";

    const auto cpu_src = "generated/" + name + "_cpu.cpp";
    std::ofstream cpu_src_f(cpu_src);

    cpu_src_f << "#include \"" << name << "_cpu.h\"\n\n";
    cpu_src_f << "#include \"" << name << "_utils.h\"\n\n";
    cpu_src_f << "#include <tbb/parallel_for.h>\n";
    cpu_src_f << "#include <tbb/parallel_for.h>\n\n";

    cpu_src_f << "template<typename index_t, typename value_t>\n";
    cpu_src_f << "void " << "cpu_" << name << "(";
    print_args(cpu_src_f, "index_t", "value_t", true);
    cpu_src_f << ") {\n    // TODO: GENERATE CPU STITCH CODE\n\n";

    for (const auto &kernel : kernels) {
        const llir::Function *func = kernel.as<llir::Function>();
        internal_assert(func) << kernel;

        // TODO: These need to be stitched together in the right order + some
        // fillers.

        cpu_src_f
            << "    tbb::parallel_for(0, max_thread_id, [&](int thread_id) {\n";
        cpu_src_f << "        " << name << "::" << func->name << "(";

        for (size_t i = 0, e = func->args.size(); i < e; i++) {
            if (i != 0) {
                cpu_src_f << ", ";
            }
            cpu_src_f << func->args[i].name;
        }

        cpu_src_f << ");\n";
        cpu_src_f << "    });\n\n";
    }

    cpu_src_f << "}\n\n";

    cpu_src_f << "void " << "cpu_" << name << "_i32_f32(";
    print_args(cpu_src_f, "int32_t", "float", false);
    cpu_src_f << ") {\n    cpu_" << name << "(";
    print_arg_names(cpu_src_f);
    cpu_src_f << ");\n}\n";

    cpu_src_f << "void " << "cpu_" << name << "_i64_f32(";
    print_args(cpu_src_f, "int64_t", "float", false);
    cpu_src_f << ") {\n    cpu_" << name << "(";
    print_arg_names(cpu_src_f);
    cpu_src_f << ");\n}\n";
}

void test() {
    // std::cout << "Basic tests\n";
    Format csr_1 = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed_unique},
    });
    Format csc_1 = Format::ordered({
        {"j", LevelFormat::Dense},
        {"i", LevelFormat::Compressed_unique},
    });
    Format csr_2 = Format::ordered({
        {"j", LevelFormat::Dense},
        {"k", LevelFormat::Compressed_unique},
    });

    Format csr_3 = Format::ordered({
        {"i", LevelFormat::Dense},
        {"k", LevelFormat::Compressed_unique},
    });

    Format dcsr = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
    });

    Format csf = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
        {"k", LevelFormat::Compressed_unique},
    });

    Format coo = Format::ordered({
        {"i", LevelFormat::Compressed_non_unique},
        {"j", LevelFormat::Singleton_unique},
    });

    Format coo_2 = Format::ordered({
        {"j", LevelFormat::Compressed_non_unique},
        {"k", LevelFormat::Singleton_unique},
    });

    Format coo3d = Format::ordered({
        {"i", LevelFormat::Compressed_non_unique},
        {"j", LevelFormat::Singleton_non_unique},
        {"k", LevelFormat::Singleton_unique},
    });

    Format s = Format::ordered({
        {"k", LevelFormat::Compressed_unique}
    });

    Format d = Format::ordered({
        {"i", LevelFormat::Dense}
    });

    Format scalar = Format::ordered({}); // result of inner product

    auto add_ttype = [&](const TensorType &ttype, const std::string &name) {
        nacho_names[get_hacky_field_names(
            backend::TensorLowerer("", ttype, {}, false)
                .lower_tensor_struct_definition())] = name;
    };

    TensorType csr_f32 = TensorType(csr_1, dType::Float32);
    add_ttype(csr_f32, "CSR");
    TensorType csc_f32 = TensorType(csc_1, dType::Float32);
    add_ttype(csr_f32, "CSC");
    TensorType csr2_f32 = TensorType(csr_2, dType::Float32);
    add_ttype(csr2_f32, "CSR");
    TensorType csr3_f32 = TensorType(csr_3, dType::Float32);
    add_ttype(csr3_f32, "CSR");
    TensorType csf_f32 = TensorType(csf, dType::Float32);
    add_ttype(csf_f32, "CSF");
    TensorType coo_f32 = TensorType(coo, dType::Float32);
    add_ttype(coo_f32, "COO");
    TensorType coo_2_f32 = TensorType(coo_2, dType::Float32);
    add_ttype(coo_2_f32, "COO");
    TensorType coo3d_f32 = TensorType(coo3d, dType::Float32);
    add_ttype(coo3d_f32, "COO3");
    TensorType dcsr_f32 = TensorType(dcsr, dType::Float32);
    add_ttype(dcsr_f32, "DCSR");
    TensorType s_f32 = TensorType(s, dType::Float32);
    add_ttype(s_f32, "CVector");
    TensorType d_f32 = TensorType(d, dType::Float32);
    add_ttype(d_f32, "DVector");
    TensorType scalar_f32(scalar, dType::Float32);
    add_ttype(scalar_f32, "Scalar");

    Expr a_cv = Tensor::make(s_f32, "a");
    Expr b_cv = Tensor::make(s_f32, "b");
    Expr a_csr_ij = Tensor::make(csr_f32, "a");
    Expr b_csr_ij = Tensor::make(csr_f32, "b");
    Expr c_csr_ij = Tensor::make(csr_f32, "c");
    Expr a_dcsr_ij = Tensor::make(dcsr_f32, "a");
    Expr b_dcsr_ij = Tensor::make(dcsr_f32, "b");
    Expr a_csc_ji = Tensor::make(csc_f32, "a");
    Expr b_csc_ji = Tensor::make(csc_f32, "b");
    Expr a_csr_jk = Tensor::make(csr2_f32, "a");
    Expr b_csr_jk = Tensor::make(csr2_f32, "b");
    Expr a_csr_ik = Tensor::make(csr3_f32, "a");
    Expr b_csr_ik = Tensor::make(csr3_f32, "b");
    Expr a_csf_ijk = Tensor::make(csf_f32, "a");
    Expr b_csf_ijk = Tensor::make(csf_f32, "b");
    Expr a_coo_ij = Tensor::make(coo_f32, "a");
    Expr b_coo_ij = Tensor::make(coo_f32, "b");
    Expr a_coo_ijk = Tensor::make(coo3d_f32, "a");
    Expr b_coo_ijk = Tensor::make(coo3d_f32, "b");
    Expr c_csr_ik = Tensor::make(csr3_f32, "c");

    // CSR add
    {
        ScopeTimer timer("CSR add");
        auto csr_add = a_csr_ij + b_csr_ij;
        // nacho::backend::CINLowerer(compile_to_cin(csr_add), std::cout)
        //     .lower_cin();
        compile_func("csr_add", csr_add);
    }

    // CSR + CSR + CSR
    {
        ScopeTimer timer("CSR + CSR + CSR");
        auto csr_add3 = a_csr_ij + b_csr_ij + c_csr_ij;
        // nacho::backend::CINLowerer(compile_to_cin(csr_add3), std::cout)
        //     .lower_cin();
        compile_func("csr_add3", csr_add3);
    }

    {
        ScopeTimer timer("CSR mul");
        auto csr_mul = a_csr_ij * b_csr_ij;
        // nacho::backend::CINLowerer(compile_to_cin(csr_mul), std::cout)
        //     .lower_cin();
        compile_func("csr_mul", csr_mul);
    }

    {
        ScopeTimer timer("COO add");
        auto coo_add = a_coo_ij + b_coo_ij;
        // nacho::backend::CINLowerer(compile_to_cin(coo_add), std::cout)
        //     .lower_cin();
        compile_func("coo_add", coo_add);
    }

    {
        ScopeTimer timer("COO mul");
        auto coo_mul = a_coo_ij * b_coo_ij;
        // nacho::backend::CINLowerer(compile_to_cin(coo_mul), std::cout)
        //     .lower_cin();
        compile_func("coo_mul", coo_mul);
    }

    {
        ScopeTimer timer("CSR + COO");
        auto csr_add_coo = a_csr_ij + b_coo_ij;
        // nacho::backend::CINLowerer(compile_to_cin(csr_add_coo), std::cout)
        //     .lower_cin();
        compile_func("csr_add_coo", csr_add_coo);
    }

    {
        ScopeTimer timer("DCSR add");
        auto dcsr_add = a_dcsr_ij + b_dcsr_ij;
        // nacho::backend::CINLowerer(compile_to_cin(dcsr_add), std::cout)
        //     .lower_cin();
        compile_func("dcsr_add", dcsr_add);
    }

    {
        ScopeTimer timer("DCSR mul");
        auto dcsr_mul = a_dcsr_ij * b_dcsr_ij;
        // nacho::backend::CINLowerer(compile_to_cin(dcsr_mul), std::cout)
        //     .lower_cin();
        compile_func("dcsr_mul", dcsr_mul);
    }

    // CSF add
    {
        ScopeTimer timer("CSF add");
        // nacho::backend::CINLowerer(compile_to_cin(a_csf_ijk + b_csf_ijk),
        //                            std::cout)
        //     .lower_cin();
        compile_func("csf_add", a_csf_ijk + b_csf_ijk);
    }

    {
        ScopeTimer timer("SpGEMM");
        auto spgemm = Sum::make("j", a_csr_ij * b_csr_jk);
        // nacho::backend::CINLowerer(compile_to_cin(spgemm), std::cout)
        //     .lower_cin();
        compile_func("spgemm", spgemm);
    }

    {
        ScopeTimer timer("SSSMM");
        auto sssmm = Sum::make("j", a_csr_ij * b_csr_jk * c_csr_ik);
        // nacho::backend::CINLowerer(compile_to_cin(sssmm), std::cout)
        //     .lower_cin();
        compile_func("sssmm", sssmm);
    }

    {
        ScopeTimer timer("Inner Product");
        auto innerprod = Sum::make(
            "i", Sum::make("j", Sum::make("k", a_csf_ijk * b_csf_ijk)));
        // nacho::backend::CINLowerer(compile_to_cin(innerprod), std::cout)
        //     .lower_cin();
        compile_func("inner_prod", innerprod);
    }

    return;
}

void spmspv() {
    std::cout << "SpMSpV test\n";
    // A is CSR: (Dense, Compressed_unique) over (i, j)
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed_unique},
    });

    // x is a sparse vector: (Compressed_unique) over (j)
    Format sparse_vec = Format::ordered({
        {"j", LevelFormat::Compressed_unique},
    });

    TensorType csr_f32 = TensorType(csr, dType::Float32);
    TensorType svec_f32 = TensorType(sparse_vec, dType::Float32);

    Expr A_ij = Tensor::make(csr_f32, "A");
    Expr x_j = Tensor::make(svec_f32, "x");

    // z_i = sum_j(A_ij * x_j) — sparse matrix times sparse vector
    Expr z_i = sum("j", A_ij * x_j);

    std::cout << z_i << "\n";
    std::cout << "Expect dense vector: " << z_i.type().format << "\n";

    CIN cin = compile_to_cin(z_i);
    std::cout << cin << "\n";
    nacho::backend::CINLowerer(compile_to_cin(z_i), std::cout).lower_cin();
    std::cout << "\n\n";
}

void spgemm() {
    std::cout << "SpGEMM test\n";
    Format csr0 = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed_unique},
    });

    Format csr1 = Format::ordered({
        {"j", LevelFormat::Dense},
        {"k", LevelFormat::Compressed_unique},
    });

    TensorType csr0_f32 = TensorType(csr0, dType::Float32);
    TensorType csr1_f32 = TensorType(csr1, dType::Float32);

    Expr a_ij = Tensor::make(csr0_f32, "a");
    Expr b_jk = Tensor::make(csr1_f32, "b");

    // Implicit broadcasting.
    Expr z_ik = sum("j", a_ij * b_jk);

    std::cout << z_ik << "\n";
    std::cout << "Expect CSR: " << z_ik.type().format << "\n";

    CIN cin = compile_to_cin(z_ik);
    nacho::backend::CINLowerer(compile_to_cin(z_ik), std::cout).lower_cin();
    std::cout << cin << "\n";
    std::cout << "\n\n";

}

void sss_s_s(){
    Format sss = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
        {"l", LevelFormat::Compressed_unique},
        {"m", LevelFormat::Dense},
    });
    Format s1 = Format::ordered({
        {"j", LevelFormat::Compressed_unique},
    });
    Format s2 = Format::ordered({
        {"k", LevelFormat::Compressed_unique},
    });
    Format s3 = Format::ordered({
        {"o", LevelFormat::Compressed_unique},
    });
    
    TensorType sss_f32 = TensorType(sss, dType::Float32);
    TensorType s1_f32 = TensorType(s1, dType::Float32);
    TensorType s2_f32 = TensorType(s2, dType::Float32);
    TensorType s3_f32 = TensorType(s3, dType::Float32);

    Expr a_ijlm = Tensor::make(sss_f32, "a");
    Expr b_j = Tensor::make(s1_f32, "b");
    Expr c_k = Tensor::make(s2_f32, "c");
    Expr d_o = Tensor::make(s3_f32, "d");
    Expr z_ijk = a_ijlm *  b_j * c_k * d_o;

    CIN cin = compile_to_cin(z_ijk);
    std::cout << cin << "\n";
    std::cout << "\n\n";
    nacho::backend::CINLowerer(compile_to_cin(z_ijk), std::cout).lower_cin();
}

void test_format_inf() {
    std::cout << "Format inference test\n";
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed_unique},
    });

    Format dcsr = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
    });

    TensorType csr_f32 = TensorType(csr, dType::Float32);
    TensorType dcsr_f32 = TensorType(dcsr, dType::Float32);

    Expr a_ij = Tensor::make(csr_f32, "a");
    Expr b_ij = Tensor::make(dcsr_f32, "b");

    std::cout << "Expect CSR: " << (a_ij + b_ij).type().format << "\n";
    std::cout << compile_to_cin(a_ij + b_ij) << "\n";
    std::cout << "Expect DCSR: " << (a_ij * b_ij).type().format << "\n";
    std::cout << compile_to_cin(a_ij * b_ij) << "\n";

    std::cout << "Expect [DC]{D}: "
              << (bc("k", a_ij) + bc("k", b_ij)).type().format << "\n";
    std::cout << "Expect [DC]{D}: "
              << (bc("k", a_ij) * bc("k", b_ij)).type().format << "\n";

    std::cout << "\n\n";
}

void test_lattice() {
    std::cout << "Lattice test\n";
    Format sparse = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
    });

    Format dense = Format::ordered({
        {"i", LevelFormat::Dense},
    });

    TensorType sparse_f32 = TensorType(sparse, dType::Float32);
    TensorType dense_f32 = TensorType(dense, dType::Float32);

    Seq i_a = Index::make("a", sparse_f32, 0);
    Seq i_b = Index::make("b", sparse_f32, 0);
    Seq i_c = Index::make("c", sparse_f32, 0);
    Seq i_d = Index::make("d", dense_f32, 0);

    std::cout << "ALL SPARSE LATTICE TESTS\n";
    {
        Seq seq = Union::make(i_a, i_b);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    {
        Seq seq = Union::make(i_a, i_b);
        seq = Union::make(seq, i_c);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    {
        Seq seq = Union::make(i_a, i_b);
        seq = Intersect::make(seq, i_c);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    {
        Seq seq = Intersect::make(i_a, i_b);
        seq = Union::make(seq, i_c);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    {
        Seq seq = Intersect::make(i_a, i_b);
        seq = Intersect::make(seq, i_c);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    std::cout << "ONE DENSE LATTICE TESTS\n";
    {
        Seq seq = Intersect::make(i_a, i_d);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }
    {
        Seq seq = Union::make(i_a, i_d);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }
    {
        Seq seq = Union::make(i_a, i_b);
        seq = Union::make(seq, i_d);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    {
        Seq seq = Union::make(i_a, i_b);
        seq = Intersect::make(seq, i_d);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    {
        Seq seq = Intersect::make(i_a, i_d);
        seq = Union::make(seq, i_c);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }

    {
        Seq seq = Intersect::make(i_a, i_d);
        seq = Intersect::make(seq, i_c);
        Lattice lattice = Lattice::build(seq);
        lattice.dump(std::cout);
    }
}

void test_locator_optimization() {
    std::cout << "Locator test\n";
    Format sparse = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
    });

    Format dense = Format::ordered({
        {"i", LevelFormat::Dense},
    });

    TensorType sparse_f32 = TensorType(sparse, dType::Float32);
    TensorType dense_f32 = TensorType(dense, dType::Float32);

    Seq i_a = Index::make("a", sparse_f32, 0);
    Seq i_b = Index::make("b", sparse_f32, 0);
    Seq i_c = Index::make("c", sparse_f32, 0);
    Seq i_d = Index::make("d", dense_f32, 0);
    Seq i_e = Index::make("e", dense_f32, 0);

    std::cout << "a, b, c are sparse, d, e are dense.\n";

    auto print_list = [](std::ostream &os, const std::vector<Seq> &seqs) {
        bool first = true;
        os << "{";
        for (const auto &s : seqs) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << s;
        }
        os << "}";
    };

    auto check = [&print_list](const Seq &seq) {
        std::cout << "\n";
        auto [iters, locs, has_universe_iter] = partition_iterators_locators(seq);
        std::cout << seq << " -> iterators: ";
        print_list(std::cout, iters);
        std::cout << " with locators: ";
        print_list(std::cout, locs);
        std::cout << " (has universe iter: " << has_universe_iter << ")" << std::endl;
    };

    Seq seq = Union::make(i_a, i_b);
    check(seq);

    seq = Union::make(seq, i_c);
    check(seq);

    seq = Union::make(i_d, i_e);
    check(seq);

    seq = Union::make(i_a, i_d);
    check(seq);

    seq = Intersect::make(i_d, i_e);
    check(seq);

    seq = Intersect::make(i_a, i_d);
    check(seq);

    seq = Union::make(i_a, i_d);
    seq = Intersect::make(seq, i_b);
    check(seq);

    seq = Union::make(i_e, i_d);
    seq = Intersect::make(seq, i_b);
    check(seq);
}

// void make_binary_search() {
//     llir::Function func;
//     func.generics.push_back("index_t");

//     func.attributes.push_back(llir::Function::device);
//     func.attributes.push_back(llir::Function::inline_);

//     llir::lType index_t = llir::Generic_t::make("index_t");
//     func.ret_type = index_t;

//     func.name = "lower_bound";

//     llir::lType ptr_index_t = llir::Ptr_t::make(index_t);
//     func.args.push_back(
//         {.mutating = false, .type = ptr_index_t, .name = "crds"});
//     func.args.push_back({.mutating = false, .type = index_t, .name = "n"});
//     func.args.push_back({.mutating = false, .type = index_t, .name = "c"});

//     {
//         llir::lExpr crds = llir::lVar::make(ptr_index_t, "crds");
//         llir::lExpr n = llir::lVar::make(index_t, "n");
//         llir::lExpr c = llir::lVar::make(index_t, "c");

//         std::vector<llir::lStmt> stmts;

//         // index_t low = 0;
//         stmts.push_back(
//             llir::Declare::make(index_t, "low", llir::lConst::make(0)));
//         llir::lExpr low = llir::lVar::make(index_t, "low");
//         // index_t high = n;
//         stmts.push_back(llir::Declare::make(index_t, "high", n));
//         llir::lExpr high = llir::lVar::make(index_t, "high");

//         std::vector<llir::lStmt> while_body;

//         while_body.push_back(
//             // index_t mid = low + (high - low) / 2;
//             llir::Declare::make(index_t, "mid",
//                                 low + (high - low) / llir::lConst::make(2)));
//         llir::lExpr mid = llir::lVar::make(index_t, "mid");

//         while_body.push_back(
//             // if (a->indices[mid] <= crd) {
//             //   low = mid + 1;
//             // } else {
//             //   high = mid
//             // }
//             llir::IfElse::make(
//                 crds[mid] <= c,
//                 llir::Store::make("low", mid + llir::lConst::make(1)),
//                 llir::Store::make("high", mid)));

//         // while (low < high) { ... }
//         llir::lStmt while_loop = llir::While::make(
//             low < high, llir::Sequence::make(std::move(while_body)));

//         stmts.push_back(std::move(while_loop));

//         // return low;
//         stmts.push_back(llir::Return::make(low));

//         func.body = llir::Sequence::make(std::move(stmts));
//     }

//     func.print(std::cout);
// }

// void make_binary_partition() {
//     auto funcs = generate_nary_1d_partition({"a", "b"});
//     assert(funcs.size() == 1);
//     funcs[0].print(std::cout);
// }

// TODO: write a parser.
int main(int argc, char **argv) {
    test();
    // test_format_inf();
    // test_vec();
    // spgemm();
    // spmspv();
    // sss_s_s();
    // make_binary_search();
    // make_binary_partition();
    // test_lattice();
    // test_locator_optimization();

    return 0;
}
