#include "Nacho.h"

#include "CPUPrinter.h"
#include "Lattice.h"
#include "Simplify.h"
#include "backend/compile.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>

using namespace nacho;

// ---------------------------------------------------------------------------
// Helper: full pipeline from expression to generated CUDA code.
// ---------------------------------------------------------------------------
static void compile_and_lower(const std::string &name, Expr expr,
                               const std::string &target = "gpu") {
    std::cout << "=== " << name << " ===\n";
    std::cout << "Expression: " << expr << "\n";
    std::cout << "Result format: " << expr.type().format << "\n\n";

    CIN cin = compile_to_cin(expr);
    std::cout << "CIN:\n" << cin << "\n\n";

    if (target == "cpu") {
        std::cout << "Generated CPU C++:\n";
        CPUPrinter printer(std::cout);
        backend::CINLowerer(cin, printer).lower_cin();
    } else {
        std::cout << "Generated CUDA:\n";
        Printer printer(std::cout);
        backend::CINLowerer(cin, printer).lower_cin();
    }
    std::cout << "\n";
}

// ===========================================================================
// 1D Sparse Vector Tests
// ===========================================================================

// a ∩ b  (element-wise multiply, single phase)
void test_sparse_vec_mul() {
    Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
    TensorType s_f32(s, dType::Float32);

    Expr a = Tensor::make(s_f32, "a");
    Expr b = Tensor::make(s_f32, "b");
    compile_and_lower("Sparse Vector Mul: a*b", a * b);
}

// a ∪ b  (element-wise add, single phase)
void test_sparse_vec_add() {
    Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
    TensorType s_f32(s, dType::Float32);

    Expr a = Tensor::make(s_f32, "a");
    Expr b = Tensor::make(s_f32, "b");
    compile_and_lower("Sparse Vector Add: a+b", a + b);
}

// Runtime parity: runtime/src/sparse_vector/sparse_vector.cu
// (A+B)*C — union then intersect, fused 3-operand expression.
void test_sparse_vec_apb_c() {
    Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
    TensorType s_f32(s, dType::Float32);

    Expr a = Tensor::make(s_f32, "a");
    Expr b = Tensor::make(s_f32, "b");
    Expr c = Tensor::make(s_f32, "c");
    compile_and_lower("Sparse Vector (a+b)*c  [runtime: sparse_vector.cu]",
                      (a + b) * c);
}

// Runtime parity: runtime/src/sparse_vector/sp_ab_c.cu
// (A*B)+C — intersect then union, fused 3-operand expression.
void test_sparse_vec_ab_pc() {
    Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
    TensorType s_f32(s, dType::Float32);

    Expr a = Tensor::make(s_f32, "a");
    Expr b = Tensor::make(s_f32, "b");
    Expr c = Tensor::make(s_f32, "c");
    compile_and_lower("Sparse Vector (a*b)+c  [runtime: sp_ab_c.cu]",
                      (a * b) + c);
}

// ===========================================================================
// 2D Matrix Tests (DCSR — all dimensions sparse)
// ===========================================================================

// Element-wise multiply: A ∩ B  (multi-phase: i then j)
void test_dcsr_mul() {
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
    TensorType dcsr_f32(dcsr, dType::Float32);

    Expr a = Tensor::make(dcsr_f32, "A");
    Expr b = Tensor::make(dcsr_f32, "B");
    compile_and_lower("DCSR Mul: A*B", a * b);
}

// Element-wise add: A ∪ B  (multi-phase: i then j)
// Runtime parity: runtime/src/csr_add/csr_add.cu (DCSR variant)
void test_dcsr_add() {
    Format dcsr = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
    });
    TensorType dcsr_f32(dcsr, dType::Float32);

    Expr a = Tensor::make(dcsr_f32, "A");
    Expr b = Tensor::make(dcsr_f32, "B");
    compile_and_lower("DCSR Add: A+B  [runtime: csr_add.cu]", a + b);
}

// Element-wise add: A ∪ B  (dense i, sparse j)
void test_csr_add() {
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed_unique},
    });
    TensorType csr_f32(csr, dType::Float32);

    Expr a = Tensor::make(csr_f32, "A");
    Expr b = Tensor::make(csr_f32, "B");
    compile_and_lower("CSR Add: A+B  [generated]", a + b);
}

// ===========================================================================
// 3D Tensor Tests (TCSF — all dimensions sparse)
// ===========================================================================

// Element-wise add: A ∪ B over (i,j,k), all compressed.
// This exercises non-innermost sparse row/plane completion guards.
void test_tcsf_add() {
    Format tcsf = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
        {"k", LevelFormat::Compressed_unique},
    });
    TensorType tcsf_f32(tcsf, dType::Float32);

    Expr a = Tensor::make(tcsf_f32, "A");
    Expr b = Tensor::make(tcsf_f32, "B");
    compile_and_lower("TCSF Add: A+B  [3D generated]", a + b);
}

// ===========================================================================
// Format Inference Tests
// ===========================================================================

void test_format_inference() {
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed_unique},
    });
    Format dcsr = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
    });
    Format sparse = Format::ordered({{"i", LevelFormat::Compressed_unique}});
    Format dense = Format::ordered({{"i", LevelFormat::Dense}});

    TensorType csr_f32(csr, dType::Float32);
    TensorType dcsr_f32(dcsr, dType::Float32);
    TensorType s_f32(sparse, dType::Float32);
    TensorType d_f32(dense, dType::Float32);

    Expr a_ij = Tensor::make(csr_f32, "a");
    Expr b_ij = Tensor::make(dcsr_f32, "b");
    Expr a_i = Tensor::make(s_f32, "a_vec");
    Expr b_i = Tensor::make(s_f32, "b_vec");
    Expr c_i = Tensor::make(d_f32, "c_vec");
    Expr d_i = Tensor::make(d_f32, "d_vec");

    std::cout << "=== Format Inference ===\n";

    // Matrix format inference
    std::cout << "CSR + DCSR  -> " << (a_ij + b_ij).type().format
              << "  (expect Dense,Compressed)\n";
    std::cout << "CSR * DCSR  -> " << (a_ij * b_ij).type().format
              << "  (expect Compressed,Compressed)\n";

    // Vector format inference
    std::cout << "sparse * sparse -> " << (a_i * b_i).type().format
              << "  (expect Compressed)\n";
    std::cout << "sparse + sparse -> " << (a_i + b_i).type().format
              << "  (expect Compressed)\n";
    std::cout << "sparse * dense  -> " << (a_i * c_i).type().format
              << "  (expect Compressed)\n";
    std::cout << "sparse + dense  -> " << (a_i + c_i).type().format
              << "  (expect Dense)\n";
    std::cout << "dense  * dense  -> " << (c_i * d_i).type().format
              << "  (expect Dense)\n";
    std::cout << "dense  + dense  -> " << (c_i + d_i).type().format
              << "  (expect Dense)\n";

    // Broadcast format inference
    std::cout << "bc(k, CSR) + bc(k, DCSR) -> "
              << (bc("k", a_ij) + bc("k", b_ij)).type().format
              << "  (expect [DC]{D})\n";

    std::cout << "\n";
}

// ===========================================================================
// Lattice Tests
// ===========================================================================

void test_lattice() {
    Format sparse = Format::ordered({{"i", LevelFormat::Compressed_unique}});
    Format dense = Format::ordered({{"i", LevelFormat::Dense}});

    TensorType sparse_f32(sparse, dType::Float32);
    TensorType dense_f32(dense, dType::Float32);

    Seq i_a = Index::make("a", sparse_f32, 0);
    Seq i_b = Index::make("b", sparse_f32, 0);
    Seq i_c = Index::make("c", sparse_f32, 0);
    Seq i_d = Index::make("d", dense_f32, 0);

    std::cout << "=== Lattice Tests ===\n";

    auto dump = [](const char *label, Seq seq) {
        std::cout << label << ": " << seq << "\n";
        Lattice::build(seq).dump(std::cout);
    };

    // All sparse
    dump("a ∪ b", Union::make(i_a, i_b));
    dump("a ∪ b ∪ c", Union::make(Union::make(i_a, i_b), i_c));
    dump("(a ∪ b) ∩ c", Intersect::make(Union::make(i_a, i_b), i_c));
    dump("(a ∩ b) ∪ c", Union::make(Intersect::make(i_a, i_b), i_c));
    dump("a ∩ b ∩ c", Intersect::make(Intersect::make(i_a, i_b), i_c));

    // One dense
    dump("a ∩ d", Intersect::make(i_a, i_d));
    dump("a ∪ d", Union::make(i_a, i_d));
    dump("a ∪ b ∪ d", Union::make(Union::make(i_a, i_b), i_d));
    dump("(a ∪ b) ∩ d", Intersect::make(Union::make(i_a, i_b), i_d));
    dump("(a ∩ d) ∪ c", Union::make(Intersect::make(i_a, i_d), i_c));
    dump("(a ∩ d) ∩ c", Intersect::make(Intersect::make(i_a, i_d), i_c));

    std::cout << "\n";
}

// ===========================================================================
// Locator Optimization Tests
// ===========================================================================

void test_locator_optimization() {
    Format sparse = Format::ordered({{"i", LevelFormat::Compressed_unique}});
    Format dense = Format::ordered({{"i", LevelFormat::Dense}});

    TensorType sparse_f32(sparse, dType::Float32);
    TensorType dense_f32(dense, dType::Float32);

    Seq i_a = Index::make("a", sparse_f32, 0);
    Seq i_b = Index::make("b", sparse_f32, 0);
    Seq i_c = Index::make("c", sparse_f32, 0);
    Seq i_d = Index::make("d", dense_f32, 0);
    Seq i_e = Index::make("e", dense_f32, 0);

    std::cout << "=== Locator Optimization Tests ===\n";
    std::cout << "a, b, c are sparse; d, e are dense.\n";

    auto check = [](Seq seq) {
        auto [iters, locs, has_universe_iter] = partition_iterators_locators(seq);
        std::cout << seq << " -> iterators: {";
        for (size_t i = 0; i < iters.size(); ++i)
            std::cout << (i ? ", " : "") << iters[i];
        std::cout << "} locators: {";
        for (size_t i = 0; i < locs.size(); ++i)
            std::cout << (i ? ", " : "") << locs[i];
        std::cout << "}\n";
    };

    check(Union::make(i_a, i_b));
    check(Union::make(Union::make(i_a, i_b), i_c));
    check(Union::make(i_d, i_e));
    check(Union::make(i_a, i_d));
    check(Intersect::make(i_d, i_e));
    check(Intersect::make(i_a, i_d));
    check(Intersect::make(Union::make(i_a, i_d), i_b));
    check(Intersect::make(Union::make(i_e, i_d), i_b));

    std::cout << "\n";
}

// ===========================================================================
// Expression Registry (for --emit mode)
// ===========================================================================

using ExprBuilder = std::function<Expr()>;

// clang-format off
const std::map<std::string, ExprBuilder> EXPRESSIONS = {
    {"sparse_vec_mul", []() {
        Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
        TensorType s_f32(s, dType::Float32);
        return Tensor::make(s_f32, "a") * Tensor::make(s_f32, "b");
    }},
    {"sparse_vec_add", []() {
        Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
        TensorType s_f32(s, dType::Float32);
        return Tensor::make(s_f32, "a") + Tensor::make(s_f32, "b");
    }},
    {"sparse_vec_apb_c", []() {
        Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
        TensorType s_f32(s, dType::Float32);
        Expr a = Tensor::make(s_f32, "a");
        Expr b = Tensor::make(s_f32, "b");
        Expr c = Tensor::make(s_f32, "c");
        return (a + b) * c;
    }},
    {"sparse_vec_ab_pc", []() {
        Format s = Format::ordered({{"i", LevelFormat::Compressed_unique}});
        TensorType s_f32(s, dType::Float32);
        Expr a = Tensor::make(s_f32, "a");
        Expr b = Tensor::make(s_f32, "b");
        Expr c = Tensor::make(s_f32, "c");
        return (a * b) + c;
    }},
    {"dcsr_mul", []() {
        Format dcsr = Format::ordered({
            {"i", LevelFormat::Compressed_unique},
            {"j", LevelFormat::Compressed_unique},
        });
        TensorType dcsr_f32(dcsr, dType::Float32);
        return Tensor::make(dcsr_f32, "A") * Tensor::make(dcsr_f32, "B");
    }},
    {"dcsr_add", []() {
        Format dcsr = Format::ordered({
            {"i", LevelFormat::Compressed_unique},
            {"j", LevelFormat::Compressed_unique},
        });
        TensorType dcsr_f32(dcsr, dType::Float32);
        return Tensor::make(dcsr_f32, "A") + Tensor::make(dcsr_f32, "B");
    }},
    {"csr_add", []() {
        Format csr = Format::ordered({
            {"i", LevelFormat::Dense},
            {"j", LevelFormat::Compressed_unique},
        });
        TensorType csr_f32(csr, dType::Float32);
        return Tensor::make(csr_f32, "A") + Tensor::make(csr_f32, "B");
    }},
    {"tcsf_add", []() {
        Format tcsf = Format::ordered({
            {"i", LevelFormat::Compressed_unique},
            {"j", LevelFormat::Compressed_unique},
            {"k", LevelFormat::Compressed_unique},
        });
        TensorType tcsf_f32(tcsf, dType::Float32);
        return Tensor::make(tcsf_f32, "A") + Tensor::make(tcsf_f32, "B");
    }},
};
// clang-format on

// ===========================================================================
// Runtime Wrapper Code Generation
// ===========================================================================
// Generates nacho_ops.{h,hpp,cpp} that bridge the runtime's nanobind structs
// (CVector, CSR, DCSR, TCSF) to the flat-API wrappers in the generated .cu
// files.  The mapping from compiler struct fields to runtime struct accessors
// is determined entirely by the tensor format.

enum class RuntimeType { CVector, CSR, DCSR, TCSF };

static RuntimeType get_runtime_type(const Format &fmt) {
    int n = (int)fmt.levels.size();
    int s = 0;
    for (auto &l : fmt.levels)
        if (is_sparse_format(l.format)) s++;
    if (n == 1 && s == 1) return RuntimeType::CVector;
    if (n == 2 && s == 1) return RuntimeType::CSR;
    if (n == 2 && s == 2) return RuntimeType::DCSR;
    if (n == 3 && s == 3) return RuntimeType::TCSF;
    std::cerr << "Unsupported format for runtime wrapper generation\n";
    exit(1);
}

static std::string runtime_type_str(RuntimeType rt) {
    switch (rt) {
    case RuntimeType::CVector: return "CVector<int, int, float>";
    case RuntimeType::CSR:     return "CSR<int, float>";
    case RuntimeType::DCSR:    return "DCSR<int, float>";
    case RuntimeType::TCSF:    return "TCSF<int, float>";
    }
    return "";
}

// Map a compiler struct field kind + dimension position to a runtime struct
// accessor expression.  `var` is the wrapper parameter name (e.g. "A").
// `kind` is one of: size, length, indices, offsets, values, nnz.
// `dim` is the 0-based dimension position in the format.
static std::string rt_field(RuntimeType rt, const std::string &var,
                            const std::string &kind, int dim = 0) {
    if (kind == "values") return "(float *)" + var + ".data.data()";
    if (kind == "nnz") {
        switch (rt) {
        case RuntimeType::CVector: return "(int)" + var + ".indices.shape(0)";
        case RuntimeType::CSR:     return "(int)" + var + ".indices.shape(0)";
        case RuntimeType::DCSR:    return "(int)" + var + ".col_indices.shape(0)";
        case RuntimeType::TCSF:    return "(int)" + var + ".k_indices.shape(0)";
        }
    }
    if (kind == "size") {
        switch (rt) {
        case RuntimeType::CVector: return var + ".size";
        case RuntimeType::CSR:     return var + ".shape(" + std::to_string(dim) + ")";
        case RuntimeType::DCSR:    return dim == 0 ? var + ".nrows" : var + ".ncols";
        case RuntimeType::TCSF: {
            const char *f[] = {"dim_i_size", "dim_j_size", "dim_k_size"};
            return var + "." + f[dim];
        }
        }
    }
    if (kind == "length") {
        switch (rt) {
        case RuntimeType::CVector: return "(int)" + var + ".indices.shape(0)";
        case RuntimeType::CSR:     return "(int)" + var + ".indices.shape(0)";
        case RuntimeType::DCSR:
            return "(int)" + var + (dim == 0 ? ".row_indices" : ".col_indices") + ".shape(0)";
        case RuntimeType::TCSF: {
            const char *f[] = {"i_indices", "j_indices", "k_indices"};
            return "(int)" + var + "." + f[dim] + ".shape(0)";
        }
        }
    }
    if (kind == "indices") {
        switch (rt) {
        case RuntimeType::CVector: return "(int *)" + var + ".indices.data()";
        case RuntimeType::CSR:     return "(int *)" + var + ".indices.data()";
        case RuntimeType::DCSR:
            return "(int *)" + var + (dim == 0 ? ".row_indices" : ".col_indices") + ".data()";
        case RuntimeType::TCSF: {
            const char *f[] = {"i_indices", "j_indices", "k_indices"};
            return "(int *)" + var + "." + f[dim] + ".data()";
        }
        }
    }
    if (kind == "offsets") {
        switch (rt) {
        case RuntimeType::CVector: return "";
        case RuntimeType::CSR:     return "(int *)" + var + ".indptr.data()";
        case RuntimeType::DCSR:    return "(int *)" + var + ".row_offsets.data()";
        case RuntimeType::TCSF: {
            const char *f[] = {"", "j_offsets", "k_offsets"};
            return "(int *)" + var + "." + f[dim] + ".data()";
        }
        }
    }
    return "";
}

// Build flat-API argument expressions for one operand, in the same order
// as TensorLowerer::lower_tensor_struct_definition().
static std::vector<std::string> build_operand_args(const Format &fmt,
                                                    const std::string &var) {
    RuntimeType rt = get_runtime_type(fmt);
    std::vector<std::string> args;
    // Size fields (forward order)
    for (int i = 0; i < (int)fmt.levels.size(); i++)
        args.push_back(rt_field(rt, var, "size", i));
    // Data fields: built innermost-to-outermost, then reversed
    std::vector<std::string> data;
    data.push_back(rt_field(rt, var, "nnz"));
    data.push_back(rt_field(rt, var, "values"));
    for (int i = (int)fmt.levels.size() - 1; i >= 0; i--) {
        if (is_sparse_format(fmt.levels[i].format)) {
            data.push_back(rt_field(rt, var, "indices", i));
            data.push_back(rt_field(rt, var, "length", i));
            if (i != 0)
                data.push_back(rt_field(rt, var, "offsets", i));
        }
    }
    for (auto it = data.rbegin(); it != data.rend(); ++it)
        args.push_back(*it);
    return args;
}

// Find the innermost sparse level index in a format.
static int find_innermost_sparse(const Format &fmt) {
    for (int i = (int)fmt.levels.size() - 1; i >= 0; i--)
        if (is_sparse_format(fmt.levels[i].format)) return i;
    return -1;
}

// Uppercase the first character of a string (for wrapper param names).
static std::string to_upper_first(const std::string &s) {
    std::string r = s;
    r[0] = (char)toupper(r[0]);
    return r;
}

// ---------------------------------------------------------------------------
// .h  — flat API forward declarations
// ---------------------------------------------------------------------------
static void emit_flat_forward_decl(std::ostream &os, const std::string &op,
                                    const backend::CINLowerer &lowerer) {
    const auto &rfmt = lowerer.result_tensor.tensor_type.format;
    std::vector<std::string> params;

    // Operand struct fields (same order as lower_tensor_struct_definition)
    for (const auto &[name, tensor] : lowerer.operand_tensors) {
        const auto &fmt = tensor.tensor_type.format;
        for (int i = 0; i < (int)fmt.levels.size(); i++)
            params.push_back("index_t " + name + "_dim_" + fmt.levels[i].index + "_size");
        std::vector<std::string> dp;
        dp.push_back("index_t " + name + "_nnz");
        dp.push_back("value_t *" + name + "_values");
        for (int i = (int)fmt.levels.size() - 1; i >= 0; i--) {
            auto idx = fmt.levels[i].index;
            if (is_sparse_format(fmt.levels[i].format)) {
                dp.push_back("index_t *" + name + "_dim_" + idx + "_indices");
                dp.push_back("index_t " + name + "_dim_" + idx + "_length");
                if (i != 0)
                    dp.push_back("index_t *" + name + "_dim_" + idx + "_offsets");
            }
        }
        for (auto it = dp.rbegin(); it != dp.rend(); ++it)
            params.push_back(*it);
    }
    // Result size fields
    for (auto &lvl : rfmt.levels)
        params.push_back("index_t result_dim_" + lvl.index + "_size");
    // Output references
    params.push_back("index_t &out_nnz");
    int innermost = find_innermost_sparse(rfmt);
    for (int i = 0; i < (int)rfmt.levels.size(); i++)
        if (is_sparse_format(rfmt.levels[i].format) && i != innermost)
            params.push_back("index_t &out_dim_" + rfmt.levels[i].index + "_length");
    for (int i = (int)rfmt.levels.size() - 1; i >= 0; i--) {
        auto idx = rfmt.levels[i].index;
        if (is_sparse_format(rfmt.levels[i].format)) {
            params.push_back("index_t *&out_dim_" + idx + "_indices");
            if (i != 0)
                params.push_back("index_t *&out_dim_" + idx + "_offsets");
        }
    }
    params.push_back("value_t *&out_values");

    // Emit
    os << "template <typename index_t, typename value_t>\n";
    os << "void " << op << "(";
    std::string pad(op.size() + 6, ' ');
    for (size_t i = 0; i < params.size(); i++) {
        if (i > 0) os << ",\n" << pad;
        os << params[i];
    }
    os << ");\n";
}

// ---------------------------------------------------------------------------
// .hpp — wrapper function declarations
// ---------------------------------------------------------------------------
static void emit_wrapper_decl(std::ostream &os, const std::string &op,
                               const backend::CINLowerer &lowerer) {
    std::string rt_str = runtime_type_str(
        get_runtime_type(lowerer.result_tensor.tensor_type.format));
    std::string sig = rt_str + " nacho_" + op + "_nb(";
    std::string pad(sig.size(), ' ');

    os << sig;
    bool first = true;
    for (const auto &[name, tensor] : lowerer.operand_tensors) {
        if (!first) os << ",\n" << pad;
        first = false;
        os << runtime_type_str(get_runtime_type(tensor.tensor_type.format))
           << " " << to_upper_first(name);
    }
    os << ");\n";
}

// ---------------------------------------------------------------------------
// .cpp — wrapper function implementations
// ---------------------------------------------------------------------------
static void emit_wrapper_impl(std::ostream &os, const std::string &op,
                               const backend::CINLowerer &lowerer) {
    const auto &rfmt = lowerer.result_tensor.tensor_type.format;
    RuntimeType result_rt = get_runtime_type(rfmt);
    int innermost = find_innermost_sparse(rfmt);

    // Collect wrapper param names and runtime types for each operand
    std::vector<std::string> pnames;
    std::vector<RuntimeType> prts;
    for (const auto &[name, tensor] : lowerer.operand_tensors) {
        pnames.push_back(to_upper_first(name));
        prts.push_back(get_runtime_type(tensor.tensor_type.format));
    }

    // --- Function signature ---
    std::string rt_str = runtime_type_str(result_rt);
    std::string sig = rt_str + " nacho_" + op + "_nb(";
    std::string pad(sig.size(), ' ');
    os << sig;
    for (int idx = 0; idx < (int)pnames.size(); idx++) {
        if (idx > 0) os << ",\n" << pad;
        os << runtime_type_str(prts[idx]) << " " << pnames[idx];
    }
    os << ") {\n";

    // --- Declare output variables ---
    os << "    int out_nnz;\n";
    for (int i = 0; i < (int)rfmt.levels.size(); i++)
        if (is_sparse_format(rfmt.levels[i].format) && i != innermost)
            os << "    int out_dim_" << rfmt.levels[i].index << "_length;\n";
    for (int i = (int)rfmt.levels.size() - 1; i >= 0; i--)
        if (is_sparse_format(rfmt.levels[i].format)) {
            os << "    int *out_dim_" << rfmt.levels[i].index << "_indices;\n";
            if (i != 0)
                os << "    int *out_dim_" << rfmt.levels[i].index << "_offsets;\n";
        }
    os << "    float *out_values;\n\n";

    // --- Build flat-API call arguments ---
    std::vector<std::string> call_args;
    {
        int idx = 0;
        for (const auto &[name, tensor] : lowerer.operand_tensors) {
            auto a = build_operand_args(tensor.tensor_type.format, pnames[idx]);
            call_args.insert(call_args.end(), a.begin(), a.end());
            idx++;
        }
    }
    // Result sizes from first operand
    for (int i = 0; i < (int)rfmt.levels.size(); i++)
        call_args.push_back(rt_field(prts[0], pnames[0], "size", i));
    // Output refs
    call_args.push_back("out_nnz");
    for (int i = 0; i < (int)rfmt.levels.size(); i++)
        if (is_sparse_format(rfmt.levels[i].format) && i != innermost)
            call_args.push_back("out_dim_" + rfmt.levels[i].index + "_length");
    for (int i = (int)rfmt.levels.size() - 1; i >= 0; i--)
        if (is_sparse_format(rfmt.levels[i].format)) {
            call_args.push_back("out_dim_" + rfmt.levels[i].index + "_indices");
            if (i != 0)
                call_args.push_back("out_dim_" + rfmt.levels[i].index + "_offsets");
        }
    call_args.push_back("out_values");

    // --- Emit flat-API call ---
    os << "    " << op << "<int, float>(\n";
    for (size_t i = 0; i < call_args.size(); i++) {
        os << "        " << call_args[i];
        if (i + 1 < call_args.size()) os << ",";
        os << "\n";
    }
    os << "    );\n\n";

    // --- Empty-result handling ---
    os << "    if (out_nnz == 0) {\n";
    switch (result_rt) {
    case RuntimeType::CVector: {
        auto idx = rfmt.levels[0].index;
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << idx
           << "_indices, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_values, sizeof(float)));\n";
        break;
    }
    case RuntimeType::CSR: {
        auto idx = rfmt.levels[1].index;
        std::string nrows = rt_field(prts[0], pnames[0], "size", 0);
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << idx
           << "_offsets, sizeof(int) * (" << nrows << " + 1)));\n";
        os << "        CHECK_CUDA(cudaMemset(out_dim_" << idx
           << "_offsets, 0, sizeof(int) * (" << nrows << " + 1)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << idx
           << "_indices, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_values, sizeof(float)));\n";
        break;
    }
    case RuntimeType::DCSR: {
        auto i = rfmt.levels[0].index, j = rfmt.levels[1].index;
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << i
           << "_indices, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << j
           << "_offsets, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMemset(out_dim_" << j
           << "_offsets, 0, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << j
           << "_indices, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_values, sizeof(float)));\n";
        os << "        out_dim_" << i << "_length = 0;\n";
        break;
    }
    case RuntimeType::TCSF: {
        auto i = rfmt.levels[0].index, j = rfmt.levels[1].index,
             k = rfmt.levels[2].index;
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << i
           << "_indices, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << j
           << "_offsets, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMemset(out_dim_" << j
           << "_offsets, 0, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << j
           << "_indices, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << k
           << "_offsets, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMemset(out_dim_" << k
           << "_offsets, 0, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_dim_" << k
           << "_indices, sizeof(int)));\n";
        os << "        CHECK_CUDA(cudaMalloc((void **)&out_values, sizeof(float)));\n";
        os << "        out_dim_" << i << "_length = 0;\n";
        os << "        out_dim_" << j << "_length = 0;\n";
        break;
    }
    }
    os << "    }\n\n";

    // --- Return result ---
    // Build constructor call matching the pointer-based ctors in nb_utils.hpp
    os << "    return ";
    switch (result_rt) {
    case RuntimeType::CVector: {
        auto idx = rfmt.levels[0].index;
        os << rt_str << "(out_dim_" << idx << "_indices, out_values, "
           << rt_field(prts[0], pnames[0], "size", 0) << ", out_nnz)";
        break;
    }
    case RuntimeType::CSR: {
        auto idx = rfmt.levels[1].index;
        os << rt_str << "(out_dim_" << idx << "_offsets, out_dim_" << idx
           << "_indices, out_values, "
           << rt_field(prts[0], pnames[0], "size", 0) << ", "
           << rt_field(prts[0], pnames[0], "size", 1) << ", out_nnz)";
        break;
    }
    case RuntimeType::DCSR: {
        auto i = rfmt.levels[0].index, j = rfmt.levels[1].index;
        os << rt_str << "(out_dim_" << i << "_indices, out_dim_" << j
           << "_offsets, out_dim_" << j << "_indices, out_values, "
           << rt_field(prts[0], pnames[0], "size", 0) << ", "
           << rt_field(prts[0], pnames[0], "size", 1) << ", out_dim_"
           << i << "_length, out_nnz)";
        break;
    }
    case RuntimeType::TCSF: {
        auto i = rfmt.levels[0].index, j = rfmt.levels[1].index,
             k = rfmt.levels[2].index;
        os << rt_str << "(out_dim_" << i << "_indices, out_dim_" << j
           << "_offsets, out_dim_" << j << "_indices, out_dim_" << k
           << "_offsets, out_dim_" << k << "_indices, out_values, "
           << rt_field(prts[0], pnames[0], "size", 0) << ", "
           << rt_field(prts[0], pnames[0], "size", 1) << ", "
           << rt_field(prts[0], pnames[0], "size", 2) << ", out_dim_"
           << i << "_length, out_dim_" << j << "_length, out_nnz)";
        break;
    }
    }
    os << ";\n}\n";
}

// ---------------------------------------------------------------------------
// Top-level: generate nacho_ops.{h,hpp,cpp} for all registered expressions.
// ---------------------------------------------------------------------------
static void emit_runtime_wrappers(const std::string &dir) {
    std::filesystem::create_directories(dir);

    std::ofstream h(dir + "/nacho_ops.h");
    std::ofstream hpp(dir + "/nacho_ops.hpp");
    std::ofstream cpp(dir + "/nacho_ops.cpp");

    h   << "#pragma once\n\n"
        << "// Auto-generated by nacho compiler. Do not edit.\n\n";

    hpp << "#pragma once\n"
        << "// Auto-generated by nacho compiler. Do not edit.\n"
        << "#include \"nb_utils.hpp\"\n\n";

    cpp << "// Auto-generated by nacho compiler. Do not edit.\n"
        << "#include \"nacho_ops.h\"\n"
        << "#include \"nacho_ops.hpp\"\n"
        << "#include \"cuda_utils/cuda_utils.h\"\n\n";

    // CINLowerer needs a Printer; discard output since we only need metadata.
    std::ostringstream devnull;
    Printer devnull_printer(devnull);

    for (const auto &[name, builder] : EXPRESSIONS) {
        Expr expr = builder();
        CIN cin = compile_to_cin(expr);
        backend::CINLowerer lowerer(cin, devnull_printer);

        emit_flat_forward_decl(h, name, lowerer);
        h << "\n";

        emit_wrapper_decl(hpp, name, lowerer);
        hpp << "\n";

        emit_wrapper_impl(cpp, name, lowerer);
        cpp << "\n";

        devnull.str("");
    }

    std::cout << "Generated " << dir << "/nacho_ops.{h,hpp,cpp}\n";
}

// ---------------------------------------------------------------------------
// Helper: emit generated CUDA code to a file with a flat-API wrapper.
// ---------------------------------------------------------------------------
static void emit_to_file(const std::string &dir, const std::string &op_name,
                         Expr expr) {
    std::filesystem::create_directories(dir);
    std::string filepath = dir + "/" + op_name + ".cu";

    std::ofstream ofs(filepath);
    if (!ofs) {
        std::cerr << "Failed to open " << filepath << " for writing\n";
        exit(1);
    }

    ofs << "#pragma once\n\n";
    ofs << "#include <cuda_runtime.h>\n";
    ofs << "#include <cub/cub.cuh>\n\n";

    // Wrap internal symbols in a unique namespace to avoid collisions
    // when multiple generated .cu files are compiled into the same library.
    ofs << "namespace " << op_name << "_ns {\n\n";

    CIN cin = compile_to_cin(expr);
    Printer printer(ofs);
    backend::CINLowerer lowerer(cin, printer);
    lowerer.lower_cin();
    ofs << "\n";

    ofs << "} // namespace " << op_name << "_ns\n\n";
    ofs << "using namespace " << op_name << "_ns;\n\n";

    // Flat wrapper at global scope (uses internal types via using namespace)
    lowerer.lower_flat_wrapper(op_name);
    ofs << "\n";

    // Emit explicit template instantiation for <int, float>.
    // Build the type list by iterating operand/result tensor formats.
    ofs << "// Explicit template instantiation\n";
    ofs << "template void " << op_name << "<int, float>(";
    bool first_arg = true;
    // Operand args (all struct fields)
    for (const auto &[name, tensor] : lowerer.operand_tensors) {
        llir::lType struct_type = tensor.lower_tensor_struct_definition();
        const auto *st = struct_type.as<llir::Struct_t>();
        for (const auto &[field_name, field_type] : st->fields) {
            if (!first_arg) ofs << ", ";
            first_arg = false;
            if (field_type.is<llir::Ptr_t>()) {
                const auto *pt = field_type.as<llir::Ptr_t>();
                if (pt->type.is<llir::Generic_t>()) {
                    const auto *gt = pt->type.as<llir::Generic_t>();
                    ofs << (gt->name == "value_t" ? "float" : "int") << "*";
                }
            } else {
                ofs << "int";
            }
        }
    }
    // Result size args
    for (size_t i = 0; i < lowerer.result_tensor.tensor_type.format.levels.size(); i++) {
        if (!first_arg) ofs << ", ";
        first_arg = false;
        ofs << "int";
    }
    // Output refs: nnz
    ofs << ", int&";
    // Output refs: length for non-innermost sparse dims
    {
        int innermost_sparse = -1;
        for (int i = (int)lowerer.result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
            auto idx = lowerer.result_tensor.tensor_type.format.levels[i].index;
            if (is_sparse_format(lowerer.result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                innermost_sparse = i;
                break;
            }
        }
        for (int i = 0; i < (int)lowerer.result_tensor.tensor_type.format.levels.size(); i++) {
            auto idx = lowerer.result_tensor.tensor_type.format.levels[i].index;
            if (is_sparse_format(lowerer.result_tensor.tensor_type.format.lvlfmt_of(idx)) && i != innermost_sparse) {
                ofs << ", int&";
            }
        }
    }
    // Output refs: indices (and offsets for non-outermost)
    for (int i = (int)lowerer.result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
        auto idx = lowerer.result_tensor.tensor_type.format.levels[i].index;
        if (is_sparse_format(lowerer.result_tensor.tensor_type.format.lvlfmt_of(idx))) {
            ofs << ", int*&";
            if (i != 0) ofs << ", int*&";
        }
    }
    // Output ref: values
    ofs << ", float*&";
    ofs << ");\n";

    ofs.close();
    std::cout << "Generated " << filepath << "\n";
}

// ---------------------------------------------------------------------------
// Helper: emit generated CPU code to a file with a flat-API wrapper.
// ---------------------------------------------------------------------------
static void emit_to_file_cpu(const std::string &dir, const std::string &op_name,
                              Expr expr) {
    std::filesystem::create_directories(dir);
    std::string filepath = dir + "/" + op_name + ".cpp";

    std::ofstream ofs(filepath);
    if (!ofs) {
        std::cerr << "Failed to open " << filepath << " for writing\n";
        exit(1);
    }

    ofs << "#pragma once\n\n";
    ofs << "#include \"cpu_runtime.h\"\n\n";

    // Wrap internal symbols in a unique namespace to avoid collisions
    ofs << "namespace " << op_name << "_ns {\n\n";

    CIN cin = compile_to_cin(expr);
    CPUPrinter printer(ofs);
    backend::CINLowerer lowerer(cin, printer);
    lowerer.lower_cin();
    ofs << "\n";

    ofs << "} // namespace " << op_name << "_ns\n\n";
    ofs << "using namespace " << op_name << "_ns;\n\n";

    // Flat wrapper at global scope
    lowerer.lower_flat_wrapper(op_name);
    ofs << "\n";

    // Emit explicit template instantiation for <int, float>.
    ofs << "// Explicit template instantiation\n";
    ofs << "template void " << op_name << "<int, float>(";
    bool first_arg = true;
    for (const auto &[name, tensor] : lowerer.operand_tensors) {
        llir::lType struct_type = tensor.lower_tensor_struct_definition();
        const auto *st = struct_type.as<llir::Struct_t>();
        for (const auto &[field_name, field_type] : st->fields) {
            if (!first_arg) ofs << ", ";
            first_arg = false;
            if (field_type.is<llir::Ptr_t>()) {
                const auto *pt = field_type.as<llir::Ptr_t>();
                if (pt->type.is<llir::Generic_t>()) {
                    const auto *gt = pt->type.as<llir::Generic_t>();
                    ofs << (gt->name == "value_t" ? "float" : "int") << "*";
                }
            } else {
                ofs << "int";
            }
        }
    }
    for (size_t i = 0; i < lowerer.result_tensor.tensor_type.format.levels.size(); i++) {
        if (!first_arg) ofs << ", ";
        first_arg = false;
        ofs << "int";
    }
    ofs << ", int&";
    {
        int innermost_sparse = -1;
        for (int i = (int)lowerer.result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
            auto idx = lowerer.result_tensor.tensor_type.format.levels[i].index;
            if (is_sparse_format(lowerer.result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                innermost_sparse = i;
                break;
            }
        }
        for (int i = 0; i < (int)lowerer.result_tensor.tensor_type.format.levels.size(); i++) {
            auto idx = lowerer.result_tensor.tensor_type.format.levels[i].index;
            if (is_sparse_format(lowerer.result_tensor.tensor_type.format.lvlfmt_of(idx)) && i != innermost_sparse) {
                ofs << ", int&";
            }
        }
    }
    for (int i = (int)lowerer.result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
        auto idx = lowerer.result_tensor.tensor_type.format.levels[i].index;
        if (is_sparse_format(lowerer.result_tensor.tensor_type.format.lvlfmt_of(idx))) {
            ofs << ", int*&";
            if (i != 0) ofs << ", int*&";
        }
    }
    ofs << ", float*&";
    ofs << ");\n";

    ofs.close();
    std::cout << "Generated " << filepath << "\n";
}

// ===========================================================================
// Test Registry
// ===========================================================================

using TestFunc = void (*)();

// clang-format off
const std::map<std::string, TestFunc> TESTS = {
    {"sparse_vec_mul",    test_sparse_vec_mul},
    {"sparse_vec_add",    test_sparse_vec_add},
    {"sparse_vec_apb_c",  test_sparse_vec_apb_c},
    {"sparse_vec_ab_pc",  test_sparse_vec_ab_pc},
    {"dcsr_mul",          test_dcsr_mul},
    {"dcsr_add",          test_dcsr_add},
    {"csr_add",           test_csr_add},
    {"tcsf_add",          test_tcsf_add},
    {"format_inference",  test_format_inference},
    {"lattice",           test_lattice},
    {"locator",           test_locator_optimization},
};
// clang-format on

int main(int argc, char **argv) {
    // Usage: compiler --test <name>        Run a single test
    //        compiler --list               List available tests
    //        compiler --emit <dir> --name <op_name>   Generate .cu file
    //        compiler                      Run all tests
    if (argc >= 2 && std::strcmp(argv[1], "--list") == 0) {
        std::cout << "Tests:\n";
        for (const auto &[name, _] : TESTS)
            std::cout << "  " << name << "\n";
        std::cout << "Expressions (for --emit):\n";
        for (const auto &[name, _] : EXPRESSIONS)
            std::cout << "  " << name << "\n";
        return 0;
    }

    if (argc >= 3 && std::strcmp(argv[1], "--test") == 0) {
        auto it = TESTS.find(argv[2]);
        if (it == TESTS.end()) {
            std::cerr << "Unknown test: " << argv[2] << "\n";
            std::cerr << "Run with --list to see available tests.\n";
            return 1;
        }
        it->second();
        return 0;
    }

    // --emit-wrappers <dir>
    if (argc >= 3 && std::strcmp(argv[1], "--emit-wrappers") == 0) {
        emit_runtime_wrappers(argv[2]);
        return 0;
    }

    // --emit <dir> --name <op_name> [--target cpu|gpu]
    if (argc >= 2 && std::strcmp(argv[1], "--emit") == 0) {
        std::string dir, name, target = "gpu";
        // Parse remaining args
        for (int i = 2; i < argc; i++) {
            if (std::strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
                name = argv[++i];
            } else if (std::strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
                target = argv[++i];
            } else if (dir.empty()) {
                dir = argv[i];
            }
        }
        if (dir.empty() || name.empty()) {
            std::cerr << "Usage: compiler --emit <dir> --name <op_name> [--target cpu|gpu]\n";
            std::cerr << "Run with --list to see available expressions.\n";
            return 1;
        }
        auto it = EXPRESSIONS.find(name);
        if (it == EXPRESSIONS.end()) {
            std::cerr << "Unknown expression: " << name << "\n";
            std::cerr << "Run with --list to see available expressions.\n";
            return 1;
        }
        if (target == "cpu") {
            emit_to_file_cpu(dir, name, it->second());
        } else {
            emit_to_file(dir, name, it->second());
        }
        return 0;
    }

    // No args: run everything.
    for (const auto &[name, func] : TESTS)
        func();

    return 0;
}
