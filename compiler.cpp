#include "Nacho.h"

#include "Lattice.h"
#include "Simplify.h"
#include "backend/compile.h"

#include <cstring>
#include <map>

using namespace nacho;

// ---------------------------------------------------------------------------
// Helper: full pipeline from expression to generated CUDA code.
// ---------------------------------------------------------------------------
static void compile_and_lower(const std::string &name, Expr expr) {
    std::cout << "=== " << name << " ===\n";
    std::cout << "Expression: " << expr << "\n";
    std::cout << "Result format: " << expr.type().format << "\n\n";

    CIN cin = compile_to_cin(expr);
    std::cout << "CIN:\n" << cin << "\n\n";

    std::cout << "Generated CUDA:\n";
    backend::CINLowerer(cin, std::cout).lower_cin();
    std::cout << "\n";
}

// ===========================================================================
// 1D Sparse Vector Tests
// ===========================================================================

// a ∩ b  (element-wise multiply, single phase)
void test_sparse_vec_mul() {
    Format s = Format::ordered({{"i", LevelFormat::Compressed}});
    TensorType s_f32(s, dType::Float32);

    Expr a = Tensor::make(s_f32, "a");
    Expr b = Tensor::make(s_f32, "b");
    compile_and_lower("Sparse Vector Mul: a*b", a * b);
}

// a ∪ b  (element-wise add, single phase)
void test_sparse_vec_add() {
    Format s = Format::ordered({{"i", LevelFormat::Compressed}});
    TensorType s_f32(s, dType::Float32);

    Expr a = Tensor::make(s_f32, "a");
    Expr b = Tensor::make(s_f32, "b");
    compile_and_lower("Sparse Vector Add: a+b", a + b);
}

// Runtime parity: runtime/src/sparse_vector/sparse_vector.cu
// (A+B)*C — union then intersect, fused 3-operand expression.
void test_sparse_vec_apb_c() {
    Format s = Format::ordered({{"i", LevelFormat::Compressed}});
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
    Format s = Format::ordered({{"i", LevelFormat::Compressed}});
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
        {"i", LevelFormat::Compressed},
        {"j", LevelFormat::Compressed},
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
        {"i", LevelFormat::Compressed},
        {"j", LevelFormat::Compressed},
    });
    TensorType dcsr_f32(dcsr, dType::Float32);

    Expr a = Tensor::make(dcsr_f32, "A");
    Expr b = Tensor::make(dcsr_f32, "B");
    compile_and_lower("DCSR Add: A+B  [runtime: csr_add.cu]", a + b);
}

// ===========================================================================
// Format Inference Tests
// ===========================================================================

void test_format_inference() {
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed},
    });
    Format dcsr = Format::ordered({
        {"i", LevelFormat::Compressed},
        {"j", LevelFormat::Compressed},
    });
    Format sparse = Format::ordered({{"i", LevelFormat::Compressed}});
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
    Format sparse = Format::ordered({{"i", LevelFormat::Compressed}});
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
    Format sparse = Format::ordered({{"i", LevelFormat::Compressed}});
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
        auto [iters, locs] = partition_iterators_locators(seq);
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
    {"format_inference",  test_format_inference},
    {"lattice",           test_lattice},
    {"locator",           test_locator_optimization},
};
// clang-format on

int main(int argc, char **argv) {
    // Usage: compiler --test <name>   Run a single test
    //        compiler --list          List available tests
    //        compiler                 Run all tests
    if (argc >= 2 && std::strcmp(argv[1], "--list") == 0) {
        for (const auto &[name, _] : TESTS)
            std::cout << name << "\n";
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

    // No args: run everything.
    for (const auto &[name, func] : TESTS)
        func();

    return 0;
}
