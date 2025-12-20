#include "Nacho.h"

// Temporary, for make_binary_search example
#include "llir/Function.h"
#include "llir/LLIR.h"

using namespace nacho;

void test() {
    std::cout << "Basic tests\n";
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed},
    });

    TensorType csr_f32 = TensorType(csr, dType::Float32);

    Expr a_ij = Tensor::make(csr_f32, "a");
    Expr b_ij = Tensor::make(csr_f32, "b");

    Expr z_ij = a_ij + b_ij;

    std::cout << z_ij << "\n";
    std::cout << compile_to_cin(z_ij) << "\n";

    Expr z_i = sum("i", z_ij);

    std::cout << z_i << "\n";
    std::cout << compile_to_cin(z_i) << "\n";

    z_ij = a_ij * b_ij;

    std::cout << z_ij << "\n";
    std::cout << compile_to_cin(z_ij) << "\n";

    std::cout << "\n\n";
}

void test_vec() {
    std::cout << "Vector math test\n";
    Format dense = Format::ordered({
        {"i", LevelFormat::Dense},
    });
    Format sparse = Format::ordered({
        {"i", LevelFormat::Compressed},
    });

    TensorType dense_f32 = TensorType(dense, dType::Float32);
    TensorType sparse_f32 = TensorType(sparse, dType::Float32);

    Expr a_i = Tensor::make(sparse_f32, "a");
    Expr b_i = Tensor::make(sparse_f32, "b");
    Expr c_i = Tensor::make(dense_f32, "c");
    Expr d_i = Tensor::make(dense_f32, "d");

    Expr e = a_i * b_i;
    std::cout << "Expect sparse: " << e.type().format << "\n";
    std::cout << compile_to_cin(e) << "\n";

    e = c_i * d_i;
    std::cout << "Expect dense: " << e.type().format << "\n";
    std::cout << compile_to_cin(e) << "\n";

    e = a_i * c_i;
    std::cout << "Expect sparse: " << e.type().format << "\n";
    std::cout << compile_to_cin(e) << "\n";

    e = a_i + b_i;
    std::cout << "Expect sparse: " << e.type().format << "\n";
    std::cout << compile_to_cin(e) << "\n";

    e = c_i + d_i;
    std::cout << "Expect dense: " << e.type().format << "\n";
    std::cout << compile_to_cin(e) << "\n";

    e = a_i + c_i;
    std::cout << "Expect dense: " << e.type().format << "\n";
    std::cout << compile_to_cin(e) << "\n";

    std::cout << "\n\n";
}

void spgemm() {
    std::cout << "SpGEMM test\n";
    Format csr0 = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed},
    });

    Format csr1 = Format::ordered({
        {"j", LevelFormat::Dense},
        {"k", LevelFormat::Compressed},
    });

    TensorType csr0_f32 = TensorType(csr0, dType::Float32);
    TensorType csr1_f32 = TensorType(csr1, dType::Float32);

    Expr a_ij = Tensor::make(csr0_f32, "a");
    Expr b_jk = Tensor::make(csr1_f32, "b");

    // Implicit broadcasting.
    Expr z_ik = sum("j", a_ij * b_jk);

    std::cout << z_ik << "\n";
    std::cout << "Expect CSR: " << z_ik.type().format << "\n";

    std::cout << compile_to_cin(z_ik) << "\n";
    std::cout << "\n\n";
}

void test_format_inf() {
    std::cout << "Format inference test\n";
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed},
    });

    Format dcsr = Format::ordered({
        {"i", LevelFormat::Compressed},
        {"j", LevelFormat::Compressed},
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

void make_binary_search() {
    llir::Function func;
    func.generics.push_back("index_t");

    func.attributes.push_back(llir::Function::device);
    func.attributes.push_back(llir::Function::inline_);

    llir::lType index_t = llir::Generic_t::make("index_t");
    func.ret_type = index_t;

    func.name = "lower_bound";

    llir::lType ptr_index_t = llir::Ptr_t::make(index_t);
    func.args.push_back(
        {.mutating = false, .type = ptr_index_t, .name = "crds"});
    func.args.push_back({.mutating = false, .type = index_t, .name = "n"});
    func.args.push_back({.mutating = false, .type = index_t, .name = "c"});

    {
        llir::lExpr crds = llir::lVar::make(ptr_index_t, "crds");
        llir::lExpr n = llir::lVar::make(index_t, "n");
        llir::lExpr c = llir::lVar::make(index_t, "c");

        std::vector<llir::lStmt> stmts;

        // index_t low = 0;
        stmts.push_back(
            llir::Declare::make(index_t, "low", llir::lConst::make(0)));
        llir::lExpr low = llir::lVar::make(index_t, "low");
        // index_t high = n;
        stmts.push_back(llir::Declare::make(index_t, "high", n));
        llir::lExpr high = llir::lVar::make(index_t, "high");

        std::vector<llir::lStmt> while_body;

        while_body.push_back(
            // index_t mid = low + (high - low) / 2;
            llir::Declare::make(index_t, "mid",
                                low + (high - low) / llir::lConst::make(2)));
        llir::lExpr mid = llir::lVar::make(index_t, "mid");

        while_body.push_back(
            // if (a->indices[mid] <= crd) {
            //   low = mid + 1;
            // } else {
            //   high = mid
            // }
            llir::IfElse::make(
                llir::lLoad::make(crds, mid) <= c,
                llir::Store::make("low", mid + llir::lConst::make(1)),
                llir::Store::make("high", mid)));

        // while (low < high) { ... }
        llir::lStmt while_loop = llir::While::make(
            low < high, llir::Sequence::make(std::move(while_body)));

        stmts.push_back(std::move(while_loop));

        // return low;
        stmts.push_back(llir::Return::make(low));

        func.body = llir::Sequence::make(std::move(stmts));
    }

    func.print(std::cout);
}

// TODO: write a parser.
int main(int argc, char **argv) {
    test();
    test_format_inf();
    test_vec();
    spgemm();

    make_binary_search();

    return 0;
}
