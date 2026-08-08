#include "Nacho.h"

// Temporary, for make_binary_search example
#include "GeneratePartition.h"
#include "Lattice.h"
#include "Simplify.h"
#include "llir/Function.h"
#include "llir/LLIR.h"

#include "backend/compile.h"
#include "backend/tensor.h"

using namespace nacho;

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

    TensorType csr_f32 = TensorType(csr_1, dType::Float32);
    TensorType csc_f32 = TensorType(csc_1, dType::Float32);
    TensorType csr2_f32 = TensorType(csr_2, dType::Float32);
    TensorType csr3_f32 = TensorType(csr_3, dType::Float32);
    TensorType csf_f32 = TensorType(csf, dType::Float32);
    TensorType coo_f32 = TensorType(coo, dType::Float32);
    TensorType coo_2_f32 = TensorType(coo_2, dType::Float32);
    TensorType coo3d_f32 = TensorType(coo3d, dType::Float32);
    TensorType dcsr_f32 = TensorType(dcsr, dType::Float32);
    TensorType s_f32 = TensorType(s, dType::Float32);
    TensorType d_f32 = TensorType(d, dType::Float32);


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
    // nacho::backend::CINLowerer(compile_to_cin(a_csr_ij + b_csr_ij), std::cout).lower_cin();

    // // CSR + CSR + CSR
    // nacho::backend::CINLowerer(compile_to_cin(a_csr_ij + b_csr_ij + c_csr_ij), std::cout).lower_cin();

    // CSR Mul
    nacho::backend::CINLowerer(compile_to_cin(a_csr_ij * b_csr_ij), "csr_mul").lower_cin();
    nacho::backend::CINLowerer(compile_to_cin(a_csr_ij * b_csr_ij), "csr_mul", true).lower_cin();

    // // COO add
    // nacho::backend::CINLowerer(compile_to_cin(a_coo_ij + b_coo_ij), std::cout).lower_cin();

    // // COO Mul
    // nacho::backend::CINLowerer(compile_to_cin(a_coo_ij * b_coo_ij), std::cout).lower_cin();

    // // CSR + COO
    // nacho::backend::CINLowerer(compile_to_cin(a_csr_ij + b_coo_ij), std::cout).lower_cin();

    // // DCSR add
    // nacho::backend::CINLowerer(compile_to_cin(a_dcsr_ij + b_dcsr_ij), std::cout).lower_cin();

    // DCSR Mul
    nacho::backend::CINLowerer(compile_to_cin(a_dcsr_ij * b_dcsr_ij), "dcsr_mul").lower_cin();
    nacho::backend::CINLowerer(compile_to_cin(a_dcsr_ij * b_dcsr_ij), "dcsr_mul", true).lower_cin();

    // // CSF add
    // nacho::backend::CINLowerer(compile_to_cin(a_csf_ijk + b_csf_ijk), std::cout).lower_cin();

    // // Spgemm
    // nacho::backend::CINLowerer(compile_to_cin(Sum::make("j", a_csr_ij * b_csr_jk)), std::cout).lower_cin();

    // // SSSMM
    // nacho::backend::CINLowerer(compile_to_cin(Sum::make("j", a_csr_ij * b_csr_jk * c_csr_ik)), std::cout).lower_cin();

    // // Inner Prod
    // nacho::backend::CINLowerer(compile_to_cin(Sum::make("i", Sum::make("j", Sum::make("k" ,a_csf_ijk * b_csf_ijk)))), std::cout).lower_cin();

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
    nacho::backend::CINLowerer(compile_to_cin(z_i), "spmspv").lower_cin();
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
    nacho::backend::CINLowerer(compile_to_cin(z_ik), "spgemm").lower_cin();
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
    nacho::backend::CINLowerer(compile_to_cin(z_ijk), "sssmm").lower_cin();
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
