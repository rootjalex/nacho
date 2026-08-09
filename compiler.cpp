#include "Nacho.h"

// Temporary, for make_binary_search example
#include "GeneratePartition.h"
#include "Lattice.h"
#include "Simplify.h"
#include "llir/Function.h"
#include "llir/LLIR.h"

#include "backend/compile.h"
#include "backend/generate_bindings.h"
#include "backend/output.h"
#include "backend/tensor.h"

using namespace nacho;

void test() {
    // Layout names become the Python class names of the generated bindings. Formats
    // sharing a level structure share a name: a CSC matrix is a CSR class over
    // transposed data, and b_jk in CSR is the same layout as a_ij in CSR.
    Format csr_1 = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed_unique},
    }).named("CSR");
    Format csc_1 = Format::ordered({
        {"j", LevelFormat::Dense},
        {"i", LevelFormat::Compressed_unique},
    }).named("CSR");
    Format csr_2 = Format::ordered({
        {"j", LevelFormat::Dense},
        {"k", LevelFormat::Compressed_unique},
    }).named("CSR");

    Format csr_3 = Format::ordered({
        {"i", LevelFormat::Dense},
        {"k", LevelFormat::Compressed_unique},
    }).named("CSR");

    Format dcsr = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
    }).named("DCSR");

    Format csf = Format::ordered({
        {"i", LevelFormat::Compressed_unique},
        {"j", LevelFormat::Compressed_unique},
        {"k", LevelFormat::Compressed_unique},
    }).named("CSF3");

    Format coo = Format::ordered({
        {"i", LevelFormat::Compressed_non_unique},
        {"j", LevelFormat::Singleton_unique},
    }).named("COO");

    Format coo_2 = Format::ordered({
        {"j", LevelFormat::Compressed_non_unique},
        {"k", LevelFormat::Singleton_unique},
    }).named("COO");

    Format coo3d = Format::ordered({
        {"i", LevelFormat::Compressed_non_unique},
        {"j", LevelFormat::Singleton_non_unique},
        {"k", LevelFormat::Singleton_unique},
    }).named("COO3");

    Format s = Format::ordered({
        {"k", LevelFormat::Compressed_unique}
    }).named("SparseVector");

    Format d = Format::ordered({
        {"i", LevelFormat::Dense}
    }).named("DenseVector");

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


    // Element-wise kernels.

    Kernel("csr_add").expr(a_csr_ij + b_csr_ij).emit();

    Kernel("csr_add_3").expr(a_csr_ij + b_csr_ij + c_csr_ij).emit();

    Kernel("csr_mul").expr(a_csr_ij * b_csr_ij).emit();

    Kernel("coo_add").expr(a_coo_ij + b_coo_ij).emit();

    Kernel("coo_mul").expr(a_coo_ij * b_coo_ij).emit();

    Kernel("coo_csr_add").expr(a_csr_ij + b_coo_ij).emit();

    Kernel("dcsr_add").expr(a_dcsr_ij + b_dcsr_ij).emit();

    Kernel("dcsr_mul").expr(a_dcsr_ij * b_dcsr_ij).emit();

    // Same product, partitioned once over the whole loop nest. Comparing the two shows
    // what re-partitioning at each sparse intersection level is worth.
    Kernel("dcsr_mul_without_recursive_partitioning")
        .expr(a_dcsr_ij * b_dcsr_ij)
        .recursive_partitioning(false)
        .emit();

    Kernel("csf_add").expr(a_csf_ijk + b_csf_ijk).emit();

    // TODO(atharvac) Reduction kernels. The Accumulate path splits the result into
    // <Z>_temp and <Z>, and the entry point exposes only the former, so there is nothing
    // for the bindings to return yet.
    //
    // Kernel("spgemm").expr(Sum::make("j", a_csr_ij * b_csr_jk)).emit();
    // Kernel("sssmm").expr(Sum::make("j", a_csr_ij * b_csr_jk * c_csr_ik)).emit();
    // Kernel("inner_prod").expr(Sum::make("i", Sum::make("j", Sum::make("k", a_csf_ijk * b_csf_ijk)))).emit();

    return;
}

int main(int argc, char **argv) {
    if (!nacho::parse_command_line(argc, argv)) {
        return 1;
    }
    nacho::backend::reset_output_directory();
    test();
    nacho::backend::finalize_bindings();
    return 0;
}
