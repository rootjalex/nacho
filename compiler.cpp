#include "Nacho.h"

using namespace nacho;

void test() {
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed},
    });

    TensorType csr_f32 = TensorType(csr, dType::Float32);

    Expr a_ij = Tensor::make(csr_f32, "a");
    Expr b_ij = Tensor::make(csr_f32, "b");

    Expr z_ij = a_ij + b_ij;

    std::cout << z_ij << "\n";

    Expr z_i = sum("i", z_ij);

    std::cout << z_i << "\n";
}

void spgemm() {
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
}

void test_format_inf() {
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
    std::cout << "Expect DCSR: " << (a_ij * b_ij).type().format << "\n";

    std::cout << "Expect [DC]{D}: "
              << (bc("k", a_ij) + bc("k", b_ij)).type().format << "\n";
    std::cout << "Expect [DC]{D}: "
              << (bc("k", a_ij) * bc("k", b_ij)).type().format << "\n";
}

// TODO: write a parser.
int main(int argc, char **argv) {
    test();
    test_format_inf();
    spgemm();

    return 0;
}
