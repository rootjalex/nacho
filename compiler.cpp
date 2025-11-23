#include "Nacho.h"

using namespace nacho;

// TODO: write a parser.
int main(int argc, char **argv) {
    Format csr = Format::ordered({
        {"i", LevelFormat::Dense},
        {"j", LevelFormat::Compressed},
    });

    TensorType csr_f32 = TensorType{csr, dType::Float32};

    Expr a_ij = Tensor::make(csr_f32, "a");
    Expr b_ij = Tensor::make(csr_f32, "b");

    Expr z_ij = a_ij + b_ij;

    std::cout << z_ij << "\n";

    Expr z_i = sum("i", z_ij);

    std::cout << z_i << "\n";

    return 0;
}
