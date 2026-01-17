#include "IRFwdDecl.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Error.h"
#include "Format.h"
#include "Type.h"

// TensorStructLowerer is responsible for lowering a tensor's structure into the LLIR.
struct TensorStructLowerer {
    std::string tensor_name;
    TensorType tensor_type;
    std::vector<std::string> generics;
    llir::lType struct_format;

    TensorStructLowerer(const std::string &tensor_name, const TensorType &type);

    void print_tensor_struct(std::ostream &os) {
        Printer printer(os);
        printer.print(struct_format);
    }
        
};
