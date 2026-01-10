#include "backend/compile.h"
#include "backend/partition.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Error.h"
#include "Format.h"
#include "Type.h"
#include "backend/tensor.h"
#include "Printer.h"

namespace nacho {
namespace backend {

    void CINLowerer::lower_cin() {
        this->lower_struct_definitions();
        this->lower_partition_function();
    }

    // lower_struct_definitions loweres all the initial struct definitions for the program
    // this includes tensor struct definitions for both the operand and result tensors
    void CINLowerer::lower_struct_definitions() {
        struct TensorVisitor : Visitor {
            Printer& printer;
            TensorVisitor(Printer& printer) : printer(printer) {}

            void print(std::string str, TensorType type) {
                TensorLowerer lowerer(str, type);
                auto struct_def = lowerer.lower_tensor_struct_definition();
                printer.print(struct_def);
            }

            void visit(const cTensor *node) override {
                print(node->name, node->type);
            }

            void visit(const Accumulate *node) override { 
                print(node->tensor, node->type);
                node->expr.accept(this); 
            }

            void visit(const Assign *node) override { 
                print(node->tensor, node->type);
                node->expr.accept(this); 
            }
        };

        TensorVisitor visitor(printer);
        cin.accept(&visitor);
    }

    void CINLowerer::lower_partition_function() {
        // Implementation for lowering partition function
    }

}

} // namespace nacho
