#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "Printer.h"
#include "Type.h"
#include "backend/tensor.h"
#include <string>


namespace nacho {
namespace backend {

        struct CINLowerer {

            CIN cin;
            Printer printer;
            TensorLowerer result_tensor;
            std::vector<TensorLowerer> operand_tensors;
            CINLowerer(CIN cin, std::ostream &os);

            std::vector<std::string> get_loop_order();
            void lower_cin();
            void lower_struct_definitions();
            void lower_partition_function();
            void lower_work_functions();
            // is_innermost_sparse_intersection checks if the given CIN represents an innermost sparse intersection. 
            // this also returns true if the CIN does not have any sparse intersection.
            bool is_innermost_sparse_intersection();
        };



} // namespace backend

} // namespace nacho
