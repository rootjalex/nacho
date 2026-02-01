#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "Printer.h"
#include "Type.h"
#include "backend/tensor.h"
#include <string>
#include <map>


namespace nacho {
namespace backend {

        struct CINLowerer {

            CIN cin;
            Printer printer;
            TensorLowerer result_tensor;
            std::map<std::string, TensorLowerer> operand_tensors;
            CINLowerer(CIN cin, std::ostream &os);

            std::vector<std::string> get_loop_order();
            std::vector<CIN> get_forall_list();
            void lower_cin();
            void lower_struct_definitions();
            std::vector<int> get_all_sparse_intersection_levels(CIN& cin);
            CIN get_modified_cin_for_sparse_intersection(int target_level, CIN& cin);
            void lower_innermost_sparse_intersection();
            void lower_work_functions();
            // is_innermost_sparse_intersection checks if the given CIN represents an innermost sparse intersection. 
            // this also returns true if the CIN does not have any sparse intersection.
            bool is_innermost_sparse_intersection();
            void lower_binary_search_function();

            std::map<std::string, TensorLowerer> get_included_tensors_for_level(int level);

        };



} // namespace backend

} // namespace nacho
