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

            llir::lType index_t  = llir::Generic_t::make("index_t");

            CIN cin;
            Printer printer;
            TensorLowerer result_tensor;
            std::map<std::string, TensorLowerer> operand_tensors;
            CINLowerer(CIN cin, std::ostream &os);

            std::vector<std::string> get_loop_order();
            std::vector<CIN> get_forall_list();
            void lower_cin();
            void lower_struct_definitions(int last_sparse_intersection);
            std::vector<int> get_all_sparse_intersection_levels(CIN& cin);
            CIN get_modified_cin_for_sparse_intersection(int target_level, CIN& cin);

            void lower_binary_search_function();
            llir::lType lower_result_pos_to_operand_pos_map_struct(int last_sparse_intersection);
            std::map<std::string, TensorLowerer> get_included_tensors_for_level(int level);

        };



} // namespace backend

} // namespace nacho
