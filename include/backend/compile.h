#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "Printer.h"
#include "Type.h"
#include "backend/tensor.h"
#include "backend/stitch_and_generate.h"
#include <string>
#include <map>


namespace nacho {
namespace backend {

        struct CINLowerer {

            llir::lType index_t  = llir::Generic_t::make("index_t");

            CIN cin;
            std::string name;
            TensorLowerer result_tensor;
            TensorLowerer reduced_result_tensor;
            std::map<std::string, TensorLowerer> operand_tensors;
            std::vector<TensorIndex> loop_order;
            // TODO: For multiple sums (inner-sum) this would be a vector of all reduction loops
            std::vector<LoopNum> reductionLoops;
            bool is_scatter_reduction = false;

        // When false, the kernel partitions work once over the whole loop nest instead of
        // re-partitioning at every sparse intersection level. Set before lower_cin().
        bool recursive_partitioning = true;

        // The tensor the entry point returns. A reduction fills result_tensor (`<Z>_temp`,
        // which still carries the reduced dimensions) and contracts it into
        // reduced_result_tensor (`<Z>`).
        const TensorLowerer &output_tensor() const {
            return reduced_result_tensor.tensor_name.empty() ? result_tensor
                                                             : reduced_result_tensor;
        }

            StitchAndGenerate* stitch_and_generate;

            CINLowerer(CIN cin, std::string name, bool is_cpu = false,
                       std::vector<std::string> operand_ordering = {});

            // The operand names in the order the entry point takes them, after any
            // requested ordering has been validated against the expression.
            const std::vector<std::string> &operand_ordering() const {
                return stitch_and_generate->operand_ordering;
            }

            std::vector<TensorIndex> get_loop_order();
            std::vector<CIN> get_forall_list();
            void lower_cin();
            void lower_struct_definitions(LoopNum last_sparse_intersection);
            std::vector<LoopNum> get_all_sparse_intersection_levels(CIN& cin);
            CIN get_modified_cin_for_sparse_intersection(LoopNum target_loop, CIN& cin);

            void lower_binary_search_function(bool is_upper_bound);
            llir::lType lower_result_pos_to_operand_pos_map_struct(LoopNum last_sparse_intersection);
            std::map<std::string, TensorLowerer> get_included_tensors_for_level(LoopNum loop_num);
            void lower_merged_index_equality_func(TensorIndex idx);
            void lower_get_tuple_func_for_merged_index(TensorIndex idx);


        };



} // namespace backend

} // namespace nacho
