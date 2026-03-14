#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Error.h"
#include "Format.h"
#include "Type.h"
#include "Seq.h"
#include "backend/tensor.h"
#include "Printer.h"
#include <map>

namespace nacho {
    namespace backend {

        struct BaseKernelLowerer{
            std::map<std::string, TensorLowerer> &operand_tensors;
            TensorLowerer &result_tensor;

            std::map<std::string, TensorLowerer> &active_phase_tensors;
            std::vector<CIN> forall_loops;
            int previous_sparse_intersection_level;
            int current_sparse_intersection_level;
            int next_sparse_intersection_level;

            llir::lType index_t = llir::Generic_t::make("index_t");
            llir::lType value_t = llir::Generic_t::make("value_t");


            BaseKernelLowerer(
                std::map<std::string, TensorLowerer> &operand_tensors,
                TensorLowerer &result_tensor,
                std::map<std::string, TensorLowerer> &active_phase_tensors,
                const std::vector<CIN> &forall_loops, 
                int previous_sparse_intersection_level, int current_sparse_intersection_level,int next_sparse_intersection_level)
                : operand_tensors(operand_tensors), result_tensor(result_tensor), active_phase_tensors(active_phase_tensors),
                  forall_loops(forall_loops), 
                  previous_sparse_intersection_level(previous_sparse_intersection_level),
                  current_sparse_intersection_level(current_sparse_intersection_level),
                  next_sparse_intersection_level(next_sparse_intersection_level) {}

            
            // This is just a helper function to construct the suffix used in giving unique names to various generated kernels
            // The suffix is just the concatenation of all loop indices that the kernel operates on
            inline std::string get_loop_suffix_up_to_level(int level) {
                std::string all_loops_string = "";
                for(int i=0;i<=level;i++)
                    all_loops_string += forall_loops[i].as<Forall>()->idx;
                return all_loops_string;
            }

            inline std::string get_partition_function_name() {
                return  "partition_" + get_loop_suffix_up_to_level(current_sparse_intersection_level) + "_kernel";
            }

            inline llir::lExpr get_partition_struct_current_thread_field(std::string field_name) {
                return  llir::lArrayAccess::make(
                                        llir::lFieldAccess::make(
                                            llir::lVar::make(
                                                llir::Generic_t::make(
                                                    get_partition_struct_name()),
                                                "partitions"),
                                            field_name),
                                        llir::lVar::make(index_t, "thread_id"));
            }

            inline std::string get_partition_struct_name() {
                return "partition_" + get_loop_suffix_up_to_level(current_sparse_intersection_level);
            }

            inline std::string get_counts_struct_name() {
                return "result_per_thread_count";
            }

            inline std::string get_counts_field_name(const std::string &index) {
                return "dim_" + index + "_count";
            }

            inline std::string get_precompute_function_name() {
                return  "precompute_" + get_loop_suffix_up_to_level(current_sparse_intersection_level) + "_kernel";
            }

            inline std::string get_compute_function_name() {
                return  "compute_" + get_loop_suffix_up_to_level(current_sparse_intersection_level) + "_kernel";
            }


            llir::lType lower_result_per_thread_count_struct();

            inline std::string get_result_to_operand_pos_map_struct_name() {
                return "result_to_operand_pos_map";
            }

            inline std::string get_result_to_operand_pos_map_var_name() {
                return result_tensor.tensor_name + "_pos_map";
            }

            bool has_result_to_operand_pos_field(const Forall* forall, TensorLowerer& operand_tensor);
            llir::lExpr build_operand_position_from_result_position(const Forall* forall, TensorLowerer& operand_tensor, llir::lExpr result_tensor_pos, llir::lExpr result_tensor_coord  = llir::lExpr());

            llir::lExpr build_partition_boundary_initializer_expr(const int forall_level, TensorLowerer& tensor, bool is_last_thread);
        };
    }
}


