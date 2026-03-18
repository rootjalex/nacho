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
            std::map<std::string, TensorLowerer> &included_tensors;
            std::vector<CIN> forall_list;
            LoopNum previous_sparse_intersection;
            LoopNum current_sparse_intersection;
            LoopNum next_sparse_intersection;
            LoopNum reduction_loop;
            llir::lType index_t = llir::Generic_t::make("index_t");
            llir::lType value_t = llir::Generic_t::make("value_t");


            BaseKernelLowerer(
                std::map<std::string, TensorLowerer> &operand_tensors,
                TensorLowerer &result_tensor,
                std::map<std::string, TensorLowerer> &included_tensors,
                const std::vector<CIN> &forall_list, 
                LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, LoopNum next_sparse_intersection,
                LoopNum reduction_loop)
                : operand_tensors(operand_tensors), result_tensor(result_tensor), included_tensors(included_tensors),
                  forall_list(forall_list), 
                  previous_sparse_intersection(previous_sparse_intersection),
                  current_sparse_intersection(current_sparse_intersection),
                  next_sparse_intersection(next_sparse_intersection),
                  reduction_loop(reduction_loop) {}
            
            // This is just a helper function to construct the suffix used in giving unique names to various generated kernels
            // The suffix is just the concatenation of all loop indices that the kernel operates on
            inline std::string get_all_loops_string(LoopNum loop_num) {
                std::string all_loops_string = "";
                for(LoopNum i=BEFORE_FIRST_LOOP+1;i<=loop_num;++i)
                    all_loops_string += forall_list[i.get()].as<Forall>()->idx;
                return all_loops_string;
            }

            TensorLowerer get_tensor(std::string tensor_name) {
                if(operand_tensors.find(tensor_name) != operand_tensors.end()) {
                    return operand_tensors[tensor_name];
                } else if(result_tensor.tensor_name == tensor_name) {
                    return result_tensor;
                } else {
                    internal_assert(false) << "Tensor " << tensor_name << " not found in either operand tensors or result tensor.";
                    return TensorLowerer();
                }
            }

            inline std::string get_partition_function_name() {
                return  "partition_" + get_all_loops_string(current_sparse_intersection) + "_kernel";
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
                return "partition_" + get_all_loops_string(current_sparse_intersection);
            }

            inline std::string get_counts_struct_name() {
                return "result_per_thread_count";
            }

            inline std::string get_counts_field_name(const std::string &index) {
                return "dim_" + index + "_count";
            }

            inline std::string get_precompute_function_name() {
                return  "precompute_" + get_all_loops_string(current_sparse_intersection) + "_kernel";
            }

            inline std::string get_compute_function_name() {
                return  "compute_" + get_all_loops_string(current_sparse_intersection) + "_kernel";
            }


            llir::lType lower_result_per_thread_count_struct();

            inline std::string get_result_to_operand_pos_map_struct_name() {
                return "result_to_operand_pos_map";
            }

            inline std::string get_result_to_operand_pos_map_var_name() {
                return result_tensor.tensor_name + "_pos_map";
            }

            bool exists_field_in_result_to_operand_pos_map(const Forall* forall, TensorLowerer& operand_tensor);
            llir::lExpr map_result_pos_to_operand_pos(const Forall* forall, TensorLowerer& operand_tensor, llir::lExpr result_tensor_pos, llir::lExpr result_tensor_coord  = llir::lExpr());

            llir::lExpr get_partition_initializer_expr_for_boundary_cases(LoopNum forall_loop_num, TensorLowerer& tensor, bool is_last_thread);
        };
    }
}


