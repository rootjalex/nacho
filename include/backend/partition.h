#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "backend/tensor.h"
#include "Simplify.h"
#include "Error.h"
#include <numeric>
#include <map>
namespace nacho {
namespace backend {

struct PartitionFunctionLowerer {
    std::map<std::string, TensorLowerer> &operand_tensors;
    TensorLowerer &result_tensor;

    std::map<std::string, TensorLowerer> &included_tensors;
    std::map<std::string, llir::lExpr> tensor_partitioned_vars;
    std::vector<CIN> forall_list;
    int previous_sparse_intersect_forall_id;
    int next_sparse_intersect_forall_id;

    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");

    llir::lExpr work_var = llir::lVar::make(index_t, "work");
    llir::lExpr rem_count_var = llir::lVar::make(index_t, "rem_count");


    // Lower the partitioning information from the CIN to the LLIR.
    PartitionFunctionLowerer(std::map<std::string, TensorLowerer> &operand_tensors, TensorLowerer& result_tensor, std::map<std::string, TensorLowerer> &included_tensors, const std::vector<CIN> &forall_list, int previous_sparse_intersect_forall_id, int next_sparse_intersect_forall_id)
        : operand_tensors(operand_tensors), result_tensor(result_tensor), included_tensors(included_tensors), forall_list(forall_list) {
            this->previous_sparse_intersect_forall_id = previous_sparse_intersect_forall_id;
            this->next_sparse_intersect_forall_id = next_sparse_intersect_forall_id;

            for (const auto& it : operand_tensors) {
                tensor_partitioned_vars[it.second.tensor_name] = llir::lVar::make(llir::Generic_t::make("bool"), "is_" + it.second.tensor_name + "_partitioned");
            }
            internal_assert(included_tensors.size()>0) << "Expected atleast 1 included tensor for partitioning";
        }

    inline std::string get_partition_all_loops_string() {
        std::string all_loops_string = "";
        for(int i=0;i<=next_sparse_intersect_forall_id; i++) {
            all_loops_string += forall_list[i].as<Forall>()->idx;
        }
        return all_loops_string;
    }
    inline std::string get_partition_function_name() {
        return  "partition_" + get_partition_all_loops_string() + "_kernel";
    }

    inline llir::lExpr get_partition_struct_current_thread_field(std::string field_name) {
       return  llir::lArrayAccess::make(
                            llir::lFieldAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(
                                        get_partition_function_name()),
                                    "partitions"),
                                field_name),
                            llir::lVar::make(index_t, "thread_id"));
    }

    llir::lStmt lower_partition_kernel();

    llir::lType lower_partition_struct_definition();

    llir::lStmt lower_partition_loop(int loop_index, bool is_last_loop, bool need_to_exclude_tensors_at_runtime);

    llir::lStmt lower_partition_loop_from_work_offsets(int loop_index, bool need_to_exclude_tensors_at_runtime);

    llir::lStmt lower_mergepath_partition_loop(int loop_index, bool is_last_loop, std::vector<TensorLowerer>& tensors_with_curr_dim_sparse);

    llir::lStmt lower_trivial_partition_loop(int loop_index, bool is_last_loop);

    llir::lStmt get_store_partition_statements(int loop_index, llir::lExpr index_value, bool need_to_exclude_tensors_at_runtime, bool is_last_loop);

    llir::lExpr get_call_work_function_expr(int loop_index, bool is_last_loop, TensorLowerer& tensor, llir::lExpr index_value);

    llir::lStmt get_statements_to_find_sparse_position(int loop_index, TensorLowerer& tensor, llir::lExpr index_value, bool need_to_exclude_tensors_at_runtime);

    llir::lExpr get_sparse_dim_start_expr(TensorLowerer& tensor, const std::string& forall_idx);
    llir::lExpr get_sparse_dim_end_expr(TensorLowerer& tensor, const std::string& forall_idx);

    inline std::string get_partition_struct_name() {
        return "partition";
    }

};


    
} // namespace backend

} // namespace nacho
