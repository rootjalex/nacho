#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "backend/tensor.h"
#include "Simplify.h"
#include "Error.h"
#include <numeric>
#include <map>
#include "backend/base_lowerer.h"
namespace nacho {
namespace backend {

struct PartitionKernelLowerer: public BaseKernelLowerer {

    std::map<std::string, llir::lExpr> tensor_partitioned_vars;

    llir::lExpr work_var = llir::lVar::make(index_t, "work");
    llir::lExpr rem_count_var = llir::lVar::make(index_t, "rem_count");


    // Lower the partitioning information from the CIN to the LLIR.
    PartitionKernelLowerer(std::map<std::string, TensorLowerer> &operand_tensors, TensorLowerer& result_tensor, std::map<std::string, TensorLowerer> &active_phase_tensors, 
        const std::vector<CIN> &forall_loops, 
        int previous_sparse_intersection_level, int current_sparse_intersection_level, int next_sparse_intersection_level)
        : BaseKernelLowerer(operand_tensors, result_tensor, active_phase_tensors, forall_loops, previous_sparse_intersection_level, current_sparse_intersection_level, next_sparse_intersection_level) {
            for (const auto& it : operand_tensors) {
                tensor_partitioned_vars[it.second.tensor_name] = llir::lVar::make(llir::Generic_t::make("bool"), "is_" + it.second.tensor_name + "_partitioned");
            }
            internal_assert(active_phase_tensors.size()>0) << "Expected atleast 1 included tensor for partitioning";
        }

    llir::lStmt lower_partition_kernel();
    std::vector<llir::Function::Argument> get_kernel_args();

    /// Lower the struct definition which holds the partition values for this kernel
    llir::lType lower_partition_struct_definition();

    llir::lStmt lower_partition_loop(int loop_index, bool is_last_loop, bool should_exclude_tensors_at_runtime);

    llir::lStmt lower_partition_loop_using_work_offsets(int loop_index, bool should_exclude_tensors_at_runtime);

    llir::lStmt lower_mergepath_partition_loop(int loop_index, bool is_last_loop, std::vector<TensorLowerer>& tensors_with_curr_dim_sparse);

    llir::lStmt lower_simple_partition_loop(int loop_index, bool is_last_loop);

    llir::lStmt lower_row_aligned_last_loop_partition(int loop_index);

    llir::lStmt build_partition_store_statements(int loop_index, llir::lExpr index_value, bool should_exclude_tensors_at_runtime, bool is_last_loop);

    llir::lExpr build_work_function_call_expr(int loop_index, bool is_last_loop, TensorLowerer& tensor, llir::lExpr index_value);

    llir::lStmt build_sparse_position_search_statements(int loop_index, TensorLowerer& tensor, llir::lExpr index_value, bool should_exclude_tensors_at_runtime);

    llir::lExpr get_sparse_dim_start_expr(TensorLowerer& tensor, const std::string& forall_idx);
    llir::lExpr get_sparse_dim_end_expr(TensorLowerer& tensor, const std::string& forall_idx);

};


    
} // namespace backend

} // namespace nacho
