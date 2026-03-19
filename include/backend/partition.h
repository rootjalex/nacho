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
    PartitionKernelLowerer(std::map<std::string, TensorLowerer> &operand_tensors, TensorLowerer& result_tensor, std::map<std::string, TensorLowerer> &included_tensors, 
        const std::vector<CIN> &forall_list, 
        LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, LoopNum next_sparse_intersection,
        LoopNum reduction_loop)
        : BaseKernelLowerer(operand_tensors, result_tensor, included_tensors, forall_list, previous_sparse_intersection, current_sparse_intersection, next_sparse_intersection, reduction_loop) {
            for (const auto& it : operand_tensors) {
                tensor_partitioned_vars[it.second.tensor_name] = llir::lVar::make(llir::Generic_t::make("bool"), "is_" + it.second.tensor_name + "_partitioned");
            }
            internal_assert(included_tensors.size()>0) << "Expected atleast 1 included tensor for partitioning";
        }

    llir::lStmt lower_partition_kernel();

    /// Lower the struct definition which holds the partition values for this kernel
    llir::lType lower_partition_struct_definition();

    llir::lStmt lower_partition_loop(LoopNum loop_num, bool is_last_loop, bool need_to_exclude_tensors_at_runtime);

    llir::lStmt lower_partition_loop_from_work_offsets(LoopNum loop_num, bool need_to_exclude_tensors_at_runtime);

    llir::lStmt lower_mergepath_partition_loop(LoopNum loop_num, bool is_last_loop, std::vector<TensorLowerer>& tensors_with_curr_dim_sparse);

    llir::lStmt lower_trivial_partition_loop(LoopNum loop_num, bool is_last_loop);

    llir::lStmt get_store_partition_statements(LoopNum loop_num, llir::lExpr index_value, bool need_to_exclude_tensors_at_runtime, bool is_last_loop);

    llir::lExpr get_call_work_function_expr(LoopNum loop_num, TensorLowerer& tensor, llir::lExpr index_value);

    llir::lStmt get_statements_to_find_sparse_position(LoopNum loop_num, TensorLowerer& tensor, llir::lExpr index_value, bool need_to_exclude_tensors_at_runtime);

    llir::lExpr get_sparse_dim_start_expr(TensorLowerer& tensor, const std::string& forall_idx);
    llir::lExpr get_sparse_dim_end_expr(TensorLowerer& tensor, const std::string& forall_idx);

    std::vector<llir::Function::Argument> get_kernel_args();

};


    
} // namespace backend

} // namespace nacho
