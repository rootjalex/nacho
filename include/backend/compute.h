#pragma once

#include "CIN.h"
#include "Equality.h"
#include "IRFwdDecl.h"
#include "Lattice.h"
#include "Seq.h"

#include "backend/tensor.h"
#include "backend/partition.h"
#include <numeric>
#include <map>
#include "backend/base_lowerer.h"


namespace nacho {
namespace backend {
    struct ComputeKernelLowerer : public BaseKernelLowerer {
        CIN cin;
        // Memoized lattices, used for both compute and precompute.
        std::map<Seq, Lattice, SeqLessThan> lattices;
        bool need_operand_pos_map;
        TensorLowerer output_write_tensor;

        ComputeKernelLowerer(
            std::map<std::string, TensorLowerer> &operand_tensors,
            TensorLowerer &result_tensor,
            std::map<std::string, TensorLowerer> &included_tensors,
            const std::vector<CIN> &forall_list, const CIN &cin,
            LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, LoopNum next_sparse_intersection,
            std::vector<LoopNum> reduction_loops, bool &is_scatter_reduction, TensorLowerer &reduced_result_tensor)
            : BaseKernelLowerer(operand_tensors, result_tensor, included_tensors, forall_list, previous_sparse_intersection, current_sparse_intersection, next_sparse_intersection, reduction_loops, is_scatter_reduction, reduced_result_tensor), cin(cin) {
                output_write_tensor = next_sparse_intersection.get() == forall_list.size() && reduction_loops.size() > 0 && !is_scatter_reduction 
                    ? reduced_result_tensor : result_tensor;
            }

        llir::lStmt lower_precompute_function();

        llir::lStmt lower_compute_function();

        llir::lStmt lower_loop(CIN loop,
            const std::set<Seq, SeqLessThan> &defined,
            bool is_precompute, LoopNum loop_num, llir::lExpr offset_write_cond);

        llir::lStmt lower_assign_statement(CIN assign, bool is_precompute);

        void add_partition_assignments(std::vector<llir::lStmt> &stmts);
    };

} // namespace backend

} // namespace nacho
