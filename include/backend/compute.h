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
        bool requires_operand_pos_map;
        // Track iterator symbols already declared within the generated kernel
        // body so nested lowering can assign without redeclaring.
        std::set<std::string> declared_iter_symbols;
        std::set<std::string> declared_stop_symbols;

        ComputeKernelLowerer(
            std::map<std::string, TensorLowerer> &operand_tensors,
            TensorLowerer &result_tensor,
            std::map<std::string, TensorLowerer> &active_phase_tensors,
            const std::vector<CIN> &forall_loops, const CIN &cin,
            int previous_sparse_intersection_level, int current_sparse_intersection_level, int next_sparse_intersection_level)
            : BaseKernelLowerer(operand_tensors, result_tensor, active_phase_tensors, forall_loops, previous_sparse_intersection_level, current_sparse_intersection_level, next_sparse_intersection_level), cin(cin) {}

        llir::lStmt lower_precompute_kernel();
        std::vector<llir::Function::Argument> get_precompute_kernel_args();

        llir::lStmt lower_compute_kernel();
        std::vector<llir::Function::Argument> get_compute_kernel_args();

        llir::lStmt lower_loop(CIN loop,
            const std::set<Seq, SeqLessThan> &defined,
            bool is_precompute, int loop_level);

        llir::lStmt lower_assignment_statement(CIN assign, bool is_precompute);

        void append_partition_load_statements(std::vector<llir::lStmt> &stmts);
    };

} // namespace backend

} // namespace nacho
