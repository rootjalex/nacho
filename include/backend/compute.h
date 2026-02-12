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

namespace nacho {
namespace backend {

    struct ComputeFunctionLowerer {
        std::map<std::string, TensorLowerer> operand_tensors;
        TensorLowerer result_tensor;
        std::vector<CIN> forall_list;
        CIN cin;
        // Memoized lattices, used for both compute and precompute.
        std::map<Seq, Lattice, SeqLessThan> lattices;
        int previous_sparse_intersection;
        int current_sparse_intersection;
        int next_sparse_intersection;
        std::map<std::string, TensorLowerer> included_tensors;

        ComputeFunctionLowerer(
            const std::map<std::string, TensorLowerer> &operand_tensors,
            const TensorLowerer &result_tensor,
            const std::map<std::string, TensorLowerer> &included_tensors,
            const std::vector<CIN> &forall_list, const CIN &cin, 
            int previous_sparse_intersection, int current_sparse_intersection,int next_sparse_intersection)
            : operand_tensors(operand_tensors), result_tensor(result_tensor),
              forall_list(forall_list), cin(cin), 
              previous_sparse_intersection(previous_sparse_intersection),
              current_sparse_intersection(current_sparse_intersection),
              next_sparse_intersection(next_sparse_intersection),
              included_tensors(included_tensors) {}

        inline std::string get_all_loops_string(int level) {
            std::string all_loops_string = "";
            for(int i=0;i<=level;i++)
                all_loops_string += forall_list[i].as<Forall>()->idx;
            return all_loops_string;
        }

        inline std::string get_counts_struct_name() {
            return "result_per_thread_count";
        }

        inline std::string get_counts_field_name(const std::string &index) {
            return "dim_" + index + "_count";
        }

        inline std::string get_iterator_name(TensorLowerer& tensor, const std::string &index) {
            return "idx_"+tensor.tensor_name+"_"+index;
        }

        inline std::string get_end_iterator_name(TensorLowerer& tensor, const std::string &index) {
            return "end_"+tensor.tensor_name+"_"+index;
        }

        inline std::string get_precompute_function_name() {
            return  "precompute_" + get_all_loops_string(current_sparse_intersection) + "_kernel";
        }

        inline std::string get_compute_function_name() {
            return  "compute_" + get_all_loops_string(current_sparse_intersection) + "_kernel";
        }

        inline std::string get_start_name(const std::string &suffix) const {
            return "start_" + suffix;
        }

        inline std::string get_end_name(const std::string &suffix) const {
            return "end_" + suffix;
        }

        llir::lType lower_result_per_thread_count_struct();

        llir::lStmt lower_precompute_function();

        llir::lStmt lower_compute_function();

        llir::lStmt lower_loop(CIN loop,
                               const std::set<Seq, SeqLessThan> &defined,
                               const std::set<Seq, SeqLessThan> &lookups,
                               bool is_precompute);

        llir::lStmt lower_assign_statement(CIN assign, bool is_precompute);

        void add_partition_assignments(std::vector<llir::lStmt> &stmts);

        // TODO: deduplicate with
        // PartitionFunctionLowerer::get_partition_struct_name()
        inline std::string get_partition_struct_name() const {
            std::string partition_argtype_name = "partition_";
            CIN strip_idxs = cin;
            while (const auto *forall = strip_idxs.as<Forall>()) {
                partition_argtype_name += forall->idx;
                strip_idxs = forall->body;
            }
            return partition_argtype_name;
        }
    };

} // namespace backend

} // namespace nacho
