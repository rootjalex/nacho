#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
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

        ComputeFunctionLowerer(const std::map<std::string, TensorLowerer> &operand_tensors, const TensorLowerer &result_tensor, const std::vector<CIN> &forall_list, const CIN &cin)
            : operand_tensors(operand_tensors), result_tensor(result_tensor), forall_list(forall_list), cin(cin) {}

        inline std::string get_all_loops_string() {
            std::string all_loops_string = std::accumulate(forall_list.begin(), forall_list.end(), std::string(""),
                [](const std::string &acc, const CIN &c) {
                    return acc + c.as<Forall>()->idx;
                });
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
            return  "precompute_" + get_all_loops_string() + "_kernel";
        }

        inline std::string get_compute_function_name() {
            return  "compute_" + get_all_loops_string() + "_kernel";
        }

        llir::lType lower_result_per_thread_count_struct();

        llir::lStmt lower_precompute_function_for_innermost_sparse_intersection();

        llir::lStmt lower_compute_function_for_innermost_sparse_intersection();

        void add_common_function_body_for_initialization(std::vector<llir::lStmt>& stmts);

    };

} // namespace backend

} // namespace nacho
