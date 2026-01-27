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
    std::map<std::string, TensorLowerer> included_tensors;
    std::vector<CIN> forall_list;
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");

    llir::lExpr total_work_var = llir::lVar::make(index_t, "total_work");
    llir::lExpr rem_count_var = llir::lVar::make(index_t, "rem_count");


    // Lower the partitioning information from the CIN to the LLIR.
    PartitionFunctionLowerer(std::map<std::string, TensorLowerer> &operand_tensors, const std::vector<CIN> &forall_list)
        : operand_tensors(operand_tensors), forall_list(forall_list) {
            auto last_forall = forall_list.back().as<Forall>();
            std::vector<Seq> locators = get_dense_locators(last_forall->seq);

            // included tensors are the tensors which are included in the work
            // calculation. Non-included tensors are not co-iterated and instead looked up.
            std::map<std::string, TensorLowerer> excluded_tensors;
            for(const auto &loc : locators) {
                const auto *index = loc.as<Index>();
                if (!index){
                    internal_assert(false) << "Expected Index node in locator sequence: " << loc;
                }
                for(const auto &it : operand_tensors) {
                    if (it.second.tensor_name == index->tensor) {
                        excluded_tensors[it.second.tensor_name] = it.second;
                    }
                }
            }
            for(auto it : operand_tensors) {
                if(excluded_tensors.find(it.first) == excluded_tensors.end()) {
                    included_tensors[it.first] = it.second;
                }
            }

            internal_assert(included_tensors.size()>0) << "Expected atleast 1 included tensor for partitioning";
        }

    inline std::string get_partition_all_loops_string() {
        std::string all_loops_string = std::accumulate(forall_list.begin(), forall_list.end(), std::string(""),
            [](const std::string &acc, const CIN &c) {
                return acc + c.as<Forall>()->idx;
            });
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

    llir::lStmt lower_partition_kernel_for_innermost_sparse_intersection();

    llir::lType lower_partition_struct_definition();

    llir::lStmt lower_partition_loop(int loop_index, bool is_last_loop);

    llir::lStmt lower_mergepath_partition_loop(int loop_index, bool is_last_loop, std::vector<TensorLowerer>& tensors_with_curr_dim_sparse);

    llir::lStmt lower_trivial_partition_loop(int loop_index, bool is_last_loop);

    llir::lStmt get_store_partition_statements(int loop_index, bool is_last_loop, llir::lExpr index_value);

    llir::lExpr get_call_work_function_expr(int loop_index, bool is_last_loop, TensorLowerer& tensor, llir::lExpr index_value);

    llir::lStmt get_statements_to_find_sparse_position(int loop_index, bool is_last_loop, TensorLowerer& tensor, llir::lExpr index_value);

    inline std::string get_partition_struct_name() {
        return "partition_" + get_partition_all_loops_string();
    }

};


    
} // namespace backend

} // namespace nacho
