#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "backend/tensor.h"
#include <numeric>
#include <map>
namespace nacho {
namespace backend {

struct PartitionFunctionLowerer {
    std::map<std::string, TensorLowerer> &operand_tensors;
    std::vector<CIN> forall_list;
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    // Lower the partitioning information from the CIN to the LLIR.
    PartitionFunctionLowerer(std::map<std::string, TensorLowerer> &operand_tensors, const std::vector<CIN> &forall_list)
        : operand_tensors(operand_tensors), forall_list(forall_list) {}

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

    llir::lStmt lower_partition_loop(int loop_index, llir::lExpr total_work_var, llir::lExpr rem_count_var, bool is_last_loop);

    inline std::string get_partition_struct_name() {
        return "partition_" + get_partition_all_loops_string();
    }

};


    
} // namespace backend

} // namespace nacho
