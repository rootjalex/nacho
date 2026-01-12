#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "backend/tensor.h"
#include <numeric>
namespace nacho {
namespace backend {

struct PartitionFunctionLowerer {
    std::vector<TensorLowerer> operand_tensors;
    std::vector<std::string> partitioning_loops_order;
    // Lower the partitioning information from the CIN to the LLIR.
    PartitionFunctionLowerer(const std::vector<TensorLowerer> &operand_tensors, const std::vector<std::string> &partitioning_loops_order)
        : operand_tensors(operand_tensors), partitioning_loops_order(partitioning_loops_order) {}

    inline std::string get_partition_all_loops_string() {
        std::string all_loops_string = std::accumulate(partitioning_loops_order.begin(), partitioning_loops_order.end(), std::string(""));
        return all_loops_string;
    }
    inline std::string get_partition_function_name() {
        return  "partition_" + get_partition_all_loops_string() + "_kernel";
    }
    llir::lStmt lower_innermost_sparse_intersection();

};


    
} // namespace backend

} // namespace nacho
