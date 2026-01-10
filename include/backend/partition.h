#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
namespace nacho {
namespace backend {

struct PartitionFunctionLowerer {
    CIN cin;
    // Lower the partitioning information from the CIN to the LLIR.
    PartitionFunctionLowerer(const CIN &cin): cin(cin) {}

    // is_innermost_sparse_intersection checks if the given CIN represents an innermost sparse intersection. 
    // this also returns true if the CIN does not have any sparse intersection.
    bool is_innermost_sparse_intersection();

    llir::lStmt lower_innermost_sparse_intersection();

};


    
} // namespace backend

} // namespace nacho
