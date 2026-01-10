#include "backend/partition.h"
#include "Visitor.h"
#include "Error.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
namespace nacho {
namespace backend {

// Check if the CIN represents an innermost sparse intersection
// This also returns true if the CIN does not have any sparse intersection.
// TODO : There are some exceptions to this like (A: DCSR , B:Dense-Dense) a_ij*b_ij
bool PartitionFunctionLowerer::is_innermost_sparse_intersection() {
    struct Checker : public Visitor {
        bool is_innermost_sparse = true;
        bool found_sparse_intersection = false;

        void visit(const Intersect * node) override{

            // This means this is a sparse intersection
            if(node->is_sparse) {
                found_sparse_intersection = true;
            }
            node->a.accept(this);
            node->b.accept(this);
        }

        void visit(const Forall * node) override{
            // if we already found a sparse intersection, 
            // that means this forall is inside a sparse intersection forall
            // hence cin is not innermost sparse
            if(found_sparse_intersection) {
                is_innermost_sparse = false;
            }
            node->seq.accept(this);
            node->body.accept(this);
        }
    };

    Checker checker;

    if (const Forall *forall = cin.as<Forall>()) {
        forall->body.accept(&checker);
    } else {
        internal_assert(false) << "Root node of CIN is not a Forall.";
    }

    return checker.is_innermost_sparse;
    
}


// llir::lStmt PartitionFunctionLowerer::lower_innermost_sparse_intersection() {
//     llir::lType index_t = llir::Generic_t::make("index_t");
//     llir::lType value_t = llir::Generic_t::make("value_t");
//     std::vector<std::string> generics = {"index_t", "value_t"};

//     std::vector<llir::Function::Attribute> attributes = {
//         llir::Function::global};

//     std::vector<llir::Function::Argument> args;
//     llir::lType ret_type;
//     std::string name;
//     llir::lStmt body;



//     return llir::Function::make(generics, args, ret_type, name, body);
// }

} // namespace backend

} // namespace nacho
