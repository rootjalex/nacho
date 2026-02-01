#include "backend/compile.h"
#include "backend/partition.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Error.h"
#include "Format.h"
#include "Type.h"
#include "backend/tensor.h"
#include "backend/compute.h"
#include "Printer.h"
#include <map>

namespace nacho {
namespace backend {

    CINLowerer::CINLowerer(CIN cin, std::ostream &os) : cin(std::move(cin)), printer(os) {
        struct TensorVisitor : Visitor {
            std::map<std::string, TensorLowerer> &operand_tensors;
            TensorLowerer &result_tensor;
            TensorVisitor(std::map<std::string, TensorLowerer> &operand_tensors, TensorLowerer &result_tensor)
                : operand_tensors(operand_tensors), result_tensor(result_tensor) {}

            void add_tensor(std::string str, TensorType type) {
                TensorLowerer lowerer(str, type);
                operand_tensors[str] = lowerer;
            }

            void visit(const cTensor *node) override {
                add_tensor(node->name, node->type);
            }

            void visit(const Accumulate *node) override { 
                result_tensor = TensorLowerer(node->tensor, node->type, true);
                node->expr.accept(this); 
            }

            void visit(const Assign *node) override { 
                result_tensor = TensorLowerer(node->tensor, node->type, true);
                node->expr.accept(this); 
            }
        };

        TensorVisitor visitor(operand_tensors, result_tensor);
        this->cin.accept(&visitor);
    }

    std::vector<std::string> CINLowerer::get_loop_order() {
        std::vector<std::string> loop_order;
        struct ForallVisitor : Visitor {
            std::vector<std::string> &loop_order;
            ForallVisitor(std::vector<std::string> &loop_order) : loop_order(loop_order) {}

            void visit(const Forall *node) override {
                loop_order.push_back(node->idx);
                node->body.accept(this);
            }
        };
        ForallVisitor visitor(loop_order);
        cin.accept(&visitor);
        return loop_order;
    }

    std::vector<CIN> CINLowerer::get_forall_list() {
        std::vector<CIN> forall_list;
        struct ForallVisitor : Visitor {
            std::vector<CIN> &forall_list;
            ForallVisitor(std::vector<CIN> &forall_list) : forall_list(forall_list) {}

            void visit(const Forall *node) override {
                forall_list.push_back(node);
                node->body.accept(this);
            }
        };
        ForallVisitor visitor(forall_list);
        cin.accept(&visitor);
        return forall_list;
    }

    void CINLowerer::lower_cin() {
        this->lower_binary_search_function();
        this->lower_struct_definitions();
        this->lower_work_functions();
        this->lower_innermost_sparse_intersection();
    }

    // lower_struct_definitions loweres all the initial struct definitions for the program
    // this includes tensor struct definitions for both the operand and result tensors
    void CINLowerer::lower_struct_definitions() {
        for (auto it : operand_tensors) {
            printer.print(it.second.lower_tensor_struct_definition());
            // printer.print(it.second.lower_tensor_index_definition());
        }
        printer.print(result_tensor.lower_tensor_struct_definition());
    }

    void CINLowerer::lower_work_functions() {
        auto loop_order = get_loop_order();
        for (auto it : operand_tensors) {

            for (int i=0; i<loop_order.size(); i++) {
                auto work_function = it.second.lower_work_function(loop_order, i);
                printer.print(work_function);
            }
            
        }
    }

    void CINLowerer::lower_innermost_sparse_intersection() {
        internal_assert(is_innermost_sparse_intersection()) << "CIN which are not innermost sparse intersection are not supported";

        PartitionFunctionLowerer partition_lowerer(operand_tensors, get_forall_list());

        printer.print(partition_lowerer.lower_partition_struct_definition());
        printer.print(partition_lowerer.lower_partition_kernel_for_innermost_sparse_intersection());

        ComputeFunctionLowerer compute_lowerer(operand_tensors, result_tensor, get_forall_list(), cin);
        
        // Precompute is required only if result tensor has atleast one sparse dim. Else we can
        // directly compute the write location, without the need of offsets.
        if(!result_tensor.tensor_type.format.are_all_lvls_dense()) {
            printer.print(compute_lowerer.lower_result_per_thread_count_struct());
            printer.print(compute_lowerer.lower_precompute_function_for_innermost_sparse_intersection());
        }
        
        printer.print(compute_lowerer.lower_compute_function_for_innermost_sparse_intersection());
    }

    // Check if the CIN represents an innermost sparse intersection
    // This also returns true if the CIN does not have any sparse intersection.
    // TODO : There are some exceptions to this like (A: DCSR , B:Dense-Dense) a_ij*b_ij
    bool CINLowerer::is_innermost_sparse_intersection() {
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

    void CINLowerer::lower_binary_search_function() {
        std::vector<std::string> generics = {"index_t"};
        auto index_t  = llir::Generic_t::make("index_t");

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::device, llir::Function::inline_};

        std::vector<llir::Function::Argument> args;
        llir::lType ret_type;
        std::string name;
        llir::lStmt body;

        name = "binary_search";

        ret_type = index_t;


        args.emplace_back(
            llir::Function::Argument{.mutating = false, .type = llir::Ptr_t::make(index_t), .name = "arr"});
            args.emplace_back(
                llir::Function::Argument{.mutating = false, .type = index_t, .name = "target_value"});
            args.emplace_back(
                llir::Function::Argument{.mutating = false, .type = llir::Int_t::make(32), .name = "start_index"});
            args.emplace_back(
                llir::Function::Argument{.mutating = false, .type = llir::Int_t::make(32), .name = "end_index"});

        std::vector<llir::lStmt> stmts;
        
        llir::lExpr start_var = llir::lVar::make(index_t, "start_index");
        llir::lExpr mid_var = llir::lVar::make(index_t, "mid");
        llir::lExpr end_var = llir::lVar::make(index_t, "end_index");
        llir::lExpr arr_var = llir::lVar::make(llir::Ptr_t::make(index_t), "arr");
        llir::lExpr target_value_var = llir::lVar::make(index_t, "target_value");

        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                "mid",
                start_var + (((end_var - start_var) + 1) / 2)
            )
        );  
        std::vector<llir::lStmt> while_stmts;

        while_stmts.emplace_back(
            llir::Store::make(
                mid_var,
                start_var + (((end_var - start_var) + 1) / 2)
            )
        );

        while_stmts.emplace_back(llir::IfElse::make(
            arr_var[mid_var] <= target_value_var,
            llir::Store::make(start_var, mid_var),
            llir::Store::make(end_var, mid_var - 1)
        ));

        stmts.emplace_back(
            llir::While::make(
                start_var < end_var,
                llir::Sequence::make(std::move(while_stmts))
            )
        );

        stmts.emplace_back(
            llir::Store::make(
                mid_var,
                start_var + (((end_var - start_var) + 1) / 2)
            )
        );
        
        stmts.emplace_back(
            llir::Return::make(mid_var)
        );
        body = llir::Sequence::make(std::move(stmts));

        printer.print(llir::Function::make(std::move(generics), std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body)));
    }


}
} // namespace nacho
