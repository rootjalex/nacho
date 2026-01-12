#include "backend/partition.h"
#include "Visitor.h"
#include "Error.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include <numeric>
namespace nacho {
namespace backend {

    llir::lStmt PartitionFunctionLowerer::lower_innermost_sparse_intersection() {

        llir::lType index_t = llir::Generic_t::make("index_t");
        llir::lType value_t = llir::Generic_t::make("value_t");
        std::vector<std::string> generics = {"index_t", "value_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::global};

        std::vector<llir::Function::Argument> args;
        llir::lType ret_type;
        std::string name;
        llir::lStmt body;

        name = get_partition_function_name();

        ret_type = llir::Generic_t::make("void");



        for(auto tensor: operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = llir::Ptr_t::make(llir::Generic_t::make(tensor.get_struct_name()+"<index_t, value_t>")), .name = tensor.tensor_name
            });
        }
        for(auto tensor:operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = true, .type = llir::Ptr_t::make(llir::Generic_t::make(tensor.get_index_struct_name()+"<index_t, value_t>")), .name = "partitions_"+tensor.tensor_name
            });
        }

        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "per_thread_work"
        });

        std::vector<llir::lStmt> stmts;

        // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        stmts.emplace_back(
            llir::Declare::make(
                llir::Int_t::make(32),
                "thread_id",
                llir::lBinOp::make(
                    llir::lBinOp::Add,
                    llir::lBinOp::make(
                        llir::lBinOp::Mul,
                        llir::lVar::make(index_t, "blockIdx.x"),
                        llir::lVar::make(index_t, "blockDim.x")
                    ),
                    llir::lVar::make(index_t, "threadIdx.x")
                )
            )
        );

        // index_t count = thread_id * per_thread_work;
        stmts.emplace_back(
            llir::Declare::make(
                llir::Generic_t::make("index_t"),
                "count",
                llir::lBinOp::make(
                    llir::lBinOp::Mul,
                    llir::lVar::make(index_t, "thread_id"),
                    llir::lVar::make(index_t, "per_thread_work")
                )
            )
        );


        stmts.emplace_back(
            llir::Return::make()
        );
        body = llir::Sequence::make(std::move(stmts));

        return llir::Function::make(std::move(generics), std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body));
    }

} // namespace backend

} // namespace nacho
