#include "backend/compute.h"
#include "IRFwdDecl.h"
#include "llir/Function.h"
#include "llir/LLIR.h"


namespace nacho {
namespace backend {

    llir::lType ComputeFunctionLowerer::lower_result_per_thread_count_struct() {
        llir::lType index_t = llir::Generic_t::make("index_t");
        std::vector<std::string> generics = {"index_t"};
        std::vector<std::pair<std::string, llir::lType>> fields;


        for (int i=0; i<result_tensor.tensor_type.format.levels.size(); i++) {
            auto index = result_tensor.tensor_type.format.levels[i].index;
            if(is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(index))) {
                fields.emplace_back(get_counts_field_name(index), llir::Ptr_t::make(index_t));
            }
        }

        return llir::Struct_t::make(get_counts_struct_name(), std::move(fields), std::move(generics));
    }

    void ComputeFunctionLowerer::add_common_function_body_for_initialization(std::vector<llir::lStmt>& stmts) {
        llir::lType index_t = llir::Generic_t::make("index_t");
        llir::lType value_t = llir::Generic_t::make("value_t");
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

        llir::lExpr thread_id_var = llir::lVar::make(index_t, "thread_id");

        // Generates the following for all tensors X index combinations
        // index_t start_B_i = partitions_B.n_p[thread_id];
        // index_t end_B_i = B.dim_i_length - 1;
        // if (thread_id < ((gridDim.x * blockDim.x) - 1)) {
        //     end_B_i = partitions_B.i_p[thread_id + 1];
        // }
        // TODO: end_B calculation might need some changing when non lowermost intersections (see spgemm.cu)
        for(int i=0; i<operand_tensors.size(); i++) {
            auto tensor = operand_tensors[i];
            for(int j=0;j<tensor.tensor_type.format.levels.size();j++) {
                auto level = tensor.tensor_type.format.levels[j];
                
                stmts.emplace_back(
                    llir::Declare::make(
                        index_t,
                        get_iterator_name(tensor, level.index),
                        llir::lArrayAccess::make(
                            llir::lFieldAccess::make(
                                llir::lVar::make(
                                    llir::Ptr_t::make(llir::Generic_t::make(tensor.get_index_struct_name())), 
                                    "partitions_"+tensor.tensor_name
                                ),
                                llir::lVar::make(index_t, level.index + (is_sparse_format(level.format) ? "_p" : ""))
                            ),
                            thread_id_var
                        )
                    )
                );
                stmts.emplace_back(
                    llir::Declare::make(
                        index_t,
                        get_end_iterator_name(tensor, level.index),
                        llir::lBinOp::make(
                            llir::lBinOp::Sub,
                            llir::lFieldAccess::make(
                                llir::lVar::make(
                                    llir::Ptr_t::make(llir::Generic_t::make(tensor.get_struct_name())), 
                                    tensor.tensor_name
                                ),
                                llir::lVar::make(
                                    index_t, 
                                    is_sparse_format(level.format) ? 
                                    tensor.get_length_field_name(level.index) : tensor.get_size_field_name(level.index))
                            ),
                            llir::lConst::make((int64_t)1)
                        )
                    )
                );
                stmts.emplace_back(
                    llir::IfElse::make(
                        llir::lBinOp::make(
                            llir::lBinOp::Lt,
                            thread_id_var,
                            llir::lBinOp::make(
                                llir::lBinOp::Sub,
                                llir::lBinOp::make(
                                    llir::lBinOp::Mul,
                                    llir::lVar::make(index_t, "gridDim.x"),
                                    llir::lVar::make(index_t, "blockDim.x")
                                ),
                                llir::lConst::make((int64_t)1)
                            )
                        ),
                        llir::Store::make(
                            llir::lVar::make(index_t,get_end_iterator_name(tensor, level.index)),
                            llir::lArrayAccess::make(
                                llir::lFieldAccess::make(
                                    llir::lVar::make(
                                        llir::Ptr_t::make(llir::Generic_t::make(tensor.get_index_struct_name())), 
                                        "partitions_"+tensor.tensor_name
                                    ),
                                    llir::lVar::make(index_t, level.index + (is_sparse_format(level.format) ? "_p" : ""))
                                ),
                                llir::lBinOp::make(
                                    llir::lBinOp::Add,
                                    thread_id_var,
                                    llir::lConst::make((int64_t)1)
                                )
                            )  
                        ),
                        nullptr
                    )
                );

            }
        }

        for(auto tensor: operand_tensors) {
            auto last_forall = forall_list[forall_list.size()-1].as<Forall>();
            if(tensor.tensor_type.format.level_exists(last_forall->idx)) {
                stmts.emplace_back(
                    llir::BaseExpr::make(
                        llir::lIncrement::make(
                            llir::lVar::make(index_t, get_iterator_name(tensor, last_forall->idx))
                        )
                    )
                );
            }
        }
        return ;
    }

    llir::lStmt ComputeFunctionLowerer::lower_precompute_function_for_innermost_sparse_intersection() {
        llir::lType index_t = llir::Generic_t::make("index_t");
        llir::lType value_t = llir::Generic_t::make("value_t");
        std::vector<std::string> generics = {"index_t", "value_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::global};

        std::vector<llir::Function::Argument> args;
        llir::lType ret_type;
        std::string name;
        llir::lStmt body;

        name = get_precompute_function_name();

        ret_type = llir::Generic_t::make("void");

        for(auto tensor: operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = llir::Generic_t::make(tensor.get_struct_name()+"<index_t, value_t>"), .name = tensor.tensor_name
            });
        }
        for(auto tensor:operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = true, .type = llir::Ptr_t::make(llir::Generic_t::make(tensor.get_index_struct_name()+"<index_t, value_t>")), .name = "partitions_"+tensor.tensor_name
            });
        }

        args.emplace_back(llir::Function::Argument{
            .mutating = true, .type = llir::Generic_t::make(get_counts_struct_name()+"<index_t>"), .name = "count_offsets"
        });

        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "per_thread_work"
        });
        
        std::vector<llir::lStmt> stmts;

        // Add common initialization statements
        add_common_function_body_for_initialization(stmts);

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


        // Declare local count variables to be used to calculat the offsets into result. These will be stored in 
        // count_offsets at the end.
        for (int i=0; i<result_tensor.tensor_type.format.levels.size(); i++) {
            auto index = result_tensor.tensor_type.format.levels[i].index;
            if(is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(index))) {
               stmts.emplace_back(
                    llir::Declare::make(
                        index_t,
                        "count_"+index,
                        llir::lConst::make((int64_t)0)
                    )
                );
            }
        }





        for (int i=0; i<result_tensor.tensor_type.format.levels.size(); i++) {
            auto index = result_tensor.tensor_type.format.levels[i].index;
            if(is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(index))) {
               stmts.emplace_back(
                    llir::Store::make(
                        llir::lArrayAccess::make(
                            llir::lFieldAccess::make(
                                llir::lVar::make(llir::Generic_t::make(get_counts_struct_name()), "count_offsets"),
                                llir::lVar::make(index_t,get_counts_field_name(index))
                            ),
                            llir::lVar::make(index_t, "thread_id")
                        ),
                        llir::lVar::make(index_t, "count_"+index)
                    )
                );
            }
        }

        stmts.emplace_back(
            llir::Return::make()
        );
        body = llir::Sequence::make(std::move(stmts));

        return llir::Function::make(std::move(generics), std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body));        
    }


    llir::lStmt ComputeFunctionLowerer::lower_compute_function_for_innermost_sparse_intersection() {
        llir::lType index_t = llir::Generic_t::make("index_t");
        llir::lType value_t = llir::Generic_t::make("value_t");
        std::vector<std::string> generics = {"index_t", "value_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::global};

        std::vector<llir::Function::Argument> args;
        llir::lType ret_type;
        std::string name;
        llir::lStmt body;

        name = get_compute_function_name();

        ret_type = llir::Generic_t::make("void");

        for(auto tensor: operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = llir::Generic_t::make(tensor.get_struct_name()+"<index_t, value_t>"), .name = tensor.tensor_name
            });
        }
        for(auto tensor:operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = true, .type = llir::Ptr_t::make(llir::Generic_t::make(tensor.get_index_struct_name()+"<index_t, value_t>")), .name = "partitions_"+tensor.tensor_name
            });
        }

        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = llir::Generic_t::make(get_counts_struct_name()+"<index_t>"), .name = "count_offsets"
        });

        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "per_thread_work"
        });

        args.emplace_back(llir::Function::Argument{
            .mutating = true, .type = llir::Ptr_t::make(llir::Generic_t::make(result_tensor.get_struct_name()+"<index_t, value_t>")), .name = "result"
        });
        
        std::vector<llir::lStmt> stmts;

        // Add common initialization statements
        add_common_function_body_for_initialization(stmts);

        stmts.emplace_back(
            llir::Return::make()
        );
        body = llir::Sequence::make(std::move(stmts));

        return llir::Function::make(std::move(generics), std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body));
    }
} // namespace backend

} // namespace nacho
