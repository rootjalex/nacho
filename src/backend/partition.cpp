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
                .mutating = false, .type = llir::Generic_t::make(tensor.get_struct_name()+"<index_t, value_t>"), .name = tensor.tensor_name
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

        // block_stmts for if count == 0 case
        std::vector<llir::lStmt> block_stmts;
        for(auto tensor: operand_tensors) {
            for(int i=0; i< tensor.tensor_type.format.levels.size(); i++) {
                std::string name = tensor.tensor_type.format.levels[i].index;
                if(is_sparse_format(tensor.tensor_type.format.levels[i].format)) {
                    name = name + "_p";
                }
                // partitions_tensor.level[thread_id] = 0;
                block_stmts.emplace_back(
                    llir::Store::make(
                        llir::lFieldAccess::make(
                            llir::lArrayAccess::make(
                                    llir::lVar::make(llir::Generic_t::make(tensor.get_index_struct_name()), "partitions_"+tensor.tensor_name),
                                    llir::lVar::make(index_t, "thread_id")
                                ),
                                llir::lVar::make(index_t, name)
                            ),
                        i == tensor.tensor_type.format.levels.size()-1 
                        ? llir::lConst::make((int64_t)-1)
                        : llir::lConst::make((int64_t)0)
                    )
                );
            }
        }
        block_stmts.emplace_back(
            llir::Return::make()
        );
            
        // if count ==0
        stmts.emplace_back(
            llir::IfElse::make(
                llir::lBinOp::make(
                    llir::lBinOp::Eq,
                    llir::lVar::make(index_t, "count"),
                    llir::lConst::make((int64_t)0)
                ),
                llir::Sequence::make(std::move(block_stmts)),
                nullptr
            )
        );

        // block_stmts for if count >= total_work case
        block_stmts = std::vector<llir::lStmt>{};
        for(auto tensor: operand_tensors) {
            for(int i=0; i< tensor.tensor_type.format.levels.size(); i++) {
                std::string field_name = tensor.tensor_type.format.levels[i].index;
                std::string field_max_value = tensor.get_size_field_name(field_name);
                if(is_sparse_format(tensor.tensor_type.format.levels[i].format)) {
                    field_max_value = tensor.get_length_field_name(field_name);
                    field_name = field_name + "_p";
                }
                // partitions_tensor.level[thread_id] = 0;
                block_stmts.emplace_back(
                    llir::Store::make(
                        llir::lFieldAccess::make(
                            llir::lArrayAccess::make(
                                    llir::lVar::make(llir::Generic_t::make(tensor.get_index_struct_name()), "partitions_"+tensor.tensor_name),
                                    llir::lVar::make(index_t, "thread_id")
                                ),
                                llir::lVar::make(index_t, field_name)
                            ),
                        llir::lFieldAccess::make(
                            llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name),
                            llir::lBinOp::make(
                                llir::lBinOp::Sub,
                                llir::lVar::make(index_t, field_max_value),
                                llir::lConst::make((int64_t)1)
                            )
                        )
                    )
                );
            }
        }
        block_stmts.emplace_back(
            llir::Return::make()
        );
        llir::lExpr total_work_expr = llir::lFieldAccess::make(
            llir::lVar::make(
                llir::Ptr_t::make(llir::Generic_t::make(operand_tensors[0].get_struct_name())),
                operand_tensors[0].tensor_name
            ),
            llir::lVar::make(
                index_t,
                "nnz"
            )
        );

        for(int i=1; i<operand_tensors.size(); i++) {
            total_work_expr = llir::lBinOp::make(
                llir::lBinOp::Add,
                total_work_expr,
                llir::lFieldAccess::make(
                    llir::lVar::make(
                        llir::Ptr_t::make(llir::Generic_t::make(operand_tensors[i].get_struct_name())),
                        operand_tensors[i].tensor_name
                    ),
                    llir::lVar::make(
                        index_t,
                        "nnz"
                    )
                )
            );
        }
        // if count ==0
        stmts.emplace_back(
            llir::IfElse::make(
                llir::lBinOp::make(
                    llir::lBinOp::Leq,
                    total_work_expr,
                    llir::lVar::make(index_t, "count")
                ),
                llir::Sequence::make(std::move(block_stmts)),
                nullptr
            )
        );

        
        llir::lExpr total_work_var = llir::lVar::make(index_t, "total_work");
        llir::lExpr rem_count_var = llir::lVar::make(index_t, "rem_count");
        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                "rem_count",
                llir::lVar::make(index_t, "count")
            )
        );
        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                "total_work",
                llir::lConst::make((int64_t)0)
            )
        );


        for(int i=0;i<forall_list.size();i++) {
            CIN forall = forall_list[i];
            std::string forall_idx = forall.as<Forall>()->idx;
            
            std::vector<TensorLowerer> tensors_with_curr_dim;
            std::vector<TensorLowerer> tensors_with_curr_dim_sparse;
            for(auto tensor : operand_tensors) {
                if(tensor.tensor_type.format.level_exists(forall_idx)){
                    tensors_with_curr_dim.push_back(tensor);
                    if(is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_idx))) {
                        tensors_with_curr_dim_sparse.push_back(tensor);
                    }
                } 
            }

            // internal_assert(tensors_with_curr_dim.size() > 0);

            llir::lExpr start_var = llir::lVar::make(index_t, "start_"+forall_idx);
            llir::lExpr end_var = llir::lVar::make(index_t, "end_"+forall_idx);
            llir::lExpr mid_var = llir::lVar::make(index_t, "mid_"+forall_idx);

            llir::lExpr start_expr, end_expr, mid_expr;
            start_expr = llir::lConst::make((int64_t)-1);
            end_expr = llir::lFieldAccess::make(
                llir::lVar::make(
                    llir::Ptr_t::make(llir::Generic_t::make(tensors_with_curr_dim[0].get_struct_name())),
                    tensors_with_curr_dim[0].tensor_name
                ),
                llir::lBinOp::make(
                    llir::lBinOp::Sub,
                    llir::lVar::make(
                        llir::Generic_t::make("index_t"),
                        tensors_with_curr_dim[0].get_size_field_name(forall_idx)
                    ),
                    llir::lConst::make((int64_t)1)
                )
            );

            mid_expr = llir::lBinOp::make(
                llir::lBinOp::Div,
                llir::lBinOp::make(
                    llir::lBinOp::Add,
                    llir::lBinOp::make(
                        llir::lBinOp::Add,
                        start_var,
                        end_var
                    ),
                    llir::lConst::make((int64_t)1)
                ),
                llir::lConst::make((int64_t)2)
            );

            stmts.emplace_back(
                llir::Declare::make(
                    index_t,
                    "start_"+forall_idx,
                    start_expr
                )
            );
            stmts.emplace_back(
                llir::Declare::make(
                    index_t,
                    "end_"+forall_idx,
                    end_expr
                )
            );
            stmts.emplace_back(
                llir::Declare::make(
                    index_t,
                    "mid_"+forall_idx,
                    mid_expr
                )
            );
            
            // mid = (star+end)/2
            std::vector<llir::lStmt> while_stmts;
            while_stmts.emplace_back(
                llir::Store::make(
                    mid_var,
                    mid_expr
                )
            );

            // for tensors where forall_idx dim is sparse we need to find the corresponding position
            for(int i=0;i<tensors_with_curr_dim_sparse.size();i++) {
                TensorLowerer tensor = tensors_with_curr_dim_sparse[i];

                llir::lExpr position_var = llir::lVar::make(index_t, tensor.tensor_name+"_"+forall_idx+"_p");
                llir::lExpr position_var_start = llir::lVar::make(index_t, tensor.tensor_name+"_"+forall_idx+"_p_start");
                llir::lExpr position_var_end = llir::lVar::make(index_t, tensor.tensor_name+"_"+forall_idx+"_p_end");
                int current_dim_level = tensor.tensor_type.format.get_level_order(forall_idx);
            
                while_stmts.emplace_back(
                    llir::Declare::make(
                        index_t,
                        tensor.tensor_name+"_"+forall_idx+"_p_start",
                        llir::lBinOp::make(
                            llir::lBinOp::Sub,
                            current_dim_level == 0 ? llir::lConst::make((int64_t)0) :
                                llir::lArrayAccess::make(
                                    llir::lFieldAccess::make(
                                        llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name),
                                        llir::lVar::make(index_t, tensor.get_offsets_field_name(forall_idx))
                                    ),
                                        tensor.get_offset_expression_for_next_sparse(
                                            tensor.tensor_type.format.get_prev_sparse_level(current_dim_level), current_dim_level-1, false, true, 
                                            llir::lArrayAccess::make(
                                                llir::lVar::make(llir::Generic_t::make(tensor.get_index_struct_name()), "partitions_"+tensor.tensor_name),
                                                llir::lVar::make(index_t, "thread_id")
                                            )
                                        )
                                ),
                            llir::lConst::make((int64_t)1)
                        )
                    )
                );
                while_stmts.emplace_back(
                    llir::Declare::make(
                        index_t,
                        tensor.tensor_name+"_"+forall_idx+"_p_end",
                        llir::lBinOp::make(
                            llir::lBinOp::Sub,
                            current_dim_level == 0 ? 
                                llir::lFieldAccess::make(
                                    llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name),
                                    llir::lVar::make(index_t, tensor.get_length_field_name(forall_idx))
                                ) :
                                llir::lArrayAccess::make(
                                    llir::lFieldAccess::make(
                                        llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name),
                                        llir::lVar::make(index_t, tensor.get_offsets_field_name(forall_idx))
                                    ),
                                    tensor.get_offset_expression_for_next_sparse(
                                        tensor.tensor_type.format.get_prev_sparse_level(current_dim_level), current_dim_level-1, true, true, 
                                        llir::lArrayAccess::make(
                                        llir::lVar::make(llir::Generic_t::make(tensor.get_index_struct_name()), "partitions_"+tensor.tensor_name),
                                        llir::lVar::make(index_t, "thread_id")
                                        )
                                    )
                                ),
                            llir::lConst::make((int64_t)1)
                        )
                    )
                );

                llir::lExpr position_var_expr = llir::lBinOp::make(
                    llir::lBinOp::Div,
                    llir::lBinOp::make(
                        llir::lBinOp::Add,
                        llir::lBinOp::make(
                            llir::lBinOp::Add,
                            position_var_start,
                            position_var_end
                        ),
                        llir::lConst::make((int64_t)1)
                    ),
                    llir::lConst::make((int64_t)2)
                );

                while_stmts.emplace_back(
                    llir::Declare::make(
                        index_t,
                        tensor.tensor_name+"_"+forall_idx+"_p",
                        position_var_expr
                    )
                );
                
                std::vector<llir::lStmt> binary_search_stmts;
                binary_search_stmts.emplace_back(
                    llir::Store::make(
                        position_var,
                        position_var_expr
                    )
                );
                // binary search if-else clause
                binary_search_stmts.emplace_back(
                    llir::IfElse::make(
                        llir::lBinOp::make(
                            llir::lBinOp::Leq,
                            llir::lArrayAccess::make(
                                llir::lFieldAccess::make(
                                    llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name),
                                    llir::lVar::make(index_t, tensor.get_indices_field_name(forall_idx))
                                ),
                                llir::lVar::make(index_t, tensor.tensor_name+"_"+forall_idx+"_p")
                            ),
                            mid_var
                        ),
                        llir::Sequence::make({
                            llir::Store::make(
                                position_var_start,
                                position_var
                            )
                        }),
                        llir::Sequence::make({
                            llir::Store::make(
                                position_var_end,
                                llir::lBinOp::make(
                                    llir::lBinOp::Sub,
                                    position_var,
                                    llir::lConst::make((int64_t)1)
                                )
                            )
                        })
                    )
                );

                while_stmts.emplace_back(
                    llir::While::make(
                        llir::lBinOp::make(
                            llir::lBinOp::Lt,
                            position_var_start,
                            position_var_end
                        ),
                        llir::Sequence::make(std::move(binary_search_stmts))
                    )
                );
                while_stmts.emplace_back(
                    llir::Store::make(
                        position_var,
                        position_var_expr
                    )
                );
            }


            

            // Now have to calculate the total work for the parition with current indices
            while_stmts.emplace_back(
                llir::Store::make(
                    total_work_var,
                    llir::lConst::make((int64_t)0)
                )
            );

            for(auto tensor : operand_tensors) {

                std::vector<llir::lExpr> work_args;
                work_args.emplace_back(llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name));
                for(int j=0;j<i;j++){
                    std::string forall_j_idx = forall_list[j].as<Forall>()->idx;
                    if(tensor.tensor_type.format.level_exists(forall_j_idx)) {
                        work_args.emplace_back(
                            llir::lFieldAccess::make(
                                llir::lArrayAccess::make(
                                    llir::lVar::make(llir::Generic_t::make(tensor.get_index_struct_name()), "partitions_"+tensor.tensor_name),
                                    llir::lVar::make(index_t, "thread_id")
                                ),
                                llir::lVar::make(
                                    index_t,
                                    forall_j_idx + (is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_j_idx)) ? "_p" : "")
                                )
                            )
                        );
                    }
                }
                if(tensor.tensor_type.format.level_exists(forall_idx)) {
                    if(is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_idx))) {
                        work_args.emplace_back(
                            llir::lVar::make(index_t, tensor.tensor_name+"_"+forall_idx+"_p")
                        );
                    } else {
                        work_args.emplace_back(mid_var);
                    }
                }

                for(int j=i+1;j<forall_list.size();j++){
                    std::string forall_j_idx = forall_list[j].as<Forall>()->idx;
                    if(!tensor.tensor_type.format.level_exists(forall_j_idx)) {
                        // pass end of the dimension for all foralls after current forall

                        auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [&](const auto& op_tensor) {
                            return op_tensor.tensor_type.format.level_exists(forall_j_idx);
                        });
                        internal_assert(it != operand_tensors.end()) << "Expected operand tensor to exist";

                        work_args.emplace_back(
                            llir::lFieldAccess::make(
                                llir::lVar::make(llir::Generic_t::make(it->get_struct_name()), it->tensor_name),
                                llir::lVar::make(
                                    index_t,
                                    it->get_size_field_name(forall_j_idx)
                                )
                            )
                        );
                    }
                }

                // call work function
                while_stmts.emplace_back(
                    llir::Store::make(
                        total_work_var,
                        llir::lBinOp::make(
                            llir::lBinOp::Add,
                            total_work_var,
                            llir::lFunctionCall::make(
                                tensor.get_work_function_name(forall_idx),
                                work_args
                            )
                        )
                    )
                );
            }

            // store calculated paritions into partition struct if needed partition is found.
            std::vector<llir::lStmt> store_partition_stmts;
            for(auto tensor : operand_tensors) {
                if(tensor.tensor_type.format.level_exists(forall_idx)) {
                    bool is_sparse = is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_idx));
                    store_partition_stmts.emplace_back(
                        llir::Store::make(
                            llir::lFieldAccess::make(
                                llir::lArrayAccess::make(
                                    llir::lVar::make(llir::Generic_t::make(tensor.get_index_struct_name()), "partitions_"+tensor.tensor_name),
                                    llir::lVar::make(index_t, "thread_id")
                                ),
                                llir::lVar::make(index_t, forall_idx + (is_sparse ? "_p" : ""))
                            ),
                            mid_var
                        )
                    );
                }
            }
            store_partition_stmts.emplace_back(
                llir::Store::make(
                    rem_count_var,
                    llir::lBinOp::make(
                        llir::lBinOp::Sub,
                        rem_count_var,
                        total_work_var
                    )
                )
            );
            store_partition_stmts.emplace_back(llir::Break::make());

            while_stmts.emplace_back(
                llir::IfElse::make(
                    llir::lBinOp::make(
                        llir::lBinOp::Leq,
                        end_var,
                        start_var
                    ),
                    llir::Sequence::make(store_partition_stmts),
                    nullptr
                )
            );

            while_stmts.emplace_back(
                    llir::IfElse::make(
                        llir::lBinOp::make(
                            llir::lBinOp::Leq,
                            total_work_var,
                            llir::lVar::make(index_t, "count")
                        ),
                        llir::Sequence::make({
                            llir::Store::make(
                                start_var,
                                mid_var
                            )
                        }),
                        llir::Sequence::make({
                            llir::Store::make(
                                end_var,
                                llir::lBinOp::make(
                                    llir::lBinOp::Sub,
                                    mid_var,
                                    llir::lConst::make((int64_t)1)
                                )
                            )
                        })
                    )
                );

            stmts.emplace_back(
                llir::While::make(
                    llir::lConst::make((bool)true),
                    llir::Sequence::make(std::move(while_stmts))
                )
            );

        }

        stmts.emplace_back(
            llir::Return::make()
        );
        body = llir::Sequence::make(std::move(stmts));

        return llir::Function::make(std::move(generics), std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body));
    }

} // namespace backend

} // namespace nacho
