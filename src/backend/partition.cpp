#include "backend/partition.h"
#include "Visitor.h"
#include "Error.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include <numeric>
namespace nacho {
namespace backend {

    llir::lType PartitionFunctionLowerer::lower_partition_struct_definition() {
        llir::lType index_t = llir::Generic_t::make("index_t");

        std::vector<std::string> generics = {"index_t"};

        std::vector<std::pair<std::string, llir::lType>> fields;


        for(int i=0;i<forall_list.size();i++) {
            const Forall* forall = forall_list[i].as<Forall>();
            std::string forall_idx = forall->idx;
            bool has_dense = false;
            for(auto it: operand_tensors) {
                auto tensor = it.second;
                // Check if the tensor has this forall index in its format levels
                if(tensor.tensor_type.format.level_exists(forall_idx)) {
                    if(!is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_idx))) {
                        has_dense = true;
                        continue;
                    }
                    std::string field_name = tensor.tensor_name + "_" + forall_idx + "_p";
                    fields.emplace_back(field_name, llir::Ptr_t::make(index_t));
                }
            }
            if(has_dense) {
                std::string field_name = forall_idx;
                fields.emplace_back(field_name, llir::Ptr_t::make(index_t));
            }
        }

        return llir::Struct_t::make(get_partition_struct_name(), std::move(fields),
                                    std::move(generics));
}

    llir::lStmt PartitionFunctionLowerer::lower_partition_kernel_for_innermost_sparse_intersection() {

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



        for(auto it: operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = llir::Generic_t::make(it.second.get_struct_name()+"<index_t, value_t>"), .name = it.second.tensor_name
            });
        }

        args.emplace_back(llir::Function::Argument{
            .mutating = true, .type = llir::Ptr_t::make(llir::Generic_t::make(get_partition_struct_name()+"<index_t>")), .name = "partitions"
        });


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


        for(int i=0;i<forall_list.size();i++) {
            const Forall* forall = forall_list[i].as<Forall>();
            std::string forall_idx = forall->idx;
            bool has_dense = false;
            for(auto it: operand_tensors) {
                auto tensor = it.second;
                // Check if the tensor has this forall index in its format levels
                if(tensor.tensor_type.format.level_exists(forall_idx)) {
                    if(!is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_idx))) {
                        has_dense = true;
                        continue;
                    }
                    std::string field_name = tensor.tensor_name + "_" + forall_idx + "_p";
                    block_stmts.emplace_back(llir::Store::make(
                        llir::lFieldAccess::make(
                            llir::lArrayAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(
                                        get_partition_function_name()),
                                    "partitions"),
                                llir::lVar::make(index_t, "thread_id")),
                            field_name),
                        i == forall_list.size() - 1
                            ? llir::lConst::make((int64_t)-1)
                            : llir::lConst::make((int64_t)0)));
                }
            }
            if(has_dense) {
                block_stmts.emplace_back(llir::Store::make(
                    llir::lFieldAccess::make(
                        llir::lArrayAccess::make(
                            llir::lVar::make(
                                llir::Generic_t::make(
                                    get_partition_function_name()),
                                "partitions"),
                            llir::lVar::make(index_t, "thread_id")),
                        forall_idx),
                    i == forall_list.size() - 1
                            ? llir::lConst::make((int64_t)-1)
                            : llir::lConst::make((int64_t)0)));
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
        for(int i=0;i<forall_list.size();i++) {
            const Forall* forall = forall_list[i].as<Forall>();
            std::string forall_idx = forall->idx;
            bool has_dense = false;
            TensorLowerer tensor_with_dense_dim;
            for(auto it: operand_tensors) {
                auto tensor = it.second;
                // Check if the tensor has this forall index in its format levels
                if(tensor.tensor_type.format.level_exists(forall_idx)) {
                    if(!is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_idx))) {
                        has_dense = true;
                        tensor_with_dense_dim = tensor;
                        continue;
                    }
                    std::string field_name = tensor.tensor_name + "_" + forall_idx + "_p";
                    std::string field_max_value = tensor.get_length_field_name(forall_idx + "_p");
                    
                    block_stmts.emplace_back(llir::Store::make(
                        llir::lFieldAccess::make(
                            llir::lArrayAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(
                                        get_partition_function_name()),
                                    "partitions"),
                                llir::lVar::make(index_t, "thread_id")),
                            field_name),
                        llir::lBinOp::make(
                            llir::lBinOp::Sub,
                            llir::lFieldAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(tensor.get_struct_name()),
                                    tensor.tensor_name),
                                field_max_value),
                            llir::lConst::make((int64_t)1))));
                }
            }
            if(has_dense) {
                block_stmts.emplace_back(llir::Store::make(
                    llir::lFieldAccess::make(
                        llir::lArrayAccess::make(
                            llir::lVar::make(
                                llir::Generic_t::make(
                                    get_partition_function_name()),
                                "partitions"),
                            llir::lVar::make(index_t, "thread_id")),
                        forall_idx),
                    llir::lBinOp::make(
                            llir::lBinOp::Sub,
                            llir::lFieldAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(tensor_with_dense_dim.get_struct_name()),
                                    tensor_with_dense_dim.tensor_name),
                                tensor_with_dense_dim.get_size_field_name(forall_idx)),
                            llir::lConst::make((int64_t)1))));
            }
        }

        block_stmts.emplace_back(
            llir::Return::make()
        );
        llir::lExpr total_work_expr = llir::lFieldAccess::make(
            llir::lVar::make(
                llir::Ptr_t::make(llir::Generic_t::make(
                    operand_tensors.begin()->second.get_struct_name())),
                operand_tensors.begin()->second.tensor_name),
            "nnz");

        for(auto it : operand_tensors) {
            total_work_expr = llir::lBinOp::make(
                llir::lBinOp::Add, total_work_expr,
                llir::lFieldAccess::make(
                    llir::lVar::make(llir::Ptr_t::make(llir::Generic_t::make(
                                         it.second.get_struct_name())),
                                     it.second.tensor_name),
                    "nnz"));
        }
        // if count >= total_work
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
            const Forall* forall = forall_list[i].as<Forall>();
            std::string forall_idx = forall->idx;

            std::vector<TensorLowerer> tensors_with_curr_dim;
            std::vector<TensorLowerer> tensors_with_curr_dim_sparse;
            for(auto it : operand_tensors) {
                if(it.second.tensor_type.format.level_exists(forall_idx)){
                    tensors_with_curr_dim.push_back(it.second);
                    if(is_sparse_format(it.second.tensor_type.format.lvlfmt_of(forall_idx))) {
                        tensors_with_curr_dim_sparse.push_back(it.second);
                    }
                } 
            }

            

            llir::lExpr start_var = llir::lVar::make(index_t, "start_"+forall_idx);
            llir::lExpr end_var = llir::lVar::make(index_t, "end_"+forall_idx);
            llir::lExpr mid_var = llir::lVar::make(index_t, "mid_"+forall_idx);

            llir::lExpr start_expr, end_expr, mid_expr;
            start_expr = llir::lConst::make((int64_t)-1);
            end_expr = llir::lBinOp::make(
                llir::lBinOp::Sub,
                llir::lFieldAccess::make(
                    llir::lVar::make(
                        llir::Ptr_t::make(llir::Generic_t::make(
                            tensors_with_curr_dim[0].get_struct_name())),
                        tensors_with_curr_dim[0].tensor_name),
                    tensors_with_curr_dim[0].get_size_field_name(forall_idx)),
                llir::lConst::make((int64_t)1));

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

                // This just stores the name of variables in which the partition of this
                // tensor is stored. This is needed to be passed as argument to the
                // get_offset_expression_for_next_sparse method call
                std::vector<llir::lExpr> dim_vars_for_offset_expression;
                for(int d=0;d<tensor.tensor_type.format.levels.size();d++) {
                    auto level = tensor.tensor_type.format.levels[d];
                    bool is_sparse = is_sparse_format(level.format);
                    dim_vars_for_offset_expression.emplace_back(
                        llir::lFieldAccess::make(
                            llir::lArrayAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(
                                        get_partition_function_name()),
                                    "partitions"),
                                    llir::lVar::make(index_t, "thread_id")
                                ),
                                is_sparse
                                    ? tensor.tensor_name + "_" + level.index + "_p"
                                    : level.index
                            )
                    );
                }

                while_stmts.emplace_back(llir::Declare::make(
                    index_t, tensor.tensor_name + "_" + forall_idx + "_p_start",
                    llir::lBinOp::make(
                        llir::lBinOp::Sub,
                        current_dim_level == 0
                            ? llir::lConst::make((int64_t)0)
                            : llir::lArrayAccess::make(
                                  llir::lFieldAccess::make(
                                      llir::lVar::make(
                                          llir::Generic_t::make(
                                              tensor.get_struct_name()),
                                          tensor.tensor_name),
                                      tensor.get_offsets_field_name(
                                          forall_idx)),
                                  tensor.get_offset_expression_for_next_sparse(
                                      tensor.tensor_type.format
                                          .get_prev_sparse_level(
                                              current_dim_level),
                                      current_dim_level - 1, false, true,
                                      dim_vars_for_offset_expression)),
                        llir::lConst::make((int64_t)1))));
                while_stmts.emplace_back(llir::Declare::make(
                    index_t, tensor.tensor_name + "_" + forall_idx + "_p_end",
                    llir::lBinOp::make(
                        llir::lBinOp::Sub,
                        current_dim_level == 0
                            ? llir::lFieldAccess::make(
                                  llir::lVar::make(
                                      llir::Generic_t::make(
                                          tensor.get_struct_name()),
                                      tensor.tensor_name),
                                  tensor.get_length_field_name(forall_idx))
                            : llir::lArrayAccess::make(
                                  llir::lFieldAccess::make(
                                      llir::lVar::make(
                                          llir::Generic_t::make(
                                              tensor.get_struct_name()),
                                          tensor.tensor_name),
                                      tensor.get_offsets_field_name(
                                          forall_idx)),
                                  tensor.get_offset_expression_for_next_sparse(
                                      tensor.tensor_type.format
                                          .get_prev_sparse_level(
                                              current_dim_level),
                                      current_dim_level - 1, true, true,
                                       dim_vars_for_offset_expression)),
                        llir::lConst::make((int64_t)1))));

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
                binary_search_stmts.emplace_back(llir::IfElse::make(
                    llir::lBinOp::make(
                        llir::lBinOp::Leq,
                        llir::lArrayAccess::make(
                            llir::lFieldAccess::make(
                                llir::lVar::make(llir::Generic_t::make(
                                                     tensor.get_struct_name()),
                                                 tensor.tensor_name),
                                tensor.get_indices_field_name(forall_idx)),
                            llir::lVar::make(index_t, tensor.tensor_name + "_" +
                                                          forall_idx + "_p")),
                        mid_var),
                    llir::Sequence::make(
                        {llir::Store::make(position_var_start, position_var)}),
                    llir::Sequence::make({llir::Store::make(
                        position_var_end,
                        llir::lBinOp::make(llir::lBinOp::Sub, position_var,
                                           llir::lConst::make((int64_t)1)))})));

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

            for(auto it : operand_tensors) {
                TensorLowerer tensor = it.second;
                std::vector<llir::lExpr> work_args;
                work_args.emplace_back(llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name));
                for(int j=0;j<i;j++){
                    std::string forall_j_idx = forall_list[j].as<Forall>()->idx;
                    if(tensor.tensor_type.format.level_exists(forall_j_idx)) {
                        work_args.emplace_back(llir::lFieldAccess::make(
                            llir::lArrayAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(
                                        get_partition_struct_name()),
                                    "partitions"),
                                llir::lVar::make(index_t, "thread_id")),
                            ( (is_sparse_format(tensor.tensor_type.format
                                                   .lvlfmt_of(forall_j_idx))
                                  ? tensor.tensor_name + "_" + forall_j_idx + "_p"
                                  : forall_j_idx))));
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

                // pass broadcast sizes for all dimensions after current forall which are not present in the tensor
                for(int j=i+1;j<forall_list.size();j++){
                    std::string forall_j_idx = forall_list[j].as<Forall>()->idx;
                    if(!tensor.tensor_type.format.level_exists(forall_j_idx)) {
                        
                        auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [&](const auto& op_tensor) {
                            return op_tensor.second.tensor_type.format.level_exists(forall_j_idx);
                        });
                        internal_assert(it != operand_tensors.end()) << "Expected operand tensor to exist";

                        work_args.emplace_back(llir::lFieldAccess::make(
                            llir::lVar::make(llir::Generic_t::make(
                                                 it->second.get_struct_name()),
                                             it->second.tensor_name),
                            it->second.get_size_field_name(forall_j_idx)));
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
            bool has_dense = false;
            for(auto it : operand_tensors) {
                TensorLowerer tensor = it.second;
                if(tensor.tensor_type.format.level_exists(forall_idx)) {
                    if(!is_sparse_format(tensor.tensor_type.format.lvlfmt_of(forall_idx))) {
                        has_dense = true;
                        continue;
                    }
                    store_partition_stmts.emplace_back(llir::Store::make(
                        llir::lFieldAccess::make(
                            llir::lArrayAccess::make(
                                llir::lVar::make(
                                    llir::Generic_t::make(
                                        get_partition_struct_name()),
                                    "partitions"),
                                llir::lVar::make(index_t, "thread_id")),
                            tensor.tensor_name + "_" + forall_idx + "_p"),
                        llir::lVar::make(index_t, tensor.tensor_name + "_" + forall_idx + "_p")));
                }
            }
            if(has_dense) {
                store_partition_stmts.emplace_back(llir::Store::make(
                    llir::lFieldAccess::make(
                        llir::lArrayAccess::make(
                            llir::lVar::make(
                                llir::Generic_t::make(
                                    get_partition_struct_name()),
                                "partitions"),
                            llir::lVar::make(index_t, "thread_id")),
                        forall_idx),
                    mid_var));
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
                            llir::lVar::make(index_t, "rem_count")
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
