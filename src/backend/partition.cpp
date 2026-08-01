#include "backend/partition.h"
#include "Visitor.h"
#include "Error.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include <numeric>
namespace nacho {
namespace backend {

    llir::lType PartitionKernelLowerer::lower_partition_struct_definition() {
        std::vector<std::string> generics = {"index_t"};

        std::vector<std::pair<std::string, llir::lType>> fields;

        for(LoopNum i=BEFORE_FIRST_LOOP+1; i<=previous_sparse_intersection;++i) {
            const Forall* forall = forall_list[i.get()].as<Forall>();
            std::string forall_idx = forall->idx;
            internal_assert(result_tensor.tensor_level_exists(forall_idx)) << "Expected forall index " << forall_idx << " to be present in the result tensor for all loops upto and including the previous sparse intersection";

            fields.emplace_back(result_tensor.get_iterator_suffix(forall_idx), llir::Ptr_t::make(index_t));
        }

        for(LoopNum i=previous_sparse_intersection+1;i<=current_sparse_intersection;++i) {
            const Forall* forall = forall_list[i.get()].as<Forall>();
            std::string forall_idx = forall->idx;
            bool has_dense = false;
            for(auto it: included_tensors) {
                auto tensor = it.second;
                // Check if the tensor has this forall index in its format levels
                if(tensor.tensor_level_exists(forall_idx)) {
                    if(!tensor.is_sparse(forall_idx)) {
                        has_dense = true;
                        continue;
                    }
                    fields.emplace_back(tensor.get_iterator_suffix(forall_idx), llir::Ptr_t::make(index_t));
                }
            }
            if(has_dense) {
                std::string field_name = forall_idx;
                fields.emplace_back(field_name, llir::Ptr_t::make(index_t));
            }
            for(auto it: operand_tensors) {
                if(included_tensors.find(it.first) == included_tensors.end()) {
                    TensorLowerer tensor = it.second;
                    if(tensor.tensor_level_exists(forall_idx)) {
                        if(tensor.is_sparse(forall_idx)) {
                            fields.emplace_back(tensor.get_iterator_suffix(forall_idx), llir::Ptr_t::make(index_t));
                        }
                    }
                }
            }
        }

        return llir::Struct_t::make(get_partition_struct_name(), std::move(fields),
                                    std::move(generics));
}

    llir::lStmt PartitionKernelLowerer::lower_partition_kernel() {

        std::vector<std::string> generics = {"index_t", "value_t"};

        std::vector<llir::Function::Attribute> attributes = {
            // llir::Function::global
            llir::Function::runnable};

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
            .mutating = false, .type = llir::Generic_t::make(result_tensor.get_struct_name()+"<index_t, value_t>"), .name = result_tensor.tensor_name
        });

        args.emplace_back(llir::Function::Argument{
            .mutating = true, .type = llir::Generic_t::make(get_partition_struct_name()+"<index_t>"), .name = "partitions"
        });


        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "per_thread_work"
        });

        args.emplace_back(
            llir::Function::Argument{
                .mutating = false, .type = index_t, .name = "total_work"
            }
        );

        if(previous_sparse_intersection != BEFORE_FIRST_LOOP)
            args.emplace_back(
                llir::Function::Argument{
                    .mutating = false, .type = llir::Ptr_t::make(index_t), .name = "T_work_offsets"
                }
            );

        bool need_operand_pos_map_arg = false;
        for(LoopNum i = BEFORE_FIRST_LOOP+1; i<=previous_sparse_intersection; ++i) {
            for(auto it: operand_tensors) {
                if(exists_field_in_result_to_operand_pos_map(forall_list[i.get()].as<Forall>(), it.second)){
                    need_operand_pos_map_arg = true;
                    break;
                }
            }
        }

        if(need_operand_pos_map_arg) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false,
                .type = llir::Generic_t::make(get_result_to_operand_pos_map_struct_name() + "<index_t>"),
                .name = get_result_to_operand_pos_map_var_name()
            });
        }

        static const llir::lType i32 = llir::Int_t::make(32);
        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = i32, .name = "thread_id"});

        std::vector<llir::lStmt> stmts;

        // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        // stmts.emplace_back(
        //     llir::Declare::make(
        //         llir::Int_t::make(32),
        //         "thread_id",
        //         (llir::lVar::make(index_t, "blockIdx.x") *
        //         llir::lVar::make(index_t, "blockDim.x")) +
        //         llir::lVar::make(index_t, "threadIdx.x")
        //     )
        // );

        // index_t count = thread_id * per_thread_work;
        stmts.emplace_back(
            llir::Declare::make(
                llir::Generic_t::make("index_t"),
                "count",
                llir::lVar::make(index_t, "thread_id") * llir::lVar::make(index_t, "per_thread_work")
            )
        );

        // block_stmts for if count == 0 case
        std::vector<llir::lStmt> block_stmts;

        for(LoopNum i=BEFORE_FIRST_LOOP+1;i<=previous_sparse_intersection;++i) {
            const Forall* forall = forall_list[i.get()].as<Forall>();
            std::string forall_idx = forall->idx;
            if(result_tensor.tensor_level_exists(forall_idx)){
                block_stmts.emplace_back(llir::Store::make(
                    get_partition_struct_current_thread_field(result_tensor.get_iterator_suffix(forall_idx)),
                    get_partition_initializer_expr_for_boundary_cases(i, result_tensor, false)
                ));
            }
        }

        for(LoopNum i=previous_sparse_intersection+1;i<=current_sparse_intersection;++i) {
            const Forall* forall = forall_list[i.get()].as<Forall>();
            std::string forall_idx = forall->idx;
            bool has_dense = false;
            TensorLowerer tensor_with_dense_dim;
            for(auto it: operand_tensors) {
                auto tensor = it.second;
                // Check if the tensor has this forall index in its format levels
                if(tensor.tensor_level_exists(forall_idx)) {
                    if(!tensor.is_sparse(forall_idx)) {
                        if(included_tensors.find(it.first) != included_tensors.end()) {
                            has_dense = true;
                            tensor_with_dense_dim = tensor;
                        }
                        continue;
                    }
                    llir::lExpr init_value_expr = get_partition_initializer_expr_for_boundary_cases(i, tensor, false);
                    block_stmts.emplace_back(llir::Store::make(
                        get_partition_struct_current_thread_field(tensor.get_iterator_suffix(forall_idx)),
                        init_value_expr));
                }
            }
            if(has_dense) {
                block_stmts.emplace_back(llir::Store::make(
                    get_partition_struct_current_thread_field(forall_idx),
                    get_partition_initializer_expr_for_boundary_cases(i, tensor_with_dense_dim, false)
                    ));
            }
        }

        block_stmts.emplace_back(
            llir::Return::make()
        );
            
        // if count ==0
        stmts.emplace_back(
            llir::IfElse::make(
                llir::lVar::make(index_t, "count") == llir::lConst::make((int64_t)0),
                llir::Sequence::make(std::move(block_stmts)),
                nullptr
            )
        );

        // block_stmts for if count >= total_work case
        block_stmts = std::vector<llir::lStmt>{};


        for(LoopNum i=BEFORE_FIRST_LOOP+1;i<=previous_sparse_intersection;++i) {
            const Forall* forall = forall_list[i.get()].as<Forall>();
            std::string forall_idx = forall->idx;
            if(result_tensor.tensor_level_exists(forall_idx)){
                block_stmts.emplace_back(llir::Store::make(
                    get_partition_struct_current_thread_field(result_tensor.get_iterator_suffix(forall_idx)),
                    get_partition_initializer_expr_for_boundary_cases(i, result_tensor, true)));
            }
        }

        for(LoopNum i=previous_sparse_intersection+1;i<=current_sparse_intersection;++i) {
            const Forall* forall = forall_list[i.get()].as<Forall>();
            std::string forall_idx = forall->idx;
            bool has_dense = false;
            TensorLowerer tensor_with_dense_dim;
            for(auto it: operand_tensors) {
                auto tensor = it.second;
                // Check if the tensor has this forall index in its format levels
                if(tensor.tensor_level_exists(forall_idx)) {
                    if(!tensor.is_sparse(forall_idx)) {
                        if(included_tensors.find(it.first) != included_tensors.end()) {
                            has_dense = true;
                            tensor_with_dense_dim = tensor;
                        }
                        continue;
                    }

                    llir::lExpr init_value_expr = get_partition_initializer_expr_for_boundary_cases(i, tensor, true);
                    block_stmts.emplace_back(llir::Store::make(
                        get_partition_struct_current_thread_field(tensor.get_iterator_suffix(forall_idx)),
                        init_value_expr
                        ));
                }
            }
            if(has_dense) {
                block_stmts.emplace_back(llir::Store::make(
                    get_partition_struct_current_thread_field(forall_idx),
                    get_partition_initializer_expr_for_boundary_cases(i, tensor_with_dense_dim, true)
                    ));
            }
        }

        block_stmts.emplace_back(
            llir::Return::make()
        );
        llir::lExpr total_work_expr = llir::lVar::make(index_t, "total_work");

        // if count >= total_work
        stmts.emplace_back(
            llir::IfElse::make(
                total_work_expr <= llir::lVar::make(index_t, "count"),
                llir::Sequence::make(std::move(block_stmts)),
                nullptr
            )
        );

        
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
                "work",
                llir::lConst::make((int64_t)0)
            )
        );


        // When a tensor has a sparse dim at any of the outer loops that are currently being partitioned
        // Then the corresponding partition for that dimension may not lie exactly on the partition
        // boundary. In that case the number of tensors to consider in the further loops has a runtime
        // dependency on whether the partition boundary aligns with the tensor's sparse dimensions.
        // We need to generate some additional bookkeeping code to keep track of these tensors which become 
        // inactive during partitioning.
        bool need_to_exclude_tensors_at_runtime = false;
        for(auto it : operand_tensors) {
            // Check if the tensor has a sparse dimension at any level other than the last level
            if(it.second.get_loop_num_for_prev_sparse_level(current_sparse_intersection) != BEFORE_FIRST_LOOP) {
                need_to_exclude_tensors_at_runtime = true;
                break;
            }
        }


        // Add bookeeping boolean variables to keep track of what tensors are currently considered
        // if tensors can get excluded at runtime (if they are already fully inside the partition boundary)
        if(need_to_exclude_tensors_at_runtime) {
            for(auto it : operand_tensors) {
                std::string var_name = "is_" + it.second.tensor_name + "_partitioned";
                stmts.emplace_back(
                    llir::Declare::make(
                        llir::Generic_t::make("bool"),
                        var_name,
                        llir::lConst::make(false)
                    )
                );
            }
        }


        for(LoopNum i=BEFORE_FIRST_LOOP+1;i<=previous_sparse_intersection; ++i) {
            stmts.emplace_back(
                lower_partition_loop_from_work_offsets(i, need_to_exclude_tensors_at_runtime)
            );
        }

        for(LoopNum i=previous_sparse_intersection+1;i<=current_sparse_intersection;++i) {
            stmts.emplace_back(
                lower_partition_loop(i, i == current_sparse_intersection, need_to_exclude_tensors_at_runtime)
            );
        }

        stmts.emplace_back(
            llir::Return::make()
        );
        body = llir::Sequence::make(std::move(stmts));

        return llir::Function::make(std::move(generics), std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body));
    }

    llir::lStmt PartitionKernelLowerer::get_store_partition_statements(LoopNum loop_num, llir::lExpr index_value, bool need_to_exclude_tensors_at_runtime, bool is_last_loop) {
        const Forall* forall = forall_list[loop_num.get()].as<Forall>();
        std::string forall_idx = forall->idx;
        std::vector<llir::lStmt> store_partition_stmts;
        bool has_dense = false;
        for(auto it : included_tensors) {
            TensorLowerer tensor = it.second;
            if(tensor.tensor_level_exists(forall_idx)) {
                if(!tensor.is_sparse(forall_idx)) {
                    has_dense = true;
                    continue;
                }
                std::string field_name = tensor.get_iterator_suffix(forall_idx);

                // If we need to recalculate sparse positions, we add the necessary statements
                // to find the sparse position for the index_value.
                if(!is_last_loop) {
                    store_partition_stmts.emplace_back(get_statements_to_find_sparse_position(
                        loop_num,
                        tensor,
                        index_value,
                        need_to_exclude_tensors_at_runtime,
                        false,
                        true
                    ));
                }

                if(!is_last_loop && need_to_exclude_tensors_at_runtime) {
                    store_partition_stmts.emplace_back(
                        llir::IfElse::make(
                            (tensor_partitioned_vars.at(tensor.tensor_name) 
                                || (llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx)) == (get_sparse_dim_start_expr(tensor, forall_idx) -1))
                                || (tensor.get_indices_field(forall_idx)[llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx))] != index_value)),
                            llir::Sequence::make({
                                    llir::Store::make(
                                        tensor_partitioned_vars.at(tensor.tensor_name),
                                        llir::lConst::make(bool(true))
                                    ),
                                    llir::BaseExpr::make(llir::lIncrement::make(
                                        llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx))
                                    )),
                                }),
                            nullptr
                        )
                    );
                }

                store_partition_stmts.emplace_back(llir::Store::make(
                    get_partition_struct_current_thread_field(field_name),
                    llir::lVar::make(index_t, field_name)));
                // // Update boolean which specifies whether this tensor will be considered for work in partitioning of next loop.
                // if(!is_last_loop && need_to_exclude_tensors_at_runtime) {
                //     store_partition_stmts.emplace_back(
                //         llir::Store::make(
                //             tensor_partitioned_vars.at(tensor.tensor_name),
                //             tensor_partitioned_vars.at(tensor.tensor_name) || (tensor.get_indices_field(forall_idx)[llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx))] != index_value)
                //         ));
                // }
            }
        }
        if(has_dense) {
            store_partition_stmts.emplace_back(llir::Store::make(
                get_partition_struct_current_thread_field(forall_idx),
                index_value));
        }

        // For tensors that are excluded, if any of them are sparse in this level then their position needs to be calculated and stored too
        // in the partitions struct based on the finalised index_value.
        for(auto it: operand_tensors) {
            TensorLowerer tensor = it.second;
            // included tensors are already handled above
            if(included_tensors.find(tensor.tensor_name) == included_tensors.end()) {
                if(tensor.tensor_level_exists(forall_idx)) {
                    if(tensor.is_sparse(forall_idx)) {
                        store_partition_stmts.emplace_back(
                            get_statements_to_find_sparse_position(
                                loop_num,
                                tensor,
                                index_value,
                                need_to_exclude_tensors_at_runtime,
                                false,
                                true
                            )
                        );
                        std::string field_name = tensor.get_iterator_suffix(forall_idx);

                        if(!is_last_loop && need_to_exclude_tensors_at_runtime) {
                            store_partition_stmts.emplace_back(
                                llir::IfElse::make(
                                    (tensor_partitioned_vars.at(tensor.tensor_name) 
                                        || (llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx)) == (get_sparse_dim_start_expr(tensor, forall_idx) -1))
                                        || (tensor.get_indices_field(forall_idx)[llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx))] != index_value)),
                                    llir::Sequence::make({
                                            llir::Store::make(
                                                tensor_partitioned_vars.at(tensor.tensor_name),
                                                llir::lConst::make(bool(true))
                                            ),
                                            llir::BaseExpr::make(llir::lIncrement::make(
                                                llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx))
                                            )),
                                        }),
                                    nullptr
                                )
                            );
                        }

                        store_partition_stmts.emplace_back(llir::Store::make(
                        get_partition_struct_current_thread_field(field_name),
                        llir::lVar::make(index_t, field_name)));
                        // // Update boolean which specifies whether this tensor will be considered for work in partitioning of next loop.
                        // if(!is_last_loop && need_to_exclude_tensors_at_runtime) {
                        //     store_partition_stmts.emplace_back(
                        //         llir::Store::make(
                        //         tensor_partitioned_vars.at(tensor.tensor_name),
                        //         tensor_partitioned_vars.at(tensor.tensor_name) || (tensor.get_indices_field(forall_idx)[llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx))] != index_value)
                        //     ));
                        // }
                    }
                }
            }
        
        }
        return llir::Sequence::make(std::move(store_partition_stmts));
    }

    // get_lower_bound gets the work for index_value -1 (sparse iterators are already assumed to be corresponding to index-1)
    llir::lExpr PartitionKernelLowerer::get_call_work_function_expr(LoopNum loop_num, TensorLowerer& tensor, llir::lExpr index_value) {
        const Forall* forall = forall_list[loop_num.get()].as<Forall>();
        std::string forall_idx = forall->idx;

        std::vector<llir::lExpr> work_args;
        work_args.emplace_back(llir::lVar::make(llir::Generic_t::make(tensor.get_struct_name()), tensor.tensor_name));
        for(LoopNum j=BEFORE_FIRST_LOOP+1;j<loop_num;++j){
            std::string forall_j_idx = forall_list[j.get()].as<Forall>()->idx;
            if(tensor.tensor_level_exists(forall_j_idx)) {
                work_args.emplace_back(
                    llir::lVar::make(
                        index_t,
                        (tensor.get_iterator_suffix(forall_j_idx))
                    ));
            }
        }
        if(tensor.tensor_level_exists(forall_idx) && tensor.is_sparse(forall_idx)) {
            work_args.emplace_back(
                llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx))
            );
        } else {
            work_args.emplace_back(index_value);
        }


        // pass broadcast sizes for all dimensions after current forall which are not present in the tensor
        for(LoopNum j=loop_num+1;j<=current_sparse_intersection;++j){
            std::string forall_j_idx = forall_list[j.get()].as<Forall>()->idx;
            if(!tensor.tensor_level_exists(forall_j_idx)) {
                
                auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [&](const auto& op_tensor) {
                    return op_tensor.second.tensor_level_exists(forall_j_idx);
                });
                internal_assert(it != operand_tensors.end()) << "Expected operand tensor to exist";

                work_args.emplace_back(it->second.get_size_field(forall_j_idx));
            }
        }

        if(tensor.is_result_tensor) {
            work_args.emplace_back(llir::lVar::make(index_t, "T_work_offsets"));
        }

        return llir::lFunctionCall::make(tensor.get_work_function_name(get_all_loops_string(tensor.is_result_tensor ? previous_sparse_intersection : current_sparse_intersection),forall_idx),work_args);
    };

    llir::lExpr PartitionKernelLowerer::get_sparse_dim_start_expr(TensorLowerer& tensor, const std::string& forall_idx) {
        internal_assert(tensor.tensor_level_exists(forall_idx)) << "Expected tensor level to exist for forall index " << forall_idx;
        TensorLevelNum current_tensor_level = tensor.loop_num_to_tensor_level(tensor.get_loop_num(forall_idx));
        std::map<TensorLevelNum, llir::lExpr> dim_vars_for_offset_expression;

        for(TensorLevelNum d=BEFORE_FIRST_LEVEL+1;d<=current_tensor_level-1;++d) {
            dim_vars_for_offset_expression[d] = llir::lVar::make(index_t, tensor.get_iterator_suffix(d));
        }
        auto offset_expr = tensor.get_level_indexing_expression(
                                current_tensor_level, false,
                                dim_vars_for_offset_expression);
        return current_tensor_level == BEFORE_FIRST_LEVEL + 1
                    ? llir::lConst::make((int64_t)0)
                    : (tensor.is_compressed(forall_idx) ? tensor.get_offsets_field(forall_idx)[offset_expr] : offset_expr);

    }

    llir::lExpr PartitionKernelLowerer::get_sparse_dim_end_expr(TensorLowerer& tensor, const std::string& forall_idx) {
        internal_assert(tensor.tensor_level_exists(forall_idx)) << "Expected tensor level to exist for forall index " << forall_idx;
        TensorLevelNum current_tensor_level = tensor.loop_num_to_tensor_level(tensor.get_loop_num(forall_idx));
        std::map<TensorLevelNum, llir::lExpr> dim_vars_for_offset_expression;

        for(TensorLevelNum d=BEFORE_FIRST_LEVEL+1;d<=current_tensor_level-1;++d) {
            if(tensor.is_sparse(d) && !tensor.is_unique(d))
                dim_vars_for_offset_expression[d] = tensor.get_seg_end(d);
            else
                dim_vars_for_offset_expression[d] = llir::lVar::make(index_t, tensor.get_iterator_suffix(d));
        }

        auto offset_expr = tensor.get_level_indexing_expression(
                                current_tensor_level, true,
                                dim_vars_for_offset_expression);
        return current_tensor_level == BEFORE_FIRST_LEVEL + 1
                    ? tensor.get_length_field(forall_idx)
                    : (tensor.is_compressed(forall_idx) ? tensor.get_offsets_field(forall_idx)[offset_expr] : offset_expr);

    }




    llir::lStmt PartitionKernelLowerer::get_statements_to_find_sparse_position(LoopNum loop_num, TensorLowerer& tensor, llir::lExpr index_value, bool need_to_exclude_tensors_at_runtime, bool is_upper_bound_search, bool add_seg_end_search) {
        const Forall* forall = forall_list[loop_num.get()].as<Forall>();
        std::string forall_idx = forall->idx;

        std::vector<llir::lStmt> stmts;


        llir::lExpr start_expr = llir::lVar::make(index_t, tensor.get_start_name(forall_idx));
        llir::lExpr end_expr = llir::lVar::make(index_t, tensor.get_end_name(forall_idx));


        auto binary_search_call = llir::lFunctionCall::make((is_upper_bound_search ? "binary_search_ub" : "binary_search_lb"), {tensor.get_indices_field(forall_idx), index_value, start_expr, end_expr});
        auto position_var_name = tensor.get_iterator_suffix(forall_idx);
        auto position_var = llir::lVar::make(index_t, position_var_name);
        stmts.emplace_back(
            llir::Store::make(position_var, binary_search_call)
        );
        if(add_seg_end_search && tensor.is_sparse(forall_idx) && !tensor.is_unique(forall_idx)) {
            auto seg_end_position_var = tensor.get_seg_end(forall_idx);
            auto seg_end_binary_search_call = llir::lFunctionCall::make("binary_search_ub", {tensor.get_indices_field(forall_idx), index_value, start_expr, end_expr});
            stmts.emplace_back(
                llir::Store::make(seg_end_position_var, seg_end_binary_search_call)
            );
        }
        if(need_to_exclude_tensors_at_runtime && loop_num != BEFORE_FIRST_LOOP+1) {
            return llir::IfElse::make(
                        llir::lOp::make(llir::lOp::Not, tensor_partitioned_vars.at(tensor.tensor_name)),
                        llir::Sequence::make(std::move(stmts)),
                        nullptr
                    );
        } else {
            return llir::Sequence::make(std::move(stmts));
        }
    }

    llir::lStmt PartitionKernelLowerer::lower_partition_loop(LoopNum loop_num, bool is_last_loop, bool need_to_exclude_tensors_at_runtime) {
        std::vector<llir::lStmt> stmts;

        const Forall* forall = forall_list[loop_num.get()].as<Forall>();
        std::string forall_idx = forall->idx;

        std::vector<TensorLowerer> tensors_with_curr_dim;
        std::vector<TensorLowerer> tensors_with_curr_dim_sparse;
        bool all_unique_levels = true;
        for(auto it : included_tensors) {
            if(it.second.tensor_level_exists(forall_idx)){
                tensors_with_curr_dim.push_back(it.second);
                if(it.second.is_sparse(forall_idx)) {
                    tensors_with_curr_dim_sparse.push_back(it.second);
                    if(!it.second.is_unique(forall_idx)) {
                        all_unique_levels = false;
                    }
                }
            } 
        }


        // Optimization Case : Can Use normal balanced mergepath to optimize partitioning
        // When no dim is sparse the general scheme is best as that will require just 1 binary search.
        if(!need_to_exclude_tensors_at_runtime && included_tensors.size()==2 && tensors_with_curr_dim_sparse.size()==2 && is_last_loop && all_unique_levels) {
            return lower_mergepath_partition_loop(loop_num, is_last_loop, tensors_with_curr_dim_sparse);
        }

        // Optimization Case: Can return partitions directly w/o binary search
        if(!need_to_exclude_tensors_at_runtime && included_tensors.size()==1 && is_last_loop && all_unique_levels) {
            return lower_trivial_partition_loop(loop_num, is_last_loop);
        }

        // Optimization Case: Can use simple partitioning when no tensor has sparse dim at current level
        if(!need_to_exclude_tensors_at_runtime && tensors_with_curr_dim_sparse.size() == 0 && is_last_loop && all_unique_levels) {
            return lower_trivial_partition_loop(loop_num, is_last_loop);
        }   

        //TODO: Optimization Case - not last loop but included tensor size == 1 and the tensor is sparse

        //TODO: Optimization Case - not last loop but all included tensors are dense and there is no sparse dim in both the tensors below this loop

        llir::lExpr start_var = llir::lVar::make(index_t, "start_"+forall_idx);
        llir::lExpr end_var = llir::lVar::make(index_t, "end_"+forall_idx);
        llir::lExpr mid_var = llir::lVar::make(index_t, forall_idx);
        

        llir::lExpr start_expr, end_expr, mid_expr;
        start_expr = llir::lConst::make((int64_t)-1);

        auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(),
            [&](const auto &pair) {
                return pair.second.tensor_level_exists(forall_idx);
            });
        end_expr = it->second.get_size_field(forall_idx) -1;

        mid_expr = is_last_loop ? start_var + ((end_var - start_var)) / llir::lConst::make((int64_t)2)
         : start_var + ((end_var - start_var) + 1) / llir::lConst::make((int64_t)2);

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
                forall_idx,
                mid_expr
            )
        );
        for(int i=0;i<tensors_with_curr_dim_sparse.size();i++) {
            stmts.emplace_back(
                llir::Declare::make(
                    index_t,
                    tensors_with_curr_dim_sparse[i].get_start_name(forall_idx),
                    get_sparse_dim_start_expr(tensors_with_curr_dim_sparse[i], forall_idx) - 1
                )
            );
            auto end_expr = get_sparse_dim_end_expr(tensors_with_curr_dim_sparse[i], forall_idx) - 1;
            if(need_to_exclude_tensors_at_runtime) {
                end_expr = llir::lSelect::make(
                    tensor_partitioned_vars.at(tensors_with_curr_dim_sparse[i].tensor_name),
                    llir::lVar::make(index_t, tensors_with_curr_dim_sparse[i].get_start_name(forall_idx)),
                    end_expr
                );
            }
            stmts.emplace_back(
                llir::Declare::make(
                    index_t,
                    tensors_with_curr_dim_sparse[i].get_end_name(forall_idx),
                    end_expr
                )
            );
            stmts.emplace_back(
                llir::Declare::make(
                    index_t,
                    tensors_with_curr_dim_sparse[i].get_iterator_suffix(forall_idx),
                    llir::lVar::make(index_t, tensors_with_curr_dim_sparse[i].get_start_name(forall_idx))
                )
            );

            if(!tensors_with_curr_dim_sparse[i].is_unique(forall_idx)) {
                stmts.emplace_back(
                    llir::Declare::make(
                        index_t,
                        tensors_with_curr_dim_sparse[i].get_seg_end_name(forall_idx),
                        llir::lVar::make(index_t, tensors_with_curr_dim_sparse[i].get_start_name(forall_idx))
                    )
                );
            }
        }
        
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
            while_stmts.emplace_back(get_statements_to_find_sparse_position(loop_num, tensor, is_last_loop ? mid_var : mid_var - 1, need_to_exclude_tensors_at_runtime, true, false));
        }


        // Now have to calculate the total work for the parition with current indices
        while_stmts.emplace_back(
            llir::Store::make(
                work_var,
                llir::lConst::make((int64_t)0)
            )
        );

        for(auto it : included_tensors) {
            TensorLowerer tensor = it.second;
            // call work function
            while_stmts.emplace_back(
                llir::Store::make(
                    work_var,
                    work_var + (need_to_exclude_tensors_at_runtime && loop_num != BEFORE_FIRST_LOOP+1 ? llir::lSelect::make(
                        tensor_partitioned_vars.at(tensor.tensor_name),
                        llir::lConst::make((int64_t)0),
                        get_call_work_function_expr(loop_num, tensor, is_last_loop ? mid_var : mid_var - 1)
                    ) : get_call_work_function_expr(loop_num, tensor, is_last_loop ? mid_var : mid_var - 1))
                )
            );
        }

        // store calculated paritions into partition struct if needed partition is found.
        while_stmts.emplace_back(
            llir::IfElse::make(
                end_var <= start_var,
                llir::Sequence::make({
                    get_store_partition_statements(loop_num, mid_var, need_to_exclude_tensors_at_runtime, is_last_loop),
                    llir::Store::make(rem_count_var, rem_count_var - work_var),
                    llir::Break::make(),
                }),
                nullptr
            )
        );

        auto start_modifier_statements = std::vector<llir::lStmt>{llir::Store::make(
                start_var,
                is_last_loop? mid_var + 1 : mid_var
            )};
        auto end_modifier_statements = std::vector<llir::lStmt>{llir::Store::make(
                        end_var,
                        is_last_loop? mid_var : mid_var - 1
                    )};

        for(int i=0;i<tensors_with_curr_dim_sparse.size();i++) {
            TensorLowerer tensor = tensors_with_curr_dim_sparse[i];
            llir::lExpr start_position_var = llir::lVar::make(index_t, tensor.get_start_name(forall_idx));
            llir::lExpr end_position_var = llir::lVar::make(index_t, tensor.get_end_name(forall_idx));
            llir::lExpr position_var = llir::lVar::make(index_t, tensor.get_iterator_suffix(forall_idx));
            auto store_stmt = llir::Store::make(start_position_var, position_var);
            if(need_to_exclude_tensors_at_runtime && loop_num != BEFORE_FIRST_LOOP+1) {
                store_stmt = llir::IfElse::make(
                    llir::lOp::make(llir::lOp::Not, tensor_partitioned_vars.at(tensor.tensor_name)),
                    store_stmt, nullptr
                );
            }
            start_modifier_statements.emplace_back(store_stmt);
            store_stmt = llir::Store::make(end_position_var,position_var);
            if(need_to_exclude_tensors_at_runtime && loop_num != BEFORE_FIRST_LOOP+1) {
                store_stmt = llir::IfElse::make(
                    llir::lOp::make(llir::lOp::Not, tensor_partitioned_vars.at(tensor.tensor_name)),
                    store_stmt, nullptr
                );
            }
            end_modifier_statements.emplace_back(store_stmt);
        }

        while_stmts.emplace_back(
                llir::IfElse::make(
                    work_var < rem_count_var,
                    llir::Sequence::make(
                        start_modifier_statements
                    ),
                    llir::Sequence::make(
                        end_modifier_statements
                    )
                )
            );

        stmts.emplace_back(
            llir::While::make(
                llir::lConst::make((bool)true),
                llir::Sequence::make(std::move(while_stmts))
            )
        );        
        return llir::Sequence::make(std::move(stmts));
    }

    llir::lStmt PartitionKernelLowerer::lower_mergepath_partition_loop(LoopNum loop_num, bool is_last_loop, std::vector<TensorLowerer>& tensors_with_curr_dim_sparse) {
        std::vector<llir::lStmt> stmts;
        internal_assert(is_last_loop) << "Mergepath partitioning can only be used in the innermost loop";
        const Forall* forall = forall_list[loop_num.get()].as<Forall>();
        std::string forall_idx = forall->idx;

        internal_assert(tensors_with_curr_dim_sparse.size() == 2);
        auto tensor1 = tensors_with_curr_dim_sparse[0];
        auto tensor2 = tensors_with_curr_dim_sparse[1];

        llir::lExpr tensor1_position_var = llir::lVar::make(index_t, tensor1.get_iterator_suffix(forall_idx));
        llir::lExpr tensor1_position_var_start = llir::lVar::make(index_t, tensor1.get_start_name(forall_idx));
        llir::lExpr tensor1_position_var_end = llir::lVar::make(index_t, tensor1.get_end_name(forall_idx));

        llir::lExpr tensor2_position_var = llir::lVar::make(index_t, tensor2.get_iterator_suffix(forall_idx));
        llir::lExpr tensor2_position_var_start = llir::lVar::make(index_t, tensor2.get_start_name(forall_idx));
        llir::lExpr tensor2_position_var_end = llir::lVar::make(index_t, tensor2.get_end_name(forall_idx));


        stmts.emplace_back(llir::Declare::make(
            index_t, tensor1.get_start_name(forall_idx),
                get_sparse_dim_start_expr(tensor1, forall_idx)-1));

        stmts.emplace_back(llir::Declare::make(
            index_t, tensor1.get_end_name(forall_idx),
                get_sparse_dim_end_expr(tensor1, forall_idx)-1));
        stmts.emplace_back(llir::Declare::make(
            index_t, tensor2.get_start_name(forall_idx),
                get_sparse_dim_start_expr(tensor2, forall_idx)-1));

        stmts.emplace_back(llir::Declare::make(
            index_t, tensor2.get_end_name(forall_idx),
                get_sparse_dim_end_expr(tensor2, forall_idx)-1));

        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                "start_"+forall_idx,
                tensor1_position_var_start
            )
        );
        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                "end_"+forall_idx,
                tensor1_position_var_end
            )
        );
        llir::lExpr start_j = llir::lVar::make(index_t, "start_"+forall_idx);
        llir::lExpr end_j = llir::lVar::make(index_t, "end_"+forall_idx);

        llir::lExpr tensor1_position_var_expr = start_j + (end_j - start_j + 1)/2;
        llir::lExpr tensor2_position_var_expr = tensor2_position_var_start + (rem_count_var - (tensor1_position_var - tensor1_position_var_start));
        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                tensor1.get_iterator_suffix(forall_idx),
                tensor1_position_var_expr
            )
        );
        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                tensor2.get_iterator_suffix(forall_idx),
                tensor2_position_var_expr
            )
        );
        
        std::vector<llir::lStmt> binary_search_stmts;
        binary_search_stmts.emplace_back(
            llir::Store::make(
                tensor1_position_var,
                tensor1_position_var_expr
            )
        );
        binary_search_stmts.emplace_back(
            llir::Store::make(
                tensor2_position_var,
                tensor2_position_var_expr
            )
        );

        std::vector<llir::lStmt> mergepath_statements;

        mergepath_statements.emplace_back(
            llir::IfElse::make(
                tensor2_position_var_start < tensor2_position_var && (tensor1_position_var+1) <= tensor1_position_var_end &&
                    tensor1.get_indices_field(forall_idx)[tensor1_position_var+1] < tensor2.get_indices_field(forall_idx)[tensor2_position_var],
                llir::Store::make(
                    start_j,
                    tensor1_position_var + 1
                ),
                llir::IfElse::make(
                    tensor1_position_var_start < tensor1_position_var && (tensor2_position_var+1) <= tensor2_position_var_end &&
                        tensor2.get_indices_field(forall_idx)[tensor2_position_var+1] < tensor1.get_indices_field(forall_idx)[tensor1_position_var],
                    llir::Store::make(
                        end_j,
                        tensor1_position_var - 1
                    ),
                    llir::Sequence::make({
                        llir::Store::make(
                            start_j,
                            tensor1_position_var
                        ),
                        llir::Store::make(
                            end_j,
                            tensor1_position_var
                        )
                    })
                    
                )
            )
        );


        binary_search_stmts.emplace_back(
            llir::IfElse::make(
                tensor2_position_var_start <= tensor2_position_var && tensor2_position_var <= tensor2_position_var_end,
                llir::Sequence::make(std::move(mergepath_statements)),
                llir::IfElse::make(
                    tensor2_position_var_end < tensor2_position_var,
                    llir::Store::make(
                        start_j,
                        tensor1_position_var + 1
                    ),
                    llir::Store::make(
                    end_j,
                    tensor1_position_var - 1
                    )
                )
            )
        ); 

        stmts.emplace_back(
            llir::While::make(
                start_j < end_j,
                llir::Sequence::make(std::move(binary_search_stmts))
            )
        );
        stmts.emplace_back(
            llir::Store::make(
                tensor1_position_var,
                tensor1_position_var_expr
            )
        );
        stmts.emplace_back(
            llir::Store::make(
                tensor2_position_var,
                tensor2_position_var_expr
            )
        );
        // balance the mergepath partitions
        stmts.emplace_back(
            llir::IfElse::make(
                tensor2_position_var_start < tensor2_position_var && (tensor1_position_var+1) <= tensor1_position_var_end &&
                    tensor1.get_indices_field(forall_idx)[tensor1_position_var+1] == tensor2.get_indices_field(forall_idx)[tensor2_position_var],
                llir::Store::make(
                    tensor1_position_var,
                    tensor1_position_var + 1
                ),
                llir::IfElse::make(
                    tensor1_position_var_start < tensor1_position_var && (tensor2_position_var+1) <= tensor2_position_var_end &&
                        tensor2.get_indices_field(forall_idx)[tensor2_position_var+1] == tensor1.get_indices_field(forall_idx)[tensor1_position_var],
                    llir::Store::make(
                        tensor2_position_var,
                        tensor2_position_var + 1
                    ),
                    nullptr
                )
            )
        );
        llir::lExpr index_value = llir::lBinOp::make(
            llir::lBinOp::Max,
            llir::lSelect::make(tensor1_position_var_start < tensor1_position_var, llir::lConst::make(-1), tensor1.get_indices_field(forall_idx)[tensor1_position_var]),
            llir::lSelect::make(tensor2_position_var_start < tensor2_position_var, llir::lConst::make(-1), tensor2.get_indices_field(forall_idx)[tensor2_position_var])
        );
        stmts.emplace_back(
            get_store_partition_statements(loop_num, index_value, false, true)
        );
        return llir::Sequence::make(std::move(stmts));
    }

    llir::lStmt PartitionKernelLowerer::lower_trivial_partition_loop(LoopNum loop_num, bool is_last_loop) {
        std::vector<llir::lStmt> stmts;

        const Forall* forall = forall_list[loop_num.get()].as<Forall>();
        std::string forall_idx = forall->idx;
        internal_assert(is_last_loop);

        // This case is when all tensors are dense
        if(included_tensors.size() > 1) {
            for(auto it: included_tensors){
                internal_assert(!it.second.is_sparse(forall_idx));
            }

            auto index_value = rem_count_var/included_tensors.size() - 1;
            return get_store_partition_statements(loop_num, index_value, false, true);
        }

        internal_assert(included_tensors.size() == 1);
        // include_tensor.size()==1 case
        TensorLowerer tensor = included_tensors.begin()->second;

        llir::lExpr index_value;
        
        if(tensor.tensor_level_exists(forall_idx) && tensor.is_sparse(forall_idx)) {
            stmts.emplace_back(
                llir::Declare::make(
                    index_t,
                    tensor.get_iterator_suffix(forall_idx),
                    (get_sparse_dim_start_expr(tensor, forall_idx) - 1) + (rem_count_var)
                )
            );
            index_value = tensor.get_indices_field(forall_idx)[(get_sparse_dim_start_expr(tensor, forall_idx) - 1) + (rem_count_var)];
            stmts.emplace_back(
                get_store_partition_statements(loop_num, index_value, false, true)
            );
        } else {
            auto index_value = rem_count_var - 1;
            stmts.emplace_back(
                get_store_partition_statements(loop_num, index_value, false, true)
            );
        }
        return llir::Sequence::make(std::move(stmts)); 
    }

    llir::lStmt PartitionKernelLowerer::lower_partition_loop_from_work_offsets(LoopNum loop_num, bool need_to_exclude_tensors_at_runtime) {
        std::vector<llir::lStmt> stmts;

        const Forall* forall = forall_list[loop_num.get()].as<Forall>();
        std::string forall_idx = forall->idx;

        std::string start_var_name = result_tensor.get_start_name(forall_idx);
        std::string end_var_name = result_tensor.get_end_name(forall_idx);
        std::string mid_var_name = result_tensor.get_iterator_suffix(forall_idx);

        llir::lExpr start_var = llir::lVar::make(index_t, start_var_name);
        llir::lExpr end_var = llir::lVar::make(index_t, end_var_name);
        llir::lExpr mid_var = llir::lVar::make(index_t, mid_var_name);

        llir::lExpr start_expr, end_expr, mid_expr;

        internal_assert(result_tensor.tensor_level_exists(forall_idx))<< "Tensor level does not exist in result tensor "<<result_tensor.tensor_name;

        if(result_tensor.is_sparse(forall_idx)) {
            start_expr = get_sparse_dim_start_expr(result_tensor, forall_idx)-1;
            end_expr = get_sparse_dim_end_expr(result_tensor, forall_idx)-1;
        } else {
            start_expr = llir::lConst::make((int64_t)-1);
            end_expr = result_tensor.get_size_field(forall_idx) -1;
        }

        mid_expr = start_var + ((end_var - start_var) + 1) / llir::lConst::make((int64_t)2);
        
        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                start_var_name,
                start_expr
            )
        );

        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                end_var_name,
                end_expr
            )
        );

        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                mid_var_name,
                mid_expr
            )
        );

        for(auto it : included_tensors) {
            if(it.second.tensor_level_exists(forall_idx)){
                if(it.second.is_sparse(forall_idx)) {
                    stmts.emplace_back(
                        llir::Declare::make(
                            index_t,
                            it.second.get_iterator_suffix(forall_idx)
                        )
                    );
                }
            }
        }

        std::vector<llir::lStmt> while_stmts;
        while_stmts.emplace_back(
            llir::Store::make(mid_var, mid_expr)
        );
        while_stmts.emplace_back(
            llir::Store::make(
                work_var,
                get_call_work_function_expr(loop_num, result_tensor, mid_var-1)
            )
        );
    

        std::vector<llir::lStmt> break_stmts;

        break_stmts.emplace_back(llir::Store::make(rem_count_var, rem_count_var - work_var));

        break_stmts.emplace_back(
            llir::Store::make(
            get_partition_struct_current_thread_field(result_tensor.get_iterator_suffix(forall_idx)),
            result_tensor.is_sparse(forall_idx) ? mid_var+1 : mid_var)
        );

        for(auto it : included_tensors) {
            if(it.second.tensor_level_exists(forall_idx)){
                if(it.second.is_sparse(forall_idx)) {
                    break_stmts.emplace_back(
                        llir::Store::make(
                            tensor_partitioned_vars.at(it.second.tensor_name),
                            tensor_partitioned_vars.at(it.second.tensor_name) || (map_result_pos_to_operand_pos(forall, it.second, result_tensor.is_sparse(forall_idx) ? mid_var+1 : mid_var) == llir::lConst::make(-1))
                        )
                    );
                    break_stmts.emplace_back(
                        llir::Store::make(
                            llir::lVar::make(index_t, it.second.get_iterator_suffix(forall_idx)),
                            llir::lSelect::make(
                                map_result_pos_to_operand_pos(forall, it.second, result_tensor.is_sparse(forall_idx) ? mid_var+1 : mid_var) == llir::lConst::make(-1),
                                it.second.get_length_field(forall_idx), // Not sure if assigning legth value is right here, mostly any value that
                                                                        // does not cause invalid writes later is fine as this wont be used anywhere else anyway.
                                map_result_pos_to_operand_pos(forall, it.second, result_tensor.is_sparse(forall_idx) ? mid_var+1 : mid_var)
                            )   
                            
                        )
                    );
                }
            }
        }

        break_stmts.emplace_back(llir::Break::make());
        while_stmts.emplace_back(
            llir::IfElse::make(
                end_var <= start_var,
                llir::Sequence::make(std::move(break_stmts)),
                nullptr
            )
        );

        while_stmts.emplace_back(
            llir::IfElse::make(
                work_var < rem_count_var,
                llir::Store::make(
                    start_var,
                    mid_var
                ),
                llir::Store::make(
                    end_var,
                    mid_var - 1
                )
            )
        );

        stmts.emplace_back(
            llir::While::make(
                llir::lConst::make((bool)true),
                llir::Sequence::make(std::move(while_stmts))
            )
        );
        if(result_tensor.is_sparse(forall_idx)) {
            stmts.emplace_back(
                llir::Declare::make(index_t, forall_idx, result_tensor.get_indices_field(forall_idx)[result_tensor.is_sparse(forall_idx) ? mid_var+1 : mid_var])
            );
        }

        return llir::Sequence::make(std::move(stmts));
    }

} // namespace backend

} // namespace nacho
