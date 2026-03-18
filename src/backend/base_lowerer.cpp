#include "backend/base_lowerer.h"
#include "llir/LLIR.h"

namespace nacho {
namespace backend {

llir::lType BaseKernelLowerer::lower_result_per_thread_count_struct() {
    std::vector<std::string> generics = {"index_t"};
    std::vector<std::pair<std::string, llir::lType>> fields;

    for (LoopNum i = BEFORE_FIRST_LOOP + 1; i < result_tensor.end_loop_num(); ++i) {
        if (result_tensor.tensor_level_exists(i) && result_tensor.is_sparse(i)) {
            fields.emplace_back(get_counts_field_name(result_tensor.loop_name(i)),
                                llir::Ptr_t::make(index_t));
        }
    }

    return llir::Struct_t::make(get_counts_struct_name(), std::move(fields),
                                std::move(generics));
}


//  check whether result_to_operand_pos_map contains a field to lookup pos for this operand_tensor at current forall sparse intersection
bool BaseKernelLowerer::exists_field_in_result_to_operand_pos_map(const Forall* forall, TensorLowerer& op_tensor) {
    std::string index = forall->idx;
    if(!op_tensor.tensor_level_exists(index)) {
        return false;
    }

    if(!op_tensor.is_sparse(index)) {
        return false;
    }

    // Optimization : if there is a single sparse tensor in the this level, and the result level is also sparse 
    // and this is the first sparse level in the result tensor
    // then the field does not need to be added (mapping is direct 1-1 from result pos to operand pos, as result dim = sparse dim)
    if (result_tensor.is_sparse(index) && result_tensor.get_loop_num_for_next_sparse_level(BEFORE_FIRST_LOOP) == result_tensor.get_loop_num(index)) {
        int num_sparse_operands = 0;
        for(auto it: operand_tensors) {
            auto tensor = it.second;
            if(tensor.tensor_level_exists(index) && tensor.is_sparse(index)) {
                num_sparse_operands++;
            }
        }
        if (num_sparse_operands == 1) {
            return false;
        }
    }
    return true;
}

// map a result tensor (pos, coord) pair to corresponding pos in operand_tensor
llir::lExpr BaseKernelLowerer::map_result_pos_to_operand_pos(const Forall* forall, TensorLowerer& operand_tensor, llir::lExpr result_tensor_pos, llir::lExpr result_tensor_coord) {
    if(exists_field_in_result_to_operand_pos_map(forall, operand_tensor)) {
        return llir::lFieldAccess::make(llir::lVar::make(
            llir::Generic_t::make(get_result_to_operand_pos_map_struct_name()),
            get_result_to_operand_pos_map_var_name()
            ),
            operand_tensor.get_iterator_suffix(forall->idx)
        )[result_tensor_pos];
    } else if (!operand_tensor.tensor_level_exists(forall->idx) || !operand_tensor.is_sparse(forall->idx)) {
        // this means operand_tensors corresponding dim is dense, so can lookup just by the actual coord value
        if(result_tensor_coord.get()!=nullptr){
            return result_tensor_coord;
        } else {
            // if this level in result is dense then result_tensor_pos = result_tensor.get_indices_field(forall->idx)[result_tensor_pos]
            return result_tensor.is_sparse(forall->idx) ? result_tensor.get_indices_field(forall->idx)[result_tensor_pos] : result_tensor_pos;
        }
    } else {
        // this means operand_tensors corresponding dim is sparse, but there is no field in result_to_operand_pos_map means, 
        // this can only happend there is only 1 sparse dim in this level, so there is a 1-1 mapping between result pos and operand pos, 
        // so can directly return result pos as operand pos without needing a field lookup
        return result_tensor_pos;
    }

    return llir::lExpr();
}

llir::lExpr BaseKernelLowerer::get_partition_initializer_expr_for_boundary_cases(LoopNum forall_loop_num, TensorLowerer& tensor, bool is_last_thread) {
    internal_assert(tensor.tensor_level_exists(forall_loop_num) && tensor.is_sparse(forall_loop_num)) << "Expected sparse tensor at level " << forall_loop_num;
    std::string forall_idx = tensor.loop_name(forall_loop_num);

    if(forall_loop_num <= previous_sparse_intersection) {
        if(is_last_thread) {
            return tensor.tensor_level_exists(forall_idx) && tensor.is_sparse(forall_idx) ? tensor.get_length_field(forall_idx) - 1 : tensor.get_size_field(forall_idx) - 1;
        } else {
            return llir::lConst::make((int64_t)0);
        }
    }

    if(!tensor.tensor_level_exists(forall_idx)) {
        internal_assert(false) << forall_idx << "Tensor level does not exist in "<<tensor.tensor_name;
    }
    // forall_level > previous_sparse_intersection

    
    llir::lExpr init_value_expr = llir::lConst::make((int64_t)0) - (forall_loop_num == current_sparse_intersection? 1 : 0);
    if(is_last_thread) {
        init_value_expr = tensor.get_length_field(forall_idx)-1;
    }
    // For some cases initializers need to be the form
    // partitions.a_j_p[thread_id] = a.dim_j_offsets[Z.i_map_a[0]]-1;
    // when the data in sparse level of the tensor is being skipped due to previous sparse intersections
    if(previous_sparse_intersection != BEFORE_FIRST_LOOP) {
        std::map<TensorLevelNum, llir::lExpr> pos_vars_for_offset_expression;
        LoopNum prev_sparse_level_loop_num =  tensor.get_loop_num_for_prev_sparse_level(forall_loop_num);
        TensorLevelNum prev_sparse_tensor_level = tensor.loop_num_to_tensor_level(prev_sparse_level_loop_num);
        // result_to_operand_map may have -1 value, we need to add a check for this to prevent 
        // illegal memory access
        llir::lExpr check_expr;
        if(prev_sparse_level_loop_num != BEFORE_FIRST_LOOP && prev_sparse_level_loop_num<=previous_sparse_intersection)
        {
            const Forall* forall_j = forall_list[prev_sparse_level_loop_num.get()].as<Forall>();
            std::string forall_j_idx = forall_j->idx;
            check_expr = map_result_pos_to_operand_pos(
                    forall_j, tensor, 
                    llir::lConst::make((int64_t)0)
                ) != llir::lConst::make((int64_t)-1);

            if(is_last_thread) {
                check_expr = map_result_pos_to_operand_pos(
                    forall_j, tensor, 
                    result_tensor.get_length_field(forall_j_idx) - 1
                ) != llir::lConst::make((int64_t)-1);
            }
        }   

        for(TensorLevelNum j=std::max(BEFORE_FIRST_LEVEL+1, prev_sparse_tensor_level);j<=tensor.loop_num_to_tensor_level(forall_loop_num)-1;++j) {
            const Forall * forall_j = forall_list[tensor.tensor_level_to_loop_num(j).get()].as<Forall>();
            std::string forall_j_idx = tensor.tensor_level_name(j);
            if(!tensor.tensor_level_exists(forall_j_idx)) {
                // if level does not exist the value does not matter, push nullptr
                pos_vars_for_offset_expression[j] = llir::lVar::make(index_t, "invalid-var");
                continue;
            }

            if(tensor.tensor_level_to_loop_num(j) <= previous_sparse_intersection) {
                pos_vars_for_offset_expression[j] = map_result_pos_to_operand_pos(
                        forall_j, tensor, 
                        is_last_thread ? (result_tensor.is_sparse(forall_j_idx) ? result_tensor.get_length_field(forall_j_idx) - 1 : result_tensor.get_size_field(forall_j_idx) - 1) : llir::lConst::make((int64_t)0) 
                );
            } else {
                pos_vars_for_offset_expression[j] = 
                    tensor.is_sparse(forall_j->idx) 
                        ? get_partition_struct_current_thread_field(tensor.get_iterator_suffix(forall_j->idx)) 
                        : ( is_last_thread ? result_tensor.get_size_field(forall_j_idx) - 1 :llir::lConst::make((int64_t)0));
            }
        }
        init_value_expr =  tensor.get_offsets_field(forall_idx)[
            tensor.get_level_indexing_expression(
                tensor.loop_num_to_tensor_level(forall_loop_num),
                false,
                pos_vars_for_offset_expression 
            ) ] - (forall_loop_num == current_sparse_intersection? 1 : 0);
        if(is_last_thread) {
            init_value_expr =  tensor.get_offsets_field(forall_idx)[
                tensor.get_level_indexing_expression(
                    tensor.loop_num_to_tensor_level(forall_loop_num),
                    true,
                    pos_vars_for_offset_expression 
                )] - 1;
        }
        if(check_expr.defined()) {
            init_value_expr = llir::lSelect::make(
                check_expr,
                init_value_expr,
                is_last_thread? llir::lConst::make((int64_t)0) : tensor.get_length_field(forall_idx) -1
            );
        }
    } 
    return init_value_expr;
}

}
}