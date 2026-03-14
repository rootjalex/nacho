#include "backend/base_lowerer.h"
#include "llir/LLIR.h"

namespace nacho {
namespace backend {

llir::lType BaseKernelLowerer::lower_result_per_thread_count_struct() {
    std::vector<std::string> generics = {"index_t"};
    std::vector<std::pair<std::string, llir::lType>> fields;

    for (int i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
        auto index = result_tensor.tensor_type.format.levels[i].index;
        if (is_sparse_format(
                result_tensor.tensor_type.format.lvlfmt_of(index))) {
            fields.emplace_back(get_counts_field_name(index),
                                llir::Ptr_t::make(index_t));
        }
    }

    return llir::Struct_t::make(get_counts_struct_name(), std::move(fields),
                                std::move(generics));
}


//  check whether result_to_operand_pos_map contains a field to lookup pos for this operand_tensor at current forall sparse intersection
bool BaseKernelLowerer::has_result_to_operand_pos_field(const Forall* forall, TensorLowerer& op_tensor) {
    std::string index = forall->idx;
    if(!op_tensor.tensor_type.format.level_exists(index)) {
        return false;
    }

    if(!op_tensor.is_sparse(index)) {
        return false;
    }

    // Optimization : if there is a single sparse tensor in the this level, and the result level is also sparse 
    // then the field does not need to be added (mapping is direct 1-1 from result pos to operand pos, as result dim = sparse dim)
    if (result_tensor.is_sparse(index)) {
        int num_sparse_operands = 0;
        for(auto it: operand_tensors) {
            auto tensor = it.second;
            if(tensor.tensor_type.format.level_exists(index) && tensor.is_sparse(index)) {
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
llir::lExpr BaseKernelLowerer::build_operand_position_from_result_position(const Forall* forall, TensorLowerer& operand_tensor, llir::lExpr result_tensor_pos, llir::lExpr result_tensor_coord) {
    if(has_result_to_operand_pos_field(forall, operand_tensor)) {
        return llir::lFieldAccess::make(llir::lVar::make(
            llir::Generic_t::make(get_result_to_operand_pos_map_struct_name()),
            get_result_to_operand_pos_map_var_name()
            ),
            operand_tensor.get_iterator_suffix(forall->idx)
        )[result_tensor_pos];
    } else if (!operand_tensor.tensor_type.format.level_exists(forall->idx) || !operand_tensor.is_sparse(forall->idx)) {
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

llir::lExpr BaseKernelLowerer::build_partition_boundary_initializer_expr(const int forall_level, TensorLowerer& tensor, bool is_last_thread) {
    std::string forall_idx = forall_loops[forall_level].as<Forall>()->idx;
    
    if(forall_level <= previous_sparse_intersection_level) {
        if(is_last_thread) {
            return tensor.is_sparse(forall_idx) ? tensor.get_length_field(forall_idx) - 1 : tensor.get_size_field(forall_idx) - 1;
        } else {
            return llir::lConst::make((int64_t)0);
        }
    }

    // forall_level > previous_sparse_intersection_level

    
    llir::lExpr init_value_expr = llir::lConst::make((int64_t)0) - (forall_level == current_sparse_intersection_level? 1 : 0);
    if(is_last_thread) {
        init_value_expr = tensor.is_sparse(forall_idx)
                              ? tensor.get_length_field(forall_idx) - 1
                              : tensor.get_size_field(forall_idx) - 1;
    }
    // For some cases initializers need to be the form
    // partitions.a_j_p[thread_id] = a.dim_j_offsets[Z.i_map_a[0]]-1;
    // when the data in sparse level of the tensor is being skipped due to previous sparse intersections
    if(previous_sparse_intersection_level!=-1) {
        std::vector<llir::lExpr> dim_vars_for_offset_expression;
        int start_level = tensor.tensor_type.format.get_prev_sparse_level(forall_level);
        // result_to_operand_map may have -1 value, we need to add a check for this to prevent 
        // illegal memory access
        llir::lExpr check_expr;
        if(start_level!=-1 && start_level<=previous_sparse_intersection_level)
        {
            const Forall* forall_j = forall_loops[start_level].as<Forall>();
            std::string forall_j_idx = forall_j->idx;
            check_expr = build_operand_position_from_result_position(
                    forall_j, tensor, 
                    llir::lConst::make((int64_t)0)
                ) != llir::lConst::make((int64_t)-1);

            if(is_last_thread) {
                check_expr = build_operand_position_from_result_position(
                    forall_j, tensor, 
                    result_tensor.is_sparse(forall_j_idx) ? result_tensor.get_length_field(forall_j_idx) - 1 : result_tensor.get_size_field(forall_j_idx) - 1
                ) != llir::lConst::make((int64_t)-1);
            }
        }   

        int end_level = forall_level-1;
        for(int j=std::max(0, start_level);j<=end_level;j++) {
            const Forall* forall_j = forall_loops[j].as<Forall>();
            std::string forall_j_idx = forall_j->idx;
            if(j<=previous_sparse_intersection_level) {
                dim_vars_for_offset_expression.push_back(
                    build_operand_position_from_result_position(
                        forall_j, tensor, 
                        is_last_thread ? (result_tensor.is_sparse(forall_j_idx) ? result_tensor.get_length_field(forall_j_idx) - 1 : result_tensor.get_size_field(forall_j_idx) - 1) : llir::lConst::make((int64_t)0) 
                ));
            } else {
                dim_vars_for_offset_expression.push_back(
                    tensor.is_sparse(forall_j->idx) ? get_partition_struct_current_thread_field(tensor.get_iterator_suffix(forall_j->idx)) : ( is_last_thread ? result_tensor.get_size_field(forall_j_idx) - 1 :llir::lConst::make((int64_t)0))
                );
            }
        }
        init_value_expr =  tensor.get_offsets_field(forall_idx)[
            tensor.get_offset_expression_for_next_sparse(
                start_level,
                end_level,
                false,
                true,
                dim_vars_for_offset_expression 
            ) ] - (forall_level == current_sparse_intersection_level? 1 : 0);
        if(is_last_thread) {
            init_value_expr =  tensor.get_offsets_field(forall_idx)[
                tensor.get_offset_expression_for_next_sparse(
                    start_level,
                    end_level,
                    true,
                    true,
                    dim_vars_for_offset_expression
                )] - 1;
        }
        if(check_expr.defined()) {
            init_value_expr = llir::lSelect::make(
                check_expr,
                init_value_expr,
                is_last_thread
                    ? llir::lConst::make((int64_t)0)
                    : (tensor.is_sparse(forall_idx)
                           ? tensor.get_length_field(forall_idx) - 1
                           : tensor.get_size_field(forall_idx) - 1)
            );
        }
    } 
    return init_value_expr;
}

}
}
