#include "backend/tensor.h"
#include "Error.h"
#include "Format.h"
#include "llir/Function.h"
#include "llir/LLIR.h"
#include <algorithm>
#include <vector>
#include <map>

namespace nacho {
namespace backend {

llir::lType TensorLowerer::lower_tensor_struct_definition() const {
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    std::vector<std::string> generics = {"index_t", "value_t"};
    std::vector<std::pair<std::string, llir::lType>> data_fields;
    data_fields.emplace_back("nnz", index_t);
    data_fields.emplace_back(get_values_field_name(), llir::Ptr_t::make(value_t));
    for (TensorLevelNum i = tensor_type.format.get_end_level()-1; i > BEFORE_FIRST_LEVEL; --i) {
        auto index = tensor_level_index(i);

        // only dimensions with sparse format need separate fields for index and
        // offsets
        // TODO: This only handles compressed dimensions as of now
        if(is_merged_level(i)) {
            for(auto it = index.indices.rbegin(); it != index.indices.rend(); ++it) {
                data_fields.emplace_back(get_indices_field_name(TensorIndex(*it)), llir::Ptr_t::make(index_t));
                data_fields.emplace_back(get_length_field_name(TensorIndex(*it)), index_t);
            }
        } else if (is_sparse(i)) {
            data_fields.emplace_back(get_indices_field_name(index),
                                     llir::Ptr_t::make(index_t));
            data_fields.emplace_back(get_length_field_name(index), index_t);
            // offsets field is not required if the outermost dimension is
            // compressed
            if (is_compressed(i) && i != BEFORE_FIRST_LEVEL+1)
                data_fields.emplace_back(get_offsets_field_name(index),
                                         llir::Ptr_t::make(index_t));
        }
    }

    std::vector<std::pair<std::string, llir::lType>> size_fields;
    for (auto i = BEFORE_FIRST_LEVEL+1; i < tensor_type.format.get_end_level(); ++i) {
         auto index = tensor_level_index(i);
        if(is_merged_level(i)) {
             for(auto idx : index.indices) {
                size_fields.emplace_back(get_size_field_name(TensorIndex(idx)), index_t);
            }
        } else {
            size_fields.emplace_back(get_size_field_name(index), index_t);
        }
    }

    std::vector<std::pair<std::string, llir::lType>> fields;
    fields.insert(fields.end(), size_fields.begin(), size_fields.end());
    fields.insert(fields.end(), data_fields.rbegin(), data_fields.rend());

    return llir::Struct_t::make(get_struct_name(), std::move(fields),
                                std::move(generics));
}



// get_level_indexing_expression gets the indexing expression for level tensor_level.
// To get the indexing expression, all the position space values of previous levels is the tensor are required.
// These position space values have to be passed as a map from the tensor levels to their expressions.
// NOTE: the level passed is level number in this tensor not the loop number among all loops
// tensor_level can also be last_tensor_level().
// if tensor_level == last_tensor_level() then this returns the indexing expression into the values array
llir::lExpr TensorLowerer::get_level_indexing_expression(TensorLevelNum tensor_level,
    bool upper_bound, std::map<TensorLevelNum, llir::lExpr> pos_vars) {
    internal_assert(tensor_level <= end_tensor_level() && tensor_level > BEFORE_FIRST_LEVEL) << "Tensor level " << tensor_level.get() << " does not exist in tensor "<< tensor_name;
    // std::cout << "Getting indexing expression for tensor level " << tensor_level.get() << " tensor " << tensor_name << tensor_level_index(tensor_level) << std::endl;
    bool is_target_sparse = tensor_level < end_tensor_level() && is_sparse(tensor_level);
    TensorLevelNum prev_sparse_level = tensor_type.format.get_prev_sparse_level(tensor_level);

    llir::lExpr expr = llir::lConst::make((int32_t)0);
    if(prev_sparse_level > BEFORE_FIRST_LEVEL) {
        internal_assert(pos_vars.find(prev_sparse_level) != pos_vars.end()) << "Position variable for level " << prev_sparse_level << " not found. Tensor " << tensor_name;
        expr = pos_vars[prev_sparse_level];
        for(TensorLevelNum i = prev_sparse_level + 1; i < tensor_level; ++i) {
            expr = expr * get_size_field(tensor_level_index(i));
        }
        if (upper_bound && prev_sparse_level + 1 == tensor_level) {
            expr = expr + 1;
        }
    }

    for(TensorLevelNum i = prev_sparse_level + 1; i < tensor_level; ++i) {
        internal_assert(pos_vars.find(i) != pos_vars.end()) << "Position variable for level " << i << " not found";
        llir::lExpr sub_expr = pos_vars[i];
        for(TensorLevelNum j = i + 1; j < tensor_level; ++j) {
            sub_expr = sub_expr * get_size_field(tensor_level_index(j));
        }
        expr = expr + sub_expr;

        if(upper_bound && is_target_sparse && i + 1 == tensor_level) {
            expr = expr + 1;
        }
    }
    return expr;
}


// get_bound gets the upper/lower bound for the position space value for level tensor_level when all the positions in all
// previous levels are fixed.
// To get the bounds, all the position space values of previous levels is the tensor are required.
// These position space values have to be passed as a map from the tensor levels to their expressions.
// NOTE: the level passed is level number in this tensor not the level number among all loops
llir::lExpr TensorLowerer::get_bound(TensorLevelNum tensor_level,
    bool upper_bound, std::map<TensorLevelNum, llir::lExpr> pos_vars) {
    internal_assert(tensor_level < end_tensor_level()) << "Tensor level " << tensor_level << " does not exist in tensor "<< tensor_name;

    if (!is_sparse(tensor_level)) {
        if (upper_bound) {
            return get_size_field(tensor_level_index(tensor_level)) - 1;
        } else {
            return llir::lConst::make(0);
        }
    }

    // this level is sparse
    if(tensor_level==BEFORE_FIRST_LEVEL+1) {
        if(upper_bound) {
            return get_length_field(tensor_level) - 1;
        } else {
            return llir::lConst::make(0);
        }
    }

    llir::lExpr offset = get_level_indexing_expression(tensor_level, upper_bound, pos_vars);
    if (is_compressed(tensor_level)) {
        llir::lExpr offset = get_level_indexing_expression(tensor_level, upper_bound, pos_vars);
        llir::lExpr array = get_offsets_field(tensor_level_index(tensor_level));
        return upper_bound ? array[offset] - 1 : array[offset];
    } else if (is_singleton(tensor_level)) {
        llir::lExpr offset = get_level_indexing_expression(tensor_level, upper_bound, pos_vars);
         return upper_bound ? offset - 1 : offset;
    } else {
        internal_assert(false) << "Unsupported format for level " << tensor_level.get() << " in tensor " << tensor_name;
    }
}


// lower_work_function returns the LLIR work function for the given tensor.
// target_dim is the target dimension for which work is being calculated.
// loop_order is the the current loop order (The loop order can be any
// prefix of the actual loop order. This is so that you can generate
// work functions for only a prefix of the loops to calculate work for the current sparse intersection loop). 
// Consider a DCSR tensor with dims i,k and the loop order is
// [i,j,k,l] (the DCSR tensor is broadcaster over j,l) And now we need to
// generate a work function which can calulate work for i=12, j=12, 0<= k <=54.
// 0 <= l <= |L| Then the args are loop_order = [i,j,k,l],target_dim = 2 (k) The
// generated work function. will take the values of i,j,k as the arguments.
//
// if is_target_dim_value_fixed is true then the generated work function calculates work for
// when target_dim is fixed to the specific value instead of target_dim being iterated from
// 0 to the the value. For example for DCSR example above if work function is generated
// with is_target_dim_value_fixed = true then the generated work function will calculate
// work for i=12, j=12, k=54, 0 <= l <= |L|.
llir::lStmt
TensorLowerer::lower_work_function(std::vector<TensorIndex> partial_loop_order,
                                   LoopNum target_loop_num, bool is_target_loop_value_fixed) {

    internal_assert(target_loop_num < LoopNum(partial_loop_order.size())) << "Target dimension has to be less than loop order size";

    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    std::vector<std::string> generics = {"index_t", "value_t"};

    std::vector<llir::Function::Attribute> attributes = {
        llir::Function::runnable};


    // Check what is the last level in the partial loop order that is present inside this tensor
    LoopNum last_level_loop_num_in_partial_loop = BEFORE_FIRST_LOOP;
    for(LoopNum i=LoopNum(partial_loop_order.size()-1); i>BEFORE_FIRST_LOOP; --i) {
        if(tensor_level_exists(i)) {
            last_level_loop_num_in_partial_loop = i;
            break;
        }
    }
    // last_tensor_level_in_partial_loop is the lowest tensor level that exists in the partial_loop_order for this work function
    TensorLevelNum last_tensor_level_in_partial_loop = loop_num_to_tensor_level(last_level_loop_num_in_partial_loop);

    // checks the partial loop order is consistent with the global loop order
    auto violates_order = [&](const std::vector<TensorIndex> &partial_loop_order) {
        for (size_t i = 0; i + 1 < partial_loop_order.size(); ++i) {
            if(partial_loop_order[i] != all_loop_indices[i])
                return true;
        }
        return false;
    };

    if (violates_order(partial_loop_order))
        internal_assert(false) << "Incompatible loop order";

    std::vector<llir::Function::Argument> args;
    args.emplace_back(llir::Function::Argument{
        .mutating = false,
        .type = llir::Generic_t::make(get_struct_name() + "<index_t, value_t>"),
        .name = tensor_name});

    // last_sparse_level_before_target_loop will store the last sparse level for this tensor coming before target_loop
    TensorLevelNum last_sparse_tensor_level_before_target_loop = BEFORE_FIRST_LEVEL;
    // last_tensor_level will store the last level for this tensor which comes before target_loop
    TensorLevelNum last_tensor_level_before_target_loop = BEFORE_FIRST_LEVEL;
    for (LoopNum i = BEFORE_FIRST_LOOP + 1; i <= target_loop_num; ++i) {

        // only add arguments for non-broadcasted levels
        if (tensor_level_exists(i)) {
            // sparse dimensions are iterated by positions while dense are
            // iterated by coordinates hence args are named accordingly
            std::string name = loop_index(i).str();
            if (is_sparse(i)) {
                // sparse dimensions are iterated by positions while dense are
                // iterated by coordinates
                name += "_p";
                last_sparse_tensor_level_before_target_loop = loop_num_to_tensor_level(i);
            }
            last_tensor_level_before_target_loop = loop_num_to_tensor_level(i);
            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = index_t, .name = name});
        }
    }

    // If target_dim is a broadcasted dimension, add it as an argument. As work
    // depends on value of the target_dim
    if (!tensor_level_exists(target_loop_num)) {
        args.emplace_back(
            llir::Function::Argument{.mutating = false,
                                     .type = index_t,
                                     .name = loop_index(target_loop_num).str()});
    }

    // broadcast Sizes of all broadcast dimensions coming after target_dim need
    // to be added as arguments. As work calculation will require multiplying
    // with these bc_sizes at end.
    for (LoopNum i = target_loop_num + 1; i < LoopNum(partial_loop_order.size()); ++i) {
        if (!tensor_level_exists(i)) {
            args.emplace_back(
                llir::Function::Argument{.mutating = false,
                                         .type = index_t,
                                         .name = "bc_size_" + loop_index(i).str()});
        }
    }

    // for result tensor, T_work_offsets needs to be added as an argument , This is required for the recurssive partitioning
    if(is_result_tensor) {
        args.emplace_back(llir::Function::Argument{.mutating = false,
                                                 .type = llir::Ptr_t::make(index_t),
                                                 .name = "T_work_offsets"});
    }

    llir::lType ret_type = llir::Generic_t::make("index_t");
    std::string all_loop_indices_string = std::accumulate(partial_loop_order.begin(), partial_loop_order.end(), std::string(""),
            [](const std::string &acc, const TensorIndex &c) {
                return acc + c.str();
            });
    std::string name = get_work_function_name(all_loop_indices_string, partial_loop_order[target_loop_num.get()]);

    // Create position variables map for each tensor level, used for get_sparse_level_indexing_expression
    std::map<TensorLevelNum, llir::lExpr> pos_vars_start, pos_vars_end;
    for(TensorLevelNum i = BEFORE_FIRST_LEVEL + 1; i <= last_tensor_level_before_target_loop; ++i) {
        if(is_sparse(i)) {
            pos_vars_start[i] = llir::lVar::make(index_t, loop_index(tensor_level_to_loop_num(i)).str() + "_p");
            pos_vars_end[i] = llir::lVar::make(index_t, loop_index(tensor_level_to_loop_num(i)).str() + "_p");
        } else {
            pos_vars_start[i] = llir::lVar::make(index_t, loop_index(tensor_level_to_loop_num(i)).str());
            pos_vars_end[i] = llir::lVar::make(index_t, loop_index(tensor_level_to_loop_num(i)).str());
        }
    }

    // if last_tensor_level_before_target_loop is same as target loop
    // and target_loop_value is not fixed then start var should be 0 for 
    // the dense case. The value for sparse case does not matter, it will
    // get overwritten to 'start' variable anyway in the logic that comes
    // after. TODO: Seems hacky, make this code more readable
    if(tensor_level_exists(target_loop_num) && !is_target_loop_value_fixed) {
        internal_assert(tensor_level_to_loop_num(last_tensor_level_before_target_loop) == target_loop_num);
        if(!is_sparse(target_loop_num)) {
            pos_vars_start[last_tensor_level_before_target_loop] = llir::lConst::make((int64_t)0);
        }
    }

    for(TensorLevelNum i = last_tensor_level_before_target_loop + 1; i <= last_tensor_level_in_partial_loop; ++i) {
        pos_vars_start[i] = llir::lConst::make((int64_t)0);
        pos_vars_end[i] = llir::lConst::make((int64_t)0);
    }

    std::vector<llir::lStmt> stmts;

    // if target_dim is a loop that comes after last_tensor_level in the
    // loop order Then iteration work for this tensor is 0 (as we are not
    // iterating on any non-zeros of the tensor)
    if (last_level_loop_num_in_partial_loop < target_loop_num) {
        stmts.emplace_back(llir::Return::make(llir::lConst::make((int64_t)0)));
        llir::lStmt body = llir::Sequence::make(std::move(stmts));
        return llir::Function::make(std::move(generics), std::move(attributes),
                                    std::move(args), std::move(ret_type), name,
                                    std::move(body));
    }

    // Body of the Work Function has to traverse down the sparse dimensions from
    // the last sparse dimension
    //  to calculate the number of non zeros that are being iterated on
    llir::lExpr count_expr;
    // We need to iterate on all the nonzeros if last_tensor_level_before_target_loop == BEFORE_FIRST_LEVEL
    if (last_tensor_level_before_target_loop == BEFORE_FIRST_LEVEL) {
        // if there is no sparse dimension in the tensor in the given loop order(i.e all dense) just
        // return 1 (as length of pos array for zero dim is 1)
        TensorLevelNum last_sparse_tensor_level_in_partial_loop = tensor_type.format.get_prev_sparse_level(last_tensor_level_in_partial_loop + 1);
        if (last_sparse_tensor_level_in_partial_loop == BEFORE_FIRST_LEVEL) {
            count_expr = llir::lConst::make((int64_t)1);
        } else {
            // We iterate on the whole last sparse dim that comes before last_level_loop_place
            count_expr = this->get_length_field(last_sparse_tensor_level_in_partial_loop);
        }
        for (TensorLevelNum i = last_sparse_tensor_level_in_partial_loop + 1;
             i <= last_tensor_level_in_partial_loop; ++i) {
            count_expr = count_expr * this->get_size_field(i);
        }
        if (!is_result_tensor) {
            stmts.emplace_back(llir::Declare::make(index_t, "count", count_expr));
        } else {
            stmts.emplace_back(llir::Declare::make(
                index_t, "count",
                llir::lVar::make(index_t, "T_work_offsets")[count_expr]));
        }
    } else {
        llir::lExpr start_expr;
        llir::lExpr end_expr;
        auto next_sparse_level = tensor_type.format.get_next_sparse_level(last_sparse_tensor_level_before_target_loop);
        // target_dim is a level inside this tensor
        if (tensor_level_exists(target_loop_num) && !is_target_loop_value_fixed) {
            // level(target_loop_num) == last_tensor_level_before_target_loop
            internal_assert(tensor_level_to_loop_num(last_tensor_level_before_target_loop) == target_loop_num);
            // last_tensor_level_before_target_loop is also sparse
            if (last_tensor_level_before_target_loop == last_sparse_tensor_level_before_target_loop) {
                start_expr = get_level_indexing_expression(last_sparse_tensor_level_before_target_loop, false, pos_vars_start);
                if(is_compressed(last_sparse_tensor_level_before_target_loop) && last_tensor_level_before_target_loop > BEFORE_FIRST_LEVEL+1)
                    start_expr = this->get_offsets_field(last_sparse_tensor_level_before_target_loop)[start_expr];
            } else {
                start_expr = get_level_indexing_expression(
                        min(next_sparse_level, last_tensor_level_in_partial_loop+1),false, pos_vars_start
                    );
            }
        } else {
            start_expr = get_level_indexing_expression(
                        min(next_sparse_level, last_tensor_level_in_partial_loop+1),false, pos_vars_start
                    );
        }
        end_expr = get_level_indexing_expression(
                        min(next_sparse_level, last_tensor_level_in_partial_loop+1),true, pos_vars_end
                    );

        stmts.emplace_back(llir::Declare::make(
            index_t, "end",
            next_sparse_level < last_tensor_level_in_partial_loop + 1 && is_compressed(next_sparse_level)
                ? this->get_offsets_field(next_sparse_level)[end_expr]
                : end_expr));

        stmts.emplace_back(llir::Declare::make(
            index_t, "start",
            next_sparse_level < last_tensor_level_in_partial_loop + 1 && is_compressed(next_sparse_level)
                ? this->get_offsets_field(next_sparse_level)[start_expr]
                : start_expr));
        
        auto end_var = llir::lVar::make(index_t, "end");
        auto start_var = llir::lVar::make(index_t, "start");

        while (next_sparse_level <= last_tensor_level_in_partial_loop) {
            auto prev_sparse_level = next_sparse_level;
            next_sparse_level =
                tensor_type.format.get_next_sparse_level(next_sparse_level);

            pos_vars_start[prev_sparse_level] = start_var;
            auto start_expr = get_level_indexing_expression(
                    min(next_sparse_level, last_tensor_level_in_partial_loop+1),
                    false, pos_vars_start
                );

            pos_vars_end[prev_sparse_level] = end_var;
            end_expr = get_level_indexing_expression(
                    min(next_sparse_level, last_tensor_level_in_partial_loop+1),
                    false, pos_vars_end
                );

            stmts.emplace_back(llir::Store::make(
                end_var,
                next_sparse_level < last_tensor_level_in_partial_loop + 1 && is_compressed(next_sparse_level)
                    ? this->get_offsets_field(next_sparse_level)[end_expr]
                    : end_expr));

            stmts.emplace_back(llir::Store::make(
                start_var,
                next_sparse_level < last_tensor_level_in_partial_loop + 1 && is_compressed(next_sparse_level)
                    ? this->get_offsets_field(next_sparse_level)[start_expr]
                    : start_expr));
        }
        if(!is_result_tensor)
            stmts.emplace_back(llir::Declare::make(
                index_t, "count",
                end_var -  start_var
                ));
        else {
            stmts.emplace_back(llir::Declare::make(
                index_t, "count",
                llir::lVar::make(index_t, "T_work_offsets")[end_var] -
                llir::lVar::make(index_t, "T_work_offsets")[start_var]
                ));
        }
    }

    llir::lExpr work_expr = llir::lVar::make(index_t, "count");

    // if target_dim is a broad cast level need to multiple by the arg of
    // target_dim
    if (!tensor_level_exists(target_loop_num) && !is_target_loop_value_fixed) {
        work_expr = work_expr * (llir::lVar::make(index_t, loop_index(target_loop_num).str())+1);
    }

    // Multiply by the required broadcast levels
    // broadcast factor is not required for broadcast levels that come after the
    // last level in the tensor
    for (auto i = target_loop_num + 1; i <= last_level_loop_num_in_partial_loop; ++i) {
        if (!tensor_level_exists(i)) {
            work_expr = work_expr * llir::lVar::make(index_t, "bc_size_" + loop_index(i).str());
        }
    }

    stmts.emplace_back(llir::Return::make(work_expr));

    llir::lStmt body = llir::Sequence::make(std::move(stmts));

    return llir::Function::make(std::move(generics), std::move(attributes),
                                std::move(args), std::move(ret_type), name,
                                std::move(body));
}



// lower the function get_tuple_<tensor_name>_<merged_index>
// This function is a helper used to generate a tuple for the
// merged index that is present in the tensor
llir::lStmt TensorLowerer::lower_func_read_tuple_for_merged_index() const {

    std::vector<llir::lStmt> stmts;

    for (auto i = BEFORE_FIRST_LEVEL+1; i < tensor_type.format.get_end_level(); ++i) {
        if(is_merged_level(i)) {
            auto merged_idx = tensor_level_index(i);
            std::vector<std::string> generics = {"index_t"};
            std::vector<llir::Function::Attribute> attributes = {llir::Function::device, llir::Function::inline_};
        
            std::vector<llir::Function::Argument> args;
            llir::lType ret_type = llir::Tuple_t::make(std::vector<llir::lType>(merged_idx.indices.size(), index_t));;
            std::string name = get_func_name_for_read_tuple_for_merged_index(merged_idx);
            llir::lStmt body;

            args.emplace_back(
                llir::Function::Argument{
                    .mutating = false,
                    .type = llir::Generic_t::make(get_struct_name() +
                                                "<index_t, value_t>"),
                    .name = tensor_name
                });

            args.emplace_back(
                llir::Function::Argument{
                    .mutating = false,
                    .type = index_t,
                    .name = get_iterator_suffix(merged_idx)
                });

            std::vector<llir::lExpr> make_tuple_args;
            for (auto idx : merged_idx.indices) {
                make_tuple_args.emplace_back(get_indices_field(TensorIndex(idx))[
                    llir::lVar::make(index_t, get_iterator_suffix(merged_idx))
                ]);
            }
            body = llir::Return::make(
                llir::lFunctionCall::make(
                    "cuda::std::make_tuple",
                    std::vector<llir::lExpr>(make_tuple_args)
                )
            );

            stmts.emplace_back(
                llir::Function::make(
                    std::move(generics),
                    std::move(attributes),
                    std::move(args),
                    std::move(ret_type),
                    std::move(name),
                    std::move(body)
                )
            );
        }
    }

    return  stmts.size() > 0 ? llir::Sequence::make(std::move(stmts)) : llir::lStmt();
}

llir::lStmt TensorLowerer::lower_func_write_tuple_for_merged_index() const {

    std::vector<llir::lStmt> stmts;

    for (auto i = BEFORE_FIRST_LEVEL+1; i < tensor_type.format.get_end_level(); ++i) {
        if(is_merged_level(i)) {
            auto merged_idx = tensor_level_index(i);
            std::vector<std::string> generics = {"index_t"};
            std::vector<llir::Function::Attribute> attributes = {llir::Function::device, llir::Function::inline_};
        
            std::vector<llir::Function::Argument> args;
            llir::lType ret_type = llir::Generic_t::make("void");
            std::string name = get_func_name_for_write_tuple_for_merged_index(merged_idx);
            llir::lStmt body;

            args.emplace_back(
                llir::Function::Argument{
                    .mutating = false,
                    .type = llir::Generic_t::make(get_struct_name() +
                                                "<index_t, value_t>"),
                    .name = tensor_name
                });

            args.emplace_back(
                llir::Function::Argument{
                    .mutating = false,
                    .type = index_t,
                    .name = get_iterator_suffix(merged_idx)
                });

            auto tuple_type = llir::Tuple_t::make(std::vector<llir::lType>(merged_idx.indices.size(), index_t));
            args.emplace_back(
                llir::Function::Argument{
                    .mutating = false,
                    .type = tuple_type,
                    .name = "tuple"
                });

            std::vector<llir::lStmt> write_stmts;
            for (size_t idx_num=0; idx_num<merged_idx.indices.size(); ++idx_num) {
                std::string idx = merged_idx.indices[idx_num];
                write_stmts.emplace_back(
                    llir::Store::make(
                        get_indices_field(TensorIndex(idx))[
                            llir::lVar::make(index_t, get_iterator_suffix(merged_idx))
                        ],
                        llir::lFunctionCall::make(
                            "cuda::std::get<" + std::to_string(idx_num) + ">",
                            std::vector<llir::lExpr>{
                                llir::lVar::make(tuple_type, "tuple")
                            }
                        )
                    )
                );
            }
            body = llir::Sequence::make(std::move(write_stmts));

            stmts.emplace_back(
                llir::Function::make(
                    std::move(generics),
                    std::move(attributes),
                    std::move(args),
                    std::move(ret_type),
                    std::move(name),
                    std::move(body)
                )
            );
        }
    }

    return stmts.size() > 0 ? llir::Sequence::make(std::move(stmts)) : llir::lStmt();
}

} // namespace backend
} // namespace nacho