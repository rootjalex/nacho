#include "backend/tensor.h"
#include "Error.h"
#include "Format.h"
#include "llir/Function.h"
#include "llir/LLIR.h"
#include <algorithm>
#include <vector>

namespace nacho {
namespace backend {

llir::lType TensorLowerer::lower_tensor_struct_definition() const {
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    std::vector<std::string> generics = {"index_t", "value_t"};
    std::vector<std::pair<std::string, llir::lType>> data_fields;
    data_fields.emplace_back("nnz", index_t);
    data_fields.emplace_back(get_values_field_name(), llir::Ptr_t::make(value_t));
    for (int i = tensor_type.format.levels.size() - 1; i >= 0; i--) {
        auto index = tensor_type.format.levels[i].index;

        // only dimensions with sparse format need separate fields for index and
        // offsets
        // TODO: This only handles compressed dimensions as of now
        if (is_sparse_format(tensor_type.format.lvlfmt_of(index))) {
            data_fields.emplace_back(get_indices_field_name(index),
                                     llir::Ptr_t::make(index_t));
            data_fields.emplace_back(get_length_field_name(index), index_t);
            // offsets field is not required if the outermost dimension is
            // sparse
            if (i != 0)
                data_fields.emplace_back(get_offsets_field_name(index),
                                         llir::Ptr_t::make(index_t));
        }
    }

    std::vector<std::pair<std::string, llir::lType>> size_fields;
    for (int i = 0; i < tensor_type.format.levels.size(); i++) {
        auto index = tensor_type.format.levels[i].index;
        size_fields.emplace_back(get_size_field_name(index), index_t);
    }

    std::vector<std::pair<std::string, llir::lType>> fields;
    fields.insert(fields.end(), size_fields.begin(), size_fields.end());
    fields.insert(fields.end(), data_fields.rbegin(), data_fields.rend());

    return llir::Struct_t::make(get_struct_name(), std::move(fields),
                                std::move(generics));
}

// Takes a sparse dimension position as input and calculates the expression for
// offseting into next sparse dimension. Returns a tuple of expression and the
// next sparse dimension level.
//
// Also optionally takes num_dense_dims which are to be considered to calculate
// the offseting expression. num_dense_dims should be <= the actual number of
// dense dimensions between sparse_dim and the next sparse dimension.
//
// Eg - consider a tensor with level order i,j,k,l where i,l are sparse and j,k
// are dense if args are sparse_dim_level = 0  dense_dims = 0
//    returned expression is  (i_p+1)*A.dim_j_size*A.dim_k_size
// if args are sparse_dim_level = 0, dense_dims = 1
//    returned expression is (i_p)*A.dim_j_size*A.dim_k_size + (j+1) *
//    A.dim_k_size
// if args are sparse_dim_level = 0, dense_dims = 2
//.   retutrn expression is (i_p)*A.dim_j_size*A.dim_k_size + (j)*A.dim_k_size +
// k+1
llir::lExpr TensorLowerer::get_offset_expression_for_next_sparse(
    int dim_level_start, int dim_level_end, bool upper_bound,
    bool use_dim_vars, std::vector<llir::lExpr> dim_vars) {
    static llir::lType index_t = llir::Generic_t::make("index_t");
    // std::cout<<"dim_level_start: "<<dim_level_start<<", dim_level_end:
    // "<<dim_level_end<<std::endl;
    if (dim_level_end < 0) {
        return llir::lConst::make((int64_t)0);
    }

    if (dim_level_start == -1) {
        dim_level_start = 0;
    }

    int next_sparse_dim_level =
        tensor_type.format.get_next_sparse_level(dim_level_start);

    internal_assert(next_sparse_dim_level > dim_level_end);

    bool is_start_dim_sparse = is_sparse_format(tensor_type.format.lvlfmt_of(
        tensor_type.format.levels[dim_level_start].index));

    llir::lExpr ip_expr = llir::lConst::make((int64_t)0);
    if (is_start_dim_sparse) {
        ip_expr =
            use_dim_vars
                ? dim_vars[dim_level_start]
                : llir::lVar::make(
                      index_t,
                      tensor_type.format.levels[dim_level_start].index + "_p");
        if (upper_bound && dim_level_start == dim_level_end) {
            ip_expr = ip_expr + 1;
        }
        for (int i = dim_level_start + 1; i <= dim_level_end; i++) {
            ip_expr = ip_expr * this->get_size_field(i);
        }
    }

    for (int i = dim_level_start + (is_start_dim_sparse ? 1 : 0);
         i <= dim_level_end; i++) {

        llir::lExpr dense_expr =
            use_dim_vars
                ? dim_vars[i]
                : llir::lVar::make(index_t, tensor_type.format.levels[i].index);
        if (i == dim_level_end && upper_bound) {
            dense_expr = dense_expr + 1;
        }
        for (int j = i + 1; j <= dim_level_end; j++) {
            dense_expr = dense_expr * this->get_size_field(j);
        }

        ip_expr = ip_expr + dense_expr;
    }

    return ip_expr;
}

// lower_work_function returns the LLIR work function for the given tensor.
// target_dim is the target dimension for which work is being calculated.
// prev_dim_positions are the positions of dimensions less than target_dim.
// target_dim_position is the position of the target_dim at which work is being
// calculated. Eg - Consider a DCSR tensor with dims i,k and the loop order is
// [i,j,k,l] (the DCSR tensor is broadcaster over j,l) And now we need to
// generate a work function which can calulate work for i=12, j=12, 0<= k <=54.
// 0 <= l <= |L| Then the args are loop_order = [i,j,k,l],target_dim = 2 (k) The
// generated work function for the example will look like following
//
llir::lStmt
TensorLowerer::lower_work_function(std::vector<std::string> loop_order,
                                   int target_dim) {

    internal_assert(target_dim < loop_order.size()) << "Target dimension has to be less than loop order size";

    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    std::vector<std::string> generics = {"index_t", "value_t"};

    std::vector<llir::Function::Attribute> attributes = {
        llir::Function::device, llir::Function::inline_};

    auto order_map = index_order_map(tensor_type.format.levels);

    // Check what is the last level in loop order that is present inside this tensor 
    int last_level_loop_place = -1;
    for(int i=loop_order.size()-1; i>=0; i--) {
        if(tensor_type.format.level_exists(loop_order[i])) {
            last_level_loop_place = i;
            break;
        }
    }

    // checks the loop order is consistent with the tensor format
    auto violates_order = [&](const std::vector<std::string> &loop_order,
                              OrderMap &level_order) {
        // Find any pair (x, y) s.t. x is before y in level_order but x is after
        // y in loop_order.
        for (size_t i = 0; i + 1 < loop_order.size(); ++i) {
            const auto &lhs = loop_order[i];
            for (size_t j = i + 1; j < loop_order.size(); ++j) {
                const auto &rhs = loop_order[j];

                auto x = level_order.find(lhs);
                auto y = level_order.find(rhs);
                if (x == level_order.end() || y == level_order.end())
                    continue;

                if (x->second > y->second)
                    return true;
            }
        }
        return false;
    };

    if (violates_order(loop_order, order_map))
        internal_assert(false) << "Incompatible loop order";

    std::vector<llir::Function::Argument> args;
    args.emplace_back(llir::Function::Argument{
        .mutating = false,
        .type = llir::Generic_t::make(get_struct_name() + "<index_t, value_t>"),
        .name = tensor_name});

    // last_sparse_level will store the last sparse level before target_dim
    int last_sparse_level = -1;
    // last_level will store the last level before target_dim
    int last_level = -1;
    for (int i = 0; i <= target_dim; i++) {

        // only add arguments for non-broadcasted levels
        if (tensor_type.format.level_exists(loop_order[i])) {
            // sparse dimensions are iterated by positions while dense are
            // iterated by coordinates hence args are named accordingly
            std::string name = loop_order[i];
            if (is_sparse_format(tensor_type.format.lvlfmt_of(loop_order[i]))) {
                // sparse dimensions are iterated by positions while dense are
                // iterated by coordinates
                name += "_p";
                last_sparse_level = order_map[loop_order[i]];
            }
            last_level = order_map[loop_order[i]];
            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = index_t, .name = name});
        }
    }

    // If target_dim is a broadcasted dimension, add it as an argument. As work
    // depends on value of the target_dim
    if (!tensor_type.format.level_exists(loop_order[target_dim])) {
        args.emplace_back(
            llir::Function::Argument{.mutating = false,
                                     .type = index_t,
                                     .name = loop_order[target_dim]});
    }

    // broadcast Sizes of all broadcast dimensions coming after target_dim need
    // to be added as arguments. As work calculation will require multiplying
    // with these bc_sizes at end.
    for (int i = target_dim + 1; i < loop_order.size(); i++) {
        if (!tensor_type.format.level_exists(loop_order[i])) {
            args.emplace_back(
                llir::Function::Argument{.mutating = false,
                                         .type = index_t,
                                         .name = "bc_size_" + loop_order[i]});
        }
    }

    llir::lType ret_type = llir::Generic_t::make("index_t");
    std::string name = get_work_function_name(loop_order, loop_order[target_dim]);

    std::vector<llir::lStmt> stmts;

    // if target_dim is a loop that comes after last level of the tensor in the
    // loop order Then iteration work for this tensor is 0 (as we are not
    // iterating on any non-zeros of the tensor)
    if (last_level_loop_place < target_dim) {
        stmts.emplace_back(llir::Return::make(llir::lConst::make((int64_t)0)));
        llir::lStmt body = llir::Sequence::make(std::move(stmts));
        return llir::Function::make(std::move(generics), std::move(attributes),
                                    std::move(args), std::move(ret_type), name,
                                    std::move(body));
    }

    // Body of the Work Function has to traverse down the sparse dimensions from
    // the last sparse dimension
    //  to calculate the number of non zeros that are being iterated on
    // std::cout<<"last_level: "<<last_level<<", last_sparse_level:
    // "<<last_sparse_level<<std::endl;
    llir::lExpr count_expr;
    // We need to iterate on all the nonzeros if last_level == -1
    if (last_level == -1) {
        // if there is no sparse dimension in the tensor in the given loop order(i.e all dense) just
        // return 1 (The 1 will get multiplied later by sizes of all dense
        // dimensions in the next for loop below)
        int last_sparse_dim = tensor_type.format.get_prev_sparse_level(order_map[loop_order[last_level_loop_place]]+1);
        if (last_sparse_dim == -1) {
            count_expr = llir::lConst::make((int64_t)1);
        } else {
            // We iterate on the whole last sparse dim that comes before last_level_loop_place
            count_expr = this->get_length_field(last_sparse_dim);
        }
        for (int i = last_sparse_dim + 1;
             i <= order_map[loop_order[last_level_loop_place]]; i++) {
            count_expr = count_expr * this->get_size_field(i);
        }
        stmts.emplace_back(llir::Declare::make(index_t, "count", count_expr));
    } else {
        llir::lExpr start_expr;
        llir::lExpr end_expr;

        // target_dim is a level inside this tensor
        if (tensor_type.format.level_exists(loop_order[target_dim])) {
            // loop_order[target_dim] == last_level

            // last_level is also sparse
            if (last_level == last_sparse_level) {
                start_expr = last_level - 1 >= 0
                                 ? get_offset_expression_for_next_sparse(
                                       tensor_type.format.get_prev_sparse_level(
                                           last_level),
                                       last_level - 1, false)
                                 : llir::lConst::make((int64_t)0);

                if (last_level - 1 >= 0)
                    start_expr = this->get_offsets_field(last_sparse_level)[start_expr];
            } else {
                // std::cout<<" last_level "<<last_level<<" last_sparse_level
                // "<<last_sparse_level<<std::endl;
                start_expr =
                    last_level - 1 >= 0
                        ? get_offset_expression_for_next_sparse(
                                  last_sparse_level, last_level - 1, false) * this->get_size_field(last_level)
                        : llir::lConst::make((int64_t)0);
            }
        } else {
            start_expr = get_offset_expression_for_next_sparse(
                last_sparse_level, last_level, false);
        }

        end_expr = get_offset_expression_for_next_sparse(last_sparse_level,
                                                         last_level, true);

        for (int i = last_level + 1;
             i < tensor_type.format.get_next_sparse_level(last_sparse_level);
             i++) {
            start_expr = start_expr * this->get_size_field(i);
            end_expr = end_expr * this->get_size_field(i);
        }

        int next_sparse_level =
            tensor_type.format.get_next_sparse_level(last_sparse_level);
        // std::cout<<" last_level "<<last_level<<" last_sparse_level
        // "<<last_sparse_level<<std::endl;
        stmts.emplace_back(llir::Declare::make(
            index_t,
            next_sparse_level < tensor_type.format.levels.size()
                ? tensor_type.format.levels[next_sparse_level].index + "_p_end"
                : "count_end",
            next_sparse_level < tensor_type.format.levels.size()
                ? this->get_offsets_field(next_sparse_level)[end_expr]
                : end_expr));

        stmts.emplace_back(llir::Declare::make(
            index_t,
            next_sparse_level < tensor_type.format.levels.size()
                ? tensor_type.format.levels[next_sparse_level].index +
                      "_p_start"
                : "count_start",
            next_sparse_level < tensor_type.format.levels.size()
                ? this->get_offsets_field(next_sparse_level)[start_expr]
                : start_expr));

        while (next_sparse_level < tensor_type.format.levels.size()) {
            int last_sparse_level = next_sparse_level;
            next_sparse_level =
                tensor_type.format.get_next_sparse_level(next_sparse_level);

            // end_expr needs upper_bound so wee need to add `1`.
            end_expr = llir::lVar::make(
                index_t, tensor_type.format.levels[last_sparse_level].index +
                            "_p_end");
            start_expr = llir::lVar::make(
                index_t, tensor_type.format.levels[last_sparse_level].index +
                             "_p_start");

            for (int i = last_sparse_level + 1; i < next_sparse_level; i++) {
                start_expr = start_expr * this->get_size_field(i);
                end_expr = end_expr * this->get_size_field(i);
            }

            stmts.emplace_back(llir::Declare::make(
                index_t,
                next_sparse_level < tensor_type.format.levels.size()
                    ? tensor_type.format.levels[next_sparse_level].index +
                          "_p_end"
                    : "count_end",
                next_sparse_level < tensor_type.format.levels.size()
                    ? this->get_offsets_field(next_sparse_level)[end_expr]
                    : end_expr));

            stmts.emplace_back(llir::Declare::make(
                index_t,
                next_sparse_level < tensor_type.format.levels.size()
                    ? tensor_type.format.levels[next_sparse_level].index +
                          "_p_start"
                    : "count_start",
                next_sparse_level < tensor_type.format.levels.size()
                    ? this->get_offsets_field(next_sparse_level)[start_expr]
                    : start_expr));
        }
        stmts.emplace_back(llir::Declare::make(
            index_t, "count",
            llir::lBinOp::make(llir::lBinOp::Sub,
                               llir::lVar::make(index_t, "count_end"),
                               llir::lVar::make(index_t, "count_start"))

                ));
    }

    llir::lExpr work_expr = llir::lVar::make(index_t, "count");

    // if target_dim is a broad cast level need to multiple by the arg of
    // target_dim
    if (!tensor_type.format.level_exists(loop_order[target_dim])) {
        work_expr = work_expr * llir::lVar::make(index_t, loop_order[target_dim]);
    }

    // Multiply by the required broadcast levels
    // broadcast factor is not required for broacast levels that come after the
    // last level in the tensor
    for (int i = target_dim + 1; i < last_level_loop_place; i++) {
        if (!tensor_type.format.level_exists(loop_order[i])) {
            work_expr = work_expr * llir::lVar::make(index_t, "bc_size_" + loop_order[i]);
        }
    }

    stmts.emplace_back(llir::Return::make(work_expr));

    llir::lStmt body = llir::Sequence::make(std::move(stmts));

    return llir::Function::make(std::move(generics), std::move(attributes),
                                std::move(args), std::move(ret_type), name,
                                std::move(body));
}

// Lower the tensor index definition to LLIR.
// This struct defines an object to specify the values of different levels to
// index into the tensor.
llir::lType TensorLowerer::lower_tensor_index_definition() {
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    std::vector<std::string> generics = {"index_t", "value_t"};

    std::vector<std::pair<std::string, llir::lType>> fields;

    for (const auto &level : tensor_type.format.levels) {
        std::string field_name = level.index;

        if (is_sparse_format(level.format)) {
            field_name = field_name + "_p";
        }

        fields.emplace_back(field_name, index_t);
    }

    return llir::Struct_t::make(get_index_struct_name(), std::move(fields),
                                std::move(generics));
}

} // namespace backend
} // namespace nacho