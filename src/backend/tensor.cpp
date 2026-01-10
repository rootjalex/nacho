#include "backend/tensor.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Format.h"
#include <algorithm>
#include <vector>
#include "Error.h"

namespace nacho {
namespace backend {

    llir::lType TensorLowerer::lower_tensor_struct_definition() {
        llir::lType index_t = llir::Generic_t::make("index_t");
        llir::lType value_t = llir::Generic_t::make("value_t");
        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<std::pair<std::string, llir::lType>> data_fields;
        data_fields.emplace_back("values", value_t);
        for (int i=tensor_type.format.levels.size()-1; i>=0; i--) {
            auto index = tensor_type.format.levels[i].index;

            // only dimensions with sparse format need separate fields for index and offsets
            // TODO: This only handles compressed dimensions as of now
            if(is_sparse_format(tensor_type.format.lvlfmt_of(index))) {
                data_fields.emplace_back(get_indices_field_name(index), llir::Ptr_t::make(index_t));
                data_fields.emplace_back(get_length_field_name(index), index_t);
                // offsets field is not required if the outermost dimension is sparse
                if(i!=0)
                    data_fields.emplace_back(
                        get_offsets_field_name(tensor_type.format.levels[i-1].index), 
                        llir::Ptr_t::make(index_t));
            }
        }

        std::vector<std::pair<std::string, llir::lType>> size_fields;
        for (int i=0; i<tensor_type.format.levels.size(); i++) {
            auto index = tensor_type.format.levels[i].index;
            size_fields.emplace_back(get_size_field_name(index), index_t);
        }

        std::vector<std::pair<std::string, llir::lType>> fields;
        fields.insert(fields.end(), size_fields.begin(), size_fields.end());
        fields.insert(fields.end(), data_fields.rbegin(), data_fields.rend());

        return llir::Struct_t::make(get_struct_name(), std::move(fields), std::move(generics));
    }
    
    

    // Takes a sparse dimension position as input and calculates the expression for offseting
    // into next sparse dimension. Returns a tuple of expression and the next sparse dimension level.
    //
    // Also optionally takes num_dense_dims which are to be considered to calculate the offseting
    // expression. num_dense_dims should be <= the actual number of dense dimensions between
    // sparse_dim and the next sparse dimension. 
    //
    // Eg - consider a tensor with level order i,j,k,l where i,l are sparse and j,k are dense
    // if args are sparse_dim_level = 0  dense_dims = 0 
    //    returned expression is  (i_p+1)*A.dim_j_size*A.dim_k_size
    // if args are sparse_dim_level = 0, dense_dims = 1  
    //    returned expression is (i_p)*A.dim_j_size*A.dim_k_size + (j+1) * A.dim_k_size
    // if args are sparse_dim_level = 0, dense_dims = 2
    //.   retutrn expression is (i_p)*A.dim_j_size*A.dim_k_size + (j)*A.dim_k_size + k+1
    std::tuple<llir::lExpr,int> TensorLowerer::get_offset_expression_for_next_sparse(
        int sparse_dim_level, int num_dense_dims) {
            llir::lType index_t = llir::Generic_t::make("index_t");


            int next_sparse_dim_level = sparse_dim_level+1;
            for(; next_sparse_dim_level < tensor_type.format.levels.size(); next_sparse_dim_level++) {
                auto curr_dim = tensor_type.format.levels[next_sparse_dim_level].index;
                if(is_sparse_format(tensor_type.format.lvlfmt_of(curr_dim))) {
                    // found next sparse dimension
                    break;
                } 
            }
            int count_dense_dims = next_sparse_dim_level - sparse_dim_level;
            internal_assert(count_dense_dims > num_dense_dims) << "num_dense_dims exceeds actual number of dense dimensions between sparse dimensions";

            
            llir::lExpr ip_expr = llir::lVar::make(index_t,sparse_dim + "_p");
            if(num_dense_dims == 0) {
                ip_expr = llir::lBinOp::make(llir::lBinOp::Add, ip_expr, llir::lConst::make((int64_t)1));
            }
            for(int i=0;i<count_dense_dims;i++) {
                ip_expr = llir::lBinOp::make(
                    llir::lBinOp::Mul, 
                    ip_expr, 
                    llir::lFieldAccess::make(
                        llir::lVar::make(index_t, tensor_name),
                        llir::lVar::make(
                            llir::Generic_t::make("index_t"), 
                            get_size_field_name(tensor_type.format.levels[sparse_dim_level + 1 + i].index)
                        )
                    )
                );
            }

            for(int i=0;i<num_dense_dims;i++) {

                llir::lExpr dense_expr = llir::lVar::make(index_t, tensor_type.format.levels[sparse_dim_level + 1 + i].index);
                if(i == num_dense_dims - 1) {
                    dense_expr = llir::lBinOp::make(
                        llir::lBinOp::Add, 
                        dense_expr, 
                        llir::lConst::make((int64_t)1)
                    );
                }
                for(int j=i+1;j<num_dense_dims;j++) {
                    dense_expr = llir::lBinOp::make(
                        llir::lBinOp::Mul, 
                        dense_expr, 
                        llir::lFieldAccess::make(
                            llir::lVar::make(index_t, tensor_name),
                            llir::lVar::make(
                                llir::Generic_t::make("index_t"), 
                                get_size_field_name(tensor_type.format.levels[sparse_dim_level + 1 + j].index)
                            )
                        )
                    );
                }

                ip_expr = llir::lBinOp::make(
                    llir::lBinOp::Add,
                    ip_expr,
                    dense_expr
                );
            }

            return std::tuple<llir::lExpr,int>(ip_expr, sparse_dim_level);
    }

    // lower_work_function returns the LLIR work function for the given tensor.
    // target_dim is the target dimension for which work is being calculated.
    // prev_dim_positions are the positions of dimensions less than target_dim.
    // target_dim_position is the position of the target_dim at which work is being calculated.
    // Eg - Consider a DCSR tensor with dims i,k and the loop order is [i,j,k,l] 
    // (the DCSR tensor is broadcaster over j,l)
    // And now we need to generate a work function which can calulate work for i=12, j=12, 0<= k <=54. 0 <= l <= |L|
    // Then the args are loop_order = [i,j,k,l],target_dim = 2 (k)
    // The generated work function for the example will look like following
    //
    llir::lStmt TensorLowerer::lower_work_function(std::vector<std::string> loop_order, int target_dim) {
        llir::lType index_t = llir::Generic_t::make("index_t");
        llir::lType value_t = llir::Generic_t::make("value_t");
        std::vector<std::string> generics = {"index_t", "value_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::device, llir::Function::inline_
        };

        auto order_map = index_order_map(tensor_type.format.levels);

        auto violates_order = [&](const std::vector<std::string> &loop_order,
            OrderMap &level_order
        ) {
            // Find any pair (x, y) s.t. x is before y in level_order but x is after y in loop_order.
            for (size_t i = 0; i + 1 < loop_order.size(); ++i) {
                const auto &lhs = loop_order[i];
                for (size_t j = i + 1; j < loop_order.size(); ++j) {
                    const auto &rhs = loop_order[j];

                    // Only meaningful if both indices also appear in L1
                    auto x = level_order.find(lhs);
                    auto y = level_order.find(rhs);
                    if (x == level_order.end() || y == level_order.end())
                        continue;

                    // x appears after y in L1 ordering
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
            .mutating = false, .type = llir::Ptr_t::make(llir::Generic_t::make(get_struct_name()+"<index_t, value_t>")), .name = tensor_name
        });

        // last_sparse_level will store the last sparse level before target_dim
        int last_sparse_level = -1;
        // last_level will store the last level before target_dim
        int last_level = -1;
        for(int i=0; i<=target_dim; i++)
        {
            // sparse dimensions are iterated by positions while dense are iterated by coordinates
            // hence args are named accordingly
            std::string name = loop_order[i];
            // only add arguments for non-broadcasted levels
            if(!tensor_type.format.is_bc_lvl(loop_order[i]))
            {
                if(is_sparse_format(tensor_type.format.lvlfmt_of(loop_order[i])))
                {
                    // sparse dimensions are iterated by positions while dense are iterated by coordinates
                    name += "_p";
                    last_sparse_level = order_map[loop_order[i]];
                }
                last_level = order_map[loop_order[i]];
            }
            

            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = index_t, .name = name
            });
        }

        // If target_dim is a broadcasted dimension, add it as an argument. As work depends
        // on value of the target_dim
        if(tensor_type.format.is_bc_lvl(loop_order[target_dim])) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = index_t, .name = loop_order[target_dim]
            });
        }
        
        
        // broadcast Sizes of all broadcast dimensions coming after target_dim need to be added as arguments.
        // As work calculation will require multiplying with bc_sizes at end.
        for(int i=target_dim+1; i < loop_order.size(); i++) {
            if(tensor_type.format.is_bc_lvl(loop_order[i])) {
                args.emplace_back(llir::Function::Argument{
                    .mutating = false, .type = index_t, .name = "bc_size_"+loop_order[i]
                });
            }
        }

        llir::lType ret_type = llir::Generic_t::make("index_t");
        std::string name = "work_function_tensor_"+tensor_name+"_dim_"+loop_order[target_dim];

        std::vector<llir::lStmt> stmts;


        // Body of the Work Function has to traverse down the sparse dimensions from the last sparse dimension
        //  to calculate the number of non zeros that are being iterated on
        stmts.emplace_back(
            llir::Declare::make(
                index_t, 
                "offset", 
                last_sparse_level == -1 ? 
                    llir::lConst::make((int64_t)0) : llir::lVar::make(index_t, tensor_type.format.levels[last_sparse_level].index+"_p")
            )
        );

        int num_dense_dims = last_level - last_sparse_level;
        while (last_sparse_level < tensor_type.format.levels.size()) {
            auto [offset_expr, next_sparse_level] = get_offset_expression_for_next_sparse(last_sparse_level, num_dense_dims);
            num_dense_dims = 0;
            
            stmts.emplace_back(
                llir::Declare::make(
                    index_t, 
                    next_sparse_level < tensor_type.format.levels.size() ? 
                        tensor_type.format.levels[next_sparse_level].index+"_p" : "count",
                    llir::lArrayAccess::make(
                        llir::lFieldAccess::make(
                            llir::lVar::make(
                                llir::Ptr_t::make(llir::Generic_t::make(get_struct_name())), 
                                tensor_name
                            ),
                            llir::lVar::make(
                                llir::Ptr_t::make(index_t), 
                                get_offsets_field_name(tensor_type.format.levels[last_sparse_level].index)
                            )
                        ),
                        offset_expr
                    )
                )
            );

            last_sparse_level = next_sparse_level;
        }

        llir::lExpr work_expr = llir::lVar::make(index_t, "count");

        // if target_dim is a broad cast level need to multiple by the arg of target_dim
        if(tensor_type.format.is_bc_lvl(loop_order[target_dim])) {
            work_expr = llir::lBinOp::make(
                llir::lBinOp::Mul,
                work_expr,
                llir::lVar::make(index_t,loop_order[target_dim])
            );
        }

        // Multiply by the required broadcast levels
        for(int i=target_dim+1; i < loop_order.size(); i++) {
            if(tensor_type.format.is_bc_lvl(loop_order[i])) {
                work_expr = llir::lBinOp::make(
                    llir::lBinOp::Mul,
                    work_expr,
                    llir::lVar::make(index_t, "bc_size_"+loop_order[i])
                );
            }
        }

        stmts.emplace_back(
            llir::Return::make(work_expr)
        );

        llir::lStmt body = llir::Sequence::make(std::move(stmts));

        return llir::Function::make(std::move(generics),std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body));
    }

}