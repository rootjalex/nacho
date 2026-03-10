#pragma once
#include "IRFwdDecl.h"
#include "CIN.h"
#include "Type.h"
#include "llir/LLIR.h"
#include <numeric>

namespace nacho {
    namespace backend {

    // TensorLowerer is responsible for lowering Tensor specific code.
    // This includes generating the tensor struct definition and other
    // tensor-access or manipulation code.
    struct TensorLowerer {
        std::string tensor_name;
        TensorType tensor_type;
        bool is_result_tensor;
        TensorLowerer() = default;
        TensorLowerer(std::string tensor_name, TensorType tensor_type, bool is_result_tensor=false)
            : tensor_name(std::move(tensor_name)), tensor_type(std::move(tensor_type)), is_result_tensor(is_result_tensor) {}
        llir::lType index_t = llir::Generic_t::make("index_t");
        inline std::string get_struct_name() const {
            return tensor_name + "_tensor_format";
        }

        inline std::string get_index_struct_name() const {
            return tensor_name + "_tensor_index";
        }

        inline std::string get_values_field_name() const {
            return "values";
        }

        inline llir::lExpr get_values_field() const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_values_field_name()
            );
        }
        
        inline std::string
        get_indices_field_name(const std::string &index) const {
            return "dim_" + index + "_indices";
        }

        inline llir::lExpr get_indices_field(const std::string &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_indices_field_name(index)
            );
        }

        inline std::string get_indices_field_name(const int level) const {
            return get_indices_field_name(level_name(level));
        }

        inline llir::lExpr get_indices_field(const int level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_indices_field_name(level)
            );
        }

        inline std::string
        get_length_field_name(const std::string &index) const {
            return "dim_" + index + "_length";
        }

        inline std::string get_length_field_name(const int level) const {
            return get_length_field_name(level_name(level));
        }
    
        inline llir::lExpr get_length_field(const std::string &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_length_field_name(index)
            );
        }

        inline llir::lExpr get_length_field(const int level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_length_field_name(level)
            );
        }

        inline std::string
        get_offsets_field_name(const std::string &index) const {
            return "dim_" + index + "_offsets";
        }

        inline llir::lExpr get_offsets_field(const std::string &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_offsets_field_name(index)
            );
        }

        inline std::string get_offsets_field_name(const int level) const {
            return get_offsets_field_name(level_name(level));
        }

        inline llir::lExpr get_offsets_field(const int level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_offsets_field_name(level)
            );
        }

        inline std::string get_size_field_name(const std::string &index) const {
            return "dim_" + index + "_size";
        }

        inline llir::lExpr get_size_field(const std::string &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_size_field_name(index)
            );
        }

        inline std::string get_size_field_name(const int level) const {
            return get_size_field_name(level_name(level));
        }

        inline llir::lExpr get_size_field(const int level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_size_field_name(level)
            );
        }

        inline std::string
        get_work_function_name(std::string prefix_string, const std::string &index) const {
            return "work_"+prefix_string+"_" + tensor_name + "_dim_" + index;
        }

        inline std::string get_type_suffix(const int level) const {
            // TODO: standardize the "_p" stuff somewhere.
            return tensor_type.format.levels[level].format ==
                           LevelFormat::Compressed
                       ? "_p"
                       : "";
        }

        inline std::string get_iterator_suffix(const int level) const {
            if(is_sparse(level)) {
                return tensor_name + "_" + level_name(level) + "_p";
            } else {
                return level_name(level);
            }
        }

        inline std::string get_iterator_suffix(std::string idx) const {
            if(is_sparse(idx)) {
                return tensor_name + "_" + idx + "_p";
            } else {
                return idx;
            }
        }

        llir::lType lower_tensor_struct_definition() const;

        // lower_work_function returns the LLIR work function for the given
        // tensor. target_dim is the target dimension for which work is being
        // calculated. prev_dim_positions are the positions of dimensions less
        // than target_dim. target_dim_position is the position of the
        // target_dim at which work is being calculated. Eg - Consider a
        // 3Dtensor with dims i,j,k and the loop order is [i,l,j,k] (the 3D
        // tensor is broadcaster over l which is the 2nd loop) And now we need
        // to calculate work for i=12, l=12, 0<= j <=54. 0 <= k <= |K| Then the
        // args are loop_order = [i,j,k,l],target_dim = 2, prev_dim_positions =
        // [12, 32], target_dim_position = 54
        llir::lStmt lower_work_function(std::vector<std::string> loop_order,
                                        int target_dim, bool is_target_dim_value_fixed = false);


        llir::lExpr get_offset_expression_for_next_sparse(
            int dim_level_start, int dim_level_end, bool upper_bound,
            bool use_dim_vars = false, std::vector<llir::lExpr> dim_vars = {});

        // Lower the tensor index definition to LLIR.
        // This struct defines an object to specify the values of different
        // levels to index into the tensor.
        llir::lType lower_tensor_index_definition();

        inline llir::lType get_lType_from_dType(dType dtype) {
            switch (dtype) {
            case dType::Float32:
                return llir::Float_t::make(32);
            case dType::Float64:
                return llir::Float_t::make(64);
            case dType::TemplateT:
                return llir::Generic_t::make("value_t");
            }
        }

        inline bool is_sparse(std::string index) const {
            return is_sparse_format(tensor_type.format.lvlfmt_of(index));
        }

        inline bool is_sparse(const int level) const {
            return is_sparse_format(tensor_type.format.levels[level].format);
        }

        inline std::string level_name(const int level) const {
            return tensor_type.format.levels[level].index;
        }

        inline std::string get_iter_name(const std::string &forall_index) const {
            return "iter_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_stop_name(const std::string &forall_index) const {
            return "stop_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_start_name(const std::string &forall_index) const {
            return "start_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_end_name(const std::string &forall_index) const {
            return "end_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_iter_name(const int level) const {
            return get_iter_name(level_name(level));
        }

        inline std::string get_stop_name(const int level) const {
            return get_stop_name(level_name(level));
        }

        inline std::string get_start_name(const int level) const {
            return get_start_name(level_name(level));
        }

        inline std::string get_end_name(const int level) const {
            return get_end_name(level_name(level));
        }

        inline llir::lExpr get_iter(const std::string &forall_index) const {
            return llir::lVar::make(index_t, get_iter_name(forall_index));
        }

        inline llir::lExpr get_stop(const std::string &forall_index) const {
            return llir::lVar::make(index_t, get_stop_name(forall_index));
        }

        inline llir::lExpr get_start(const std::string &forall_index) const {
            return llir::lVar::make(index_t, get_start_name(forall_index));
        }

        inline llir::lExpr get_end(const std::string &forall_index) const {
            return llir::lVar::make(index_t, get_end_name(forall_index));
        }

        inline llir::lExpr get_iter(const int level) const {
            return llir::lVar::make(index_t, get_iter_name(level_name(level)));
        }

        inline llir::lExpr get_stop(const int level) const {
            return llir::lVar::make(index_t, get_stop_name(level_name(level)));
        }

        inline llir::lExpr get_start(const int level) const {
            return llir::lVar::make(index_t, get_start_name(level_name(level)));
        }

        inline llir::lExpr get_end(const int level) const {
            return llir::lVar::make(index_t, get_end_name(level_name(level)));
        }

        inline std::string get_coord_name(const int level) const {
            return tensor_name + "_" + level_name(level);
        }


        llir::lExpr get_tensor_expr() const {
            llir::lType tensor_t = lower_tensor_struct_definition();
            return llir::lVar::make(tensor_t, tensor_name);
        }

        llir::lExpr get_dim_size_expr(const int level) const {
            return llir::lFieldAccess::make(get_tensor_expr(),
                                            get_size_field_name(level));
        }

        llir::lExpr get_offsets_expr(const int level) const {
            return llir::lFieldAccess::make(get_tensor_expr(),
                                            get_offsets_field_name(level));
        }

        llir::lExpr get_offset(const int level,
                               const llir::lType &index_t) const {
            llir::lExpr coord = get_iter_var(level, index_t);
            if (is_sparse(level) || level == 0) {
                return coord;
            }
            // If dense, return coordinate + size * recurse
            llir::lExpr size = get_dim_size_expr(level);
            llir::lExpr rec = get_offset(level - 1, index_t);
            return coord + size * rec;
        }

        llir::lExpr get_bound(const int level, const llir::lType &index_t,
                              const bool upper_bound) const {
            if (!is_sparse(level)) {
                if (upper_bound) {
                    return get_dim_size_expr(level);
                } else {
                    return llir::lConst::make(0);
                }
            }
            llir::lExpr offset;
            if (level == 0) {
                offset =
                    upper_bound ? llir::lConst::make(1) : llir::lConst::make(0);
            } else {
                offset = get_offset(level - 1, index_t);
                if (upper_bound) {
                    offset = offset + llir::lConst::make(1);
                }
            }
            llir::lExpr array = get_offsets_expr(level);
            return llir::lArrayAccess::make(std::move(array),
                                            std::move(offset));
        }

        llir::lExpr get_coord_var(const int level,
                                  const llir::lType &index_t) const {
            return llir::lVar::make(index_t, get_coord_name(level));
        }

        llir::lExpr get_iter_var(const int level,
                                 const llir::lType &index_t) const {
            return llir::lVar::make(index_t, get_iter_name(level));
        }

        llir::lExpr get_coord(const int level,
                              const llir::lType &index_t) const {
            llir::lExpr iter = llir::lVar::make(index_t, get_iter_name(level));
            if (!is_sparse(level)) {
                return iter;
            }
            return  get_indices_field(level)[iter];
        }

        llir::lStmt make_eval(const int level,
                              const llir::lType &index_t, bool is_loop_before_prev_intersection) const {
            
            auto val = get_coord(level, index_t);
            if(is_sparse(level) && is_loop_before_prev_intersection) {
                // If this loop is before the previous intersection, then we are going to iterate only over the result tensor index. So if this level is sparse, we need to check if the coordinate is within bounds of the size of the dimension.
                val = llir::lSelect::make(
                    llir::lVar::make(index_t, get_iter_name(level))!=llir::lConst::make(-1),
                    std::move(val),
                    get_size_field(level)
                );
            }
            return llir::Declare::make(index_t, get_coord_name(level),
                                       get_coord(level, index_t));
        }

        llir::lStmt make_inc(const int level,
                             const llir::lType &index_t) const {
            llir::lExpr iter = get_iter_var(level, index_t);
            llir::lExpr crd = get_coord_var(level, index_t);
            // Dense always increments, sparse increments if at the lowest
            // value.
            llir::lExpr value =
                is_sparse(level)
                    ? (crd == llir::lVar::make(index_t, level_name(level)))
                    : llir::lConst::make(1);
            return llir::Accumulate::make(std::move(iter), std::move(value));
        }
    };
}
}