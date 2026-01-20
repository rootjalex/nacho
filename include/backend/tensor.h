#pragma once
#include "IRFwdDecl.h"
#include "CIN.h"
#include "Type.h"
#include "llir/LLIR.h"

namespace nacho {
    namespace backend {

    // TensorLowerer is responsible for lowering Tensor specific code.
    // This includes generating the tensor struct definition and other
    // tensor-access or manipulation code.
    struct TensorLowerer {
        std::string tensor_name;
        TensorType tensor_type;
        TensorLowerer() = default;
        TensorLowerer(std::string tensor_name, TensorType tensor_type)
            : tensor_name(std::move(tensor_name)), tensor_type(std::move(tensor_type)) {}

        inline std::string get_struct_name() const {
            return tensor_name + "_tensor_format";
        }

        inline std::string get_index_struct_name() const {
            return tensor_name + "_tensor_index";
        }

        inline std::string
        get_indices_field_name(const std::string &index) const {
            return "dim_" + index + "_indices";
        }

        inline std::string get_indices_field_name(const int level) const {
            return get_indices_field_name(
                tensor_type.format.levels[level].index);
        }

        inline std::string
        get_length_field_name(const std::string &index) const {
            return "dim_" + index + "_length";
        }

        inline std::string
        get_offsets_field_name(const std::string &index) const {
            return "dim_" + index + "_offsets";
        }

        inline std::string get_offsets_field_name(const int level) const {
            return get_offsets_field_name(
                tensor_type.format.levels[level].index);
        }

        inline std::string get_size_field_name(const std::string &index) const {
            return "dim_" + index + "_size";
        }

        inline std::string get_size_field_name(const int level) const {
            return get_size_field_name(tensor_type.format.levels[level].index);
        }

        inline std::string
        get_work_function_name(const std::string &index) const {
            return "work_" + tensor_name + "_dim_" + index;
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
                                        int target_dim);

        llir::lExpr get_offset_expression_for_next_sparse(
            int dim_level_start, int dim_level_end, bool upper_bound,
            bool use_field_access = false,
            llir::lExpr field_access_var = nullptr);

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

        inline bool is_sparse(const int level) const {
            return is_sparse_format(tensor_type.format.levels[level].format);
        }

        inline std::string get_iter_name(const int level) const {
            return "iter_" + tensor_name + "_" +
                   tensor_type.format.levels[level].index +
                   (is_sparse(level) ? "p" : "");
        }

        inline std::string get_stop_name(const int level) const {
            return "stop_" + tensor_name + "_" +
                   tensor_type.format.levels[level].index;
        }

        inline std::string get_idx_name(const int level) const {
            return "idx_" + tensor_name + "_" +
                   tensor_type.format.levels[level].index;
        }

        inline std::string get_end_name(const int level) const {
            return "end_" + tensor_name + "_" +
                   tensor_type.format.levels[level].index;
        }

        inline std::string get_coord_name(const int level) const {
            return tensor_name + "_" + tensor_type.format.levels[level].index;
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
            llir::lExpr tensor = get_tensor_expr();
            llir::lExpr idxs =
                llir::lFieldAccess::make(tensor, get_indices_field_name(level));
            return llir::lArrayAccess::make(std::move(idxs), std::move(iter));
        }

        llir::lStmt make_eval(const int level,
                              const llir::lType &index_t) const {
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
                    ? (crd ==
                       llir::lVar::make(index_t,
                                        tensor_type.format.levels[level].index))
                    : llir::lConst::make(1);
            return llir::Accumulate::make(std::move(iter), std::move(value));
        }
    };
}
}