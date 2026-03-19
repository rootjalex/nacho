#pragma once
#include "IRFwdDecl.h"
#include "CIN.h"
#include "Type.h"
#include "llir/LLIR.h"
#include <numeric>
#include <ostream>
#include <map>

namespace nacho {
    namespace backend {

    class LoopNum {
        public:
        int value;
        explicit LoopNum(int v) : value(v) {}
        int get() const { return value; }
        bool operator<(const LoopNum& other) const {
            return value < other.value;
        }
        bool operator>(const LoopNum& other) const {
            return value > other.value;
        }
        bool operator<=(const LoopNum& other) const {
            return value <= other.value;
        }
        bool operator>=(const LoopNum& other) const {
            return value >= other.value;
        }
        bool operator==(const LoopNum& other) const {
            return value == other.value;
        }
        bool operator!=(const LoopNum& other) const {
            return value != other.value;
        }
        LoopNum operator+(int other) const {
            return LoopNum(value + other);
        }

        LoopNum operator-(int other) const {
            return LoopNum(value - other);
        }

        LoopNum& operator++() {
            ++value;
            return *this;
        }
        LoopNum& operator--() {
            --value;
            return *this;
        }
        LoopNum& max(const LoopNum& other) {
            if (other.value > value) {
                value = other.value;
            }
            return *this;
        }
    };

    inline static const LoopNum BEFORE_FIRST_LOOP{-1};

    inline std::ostream &operator<<(std::ostream &os, const LoopNum &l) {
        return os << l.value;
    }

    // TensorLowerer is responsible for lowering Tensor specific code.
    // This includes generating the tensor struct definition and other
    // tensor-access or manipulation code.
    struct TensorLowerer {

        TensorType tensor_type;
        std::string tensor_name;
        std::vector<std::string> all_loop_indices;
        bool is_result_tensor;
        TensorLowerer() = default;
        TensorLowerer(std::string tensor_name, TensorType tensor_type, std::vector<std::string>& all_loop_indices, bool is_result_tensor=false)
            : tensor_type(std::move(tensor_type)), tensor_name(std::move(tensor_name)), all_loop_indices(all_loop_indices), is_result_tensor(is_result_tensor) {}
        llir::lType index_t = llir::Generic_t::make("index_t");


        inline LoopNum end_loop_num() const {
            return LoopNum(static_cast<int>(all_loop_indices.size()));
        }

        inline TensorLevelNum end_tensor_level() const {
            return TensorLevelNum(static_cast<int>(tensor_type.format.levels.size()));
        }

        inline LoopNum get_loop_num(std::string loop_name) const {
            auto it = std::find(all_loop_indices.begin(), all_loop_indices.end(), loop_name);
            if (it != all_loop_indices.end()) {
                return LoopNum(std::distance(all_loop_indices.begin(), it));
            }
            internal_assert(false) << "Loop index " << loop_name << " does not exist in all loop indices.";
        }

        inline bool tensor_level_exists(LoopNum loop_num) const {
            return tensor_type.format.level_exists(loop_name(loop_num));
        }

        inline bool tensor_level_exists(std::string loop_name) const {
            return tensor_type.format.level_exists(loop_name);
        }

    
        inline TensorLevelNum loop_num_to_tensor_level(LoopNum loop_num) const {
            // Map loop levels to tensor levels
            if(loop_num == BEFORE_FIRST_LOOP || loop_num == end_loop_num()) {
                return TensorLevelNum(loop_num.get());
            }
            internal_assert(tensor_type.format.level_exists(all_loop_indices[loop_num.get()])) << "Loop index " << all_loop_indices[loop_num.get()] << " does not exist in tensor format levels.";
            return tensor_type.format.get_level_order(all_loop_indices[loop_num.get()]);
        }

        inline LoopNum tensor_level_to_loop_num(TensorLevelNum tensor_level) const {
            // Map tensor levels to loop levels
            if(tensor_level == BEFORE_FIRST_LEVEL || tensor_level == end_tensor_level()) {
                return LoopNum(tensor_level.get());
            }
            std::string idx = tensor_type.format.levels[tensor_level.get()].index;
            auto it = std::find(all_loop_indices.begin(), all_loop_indices.end(), idx);
            internal_assert(it != all_loop_indices.end()) << "Tensor level index " << idx << " does not exist in loop indices.";
            return LoopNum(std::distance(all_loop_indices.begin(), it));
        }

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

        inline std::string get_indices_field_name(const TensorLevelNum level) const {
            return get_indices_field_name(tensor_level_name(level));
        }

        inline llir::lExpr get_indices_field(const TensorLevelNum level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_indices_field_name(level)
            );
        }

        inline std::string
        get_length_field_name(const std::string &index) const {
            return "dim_" + index + "_length";
        }

        inline std::string get_length_field_name(const TensorLevelNum level) const {
            return get_length_field_name(tensor_level_name(level));
        }
    
        inline llir::lExpr get_length_field(const std::string &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_length_field_name(index)
            );
        }

        inline llir::lExpr get_length_field(const TensorLevelNum level) const {
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

        inline std::string get_offsets_field_name(const TensorLevelNum level) const {
            return get_offsets_field_name(tensor_level_name(level));
        }

        inline llir::lExpr get_offsets_field(const TensorLevelNum level) const {
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

        inline std::string get_size_field_name(const TensorLevelNum level) const {
            return get_size_field_name(tensor_level_name(level));
        }

        inline llir::lExpr get_size_field(const TensorLevelNum level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_size_field_name(level)
            );
        }

        inline std::string
        get_work_function_name(std::string prefix_string, const std::string &index) const {
            return "work_"+prefix_string+"_" + tensor_name + "_dim_" + index;
        }

        inline std::string get_type_suffix(const TensorLevelNum level) const {
            // TODO: standardize the "_p" stuff somewhere.
            return tensor_type.format.levels[level.get()].format ==
                           LevelFormat::Compressed
                       ? "_p"
                       : "";
        }

        inline std::string get_iterator_suffix(const TensorLevelNum level) const {
            if(is_sparse(level)) {
                return tensor_name + "_" + tensor_level_name(level) + "_p";
            } else {
                return tensor_level_name(level);
            }
        }

        inline std::string get_iterator_suffix(std::string idx) const {
            if(!tensor_type.format.level_exists(idx) || is_sparse(idx)) {
                return tensor_name + "_" + idx + "_p";
            } else {
                return idx;
            }
        }

        llir::lType lower_tensor_struct_definition() const;

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
        llir::lStmt lower_work_function(std::vector<std::string> partial_loop_order,
                                        LoopNum target_loop, bool is_target_loop_value_fixed = false);



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
            internal_assert(tensor_type.format.level_exists(index)) << "Index " << index << " does not exist in tensor format levels. Tensor " << tensor_name;
            return is_sparse_format(tensor_type.format.lvlfmt_of(index));
        }

        inline bool is_sparse(const LoopNum loop_num) const {
            internal_assert(tensor_level_exists(loop_num)) << "Level " << loop_num.get() << " does not exist in tensor format levels.";
            return is_sparse_format(tensor_type.format.levels[loop_num_to_tensor_level(loop_num).get()].format);
        }

        inline bool is_sparse(const TensorLevelNum level) const {
            return is_sparse_format(tensor_type.format.levels[level.get()].format);
        }

        inline std::string loop_name(const LoopNum loop_num) const {
            return all_loop_indices[loop_num.get()];
        }

        inline std::string tensor_level_name(const TensorLevelNum tensor_level) const {
            return tensor_type.format.levels[tensor_level.get()].index;
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

        inline std::string get_iter_name(const TensorLevelNum level) const {
            return get_iter_name(tensor_level_name(level));
        }

        inline std::string get_stop_name(const TensorLevelNum level) const {
            return get_stop_name(tensor_level_name(level));
        }

        inline std::string get_start_name(const TensorLevelNum level) const {
            return get_start_name(tensor_level_name(level));
        }

        inline std::string get_end_name(const TensorLevelNum level) const {
            return get_end_name(tensor_level_name(level));
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

        inline llir::lExpr get_iter(const TensorLevelNum level) const {
            return llir::lVar::make(index_t, get_iter_name(level));
        }

        inline llir::lExpr get_stop(const TensorLevelNum level) const {
            return llir::lVar::make(index_t, get_stop_name(level));
        }

        inline llir::lExpr get_start(const TensorLevelNum level) const {
            return llir::lVar::make(index_t, get_start_name(level));
        }

        inline llir::lExpr get_end(const TensorLevelNum level) const {
            return llir::lVar::make(index_t, get_end_name(level));
        }

        inline std::string get_coord_name(const TensorLevelNum level) const {
            return tensor_name + "_" + tensor_level_name(level);
        }


        // get_level_indexing_expression gets the indexing expression for level sparse_tensor_level.
        // To get the indexing expression, all the position space values of previous levels is the tensor are required.
        // These position space values have to be passed as a map from the tensor levels to their expressions.
        // NOTE: the level passed is level number in this tensor not the loop number among all loops
        // tensor_level has to be a sparse level in this tensor
        llir::lExpr get_level_indexing_expression(TensorLevelNum tensor_level,
            bool upper_bound, std::map<TensorLevelNum, llir::lExpr> pos_vars);

        // get_bound gets the upper/lower bound for the position space value for level tensor_level.
        // To get the bounds, all the position space values of previous levels is the tensor are required.
        // These position space values have to be passed as a map from the tensor levels to their expressions.
        // NOTE: the level passed is level number in this tensor not the level number among all loops
        llir::lExpr get_bound(TensorLevelNum tensor_level,
            bool upper_bound, std::map<TensorLevelNum, llir::lExpr> pos_vars);


        llir::lExpr get_coord_var(const TensorLevelNum level,
                                  const llir::lType &index_t) const {
            return llir::lVar::make(index_t, get_coord_name(level));
        }

        llir::lExpr get_iter_var(const TensorLevelNum level,
                                 const llir::lType &index_t) const {
            return llir::lVar::make(index_t, get_iter_name(level));
        }

        llir::lExpr get_coord(const TensorLevelNum level,
                              const llir::lType &index_t) const {
            llir::lExpr iter = llir::lVar::make(index_t, get_iter_name(level));
            if (!is_sparse(level)) {
                return iter;
            }
            return  get_indices_field(level)[iter];
        }

        llir::lStmt make_eval(const TensorLevelNum level,
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

        llir::lStmt make_inc(const TensorLevelNum level,
                             const llir::lType &index_t) const {
            llir::lExpr iter = get_iter_var(level, index_t);
            llir::lExpr crd = get_coord_var(level, index_t);
            // Dense always increments, sparse increments if at the lowest
            // value.
            llir::lExpr value =
                is_sparse(level)
                    ? (crd == llir::lVar::make(index_t, tensor_level_name(level)))
                    : llir::lConst::make(1);
            return llir::Accumulate::make(std::move(iter), std::move(value));
        }

        Seq get_index_sequence(const TensorLevelNum level) const {
            return Index::make(tensor_name, tensor_type, level.get());
        }

        bool are_all_lvls_dense() const {
            return tensor_type.format.are_all_lvls_dense();
        }

        LoopNum get_loop_num_for_prev_sparse_level(LoopNum loop_num) const {
            for (LoopNum i = loop_num - 1; BEFORE_FIRST_LOOP < i; --i) {
                if (tensor_level_exists(i) && is_sparse(i)) {
                    return i;
                }
            }
            return BEFORE_FIRST_LOOP;
        }

        LoopNum get_loop_num_for_next_sparse_level(LoopNum loop_num) const {
            for (LoopNum i = loop_num + 1; i < end_loop_num(); ++i) {
                if (tensor_level_exists(i) && is_sparse(i)) {
                    return i;
                }
            }
            return end_loop_num();
        }

        LoopNum get_loop_num_for_last_sparse_level() const {
            for (LoopNum i = end_loop_num() - 1; i > BEFORE_FIRST_LOOP; --i) {
                if (tensor_level_exists(i) && is_sparse(i)) {
                    return i;
                }
            }
            return BEFORE_FIRST_LOOP;
        }
    };
}
}