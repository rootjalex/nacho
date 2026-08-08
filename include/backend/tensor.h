#pragma once
#include "IRFwdDecl.h"
#include "CIN.h"
#include "Type.h"
#include "llir/LLIR.h"
#include <algorithm>
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

        private:
        TensorType tensor_type;

        public:
        // Named alongside the other field-name helpers below; the offsets field itself is
        // only present on compressed levels that have a parent.
        inline std::string
        get_offsets_field_name(const TensorIndex &index) const {
            return "dim_" + index.str() + "_offsets";
        }

        inline std::string get_offsets_field_name(const TensorLevelNum level) const {
            return get_offsets_field_name(tensor_level_index(level));
        }

        std::string tensor_name;
        std::vector<TensorIndex> all_loop_indices;
        bool is_result_tensor;
        TensorLowerer() = default;
        TensorLowerer(std::string tensor_name, TensorType tensor_type, std::vector<TensorIndex>& all_loop_indices, bool is_result_tensor=false)
            : tensor_type(std::move(tensor_type)), tensor_name(std::move(tensor_name)), all_loop_indices(all_loop_indices), is_result_tensor(is_result_tensor) {}
        llir::lType index_t = llir::Generic_t::make("index_t");


        inline LoopNum end_loop_num() const {
            return LoopNum(static_cast<int>(all_loop_indices.size()));
        }

        inline TensorLevelNum end_tensor_level() const {
            return TensorLevelNum(static_cast<int>(tensor_type.format.levels.size()));
        }

        inline LoopNum get_loop_num(TensorIndex loop_index) const {
            auto it = std::find(all_loop_indices.begin(), all_loop_indices.end(), loop_index);
            if (it != all_loop_indices.end()) {
                return LoopNum(std::distance(all_loop_indices.begin(), it));
            }
            internal_assert(false) << "Loop index " << loop_index << " does not exist in all loop indices.";
        }

        inline bool tensor_level_exists(LoopNum loop_num) const {
            return tensor_type.format.level_exists(loop_index(loop_num));
        }

        inline bool tensor_level_exists(TensorIndex loop_index) const {
            return tensor_type.format.level_exists(loop_index);
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
            TensorIndex idx = tensor_type.format.levels[tensor_level.get()].index;
            auto it = std::find(all_loop_indices.begin(), all_loop_indices.end(), idx);
            internal_assert(it != all_loop_indices.end()) << "Tensor level index " << idx << " does not exist in loop indices.";
            return LoopNum(std::distance(all_loop_indices.begin(), it));
        }

        inline std::string get_struct_name() const {
            return tensor_name + "_tensor_format";
        }

        inline const TensorType &type() const {
            return tensor_type;
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
        get_indices_field_name(const TensorIndex &index) const {
            return "dim_" + index.str() + "_indices";
        }

        inline llir::lExpr get_indices_field(const TensorIndex &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_indices_field_name(index)
            );
        }

        inline std::string get_indices_field_name(const TensorLevelNum level) const {
            return get_indices_field_name(tensor_level_index(level));
        }

        inline llir::lExpr get_indices_field(const TensorLevelNum level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_indices_field_name(level)
            );
        }

        inline llir::lExpr get_read_index_value_expr(const TensorIndex &index, llir::lExpr pos) const {
            if(index.is_merged_index()) {
                return llir::lFunctionCall::make(
                    get_func_name_for_read_tuple_for_merged_index(index),
                    std::vector<llir::lExpr>{
                        llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                        pos
                    }
                );
            }
            return get_indices_field(index)[pos];
        }

        inline llir::lStmt get_store_index_stmt(const TensorIndex &index, llir::lExpr pos, llir::lExpr value) const {
            if(index.is_merged_index()) {
                return llir::BaseExpr::make(llir::lFunctionCall::make(
                    get_func_name_for_write_tuple_for_merged_index(index),
                    std::vector<llir::lExpr>{
                        llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                        pos,
                        value
                    }
                ));
            }
            return llir::Store::make(get_indices_field(index)[pos], value);
        }

        inline std::string
        get_length_field_name(const TensorIndex &index) const {
            return "dim_" + index.str() + "_length";
        }

        inline std::string get_length_field_name(const TensorLevelNum level) const {
            return get_length_field_name(tensor_level_index(level));
        }

        inline llir::lExpr get_length_field(const TensorIndex &index) const {
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


        inline llir::lExpr get_offsets_field(const TensorIndex &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_offsets_field_name(index)
            );
        }

        

        inline llir::lExpr get_offsets_field(const TensorLevelNum level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_offsets_field_name(level)
            );
        }

        inline std::string get_size_field_name(const TensorIndex &index) const {
            return "dim_" + index.str() + "_size";
        }

        inline llir::lExpr get_size_field(const TensorIndex &index) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_size_field_name(index)
            );
        }

        inline std::string get_size_field_name(const TensorLevelNum level) const {
            return get_size_field_name(tensor_level_index(level));
        }

        inline llir::lExpr get_size_field(const TensorLevelNum level) const {
            return llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make(get_struct_name()), tensor_name),
                get_size_field_name(level)
            );
        }

        inline std::string
        get_work_function_name(std::string prefix_string, const TensorIndex &index) const {
            return "work_"+prefix_string+"_" + tensor_name + "_dim_" + index.str();
        }

        inline std::string get_type_suffix(const TensorLevelNum level) const {
            // TODO: standardize the "_p" stuff somewhere.
            return is_sparse(level)
                       ? "_p"
                       : "";
        }

        inline std::string get_iterator_suffix(const TensorLevelNum level) const {
            if(is_sparse(level)) {
                return tensor_name + "_" + tensor_level_index(level).str() + "_p";
            } else {
                return tensor_level_index(level).str();
            }
        }

        inline std::string get_iterator_suffix(TensorIndex idx) const {
            if(!tensor_type.format.level_exists(idx) || is_sparse(idx)) {
                return tensor_name + "_" + idx.str() + "_p";
            } else {
                return idx.str();
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
        llir::lStmt lower_work_function(std::vector<TensorIndex> partial_loop_order,
                                        LoopNum target_loop, bool is_target_loop_value_fixed = false);



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

        inline bool is_sparse(TensorIndex index) const {
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

        inline bool is_merged_level(const TensorLevelNum level) const {
            return is_merged_format(tensor_type.format.levels[level.get()].format);
        }

        inline bool is_singleton(TensorIndex index) const {
            internal_assert(tensor_type.format.level_exists(index)) << "Index " << index << " does not exist in tensor format levels. Tensor " << tensor_name;
            return is_singleton_format(tensor_type.format.lvlfmt_of(index));
        }

        inline bool is_singleton(LoopNum loop_num) const {
            internal_assert(tensor_level_exists(loop_num)) << "Level " << loop_num.get() << " does not exist in tensor format levels.";
            return is_singleton_format(tensor_type.format.levels[loop_num_to_tensor_level(loop_num).get()].format);
        }

        inline bool is_singleton(const TensorLevelNum level) const {
            return is_singleton_format(tensor_type.format.levels[level.get()].format);
        }

        inline bool is_unique(TensorIndex index) const {
            internal_assert(tensor_type.format.level_exists(index)) << "Index " << index << " does not exist in tensor format levels. Tensor " << tensor_name;
            return is_unique_format(tensor_type.format.lvlfmt_of(index));
        }

        inline bool is_unique(LoopNum loop_num) const {
            internal_assert(tensor_level_exists(loop_num)) << "Level " << loop_num.get() << " does not exist in tensor format levels.";
            return is_unique_format(tensor_type.format.levels[loop_num_to_tensor_level(loop_num).get()].format);
        }

        inline bool is_unique(const TensorLevelNum level) const {
            return is_unique_format(tensor_type.format.levels[level.get()].format);
        }

        inline bool is_compressed(TensorIndex index) const {
            internal_assert(tensor_type.format.level_exists(index)) << "Index " << index << " does not exist in tensor format levels. Tensor " << tensor_name;
            return is_compressed_format(tensor_type.format.lvlfmt_of(index));
        }

        inline bool is_compressed(LoopNum loop_num) const {
            internal_assert(tensor_level_exists(loop_num)) << "Level " << loop_num.get() << " does not exist in tensor format levels.";
            return is_compressed_format(tensor_type.format.levels[loop_num_to_tensor_level(loop_num).get()].format);
        }

        inline bool is_compressed(const TensorLevelNum level) const {
            return is_compressed_format(tensor_type.format.levels[level.get()].format);
        }

        inline TensorIndex loop_index(const LoopNum loop_num) const {
            return all_loop_indices[loop_num.get()];
        }

        inline TensorIndex tensor_level_index(const TensorLevelNum tensor_level) const {
            return tensor_type.format.levels[tensor_level.get()].index;
        }

        inline std::string get_iter_name(const TensorIndex &forall_index) const {
            return "iter_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_stop_name(const TensorIndex &forall_index) const {
            return "stop_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_start_name(const TensorIndex &forall_index) const {
            return "start_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_end_name(const TensorIndex &forall_index) const {
            return "end_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_seg_end_name(const TensorIndex &forall_index) const {
            return "seg_end_" + get_iterator_suffix(forall_index);
        }

        inline std::string get_iter_name(const TensorLevelNum level) const {
            return get_iter_name(tensor_level_index(level));
        }

        inline std::string get_stop_name(const TensorLevelNum level) const {
            return get_stop_name(tensor_level_index(level));
        }

        inline std::string get_start_name(const TensorLevelNum level) const {
            return get_start_name(tensor_level_index(level));
        }

        inline std::string get_end_name(const TensorLevelNum level) const {
            return get_end_name(tensor_level_index(level));
        }

        inline std::string get_seg_end_name(const TensorLevelNum level) const {
            return get_seg_end_name(tensor_level_index(level));
        }

        inline llir::lExpr get_iter(const TensorIndex &forall_index) const {
            return llir::lVar::make(index_t, get_iter_name(forall_index));
        }

        inline llir::lExpr get_stop(const TensorIndex &forall_index) const {
            return llir::lVar::make(index_t, get_stop_name(forall_index));
        }

        inline llir::lExpr get_start(const TensorIndex &forall_index) const {
            return llir::lVar::make(index_t, get_start_name(forall_index));
        }

        inline llir::lExpr get_end(const TensorIndex &forall_index) const {
            return llir::lVar::make(index_t, get_end_name(forall_index));
        }

        inline llir::lExpr get_seg_end(const TensorIndex &forall_index) const {
            return llir::lVar::make(index_t, get_seg_end_name(forall_index));
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

        inline llir::lExpr get_seg_end(const TensorLevelNum level) const {
            return llir::lVar::make(index_t, get_seg_end_name(level));
        }

        inline std::string get_coord_name(const TensorLevelNum level) const {
            return tensor_name + "_" + tensor_level_index(level).str();
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
            return  get_read_index_value_expr(tensor_level_index(level), iter);
        }

        llir::lStmt make_eval(const TensorLevelNum level,
                              const llir::lType &index_t, bool is_last_loop) {
            std::vector<llir::lStmt> stmts;
            std::map<TensorLevelNum, llir::lExpr> pos_vars;
            for(TensorLevelNum l = BEFORE_FIRST_LEVEL + 1; l < TensorLevelNum(level); ++l) {
                if(is_sparse(l) && !is_unique(l))
                    pos_vars[l] = get_seg_end(l);
                else
                    pos_vars[l] = get_iter(l);
            }
            
            auto val = get_coord(level, index_t);
            if(is_sparse(level) && !is_last_loop) {
                // if the value is outside the bounds then use max possible val for the level (size val) which
                // is never possible. We terminate iteration early if coordinate is evaluated to this value
                val = llir::lSelect::make(
                    (llir::lVar::make(index_t, get_iter_name(level)) > this->get_bound(level, true, pos_vars)),
                    get_size_field(level),
                    std::move(val)
                );
            }
            auto level_idx = tensor_level_index(level);
            stmts.push_back(llir::Declare::make(
                level_idx.is_merged_index() ? llir::Tuple_t::make(std::vector<llir::lType>(level_idx.indices.size(), index_t)) : index_t,
                get_coord_name(level), val));
            return llir::Sequence::make(std::move(stmts));
        }

        llir::lStmt make_seg_end(const TensorLevelNum level, const llir::lExpr index_value) {
            std::vector<llir::lStmt> stmts;
            std::map<TensorLevelNum, llir::lExpr> pos_vars;
            for(TensorLevelNum l = BEFORE_FIRST_LEVEL + 1; l < TensorLevelNum(level); ++l) {
                if(is_sparse(l) && !is_unique(l))
                    pos_vars[l] = get_seg_end(l);
                else
                    pos_vars[l] = get_iter(l);
            }
            internal_assert(is_sparse(level) && !is_unique(level));

            llir::lExpr binary_search_call = llir::lFunctionCall::make(
                "binary_search_ub",
                std::vector<llir::lExpr>{
                    get_indices_field(level),
                    index_value,
                    get_iter(level),
                    get_bound(level, true, pos_vars)
                }
            );
            stmts.push_back(
                llir::Declare::make(
                    index_t, get_seg_end_name(level),
                    llir::lSelect::make(
                        index_value != get_coord_var(level, index_t),
                        get_iter(level),
                        binary_search_call
                    )
                )
            );
        
            return llir::Sequence::make(std::move(stmts));
        }

        llir::lStmt make_inc(const TensorLevelNum level,
                             const llir::lType &index_t) const {
            llir::lExpr iter = get_iter_var(level, index_t);
            llir::lExpr crd = get_coord_var(level, index_t);
            // Dense always increments, sparse increments if at the lowest
            // value.
            llir::lExpr value =
                is_sparse(level)
                    ? (crd == llir::lVar::make(index_t, tensor_level_index(level).str()))
                    : llir::lConst::make(1);
            return llir::Accumulate::make(std::move(iter), std::move(value));
        }

        llir::lStmt make_seg_inc(const TensorLevelNum level,
                                 const llir::lType &index_t) const {
            internal_assert(is_sparse(level) && !is_unique(level));
            llir::lExpr iter = get_iter_var(level, index_t);
            llir::lExpr crd = get_coord_var(level, index_t);
            llir::lExpr seg_end = get_seg_end(level);
            return llir::Store::make(
                iter,
                llir::lSelect::make(
                    (crd == llir::lVar::make(index_t, tensor_level_index(level).str())),
                    seg_end,
                    iter
                )
            );
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

        std::string get_func_name_for_read_tuple_for_merged_index(TensorIndex idx) const {
            return "read_tuple_" + tensor_name + "_" + idx.str();
        }

        std::string get_func_name_for_write_tuple_for_merged_index(TensorIndex idx) const {
            return "write_tuple_" + tensor_name + "_" + idx.str();
        }

        // lower the function read_tuple_<tensor_name>_<merged_index>
        // This function is a helper used to generate a tuple for the
        // merged index that is present in the tensor
        llir::lStmt lower_func_read_tuple_for_merged_index() const;

        llir::lStmt lower_func_write_tuple_for_merged_index() const;
    };
}
}