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
        
        TensorLowerer(std::string tensor_name, TensorType tensor_type)
            : tensor_name(std::move(tensor_name)), tensor_type(std::move(tensor_type)) {}

        inline std::string get_struct_name() {
            return tensor_name + "_tensor_format";
        }

        inline std::string get_indices_field_name(const std::string &index) {
            return "dim_" + index + "_indices";
        }

        inline std::string get_length_field_name(const std::string &index) {
            return "dim_" + index + "_length";
        }

        inline std::string get_offsets_field_name(const std::string &index) {
            return "dim_" + index + "_offsets";
        }

        inline std::string get_size_field_name(const std::string &index) {
            return "dim_" + index + "_size";
        }

        llir::lType lower_tensor_struct_definition();

        // lower_work_function returns the LLIR work function for the given tensor.
        // target_dim is the target dimension for which work is being calculated.
        // prev_dim_positions are the positions of dimensions less than target_dim.
        // target_dim_position is the position of the target_dim at which work is being calculated.
        // Eg - Consider a 3Dtensor with dims i,j,k and the loop order is [i,l,j,k] 
        // (the 3D tensor is broadcaster over l which is the 2nd loop)
        // And now we need to calculate work for i=12, l=12, 0<= j <=54. 0 <= k <= |K|
        // Then the args are loop_order = [i,j,k,l],target_dim = 2, prev_dim_positions = [12, 32], target_dim_position = 54
        llir::lStmt TensorLowerer::lower_work_function(std::vector<std::string> loop_order, int target_dim);

        std::tuple<llir::lExpr,int> TensorLowerer::get_offset_expression_for_next_sparse(int sparse_dim, int num_dense_dims=0) {
    };

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
}
}
}