#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Printer.h"
#include "Type.h"
#include "backend/tensor.h"
#include <string>
#include <map>


namespace nacho {
namespace backend {

        struct KernelInfo {
            std::string name;
            std::vector<std::string> template_args;
            std::vector<llir::Function::Argument> args;
            enum Kind { Partition, Precompute, Compute } kind;
            int phase;
        };

        struct CINLowerer {

            llir::lType index_t  = llir::Generic_t::make("index_t");

            CIN cin;
            Printer printer;
            TensorLowerer result_tensor;
            std::map<std::string, TensorLowerer> operand_tensors;
            CINLowerer(CIN cin, std::ostream &os);

            std::vector<std::string> get_loop_order();
            std::vector<CIN> get_forall_loops();
            void lower_cin();
            void lower_struct_definitions(int last_sparse_intersection);
            std::vector<int> get_sparse_intersection_levels(CIN& cin);
            CIN get_phase_cin_for_sparse_intersection(int target_level, CIN& cin);

            void lower_binary_search_function();
            llir::lType lower_result_to_operand_pos_map_struct(int last_sparse_intersection);
            std::map<std::string, TensorLowerer> get_active_tensors_for_level(int level);

            // Host function generation
            std::vector<KernelInfo> kernel_infos;

            // Struct types collected during lowering for field enumeration
            struct PhaseStructInfo {
                llir::lType partition_struct;
                llir::lType counts_struct;
                int previous_sparse_intersection_level;
                int current_sparse_intersection_level;
                int next_sparse_intersection_level;
                bool has_precompute_kernel;
            };
            std::vector<PhaseStructInfo> phase_struct_infos;

            void lower_host_function();

            // Generate a flat-API wrapper function that takes raw pointers
            // instead of nacho structs. This is the stable boundary between
            // the compiler and runtime.
            void lower_flat_wrapper(const std::string &op_name);

            // COO (all-Coordinate format) specific codegen
            void lower_coo_cin();
            void lower_coo_compare_function();
            void lower_coo_partition_kernel();
            void lower_coo_precompute_kernel();
            void lower_coo_compute_kernel();
            void lower_coo_host_function();
            void lower_coo_flat_wrapper(const std::string &op_name);
        };



} // namespace backend

} // namespace nacho
