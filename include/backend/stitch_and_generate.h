#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "Printer.h"
#include "Type.h"
#include "backend/tensor.h"
#include "llir/Function.h"

#include <fstream>

namespace nacho {
namespace backend {

    struct StitchAndGenerate {

        std::map<std::string, TensorLowerer> &operand_tensors;
        TensorLowerer &result_tensor;
        std::vector<CIN> forall_list;
        TensorLowerer &reduced_result_tensor;

        struct FuncDecl {
            std::vector<std::string> generics;
            std::vector<llir::Function::Attribute> attributes;
            std::vector<llir::Function::Argument> args;
            llir::lType ret_type;
            std::string name;
            std::vector<llir::lStmt> body;
        };

        std::string name;
        std::ofstream header_file;
        std::ofstream source_file;

        llir::lExpr num_threads_var = llir::lVar::make(llir::Int_t::make(32), "num_threads");
        llir::lType sizet_type = llir::Generic_t::make("size_t");
        llir::lType void_type = llir::Generic_t::make("void");
        llir::lType index_t = llir::Generic_t::make("index_t");
        llir::lExpr cuda_stream_var = llir::lVar::make(llir::Generic_t::make("cudaStream_t"), "stream");
        llir::lExpr device_to_host = llir::lVar::make(llir::Generic_t::make("cudaMemcpyKind"), "cudaMemcpyDeviceToHost");
        llir::lExpr thread_id_var = llir::lVar::make(llir::Int_t::make(32), "thread_id");
        llir::lExpr total_work_var = llir::lVar::make(llir::Int_t::make(32), "total_work");
        llir::lExpr per_thread_work_var = llir::lVar::make(llir::Int_t::make(32), "per_thread_work");
        llir::lExpr work_offsets_var = llir::lVar::make(llir::Ptr_t::make(llir::Int_t::make(32)), "T_work_offsets");


        FuncDecl main_func;

        std::vector<llir::lStmt> main_function_stmts;

        StitchAndGenerate(std::string name,
                           std::map<std::string, TensorLowerer> &operand_tensors,
                           TensorLowerer &result_tensor,
                           std::vector<CIN> forall_list,
                           TensorLowerer &reduced_result_tensor)
            : operand_tensors(operand_tensors), result_tensor(result_tensor),
              forall_list(std::move(forall_list)), reduced_result_tensor(reduced_result_tensor),
              name(std::move(name)) {}

        // Opens header_file/source_file at "<output_directory()>/<name><header_suffix>" and
        // "<output_directory()>/<name><source_suffix>", and writes the boilerplate prologue
        // shared by every target (CPU/GPU).
        void open_files(const std::string &header_suffix, const std::string &source_suffix, const std::string &runnable_macro);

        // Adds the operand-tensor arguments shared by the CPU/GPU main function signatures,
        // and sets the return type to the result tensor's struct.
        void add_tensor_args();

        // `<Result>_tensor_format<index_t, value_t>` — the entry point's return type.
        llir::lType get_result_struct_type() const;

        // The local holding the result inside the entry point's body.
        llir::lExpr get_result_var() const;

        llir::lExpr convert_param_to_arg(const llir::Function::Argument &param, LoopNum current_sparse_intersection);


        void add_to_header_file(llir::lStmt func) {
            Printer printer(header_file);
            printer.print(func);
        }

        void add_to_header_file(llir::lType type) {
            Printer printer(header_file);
            printer.print(type);
        }

        void close_files();

        void stitch_tensor_def(llir::lType type) {add_to_header_file(type);};
        virtual void stitch_result_tensor_def(llir::lType type) {add_to_header_file(type);};

        virtual void stitch_temp_result_tensor_def(llir::lType type) {add_to_header_file(type);};
        virtual void stitch_count_offset_struct_def(llir::lType type) {add_to_header_file(type);};

        llir::lType operand_pos_map_struct_def;
        virtual void stitch_operand_pos_map_struct_def(llir::lType type) {
            operand_pos_map_struct_def = type;
            add_to_header_file(type);
        };

        void stitch_kernels(
            llir::lStmt compute_kernel, // compute_kernel from previous sparse intersect
            llir::lType partition_struct, llir::lStmt partition_kernel, llir::lStmt precompute_kernel, // partition and precompute kernels from current sparse intersection
            LoopNum previous_previous_sparse_intersection, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection,
            std::map<std::string, TensorLowerer> included_tensors
        );

        virtual void generate_memory_allocations(
            llir::lType partition_struct, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection,
            bool precompute_kernel_defined, bool compute_kernel_defined
        ) = 0;

        // Generates a single "allocate `size` bytes at `address`" statement (e.g. `address =
        // (pointer_type)malloc(size)` on CPU, `cudaMallocAsync(&address, size, stream)` on GPU).
        // `size` is always a byte count. `pointer_type` is address's pointer type
        // register_for_free should be true only for temporary/scratch allocations
        virtual llir::lStmt generate_single_memory_allocation_statement(llir::lExpr address, llir::lType pointer_type, llir::lExpr size, bool register_for_free) = 0;

        // Generates the statement setting `offsets_field[0]` to 0. The compute kernel only
        // ever writes the offset one past a coordinate it produced, so the leading entry of a
        // freshly allocated offsets array has no other writer.
        virtual llir::lStmt generate_zero_leading_offset_statement(llir::lExpr offsets_field) = 0;

        // Returns the statement allocating result tensor fields for loops between
        // previous_sparse_intersection and current_sparse_intersection. 
        llir::lStmt generate_result_tensor_allocations(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, bool is_dummy_allocation = false);

        // Returns the statement that frees the memory block(s) allocated for the sparse
        // intersection level ending at sparse_intersection (an empty Sequence if there was
        // no earlier level to free).
        virtual llir::lStmt generate_memory_free_statements(LoopNum sparse_intersection) = 0;

        void stitch_compute_kernel_call(llir::lStmt compute_kernel, LoopNum current_sparse_intersection);
        void stitch_partition_kernel_call(llir::lType partition_struct, llir::lStmt partition_kernel, LoopNum current_sparse_intersection);
        void stitch_precompute_kernel_call(llir::lStmt precompute_kernel, LoopNum current_sparse_intersection);

        virtual void generate_prefix_sum_calls(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) = 0;

        virtual void generate_work_offsets_scan(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) = 0;

        // Emits an early-return when the result of this sparse intersection level turns out to be empty.
        void generate_early_termination_for_empty_result(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection);

        // Wraps a partition/precompute kernel in the backend-specific dispatch call
        // (e.g. a tbb::parallel_for lambda on CPU, a CUDA kernel launch on GPU).
        virtual llir::lExpr wrap_kernel_with_backend_specific_call(llir::lStmt kernel, LoopNum current_sparse_intersection) = 0;

        void stitch_final_compute_kernel(
            llir::lStmt compute_kernel,
            std::map<std::string, TensorLowerer> included_tensors,
            LoopNum previous_sparse_intersection,
            bool stitch_with_scatter_reduction);

        void decalare_and_initialize_common_variables();

        void add_work_calculation_statements(std::map<std::string, TensorLowerer> included_tensors, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, bool compute_kernel_defined);

        // Generates the statement that reads total_work out of work_offsets_var[index_expr]
        // (a direct host-array read on CPU, a cudaMemcpyAsync from device memory on GPU).
        virtual llir::lStmt generate_total_work_from_offsets_statement(llir::lExpr index_expr) = 0;

        inline std::string get_all_loops_string(LoopNum loop_num) {
            std::string all_loop_indices_string = "";
            for(LoopNum i=BEFORE_FIRST_LOOP+1;i<=loop_num;++i)
                all_loop_indices_string += forall_list[i.get()].as<Forall>()->idx.str();
            return all_loop_indices_string;
        }

        inline std::string get_partition_var_name(LoopNum current_sparse_intersection) {
            return "parts_" + get_all_loops_string(current_sparse_intersection);
        }

        inline std::string get_count_offset_var_name(LoopNum current_sparse_intersection) {
            return "count_offsets_" + get_all_loops_string(current_sparse_intersection);
        }


        inline llir::lType get_partition_var_type(LoopNum current_sparse_intersection) {
            return llir::Generic_t::make("partition_" + get_all_loops_string(current_sparse_intersection) + "<int32_t>");
        }

        // ugly hack
        inline llir::lType get_count_offset_var_type() {
            return llir::Generic_t::make("result_per_thread_count<int32_t>");
        }

        inline llir::lExpr get_partition_var(LoopNum current_sparse_intersection) {
            return llir::lVar::make(get_partition_var_type(current_sparse_intersection), get_partition_var_name(current_sparse_intersection));
        }

        inline llir::lExpr get_count_offset_var(LoopNum current_sparse_intersection) {
            return llir::lVar::make(get_count_offset_var_type(), get_count_offset_var_name(current_sparse_intersection));
        }

        inline llir::lExpr get_count_offsets_field(LoopNum loop, LoopNum current_sparse_intersection) {
            return llir::lFieldAccess::make(get_count_offset_var(current_sparse_intersection), "dim_" + result_tensor.loop_index(loop).str()+"_count");
        }

        inline llir::lType get_operand_pos_map_type() {
            return llir::Generic_t::make("result_to_operand_pos_map<int32_t>");
        }

        inline std::string get_operand_pos_map_var_name() {
            return result_tensor.tensor_name + "_pos_map";
        }
    };


    struct StitchAndGenerateCPU : public StitchAndGenerate {
        StitchAndGenerateCPU(
            std::string name,
            std::map<std::string, TensorLowerer> &operand_tensors,
            TensorLowerer &result_tensor,
            std::vector<CIN> forall_list,
            TensorLowerer &reduced_result_tensor);

        // Tracks pointers registered via generate_single_memory_allocation_statement, keyed by the
        // sparse-intersection level
        std::map<std::string, std::vector<llir::lExpr>> allocated_pointers;
        LoopNum current_allocation_level = BEFORE_FIRST_LOOP;

        void generate_memory_allocations(
            llir::lType partition_struct, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection,
            bool precompute_kernel_defined, bool compute_kernel_defined
        ) override;

        llir::lStmt generate_single_memory_allocation_statement(llir::lExpr address, llir::lType pointer_type, llir::lExpr size, bool register_for_free) override;

        llir::lStmt generate_zero_leading_offset_statement(llir::lExpr offsets_field) override;

        llir::lStmt generate_memory_free_statements(LoopNum sparse_intersection) override;

        llir::lExpr wrap_kernel_with_backend_specific_call(llir::lStmt kernel, LoopNum current_sparse_intersection) override;

        void generate_prefix_sum_calls(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) override;

        void generate_work_offsets_scan(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) override;

        llir::lStmt generate_total_work_from_offsets_statement(llir::lExpr index_expr) override;
    };

    struct StitchAndGenerateGPU : public StitchAndGenerate {
        StitchAndGenerateGPU(
            std::string name,
            std::map<std::string, TensorLowerer> &operand_tensors,
            TensorLowerer &result_tensor,
            std::vector<CIN> forall_list,
            TensorLowerer &reduced_result_tensor);

        void generate_memory_allocations(
            llir::lType partition_struct, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection,
            bool precompute_kernel_defined, bool compute_kernel_defined
        ) override;

        llir::lStmt generate_single_memory_allocation_statement(llir::lExpr address, llir::lType pointer_type, llir::lExpr size, bool register_for_free) override;

        llir::lStmt generate_zero_leading_offset_statement(llir::lExpr offsets_field) override;

        llir::lStmt generate_memory_free_statements(LoopNum sparse_intersection) override;

        llir::lExpr wrap_kernel_with_backend_specific_call(llir::lStmt kernel, LoopNum current_sparse_intersection) override;

        void generate_prefix_sum_calls(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) override;

        void generate_work_offsets_scan(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) override;

        llir::lStmt generate_total_work_from_offsets_statement(llir::lExpr index_expr) override;
    };

} // namespace backend
} // namespace nacho
