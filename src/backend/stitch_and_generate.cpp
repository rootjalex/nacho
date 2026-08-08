#include "backend/stitch_and_generate.h"

#include "backend/output.h"

namespace nacho {
namespace backend {

    void StitchAndGenerate::stitch_kernels(
        llir::lStmt compute_kernel,
        llir::lType partition_struct, llir::lStmt partition_kernel, llir::lStmt precompute_kernel,
        LoopNum previous_previous_sparse_intersection,
        LoopNum previous_sparse_intersection, 
        LoopNum current_sparse_intersection,
        std::map<std::string, TensorLowerer> included_tensors
    ) {
        if(compute_kernel.defined()) {
            llir::lStmt result_tensor_allocations = this->generate_result_tensor_allocations(previous_previous_sparse_intersection, previous_sparse_intersection);
            if (result_tensor_allocations.defined()) {
                main_func.body.emplace_back(result_tensor_allocations);
            }
        }

        this->generate_memory_allocations(
            partition_struct, previous_sparse_intersection, current_sparse_intersection, precompute_kernel.defined(), compute_kernel.defined()
        );

        if(compute_kernel.defined()) {
            this->stitch_compute_kernel_call(
                compute_kernel, previous_sparse_intersection
            );
            this->generate_work_offsets_scan(previous_sparse_intersection, current_sparse_intersection);
        }

        add_work_calculation_statements(included_tensors, previous_sparse_intersection, current_sparse_intersection, compute_kernel.defined());

        if(compute_kernel.defined()) {
            llir::lStmt free_stmt = this->generate_memory_free_statements(previous_sparse_intersection);
            if (free_stmt.defined()) {
                main_func.body.emplace_back(free_stmt);
            }
        }

        this->stitch_partition_kernel_call(partition_struct, partition_kernel, current_sparse_intersection);

        if(precompute_kernel.defined()) {
            this->stitch_precompute_kernel_call(precompute_kernel, current_sparse_intersection);
            this->generate_prefix_sum_calls(previous_sparse_intersection, current_sparse_intersection);
            if(current_sparse_intersection != LoopNum(forall_list.size()-1)) {
                this->generate_early_termination_for_empty_result(previous_sparse_intersection, current_sparse_intersection);
            }
        }
    }

    void StitchAndGenerate::stitch_final_compute_kernel(
        llir::lStmt compute_kernel,
        std::map<std::string, TensorLowerer> included_tensors,
        LoopNum previous_sparse_intersection,
        bool stitch_with_scatter_reduction
    ) {
            llir::lStmt result_tensor_allocations = this->generate_result_tensor_allocations(previous_sparse_intersection, LoopNum(forall_list.size() - 1));
            if (result_tensor_allocations.defined()) {
                main_func.body.emplace_back(result_tensor_allocations);
            }

            this->stitch_compute_kernel_call(
                compute_kernel, LoopNum(forall_list.size() - 1)
            );

            llir::lStmt free_stmt = this->generate_memory_free_statements(LoopNum(forall_list.size() - 1));
            if (free_stmt.defined()) {
                main_func.body.emplace_back(free_stmt);
            }
    }

    void StitchAndGenerate::add_work_calculation_statements(std::map<std::string, TensorLowerer> included_tensors, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, bool compute_kernel_defined) {
        if(compute_kernel_defined) {
            main_func.body.emplace_back(
                generate_total_work_from_offsets_statement(result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)))
            );
        } else {
            llir::lExpr total_work_expr = llir::lConst::make(0);
            for (const auto &[tensor_name, tensor] : included_tensors) {
                total_work_expr = total_work_expr + tensor.get_length_field(tensor.loop_num_to_tensor_level(current_sparse_intersection));
            }
            main_func.body.emplace_back(
                    llir::Store::make(
                        llir::lVar::make(llir::Int_t::make(32), "total_work"),
                        total_work_expr
                ));
        }
        main_func.body.emplace_back(
                llir::Store::make(
                    llir::lVar::make(llir::Int_t::make(32), "per_thread_work"),
                    (llir::lVar::make(llir::Int_t::make(32), "total_work") / llir::lVar::make(llir::Int_t::make(32), "num_threads")) + llir::lConst::make(1)
                )
            );
    }

    llir::lExpr StitchAndGenerate::convert_param_to_arg(const llir::Function::Argument &param, LoopNum current_sparse_intersection) {
        if (param.name == "num_threads") {
            return num_threads_var;
        } else if (param.name == "thread_id") {
            return thread_id_var;
        } else if (param.name == "count_offset") {
            return get_count_offset_var(current_sparse_intersection);
        } else if (param.name == "total_work") {
            return total_work_var;
        } else if (param.name == "per_thread_work") {
            return per_thread_work_var;
        } else if (param.name == "partitions") {
            return get_partition_var(current_sparse_intersection);
        } else if (param.name == "count_offsets") {
            return get_count_offset_var(current_sparse_intersection);
        } else if (operand_tensors.find(param.name) != operand_tensors.end()) {
            return llir::lVar::make(param.type, param.name);
        } else if (param.name == result_tensor.tensor_name) {
            return llir::lVar::make(param.type, param.name);
        } else if (param.name == reduced_result_tensor.tensor_name) {
            return llir::lVar::make(param.type, param.name);
        } else if (param.name == get_operand_pos_map_var_name()) {
            return llir::lVar::make(param.type, param.name);
        } else if (param.name == "T_work_offsets") {
            return work_offsets_var;
        }
        internal_assert(false) << "Unknown parameter name: " << param.name;
        return llir::lExpr();
    }

    void StitchAndGenerate::stitch_partition_kernel_call(llir::lType partition_struct, llir::lStmt partition_kernel, LoopNum current_sparse_intersection) {
        add_to_header_file(partition_struct);
        add_to_header_file(partition_kernel);

        main_func.body.emplace_back(
            llir::BaseExpr::make(wrap_kernel_with_backend_specific_call(partition_kernel, current_sparse_intersection))
        );
    }

    void StitchAndGenerate::stitch_precompute_kernel_call(llir::lStmt precompute_kernel, LoopNum current_sparse_intersection) {
        add_to_header_file(precompute_kernel);

        main_func.body.emplace_back(
            llir::BaseExpr::make(wrap_kernel_with_backend_specific_call(precompute_kernel, current_sparse_intersection))
        );
    }

    void StitchAndGenerate::stitch_compute_kernel_call(llir::lStmt compute_kernel, LoopNum current_sparse_intersection) {
        add_to_header_file(compute_kernel);
        main_func.body.emplace_back(
            llir::BaseExpr::make(wrap_kernel_with_backend_specific_call(compute_kernel, current_sparse_intersection))
        );
    }

    void StitchAndGenerate::generate_early_termination_for_empty_result(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) {
        std::vector<llir::lStmt> early_termination_stmts;

        llir::lStmt free_stmt = this->generate_memory_free_statements(current_sparse_intersection);
        if (free_stmt.defined()) {
            early_termination_stmts.emplace_back(free_stmt);
        }

        for(LoopNum loop = previous_sparse_intersection + 1; loop <= LoopNum(forall_list.size() - 1); ++loop) {
            if(result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                early_termination_stmts.emplace_back(
                    llir::Store::make(
                        result_tensor.get_length_field(result_tensor.loop_index(loop)),
                        llir::lConst::make(0)
                    )
                );
            }
        }

        llir::lStmt dummy_allocations = this->generate_result_tensor_allocations(previous_sparse_intersection, LoopNum(forall_list.size() - 1), true);
        if (dummy_allocations.defined()) {
            early_termination_stmts.emplace_back(dummy_allocations);
        }

        early_termination_stmts.emplace_back(
            llir::Return::make(get_result_var())
        );

        main_func.body.emplace_back(
            llir::IfElse::make(
                result_tensor.get_length_field(result_tensor.loop_index(current_sparse_intersection)) == llir::lConst::make(0),
                llir::Sequence::make(early_termination_stmts),
                llir::lStmt()
            )
        );
    }

    void StitchAndGenerate::open_files(const std::string &header_suffix, const std::string &source_suffix, const std::string &runnable_macro) {
        const std::string header_name = name + header_suffix;
        header_file.open(output_directory() + "/" + header_name);
        source_file.open(output_directory() + "/" + name + source_suffix);
        internal_assert(header_file.is_open() && source_file.is_open())
            << "Could not open generated files for '" << name << "' in '" << output_directory() << "'";

        header_file << "#pragma once\n\n";
        header_file << "#include \"nacho_kernel_utils.h\"\n\n";
        header_file << "namespace " << name << " {\n\n";
        header_file << "#define __runnable__ " << runnable_macro << "\n\n";
        // Namespace scope, because the entry point names them in its signature. Inside the
        // templated kernels they are shadowed by the template parameters of the same name.
        header_file << "using index_t = int32_t;\n";
        header_file << "using value_t = float;\n\n";

        source_file << "#include \"" << header_name << "\"\n";
    }

    llir::lType StitchAndGenerate::get_result_struct_type() const {
        return llir::Generic_t::make(result_tensor.get_struct_name() + "<index_t, value_t>");
    }

    void StitchAndGenerate::add_tensor_args() {
        for (const auto &[tensor_name, tensor] : operand_tensors) {
            main_func.args.emplace_back(llir::Function::Argument{
                .mutating = false,
                .type = llir::Generic_t::make(tensor.get_struct_name() + "<index_t, value_t>"),
                .name = tensor_name});
        }

        // Arguments are emitted by value, so the result is returned rather than passed in:
        // the body writes lengths into it and allocates its buffers.
        main_func.ret_type = get_result_struct_type();
    }

    llir::lExpr StitchAndGenerate::get_result_var() const {
        return llir::lVar::make(get_result_struct_type(), result_tensor.tensor_name);
    }

    void StitchAndGenerate::close_files() {
        // Prototype in the header, so callers (e.g. the generated bindings) get the
        // signature by including it.
        Printer header_printer(header_file);
        header_file << "\n";
        header_printer.print(llir::Function::declaration(main_func.args, main_func.ret_type, main_func.name));
        header_file << "\n#undef __runnable__\n";
        header_file << "} // namespace " << name << std::endl;
        header_file.close();

        main_func.body.emplace_back(llir::Return::make(get_result_var()));

        source_file << "\nnamespace " << name << " {\n";
        Printer printer(source_file);
        printer.print(llir::Function::make(main_func.generics, main_func.attributes, main_func.args,
                                            main_func.ret_type, main_func.name,
                                            llir::Sequence::make(main_func.body)));
        source_file << "} // namespace " << name << std::endl;
        source_file.close();
    }

    void StitchAndGenerate::decalare_and_initialize_common_variables() {
        // Zero-initialized so the pointer fields are null on paths that return before
        // reaching the allocations (e.g. the empty-result early exit).
        main_func.body.emplace_back(llir::RawStmt::make(
            result_tensor.get_struct_name() + "<index_t, value_t> " + result_tensor.tensor_name + " = {};"));

        main_func.body.emplace_back(
            llir::Declare::make(
                llir::Int_t::make(32),
                "total_work"
        ));

        main_func.body.emplace_back(
            llir::Declare::make(
                llir::Int_t::make(32),
                "per_thread_work"
            )
        );

        main_func.body.emplace_back(
            llir::Declare::make(
                llir::Ptr_t::make(llir::Int_t::make(32)),
                "T_work_offsets"
            )
        );

        for(TensorLevelNum level = BEFORE_FIRST_LEVEL + 1; level < result_tensor.end_tensor_level(); ++level) {
            LoopNum loopNum = result_tensor.tensor_level_to_loop_num(level);
            auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [loopNum](const std::pair<std::string, TensorLowerer> &tensor) {
                return tensor.second.tensor_level_exists(loopNum);
            });
            internal_assert(it != operand_tensors.end()) << "No operand tensor found for loop number " << loopNum.get();
            main_func.body.emplace_back(
                llir::Store::make(
                    result_tensor.get_size_field(level),
                    it->second.get_size_field(it->second.loop_num_to_tensor_level(loopNum))
                )
            );
        }
    }

    // Returns the statement allocating result tensor fields for loops between
    // previous_sparse_intersection and current_sparse_intersection. Callers are responsible
    // for emplacing it into main_func.body.
    llir::lStmt StitchAndGenerate::generate_result_tensor_allocations(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, bool is_dummy_allocation) {
        std::vector<llir::lStmt> stmts;
        llir::lType index_t_ptr = llir::Ptr_t::make(result_tensor.index_t);
        llir::lType value_t_ptr = llir::Ptr_t::make(llir::Generic_t::make("value_t"));
        llir::lExpr index_t_size = llir::lVar::make(sizet_type, "sizeof(index_t)");
        llir::lExpr value_t_size = llir::lVar::make(sizet_type, "sizeof(value_t)");
        llir::lExpr size_multiplier = llir::lConst::make(1); // required if the last level is dense in result
        for(LoopNum loop = previous_sparse_intersection + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop)) {
                TensorLevelNum tensor_level = result_tensor.loop_num_to_tensor_level(loop);
                if(result_tensor.is_sparse(loop)) {
                    if(result_tensor.is_compressed(loop)) {
                        size_multiplier = result_tensor.get_length_field(tensor_level);
                        TensorLevelNum prev_tensor_level = tensor_level - 1;
                        llir::lExpr prev_level_len;
                        if(prev_tensor_level != BEFORE_FIRST_LEVEL) {
                            if(result_tensor.is_sparse(prev_tensor_level)) {
                                prev_level_len = result_tensor.get_length_field(prev_tensor_level);
                            } else {
                                prev_level_len = result_tensor.get_size_field(prev_tensor_level);
                            }
                            stmts.emplace_back(
                                this->generate_single_memory_allocation_statement(
                                    result_tensor.get_offsets_field(tensor_level),
                                    index_t_ptr,
                                    index_t_size * (is_dummy_allocation ? llir::lConst::make(1) : prev_level_len+1),
                                    false
                                )
                            );
                            stmts.emplace_back(
                                this->generate_zero_leading_offset_statement(
                                    result_tensor.get_offsets_field(tensor_level)
                                )
                            );
                        }
                        stmts.emplace_back(
                            this->generate_single_memory_allocation_statement(
                                result_tensor.get_indices_field(tensor_level),
                                index_t_ptr,
                                index_t_size * (is_dummy_allocation ? llir::lConst::make(1) : result_tensor.get_length_field(tensor_level)),
                                false
                            )
                        );
                    } else if (result_tensor.is_singleton(loop)) {
                        size_multiplier = result_tensor.get_length_field(tensor_level);
                        stmts.emplace_back(
                            this->generate_single_memory_allocation_statement(
                                result_tensor.get_indices_field(tensor_level),
                                index_t_ptr,
                                index_t_size * (is_dummy_allocation ? llir::lConst::make(1) : result_tensor.get_length_field(tensor_level)),
                                false
                            )
                        );
                    } else if (result_tensor.is_merged_level(tensor_level)) {
                        TensorIndex index = result_tensor.tensor_level_index(tensor_level);
                        for(auto &idx : index.indices) {
                            stmts.emplace_back(
                                this->generate_single_memory_allocation_statement(
                                    result_tensor.get_indices_field(TensorIndex(idx)),
                                    index_t_ptr,
                                    index_t_size * (is_dummy_allocation ? llir::lConst::make(1) : result_tensor.get_length_field(TensorIndex(idx))),
                                    false
                                )
                            );
                            size_multiplier = result_tensor.get_length_field(TensorIndex(idx));
                        }
                    } else {
                        internal_assert(false) << "Unexpected result tensor level format";
                    }
                } else {
                    size_multiplier = size_multiplier * result_tensor.get_size_field(tensor_level);
                }
                // values can only be allocated once every coordinate level (sparse or dense) has
                // been settled, i.e. once we've processed the last level of the result tensor.
                if (tensor_level == result_tensor.end_tensor_level() - 1) {
                    stmts.emplace_back(
                        this->generate_single_memory_allocation_statement(
                            result_tensor.get_values_field(),
                            value_t_ptr,
                            value_t_size * (is_dummy_allocation ? llir::lConst::make(1) : size_multiplier),
                            false
                        )
                    );
                }
            }
        }
        return stmts.empty() ? llir::lStmt() : llir::Sequence::make(stmts);
    }

} // namespace backend
} // namespace nacho
