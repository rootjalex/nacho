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
        std::map<std::string, TensorLowerer> included_tensors,
        bool operands_intersected_at_current
    ) {
        if(compute_kernel.defined()) {
            llir::lStmt result_tensor_allocations = this->generate_result_tensor_allocations(result_tensor, previous_previous_sparse_intersection, previous_sparse_intersection);
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
            // Only an intersected level can produce nothing from non-empty operands.
            if(operands_intersected_at_current) {
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
            llir::lStmt result_tensor_allocations = this->generate_result_tensor_allocations(result_tensor, previous_sparse_intersection, LoopNum(forall_list.size() - 1));
            if (result_tensor_allocations.defined()) {
                main_func.body.emplace_back(result_tensor_allocations);
            }

            // An append reduction accumulates into the reduced result from inside the
            // compute kernel, so its values have to exist and be zero before the call.
            if (has_reduction() && !stitch_with_scatter_reduction) {
                main_func.body.emplace_back(generate_append_reduction_allocation());
            }

            this->stitch_compute_kernel_call(
                compute_kernel, LoopNum(forall_list.size() - 1)
            );

            llir::lStmt free_stmt = this->generate_memory_free_statements(LoopNum(forall_list.size() - 1));
            if (free_stmt.defined()) {
                main_func.body.emplace_back(free_stmt);
            }

            // A scatter reduction leaves result_tensor holding one entry per reduced
            // coordinate; contracting those onto the output coordinates is a stage of its
            // own, after every compute kernel has run.
            if (stitch_with_scatter_reduction) {
                this->stitch_scatter_reduction();
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
            return work_offsets_var();
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

        // Anything the compute kernels would have filled from here on is empty. Both the
        // intermediate and, when reducing, the tensor the entry point returns need lengths
        // of zero and buffers a consumer can still walk.
        auto empty_out = [&](const TensorLowerer &tensor, LoopNum from) {
            for(LoopNum loop = from + 1; loop <= LoopNum(forall_list.size() - 1); ++loop) {
                if(tensor.tensor_level_exists(loop) && tensor.is_sparse(loop)) {
                    // A merged level stores a length per flattened dimension; zero them all.
                    for (const auto &index : tensor.stored_indices(tensor.loop_index(loop))) {
                        early_termination_stmts.emplace_back(
                            llir::Store::make(tensor.get_length_field(index), llir::lConst::make(0))
                        );
                    }
                }
            }
            llir::lStmt dummy_allocations = this->generate_result_tensor_allocations(
                tensor, from, LoopNum(forall_list.size() - 1), true);
            if (dummy_allocations.defined()) {
                early_termination_stmts.emplace_back(dummy_allocations);
            }
        };

        empty_out(result_tensor, previous_sparse_intersection);
        if (has_reduction()) {
            // The contraction never runs on this path, so the output is built here instead,
            // over all of its levels rather than only those past this intersection.
            empty_out(reduced_result_tensor, BEFORE_FIRST_LOOP);
        }

        llir::lStmt long_lived_frees = generate_long_lived_free_statements();
        if (long_lived_frees.defined()) {
            early_termination_stmts.emplace_back(long_lived_frees);
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
        return llir::Generic_t::make(output_tensor().get_struct_name() + "<index_t, value_t>");
    }

    llir::lStmt StitchAndGenerate::generate_long_lived_free_statements() {
        if (long_lived_allocations.empty()) {
            return llir::lStmt();
        }
        std::vector<llir::lStmt> free_stmts;
        for (const llir::lExpr &address : long_lived_allocations) {
            free_stmts.emplace_back(generate_free_statement(address));
        }
        return llir::Sequence::make(free_stmts);
    }

    void StitchAndGenerate::resolve_operand_ordering(std::vector<std::string> requested) {
        if (requested.empty()) {
            for (const auto &[tensor_name, tensor] : operand_tensors) {
                operand_ordering.push_back(tensor_name);
            }
            return;
        }

        internal_assert(requested.size() == operand_tensors.size())
            << "Kernel '" << name << "' lists " << requested.size()
            << " operands in its ordering but the expression has " << operand_tensors.size();
        std::set<std::string> seen;
        for (const std::string &tensor_name : requested) {
            internal_assert(operand_tensors.count(tensor_name) > 0)
                << "Kernel '" << name << "' orders an operand '" << tensor_name
                << "' that its expression does not use";
            internal_assert(seen.insert(tensor_name).second)
                << "Kernel '" << name << "' lists operand '" << tensor_name << "' twice in its ordering";
        }
        operand_ordering = std::move(requested);
    }

    void StitchAndGenerate::add_tensor_args() {
        for (const std::string &tensor_name : operand_ordering) {
            const TensorLowerer &tensor = operand_tensors.at(tensor_name);
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
        return llir::lVar::make(get_result_struct_type(), output_tensor().tensor_name);
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

        llir::lStmt long_lived_frees = generate_long_lived_free_statements();
        if (long_lived_frees.defined()) {
            main_func.body.emplace_back(long_lived_frees);
        }

        main_func.body.emplace_back(llir::Return::make(get_result_var()));

        source_file << "\nnamespace " << name << " {\n";
        Printer printer(source_file);
        printer.print(llir::Function::make(main_func.generics, main_func.attributes, main_func.args,
                                            main_func.ret_type, main_func.name,
                                            llir::Sequence::make(main_func.body)));
        source_file << "} // namespace " << name << std::endl;
        source_file.close();
    }

    void StitchAndGenerate::stitch_scatter_reduction() {
        internal_assert(false)
            << "Kernel '" << name << "': scatter reduction is only supported on GPU";
    }

    llir::lExpr StitchAndGenerate::values_extent(const TensorLowerer &tensor) const {
        llir::lExpr extent = llir::lConst::make(1);
        for (TensorLevelNum level = BEFORE_FIRST_LEVEL + 1; level < tensor.end_tensor_level(); ++level) {
            if (tensor.is_sparse(level)) {
                extent = tensor.get_length_field(level);
            } else {
                extent = extent * tensor.get_size_field(level);
            }
        }
        return extent;
    }

    llir::lStmt StitchAndGenerate::generate_append_reduction_allocation() {
        llir::lExpr byte_count = llir::lVar::make(sizet_type, "sizeof(value_t)") * values_extent(reduced_result_tensor);
        return llir::Sequence::make({
            generate_single_memory_allocation_statement(
                reduced_result_tensor.get_values_field(),
                llir::Ptr_t::make(llir::Generic_t::make("value_t")),
                byte_count,
                false
            ),
            generate_zero_range_statement(reduced_result_tensor.get_values_field(), byte_count),
        });
    }

    void StitchAndGenerate::declare_operand_pos_map_once() {
        if (operand_pos_map_declared) {
            return;
        }
        operand_pos_map_declared = true;
        main_func.body.emplace_back(
            llir::Declare::make(get_operand_pos_map_type(), get_operand_pos_map_var_name())
        );
    }

    void StitchAndGenerate::declare_and_size_tensor(const TensorLowerer &tensor) {
        // Zero-initialized so the pointer fields are null on paths that return before
        // reaching the allocations (e.g. the empty-result early exit).
        main_func.body.emplace_back(llir::RawStmt::make(
            tensor.get_struct_name() + "<index_t, value_t> " + tensor.tensor_name + " = {};"));

        for(TensorLevelNum level = BEFORE_FIRST_LEVEL + 1; level < tensor.end_tensor_level(); ++level) {
            LoopNum loopNum = tensor.tensor_level_to_loop_num(level);
            auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [loopNum](const std::pair<std::string, TensorLowerer> &operand) {
                return operand.second.tensor_level_exists(loopNum);
            });
            internal_assert(it != operand_tensors.end()) << "No operand tensor found for loop number " << loopNum.get();
            TensorLevelNum operand_level = it->second.loop_num_to_tensor_level(loopNum);

            if (tensor.is_merged_level(level)) {
                // A merged level keeps a size per dimension it flattens. Result and operand
                // flatten the same dimensions, so copy them pairwise.
                std::vector<TensorIndex> result_indices = tensor.stored_indices(level);
                std::vector<TensorIndex> operand_indices = it->second.stored_indices(operand_level);
                internal_assert(result_indices.size() == operand_indices.size())
                    << "Result and operand disagree on how many dimensions loop " << loopNum.get() << " covers";
                for (size_t i = 0; i < result_indices.size(); ++i) {
                    main_func.body.emplace_back(
                        llir::Store::make(
                            tensor.get_size_field(result_indices[i]),
                            it->second.get_size_field(operand_indices[i])
                        )
                    );
                }
            } else {
                main_func.body.emplace_back(
                    llir::Store::make(
                        tensor.get_size_field(level),
                        it->second.get_size_field(operand_level)
                    )
                );
            }
        }
    }

    void StitchAndGenerate::decalare_and_initialize_common_variables() {
        declare_and_size_tensor(result_tensor);
        // A reduction fills result_tensor (`<Z>_temp`) and then contracts it into this one,
        // which is what the entry point returns.
        if (has_reduction()) {
            declare_and_size_tensor(reduced_result_tensor);
        }

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

    }

    // Returns the statement allocating `tensor`'s fields for loops between
    // previous_sparse_intersection and current_sparse_intersection. Callers are responsible
    // for emplacing it into main_func.body.
    //
    // is_dummy_allocation covers the path where the tensor turned out empty. Coordinates
    // and values shrink to one element because the levels' lengths are all zero and nothing
    // indexes them, but the offsets keep their full extent: a consumer still walks all of
    // them, so they are allocated in full and zeroed to read back as empty everywhere.
    llir::lStmt StitchAndGenerate::generate_result_tensor_allocations(const TensorLowerer &tensor, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection, bool is_dummy_allocation) {
        std::vector<llir::lStmt> stmts;
        llir::lType index_t_ptr = llir::Ptr_t::make(tensor.index_t);
        llir::lType value_t_ptr = llir::Ptr_t::make(llir::Generic_t::make("value_t"));
        llir::lExpr index_t_size = llir::lVar::make(sizet_type, "sizeof(index_t)");
        llir::lExpr value_t_size = llir::lVar::make(sizet_type, "sizeof(value_t)");
        llir::lExpr size_multiplier = llir::lConst::make(1); // required if the last level is dense in result
        for(LoopNum loop = previous_sparse_intersection + 1; loop <= current_sparse_intersection; ++loop) {
            if (tensor.tensor_level_exists(loop)) {
                TensorLevelNum tensor_level = tensor.loop_num_to_tensor_level(loop);
                if(tensor.is_sparse(loop)) {
                    if(tensor.is_compressed(loop)) {
                        size_multiplier = tensor.get_length_field(tensor_level);
                        TensorLevelNum prev_tensor_level = tensor_level - 1;
                        llir::lExpr prev_level_len;
                        if(prev_tensor_level != BEFORE_FIRST_LEVEL) {
                            if(tensor.is_sparse(prev_tensor_level)) {
                                prev_level_len = tensor.get_length_field(prev_tensor_level);
                            } else {
                                prev_level_len = tensor.get_size_field(prev_tensor_level);
                            }
                            llir::lExpr offsets_bytes = index_t_size * (prev_level_len + 1);
                            stmts.emplace_back(
                                this->generate_single_memory_allocation_statement(
                                    tensor.get_offsets_field(tensor_level),
                                    index_t_ptr,
                                    offsets_bytes,
                                    false
                                )
                            );
                            stmts.emplace_back(
                                this->generate_zero_range_statement(
                                    tensor.get_offsets_field(tensor_level),
                                    is_dummy_allocation ? offsets_bytes : index_t_size
                                )
                            );
                        }
                        stmts.emplace_back(
                            this->generate_single_memory_allocation_statement(
                                tensor.get_indices_field(tensor_level),
                                index_t_ptr,
                                index_t_size * (is_dummy_allocation ? llir::lConst::make(1) : tensor.get_length_field(tensor_level)),
                                false
                            )
                        );
                    } else if (tensor.is_singleton(loop)) {
                        size_multiplier = tensor.get_length_field(tensor_level);
                        stmts.emplace_back(
                            this->generate_single_memory_allocation_statement(
                                tensor.get_indices_field(tensor_level),
                                index_t_ptr,
                                index_t_size * (is_dummy_allocation ? llir::lConst::make(1) : tensor.get_length_field(tensor_level)),
                                false
                            )
                        );
                    } else if (tensor.is_merged_level(tensor_level)) {
                        TensorIndex index = tensor.tensor_level_index(tensor_level);
                        for(auto &idx : index.indices) {
                            stmts.emplace_back(
                                this->generate_single_memory_allocation_statement(
                                    tensor.get_indices_field(TensorIndex(idx)),
                                    index_t_ptr,
                                    index_t_size * (is_dummy_allocation ? llir::lConst::make(1) : tensor.get_length_field(TensorIndex(idx))),
                                    false
                                )
                            );
                            size_multiplier = tensor.get_length_field(TensorIndex(idx));
                        }
                    } else {
                        internal_assert(false) << "Unexpected result tensor level format";
                    }
                } else {
                    size_multiplier = size_multiplier * tensor.get_size_field(tensor_level);
                }
                // values can only be allocated once every coordinate level (sparse or dense) has
                // been settled, i.e. once we've processed the last level of the result tensor.
                if (tensor_level == tensor.end_tensor_level() - 1) {
                    stmts.emplace_back(
                        this->generate_single_memory_allocation_statement(
                            tensor.get_values_field(),
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
