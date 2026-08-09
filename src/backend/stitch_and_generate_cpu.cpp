#include "backend/stitch_and_generate.h"

namespace nacho {
namespace backend {

    StitchAndGenerateCPU::StitchAndGenerateCPU(
        std::string name,
        std::map<std::string, TensorLowerer> &operand_tensors,
        TensorLowerer &result_tensor,
        std::vector<CIN> forall_list,
        TensorLowerer &reduced_result_tensor,
        std::vector<LoopNum> &reduction_loops)
        : StitchAndGenerate(name, operand_tensors, result_tensor, std::move(forall_list), reduced_result_tensor, reduction_loops) {
            // Worker kernels run on the host inside tbb::parallel_for lambdas.
            open_files("_cpu.h", "_cpu.cpp", "inline");
            main_func.name = name + "_cpu_i32_f32";

            source_file << "#include \"tbb/parallel_for.h\"\n";
            source_file << "#include \"tbb/info.h\"\n";
            source_file << "#include <cstdlib>\n";

            add_tensor_args();

            main_func.body.emplace_back(
                llir::Declare::make(
                    llir::Int_t::make(32),
                    "num_threads",
                    llir::lFunctionCall::make("tbb::info::default_concurrency", {})
                ));

            decalare_and_initialize_common_variables();
    }

    llir::lStmt StitchAndGenerateCPU::generate_single_memory_allocation_statement(llir::lExpr address, llir::lType pointer_type, llir::lExpr size, bool register_for_free) {
        if (register_for_free) {
            allocated_pointers[get_all_loops_string(current_allocation_level)].push_back(address);
        }
        return llir::Store::make (
            address,
            llir::Cast::make(pointer_type, llir::lFunctionCall::make("malloc", {size}))
        );
    }

    llir::lStmt StitchAndGenerateCPU::generate_zero_range_statement(llir::lExpr field, llir::lExpr byte_count) {
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("memset", {field, llir::lConst::make(0), byte_count})
        );
    }

    void StitchAndGenerateCPU::generate_memory_allocations(
        llir::lType partition_struct, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection,
        bool precompute_kernel_defined, bool compute_kernel_defined
    ) {
        current_allocation_level = current_sparse_intersection;

        if(compute_kernel_defined) {
            if(operand_pos_map_struct_def.defined()) {
                declare_operand_pos_map_once();

                for(auto &[tensor_name, tensor] : operand_tensors) {
                    auto field =    std::find_if(operand_pos_map_struct_def.as<llir::Struct_t>()->fields.begin(), operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end(),
                        [previous_sparse_intersection, &tensor](const std::pair<std::string, llir::lType> &field) {
                            return field.first == tensor.get_iterator_suffix(tensor.loop_index(previous_sparse_intersection));
                        });
                    if(field != operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end()) {
                            main_func.body.emplace_back(
                                generate_single_memory_allocation_statement(
                                    llir::lFieldAccess::make(llir::lVar::make(operand_pos_map_struct_def, get_operand_pos_map_var_name()), field->first),
                                    field->second,
                                    llir::lVar::make(sizet_type, "sizeof(int32_t)") * result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)),
                                    true
                                )
                            );
                    }
                }
            }

            // Indexed by position in the previous level of the result, plus a terminator:
            // generate_work_offsets_scan() prefix-sums [1, length] and reads [length].
            main_func.body.emplace_back(
                generate_single_memory_allocation_statement(
                    work_offsets_var,
                    llir::Ptr_t::make(llir::Int_t::make(32)),
                    llir::lVar::make(sizet_type, "sizeof(int32_t)")
                        * (result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection))
                           + llir::lConst::make(1)),
                    true
                )
            );
        }


        main_func.body.emplace_back(
            llir::Declare::make(get_partition_var_type(current_sparse_intersection), get_partition_var_name(current_sparse_intersection))
        );

        llir::lExpr partitions_var = get_partition_var(current_sparse_intersection);
        for(auto &field : partition_struct.as<llir::Struct_t>()->fields) {
            main_func.body.emplace_back(
                generate_single_memory_allocation_statement(
                    llir::lFieldAccess::make(partitions_var, field.first),
                    field.second,
                    llir::lVar::make(sizet_type, "sizeof(int32_t)") * num_threads_var,
                    true
                )
            );
        }

        if(precompute_kernel_defined) {
            main_func.body.emplace_back(
                llir::Declare::make(get_count_offset_var_type(), get_count_offset_var_name(current_sparse_intersection))
            );
            for (LoopNum loop = BEFORE_FIRST_LOOP + 1; loop <= current_sparse_intersection; ++loop) {
                if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                    main_func.body.emplace_back(
                        generate_single_memory_allocation_statement(
                            get_count_offsets_field(loop, current_sparse_intersection),
                            llir::Ptr_t::make(llir::Int_t::make(32)),
                            llir::lVar::make(sizet_type, "sizeof(int32_t)") * num_threads_var,
                            true
                        )
                    );
                }
            }
        }
    }

    llir::lExpr StitchAndGenerateCPU::wrap_kernel_with_backend_specific_call(llir::lStmt kernel, LoopNum current_sparse_intersection) {
        std::vector<llir::lExpr> args;
        for (auto &param : kernel.as<llir::Function>()->args) {
            args.emplace_back(convert_param_to_arg(param, current_sparse_intersection));
        }
        return llir::lFunctionCall::make(
            "tbb::parallel_for",
            {
                llir::lConst::make(0),
                num_threads_var,
                llir::lLambda::make("&", {{index_t, "thread_id"}}, llir::BaseExpr::make(llir::lFunctionCall::make(
                    kernel.as<llir::Function>()->name,
                    args
                )))
            }
        );
    }

    void StitchAndGenerateCPU::generate_prefix_sum_calls(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) {
        std::vector<llir::lStmt> prefix_sum_loop_stmts;
        llir::lExpr loop_var = llir::lVar::make(llir::Int_t::make(32),"t");
        for (LoopNum loop = BEFORE_FIRST_LOOP + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                std::string prefix_sum_loop_var_name = "prefix_sum_" + get_all_loops_string(current_sparse_intersection) + "_" + std::to_string(loop.get());
                std::string temp_loop_var_name = "temp_" + get_all_loops_string(current_sparse_intersection) + "_" + std::to_string(loop.get());
                main_func.body.emplace_back(
                    llir::Declare::make(
                        llir::Int_t::make(32),
                        prefix_sum_loop_var_name,
                        llir::lConst::make(0)
                    )
                );
                prefix_sum_loop_stmts.emplace_back(
                    llir::Declare::make(
                        llir::Int_t::make(32),
                        temp_loop_var_name,
                        get_count_offsets_field(loop, current_sparse_intersection)[loop_var]
                    )
                );
                prefix_sum_loop_stmts.emplace_back(
                    llir::Store::make(
                        get_count_offsets_field(loop, current_sparse_intersection)[loop_var],
                        llir::lVar::make(llir::Int_t::make(32), prefix_sum_loop_var_name)
                    )
                );
                prefix_sum_loop_stmts.emplace_back(
                    llir::Accumulate::make(
                        llir::lVar::make(llir::Int_t::make(32), prefix_sum_loop_var_name),
                        llir::lVar::make(llir::Int_t::make(32), temp_loop_var_name)
                    )
                );
            }
        }
        main_func.body.emplace_back(
            llir::For::make(
                llir::Int_t::make(32), "t",
                llir::lConst::make(0),
                loop_var < num_threads_var,
                llir::lConst::make(1),
                llir::Sequence::make(prefix_sum_loop_stmts)
            )
        );

        for (LoopNum loop = previous_sparse_intersection + 1; loop <= current_sparse_intersection; ++loop) {
            if(result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                // A merged level stores a length per flattened dimension, all equal to the
                // number of coordinate tuples this level produced.
                for (const auto &index : result_tensor.stored_indices(result_tensor.loop_index(loop))) {
                    main_func.body.emplace_back(
                        llir::Store::make(
                            result_tensor.get_length_field(index),
                            llir::lVar::make(llir::Int_t::make(32), "prefix_sum_" + get_all_loops_string(current_sparse_intersection) + "_" + std::to_string(loop.get()))
                        )
                    );
                }
            }
        }
    }

    void StitchAndGenerateCPU::generate_work_offsets_scan(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) {
        main_func.body.emplace_back(
            llir::Store::make(work_offsets_var[llir::lConst::make(0)], llir::lConst::make(0))
        );
        llir::lExpr length = result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection));
        llir::lExpr loop_var = llir::lVar::make(llir::Int_t::make(32), "t");
        main_func.body.emplace_back(
            llir::For::make(
                llir::Int_t::make(32), "t",
                llir::lConst::make(1),
                loop_var <= length,
                llir::lConst::make(1),
                llir::Accumulate::make(
                    work_offsets_var[loop_var],
                    work_offsets_var[loop_var - llir::lConst::make(1)]
                )
            )
        );
    }

    llir::lStmt StitchAndGenerateCPU::generate_total_work_from_offsets_statement(llir::lExpr index_expr) {
        return llir::Store::make(
            llir::lVar::make(llir::Int_t::make(32), "total_work"),
            work_offsets_var[index_expr]
        );
    }

    llir::lStmt StitchAndGenerateCPU::generate_memory_free_statements(LoopNum sparse_intersection) {
        auto it = allocated_pointers.find(get_all_loops_string(sparse_intersection));
        if (it == allocated_pointers.end()) {
            return llir::lStmt();
        }

        std::vector<llir::lStmt> free_stmts;
        for (auto &address : it->second) {
            free_stmts.emplace_back(
                llir::BaseExpr::make(llir::lFunctionCall::make("free", {address}))
            );
        }
        return llir::Sequence::make(free_stmts);
    }


} // namespace backend
} // namespace nacho
