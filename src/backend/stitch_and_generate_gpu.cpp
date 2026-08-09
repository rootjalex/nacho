#include "backend/stitch_and_generate.h"

namespace nacho {
namespace backend {

    StitchAndGenerateGPU::StitchAndGenerateGPU(
        std::string name,
        std::map<std::string, TensorLowerer> &operand_tensors,
        TensorLowerer &result_tensor,
        std::vector<CIN> forall_list,
        TensorLowerer &reduced_result_tensor)
        : StitchAndGenerate(name, operand_tensors, result_tensor, std::move(forall_list), reduced_result_tensor) {
            // Worker kernels are only ever called from the __global__ wrappers.
            open_files("_gpu.h", "_gpu.cu", "__device__ inline");
            main_func.name = name + "_gpu_i32_f32";

            add_tensor_args();

            main_func.args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = llir::Int_t::make(32), .name = "num_blocks"});
            main_func.args.emplace_back(llir::Function::Argument{
                .mutating = false, .type = llir::Int_t::make(32), .name = "threads_per_block"});

            main_func.body.emplace_back(llir::Declare::make(
                llir::Generic_t::make("cudaStream_t"),
                "stream",
                llir::lConst::make(0))
            );
            main_func.body.emplace_back(
                llir::Declare::make(
                    llir::Int_t::make(32),
                    "num_threads",
                    llir::lBinOp::make(llir::lBinOp::Mul,
                        llir::lVar::make(llir::Int_t::make(32), "num_blocks"),
                        llir::lVar::make(llir::Int_t::make(32), "threads_per_block")
                        )
                ));
            decalare_and_initialize_common_variables();

            main_func.body.emplace_back(
                    llir::Declare::make(
                        llir::Generic_t::make("size_t"),
                        "cub_bytes"
                    )
                );
    }

    llir::lStmt StitchAndGenerateGPU::generate_single_memory_allocation_statement(llir::lExpr address, llir::lType pointer_type, llir::lExpr size, bool register_for_free) {
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("cudaMallocAsync", {
                // cudaMallocAsync takes void**; the typed fields need an explicit cast.
                llir::Cast::make(llir::Generic_t::make("void**"), llir::lAddress::make(address)),
                size,
                cuda_stream_var
            })
        );
    }

    llir::lStmt StitchAndGenerateGPU::generate_zero_leading_offset_statement(llir::lExpr offsets_field) {
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("cudaMemsetAsync", {
                offsets_field,
                llir::lConst::make(0),
                llir::lVar::make(sizet_type, "sizeof(index_t)"),
                cuda_stream_var
            })
        );
    }

    void StitchAndGenerateGPU::generate_memory_allocations(
        llir::lType partition_struct, LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection,
        bool precompute_kernel_defined, bool compute_kernel_defined
    ) {
        // CUB deduces its iterator types from the arguments even when d_temp_storage is null
        // and it only reports the temp-storage size, so the sizing queries below pass a typed
        // null pointer matching the buffers the real scans operate on.
        llir::lExpr null_scan_iterator =
            llir::Cast::make(llir::Generic_t::make("int32_t*"), llir::lConst::make(0));

        llir::lExpr size_expr = llir::lConst::make(0);
        if(compute_kernel_defined) {
            if(operand_pos_map_struct_def.defined()) {
                for(auto &[tensor_name, tensor] : operand_tensors) {
                    auto field = std::find_if(operand_pos_map_struct_def.as<llir::Struct_t>()->fields.begin(), operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end(),
                        [previous_sparse_intersection, &tensor](const std::pair<std::string, llir::lType> &field) {
                            return field.first == tensor.get_iterator_suffix(tensor.loop_index(previous_sparse_intersection));
                        });
                    if(field != operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end()) {
                        size_expr =  size_expr + result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection));
                    }
                }
            }

        
            size_expr = size_expr + result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)) + llir::lConst::make(1);
        }

        main_func.body.emplace_back(
            llir::BaseExpr::make(
                llir::lFunctionCall::make("cub::DeviceScan::ExclusiveSum", {
                    llir::lConst::make(0),
                    llir::lVar::make(sizet_type, "cub_bytes"),
                    null_scan_iterator, null_scan_iterator,
                    num_threads_var,
                    cuda_stream_var
                })
            )
        );

        if (compute_kernel_defined) {
            std::string work_offsets_scan_bytes_name = "cub_bytes_work_offsets_scan_" + get_all_loops_string(current_sparse_intersection);
            main_func.body.emplace_back(
                llir::Declare::make(llir::Generic_t::make("size_t"), work_offsets_scan_bytes_name)
            );
            main_func.body.emplace_back(
                llir::BaseExpr::make(
                    llir::lFunctionCall::make("cub::DeviceScan::InclusiveSum", {
                        llir::lConst::make(0),
                        llir::lVar::make(sizet_type, work_offsets_scan_bytes_name),
                        null_scan_iterator, null_scan_iterator,
                        result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)),
                        cuda_stream_var
                    })
                )
            );
            main_func.body.emplace_back(
                llir::Store::make(
                    llir::lVar::make(sizet_type, "cub_bytes"),
                    llir::lBinOp::make(llir::lBinOp::Max, llir::lVar::make(sizet_type, "cub_bytes"), llir::lVar::make(sizet_type, work_offsets_scan_bytes_name))
                )
            );
        }

        int num_mem_block_fields = static_cast<int>(partition_struct.as<llir::Struct_t>()->fields.size());
        for (LoopNum loop = BEFORE_FIRST_LOOP + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                num_mem_block_fields++;
            }
        }

        size_expr = size_expr + llir::lConst::make(num_mem_block_fields) * num_threads_var;

        std::string mem_block_name = "mem_block_" + get_all_loops_string(current_sparse_intersection);
        main_func.body.emplace_back(
            llir::Declare::make(
                llir::Generic_t::make("void*"),
                mem_block_name,
                llir::lConst::make(0)
            )
        );

        // call cudaMallocAsync to allocate buffer chunk
        main_func.body.emplace_back(
            generate_single_memory_allocation_statement(
                llir::lVar::make(llir::Generic_t::make("void*"), mem_block_name),
                llir::Ptr_t::make(void_type),
                (llir::lVar::make(sizet_type, "sizeof(int)") * size_expr)
                + llir::lVar::make(sizet_type, "cub_bytes"),
                true
            )
        );

        main_func.body.emplace_back(
            llir::Declare::make(llir::Int_t::make(32), "offset_" + get_all_loops_string(current_sparse_intersection), llir::lConst::make(0))
        );
        std::string mem_block_base_name = "mem_block_base_" + get_all_loops_string(current_sparse_intersection);
        main_func.body.emplace_back(
            llir::Declare::make(llir::Generic_t::make("char*"), mem_block_base_name,
                llir::Cast::make(llir::Generic_t::make("char*"), llir::lVar::make(llir::Generic_t::make("void*"), mem_block_name)))
        );
        auto offset_var = llir::lVar::make(llir::Int_t::make(32), "offset_" + get_all_loops_string(current_sparse_intersection));
        auto mem_block_base_var = llir::lVar::make(llir::Generic_t::make("char*"), mem_block_base_name);

        // Byte-addressed slice of the shared block, cast back to the field's pointer type.
        auto mem_block_slice = [&](llir::lType pointer_type) {
            return llir::Cast::make(pointer_type, mem_block_base_var + offset_var);
        };

        if(compute_kernel_defined) {
            if(operand_pos_map_struct_def.defined()) {
                main_func.body.emplace_back(
                    llir::Declare::make(get_operand_pos_map_type(), get_operand_pos_map_var_name())
                );

                for(auto &[tensor_name, tensor] : operand_tensors) {
                    auto field = std::find_if(operand_pos_map_struct_def.as<llir::Struct_t>()->fields.begin(), operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end(),
                        [previous_sparse_intersection, &tensor](const std::pair<std::string, llir::lType> &field) {
                            return field.first == tensor.get_iterator_suffix(tensor.loop_index(previous_sparse_intersection));
                        });
                    if(field != operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end()) {
                        main_func.body.emplace_back(
                            llir::Store::make(
                                llir::lFieldAccess::make(llir::lVar::make(operand_pos_map_struct_def, get_operand_pos_map_var_name()), field->first),
                                mem_block_slice(field->second)
                            )
                        );
                        main_func.body.emplace_back(
                            llir::Accumulate::make(offset_var, llir::lVar::make(sizet_type, "sizeof(int32_t)") * result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)))
                        );
                    }
                }
            }

            main_func.body.emplace_back(
                llir::Store::make(
                    work_offsets_var,
                    mem_block_slice(llir::Ptr_t::make(llir::Int_t::make(32)))
                )
            );
            main_func.body.emplace_back(
                llir::Accumulate::make(offset_var, llir::lVar::make(sizet_type, "sizeof(int32_t)") * (result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)) + llir::lConst::make(1)))
            );
        }



        main_func.body.emplace_back(
            llir::Declare::make(get_partition_var_type(current_sparse_intersection), get_partition_var_name(current_sparse_intersection))
        );

        llir::lExpr partitions_var = get_partition_var(current_sparse_intersection);
        for(auto &field : partition_struct.as<llir::Struct_t>()->fields) {
            main_func.body.emplace_back(
                llir::Store::make(
                    llir::lFieldAccess::make(partitions_var, field.first),
                    mem_block_slice(field.second)
                )
            );
            main_func.body.emplace_back(
                llir::Accumulate::make(offset_var, (llir::lVar::make(sizet_type, "sizeof(int32_t)") * num_threads_var))
            );
        }

        if(precompute_kernel_defined) {
            main_func.body.emplace_back(
                llir::Declare::make(get_count_offset_var_type(), get_count_offset_var_name(current_sparse_intersection))
            );
            for (LoopNum loop = BEFORE_FIRST_LOOP + 1; loop <= current_sparse_intersection; ++loop) {
                if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                    main_func.body.emplace_back(
                        llir::Store::make(
                            get_count_offsets_field(loop, current_sparse_intersection),
                            mem_block_slice(llir::Ptr_t::make(llir::Int_t::make(32)))
                        )
                    );
                    main_func.body.emplace_back(
                        llir::Accumulate::make(offset_var, (llir::lVar::make(sizet_type, "sizeof(int32_t)") * num_threads_var))
                    );
                }
            }
        }

        main_func.body.emplace_back(
            llir::Declare::make(llir::Generic_t::make("void*"), "cub_temp_storage_" + get_all_loops_string(current_sparse_intersection),
                mem_block_slice(llir::Generic_t::make("void*")))
        );
    }


    llir::lExpr StitchAndGenerateGPU::wrap_kernel_with_backend_specific_call(llir::lStmt kernel, LoopNum current_sparse_intersection) {
        FuncDecl new_kernel_func;
        new_kernel_func.generics = {"index_t", "value_t"};
        new_kernel_func.attributes = {llir::Function::Attribute::global};
        new_kernel_func.ret_type = void_type;
        new_kernel_func.name = kernel.as<llir::Function>()->name;
        std::vector<llir::lExpr> kernel_call_args;
        std::vector<llir::lExpr> new_kernel_call_args;
        for(auto & param : kernel.as<llir::Function>()->args) {
            kernel_call_args.emplace_back(llir::lVar::make(param.type, param.name));
            if(param.name != "thread_id"){
                new_kernel_func.args.emplace_back(llir::Function::Argument{
                    .mutating = param.mutating, .type = param.type, .name = param.name});
                new_kernel_call_args.emplace_back(convert_param_to_arg(param, current_sparse_intersection));
            }
        }
        // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        new_kernel_func.body.emplace_back(
            llir::Declare::make(
                llir::Int_t::make(32),
                "thread_id",
                (llir::lVar::make(index_t, "blockIdx.x") * llir::lVar::make(index_t, "blockDim.x")) + llir::lVar::make(index_t, "threadIdx.x")
            )
        );

        new_kernel_func.body.emplace_back(
            llir::BaseExpr::make(llir::lFunctionCall::make(kernel.as<llir::Function>()->name, kernel_call_args))
        );

        add_to_header_file(llir::Function::make(new_kernel_func.generics, new_kernel_func.attributes, new_kernel_func.args,
                                            new_kernel_func.ret_type, new_kernel_func.name,
                                            llir::Sequence::make(new_kernel_func.body)));

        return llir::lFunctionCall::make(new_kernel_func.name+"<int32_t, float><<<num_blocks, threads_per_block, 0, stream>>>", new_kernel_call_args);
    }

    void StitchAndGenerateGPU::generate_prefix_sum_calls(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) {

        for (LoopNum loop = previous_sparse_intersection + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                main_func.body.emplace_back(
                    llir::Declare::make(
                        llir::Int_t::make(32),
                        "temp_last_value_" + get_all_loops_string(current_sparse_intersection) + "_" +std::to_string(loop.get())
                    )
                );
                main_func.body.emplace_back(
                    llir::BaseExpr::make(
                        llir::lFunctionCall::make("cudaMemcpyAsync", {
                            llir::lAddress::make(llir::lVar::make(llir::Int_t::make(32), "temp_last_value_" + get_all_loops_string(current_sparse_intersection) + "_" +std::to_string(loop.get()))),
                            get_count_offsets_field(loop, current_sparse_intersection) + (num_threads_var - 1),
                            llir::lVar::make(sizet_type, "sizeof(int32_t)"),
                            device_to_host,
                            cuda_stream_var
                        })
                    )
                );
            }
        }

        for (LoopNum loop = BEFORE_FIRST_LOOP + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                main_func.body.emplace_back(
                    llir::BaseExpr::make(
                        llir::lFunctionCall::make("cub::DeviceScan::ExclusiveSum", {
                            llir::lVar::make(void_type, "cub_temp_storage_" + get_all_loops_string(current_sparse_intersection)),
                            llir::lVar::make(sizet_type, "cub_bytes"),
                            get_count_offsets_field(loop, current_sparse_intersection),
                            get_count_offsets_field(loop, current_sparse_intersection),
                            num_threads_var,
                            cuda_stream_var
                        })
                    )
                );
            }
        }

        for (LoopNum loop = previous_sparse_intersection + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                main_func.body.emplace_back(
                    llir::BaseExpr::make(
                        llir::lFunctionCall::make("cudaMemcpyAsync", {
                            llir::lAddress::make(result_tensor.get_length_field(result_tensor.loop_index(loop))),
                            get_count_offsets_field(loop, current_sparse_intersection) + (num_threads_var - 1),
                            llir::lVar::make(sizet_type, "sizeof(int32_t)"),
                            device_to_host,
                            cuda_stream_var
                        })
                    )
                );
            }
        }
        // The host reads/writes the length fields right below (temp_last_value_* accumulate),
        // so the preceding cudaMemcpyAsync's must have actually landed by then.
        main_func.body.emplace_back(
            llir::BaseExpr::make(
                llir::lFunctionCall::make("cudaStreamSynchronize", {cuda_stream_var})
            )
        );

        for (LoopNum loop = previous_sparse_intersection + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                main_func.body.emplace_back(
                    llir::Accumulate::make(
                        result_tensor.get_length_field(result_tensor.loop_index(loop)),
                        llir::lVar::make(llir::Int_t::make(32), "temp_last_value_" + get_all_loops_string(current_sparse_intersection) + "_" +std::to_string(loop.get()))
                    )
                );

                // A merged level stores a length per flattened dimension, all equal. Only
                // the first was brought back from the device; mirror it into the rest.
                std::vector<TensorIndex> indices = result_tensor.stored_indices(result_tensor.loop_index(loop));
                for (size_t i = 1; i < indices.size(); ++i) {
                    main_func.body.emplace_back(
                        llir::Store::make(
                            result_tensor.get_length_field(indices[i]),
                            result_tensor.get_length_field(indices[0])
                        )
                    );
                }
            }
        }
    }

    void StitchAndGenerateGPU::generate_work_offsets_scan(LoopNum previous_sparse_intersection, LoopNum current_sparse_intersection) {
        main_func.body.emplace_back(
            llir::BaseExpr::make(
                llir::lFunctionCall::make("cudaMemsetAsync", {
                    work_offsets_var,
                    llir::lConst::make(0),
                    llir::lVar::make(sizet_type, "sizeof(int32_t)"),
                    cuda_stream_var
                })
            )
        );
        main_func.body.emplace_back(
            llir::BaseExpr::make(
                llir::lFunctionCall::make("cub::DeviceScan::InclusiveSum", {
                    llir::lVar::make(void_type, "cub_temp_storage_" + get_all_loops_string(current_sparse_intersection)),
                    llir::lVar::make(sizet_type, "cub_bytes"),
                    work_offsets_var + llir::lConst::make(1),
                    work_offsets_var + llir::lConst::make(1),
                    result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)),
                    cuda_stream_var
                })
            )
        );
    }

    llir::lStmt StitchAndGenerateGPU::generate_memory_free_statements(LoopNum sparse_intersection) {
        std::string mem_block_name = "mem_block_" + get_all_loops_string(sparse_intersection);
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("cudaFreeAsync", {
                llir::lVar::make(llir::Generic_t::make("void*"), mem_block_name),
                cuda_stream_var
            })
        );
    }

    llir::lStmt StitchAndGenerateGPU::generate_total_work_from_offsets_statement(llir::lExpr index_expr) {
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("cudaMemcpyAsync", {
                llir::lAddress::make(llir::lVar::make(llir::Int_t::make(32), "total_work")),
                work_offsets_var + index_expr,
                llir::lVar::make(sizet_type, "sizeof(int32_t)"),
                device_to_host,
                cuda_stream_var
            })
        );
    }

} // namespace backend
} // namespace nacho
