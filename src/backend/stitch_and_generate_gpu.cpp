#include "backend/stitch_and_generate.h"

namespace nacho {
namespace backend {

namespace {

// The contraction stage of ESC algorithm, one entry per emitted line. `$...$` are substituted with the
// tensors' actual field names.
//
//
// TODO(atharvac) : this needs to be generalized for any kind of scatter reduction
// not just dense-sparse-sparse to dense-sparse
const std::vector<std::string> &scatter_reduction_lines() {
    static const std::vector<std::string> lines = {
        "",
        "// Contract $REDUCED$ out of $TEMP$.",
        "{",
        "  const index_t contract_rows = $ROWS$;",
        "  const index_t contract_products = $PRODUCTS$;",
        "  const auto contract_policy = thrust::cuda::par_nosync.on(stream);",
        "",
        "  index_t* contract_segments = nullptr;",
        "  index_t* contract_keys_alt = nullptr;",
        "  index_t* contract_row_of = nullptr;",
        "  index_t* contract_row_ids = nullptr;",
        "  index_t* contract_unique_rows = nullptr;",
        "  index_t* contract_row_counts = nullptr;",
        "  value_t* contract_values_alt = nullptr;",
        "  value_t* contract_summed_values = nullptr;",
        "  int64_t* contract_keys = nullptr;",
        "  int64_t* contract_summed_keys = nullptr;",
        "  cudaMallocAsync((void**)&contract_segments, sizeof(index_t) * (contract_rows + 1), stream);",
        "  cudaMallocAsync((void**)&contract_keys_alt, sizeof(index_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_row_of, sizeof(index_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_row_ids, sizeof(index_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_unique_rows, sizeof(index_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_row_counts, sizeof(index_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_values_alt, sizeof(value_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_summed_values, sizeof(value_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_keys, sizeof(int64_t) * contract_products, stream);",
        "  cudaMallocAsync((void**)&contract_summed_keys, sizeof(int64_t) * contract_products, stream);",
        "",
        "  // Outer coordinate p spans inner positions",
        "  // [inner_offsets[reduced_offsets[p]], inner_offsets[reduced_offsets[p + 1]]).",
        "  {",
        "    const index_t* contract_inner_offsets = $INNER_OFFSETS$;",
        "    thrust::transform(contract_policy,",
        "                      $REDUCED_OFFSETS$, $REDUCED_OFFSETS$ + (contract_rows + 1),",
        "                      contract_segments,",
        "                      [contract_inner_offsets] __device__ (index_t position) {",
        "                        return contract_inner_offsets[position];",
        "                      });",
        "  }",
        "",
        "  size_t contract_sort_bytes = 0;",
        "  size_t contract_scan_bytes = 0;",
        "  {",
        "    cub::DoubleBuffer<index_t> probe_keys(contract_keys_alt, contract_keys_alt);",
        "    cub::DoubleBuffer<value_t> probe_values(contract_values_alt, contract_values_alt);",
        "    cub::DeviceSegmentedSort::SortPairs(nullptr, contract_sort_bytes, probe_keys, probe_values,",
        "                                        contract_products, contract_rows,",
        "                                        contract_segments, contract_segments + 1, stream);",
        "  }",
        "  cub::DeviceScan::InclusiveSum(nullptr, contract_scan_bytes, contract_row_counts,",
        "                                contract_row_counts, contract_rows, stream);",
        "  void* contract_scratch = nullptr;",
        "  const size_t contract_scratch_bytes =",
        "      contract_sort_bytes > contract_scan_bytes ? contract_sort_bytes : contract_scan_bytes;",
        "  cudaMallocAsync(&contract_scratch, contract_scratch_bytes, stream);",
        "",
        "  cub::DoubleBuffer<index_t> contract_sorted_keys($INNER_INDICES$, contract_keys_alt);",
        "  cub::DoubleBuffer<value_t> contract_sorted_values($TEMP_VALUES$, contract_values_alt);",
        "  cub::DeviceSegmentedSort::SortPairs(contract_scratch, contract_scratch_bytes,",
        "                                      contract_sorted_keys, contract_sorted_values,",
        "                                      contract_products, contract_rows,",
        "                                      contract_segments, contract_segments + 1, stream);",
        "",
        "  {",
        "    const index_t* contract_bounds = contract_segments;",
        "    const index_t contract_row_count = contract_rows;",
        "    thrust::transform(contract_policy,",
        "                      thrust::make_counting_iterator<index_t>(0),",
        "                      thrust::make_counting_iterator<index_t>(contract_products),",
        "                      contract_row_of,",
        "                      [contract_bounds, contract_row_count] __device__ (index_t position) {",
        "                        index_t low = 0;",
        "                        index_t high = contract_row_count;",
        "                        while (low < high) {",
        "                          const index_t mid = (low + high + 1) / 2;",
        "                          if (contract_bounds[mid] <= position) { low = mid; } else { high = mid - 1; }",
        "                        }",
        "                        return low;",
        "                      });",
        "    const index_t* contract_rows_of = contract_row_of;",
        "    const index_t* contract_inner = contract_sorted_keys.Current();",
        "    thrust::transform(contract_policy,",
        "                      thrust::make_counting_iterator<index_t>(0),",
        "                      thrust::make_counting_iterator<index_t>(contract_products),",
        "                      contract_keys,",
        "                      [contract_rows_of, contract_inner] __device__ (index_t position) {",
        "                        return ((int64_t)contract_rows_of[position] << 32)",
        "                             | (int64_t)(uint32_t)contract_inner[position];",
        "                      });",
        "  }",
        "",
        "  const auto contract_end = thrust::reduce_by_key(contract_policy,",
        "      contract_keys, contract_keys + contract_products,",
        "      contract_sorted_values.Current(),",
        "      contract_summed_keys, contract_summed_values,",
        "      thrust::equal_to<int64_t>(), thrust::plus<value_t>());",
        "  cudaStreamSynchronize(stream);",
        "  $OUT_LENGTH$ = (index_t)(contract_end.first - contract_summed_keys);",
        "",
        "  cudaMallocAsync((void**)&$OUT_OFFSETS$, sizeof(index_t) * (contract_rows + 1), stream);",
        "  cudaMallocAsync((void**)&$OUT_INDICES$, sizeof(index_t) * $OUT_LENGTH$, stream);",
        "  cudaMallocAsync((void**)&$OUT_VALUES$, sizeof(value_t) * $OUT_LENGTH$, stream);",
        "  cudaMemsetAsync($OUT_OFFSETS$, 0, sizeof(index_t) * (contract_rows + 1), stream);",
        "",
        "  thrust::transform(contract_policy, contract_summed_keys, contract_summed_keys + $OUT_LENGTH$,",
        "                    $OUT_INDICES$,",
        "                    [] __device__ (int64_t key) { return (index_t)(key & 0xFFFFFFFF); });",
        "  thrust::copy(contract_policy, contract_summed_values,",
        "               contract_summed_values + $OUT_LENGTH$, $OUT_VALUES$);",
        "  thrust::transform(contract_policy, contract_summed_keys, contract_summed_keys + $OUT_LENGTH$,",
        "                    contract_row_ids,",
        "                    [] __device__ (int64_t key) { return (index_t)(key >> 32); });",
        "",
        "  // Rebuild the output's offsets from how many entries each outer coordinate kept.",
        "  const auto contract_rows_end = thrust::reduce_by_key(contract_policy,",
        "      contract_row_ids, contract_row_ids + $OUT_LENGTH$,",
        "      thrust::make_constant_iterator<index_t>(1),",
        "      contract_unique_rows, contract_row_counts);",
        "  cudaStreamSynchronize(stream);",
        "  const index_t contract_unique = (index_t)(contract_rows_end.first - contract_unique_rows);",
        "  thrust::scatter(contract_policy, contract_row_counts, contract_row_counts + contract_unique,",
        "                  contract_unique_rows, $OUT_OFFSETS$ + 1);",
        "  cub::DeviceScan::InclusiveSum(contract_scratch, contract_scratch_bytes,",
        "                                $OUT_OFFSETS$ + 1, $OUT_OFFSETS$ + 1, contract_rows, stream);",
        "",
        "  // Both halves of each sort double buffer, so whichever one ended up current.",
        "  cudaFreeAsync($REDUCED_OFFSETS$, stream);",
        "  cudaFreeAsync($INNER_OFFSETS$, stream);",
        "  cudaFreeAsync($INNER_INDICES$, stream);",
        "  cudaFreeAsync($TEMP_VALUES$, stream);",
        "  cudaFreeAsync(contract_scratch, stream);",
        "  cudaFreeAsync(contract_segments, stream);",
        "  cudaFreeAsync(contract_keys_alt, stream);",
        "  cudaFreeAsync(contract_row_of, stream);",
        "  cudaFreeAsync(contract_row_ids, stream);",
        "  cudaFreeAsync(contract_unique_rows, stream);",
        "  cudaFreeAsync(contract_row_counts, stream);",
        "  cudaFreeAsync(contract_values_alt, stream);",
        "  cudaFreeAsync(contract_summed_values, stream);",
        "  cudaFreeAsync(contract_keys, stream);",
        "  cudaFreeAsync(contract_summed_keys, stream);",
        "}",
    };
    return lines;
}

} // namespace

    StitchAndGenerateGPU::StitchAndGenerateGPU(
        std::string name,
        std::map<std::string, TensorLowerer> &operand_tensors,
        TensorLowerer &result_tensor,
        std::vector<CIN> forall_list,
        TensorLowerer &reduced_result_tensor,
        std::vector<LoopNum> &reduction_loops,
        std::vector<std::string> requested_operand_ordering)
        : StitchAndGenerate(name, operand_tensors, result_tensor, std::move(forall_list), reduced_result_tensor, reduction_loops,
                            std::move(requested_operand_ordering)) {
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

    void StitchAndGenerateGPU::stitch_scatter_reduction() {
        const TensorLowerer &temp = result_tensor;
        const TensorLowerer &out = reduced_result_tensor;

        internal_assert(reduction_loops.size() == 1)
            << "Kernel '" << name << "': scatter reduction over more than one loop is not supported";
        const LoopNum reduced_loop = reduction_loops.front();

        // The loops the output keeps, split around the reduced one.
        std::vector<LoopNum> outer_loops, inner_loops;
        for (LoopNum loop = BEFORE_FIRST_LOOP + 1; loop < LoopNum(forall_list.size()); ++loop) {
            if (!temp.tensor_level_exists(loop) || loop == reduced_loop) {
                continue;
            }
            (loop < reduced_loop ? outer_loops : inner_loops).push_back(loop);
        }
        internal_assert(outer_loops.size() == 1 && inner_loops.size() == 1)
            << "Kernel '" << name << "': scatter reduction needs exactly one level on each "
            << "side of the reduced one";

        const TensorIndex outer = temp.loop_index(outer_loops.front());
        const TensorIndex reduced = temp.loop_index(reduced_loop);
        const TensorIndex inner = temp.loop_index(inner_loops.front());
        internal_assert(!temp.is_sparse(temp.loop_num_to_tensor_level(outer_loops.front())))
            << "Kernel '" << name << "': scatter reduction needs a dense outermost level";

        auto field = [](const TensorLowerer &tensor, const std::string &member) {
            return tensor.tensor_name + "." + member;
        };

        // The emitted block reads and writes only these, plus `stream` and the two structs.
        const std::map<std::string, std::string> substitutions = {
            {"$ROWS$", field(temp, temp.get_size_field_name(outer))},
            {"$PRODUCTS$", field(temp, temp.get_length_field_name(inner))},
            {"$REDUCED_OFFSETS$", field(temp, temp.get_offsets_field_name(reduced))},
            {"$INNER_OFFSETS$", field(temp, temp.get_offsets_field_name(inner))},
            {"$INNER_INDICES$", field(temp, temp.get_indices_field_name(inner))},
            {"$TEMP_VALUES$", field(temp, temp.get_values_field_name())},
            {"$OUT_OFFSETS$", field(out, out.get_offsets_field_name(inner))},
            {"$OUT_INDICES$", field(out, out.get_indices_field_name(inner))},
            {"$OUT_LENGTH$", field(out, out.get_length_field_name(inner))},
            {"$OUT_VALUES$", field(out, out.get_values_field_name())},
            {"$REDUCED$", reduced.str()},
            {"$TEMP$", temp.tensor_name},
        };

        for (const std::string &line : scatter_reduction_lines()) {
            std::string emitted = line;
            for (const auto &[placeholder, replacement] : substitutions) {
                for (size_t at = emitted.find(placeholder); at != std::string::npos;
                     at = emitted.find(placeholder, at + replacement.size())) {
                    emitted.replace(at, placeholder.size(), replacement);
                }
            }
            main_func.body.emplace_back(llir::RawStmt::make(emitted));
        }
    }

    llir::lStmt StitchAndGenerateGPU::generate_zero_range_statement(llir::lExpr field, llir::lExpr byte_count) {
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("cudaMemsetAsync", {
                field,
                llir::lConst::make(0),
                byte_count,
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
                llir::lFunctionCall::make("cub::DeviceScan::InclusiveSum", {
                    llir::lConst::make(0),
                    llir::lVar::make(sizet_type, "cub_bytes"),
                    null_scan_iterator, null_scan_iterator,
                    num_threads_var+1,
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

        size_expr = size_expr + llir::lConst::make(num_mem_block_fields) * (num_threads_var + 1);

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
                declare_operand_pos_map_once();

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
                    // The counts occupy num_threads slots, but generate_prefix_sum_calls
                    // walks the pointer back one afterwards to turn the inclusive scan into
                    // exclusive offsets. That slot is reserved here, ahead of the counts, so
                    // the shifted pointer stays inside this field's own region.
                    main_func.body.emplace_back(
                        llir::Accumulate::make(offset_var, llir::lVar::make(sizet_type, "sizeof(int32_t)"))
                    );
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

        for (LoopNum loop = BEFORE_FIRST_LOOP + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
                main_func.body.emplace_back(
                    llir::BaseExpr::make(
                        llir::lFunctionCall::make("cub::DeviceScan::InclusiveSum", {
                            llir::lVar::make(void_type, "cub_temp_storage_" + get_all_loops_string(current_sparse_intersection)),
                            llir::lVar::make(sizet_type, "cub_bytes"),
                            get_count_offsets_field(loop, current_sparse_intersection),
                            get_count_offsets_field(loop, current_sparse_intersection),
                            num_threads_var,
                            cuda_stream_var
                        })
                    )
                );
                // The last inclusive element is the total. Levels resolved by an earlier
                // intersection already have their length and are only rescanned to give
                // this phase's threads their starting offsets.
                if (loop > previous_sparse_intersection) {
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
                // Walking the base back one slot turns the inclusive scan into the
                // exclusive offsets the compute kernel wants, and the reserved slot it
                // now points at becomes the leading zero.
                main_func.body.emplace_back(
                    llir::Store::make(
                        get_count_offsets_field(loop, current_sparse_intersection),
                        get_count_offsets_field(loop, current_sparse_intersection) - 1
                    )
                );
                main_func.body.emplace_back(
                    generate_zero_range_statement(
                        get_count_offsets_field(loop, current_sparse_intersection),
                        llir::lVar::make(sizet_type, "sizeof(int32_t)")
                    )
                );
            }
        }

        // The lengths were copied back asynchronously and the host reads them right below
        // to size its allocations, so the copies have to have landed by now.
        main_func.body.emplace_back(
            llir::BaseExpr::make(
                llir::lFunctionCall::make("cudaStreamSynchronize", {cuda_stream_var})
            )
        );

        for (LoopNum loop = previous_sparse_intersection + 1; loop <= current_sparse_intersection; ++loop) {
            if (result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop)) {
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
