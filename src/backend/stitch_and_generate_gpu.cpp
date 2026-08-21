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
        "  if (contract_products == 0) {",
        "    $OUT_LENGTH$ = 0;",
        "    cudaMallocAsync((void**)&$OUT_OFFSETS$, sizeof(index_t) * (contract_rows + 1), stream);",
        "    cudaMemsetAsync($OUT_OFFSETS$, 0, sizeof(index_t) * (contract_rows + 1), stream);",
        "    cudaMallocAsync((void**)&$OUT_INDICES$, sizeof(index_t) * 1, stream);",
        "    cudaMallocAsync((void**)&$OUT_VALUES$, sizeof(value_t) * 1, stream);",
        "    ",
        "    cudaFreeAsync($REDUCED_OFFSETS$, stream);",
        "    cudaFreeAsync($REDUCED_INDICES$, stream);",
        "    cudaFreeAsync($INNER_OFFSETS$, stream);",
        "    cudaFreeAsync($INNER_INDICES$, stream);",
        "    cudaFreeAsync($TEMP_VALUES$, stream);",
        "  } else {",
        "    index_t* contract_segments = nullptr;",
        "    index_t* contract_keys_alt = nullptr;",
        "    value_t* contract_values_alt = nullptr;",
        "    int32_t* head_flags = nullptr;",
        "    int32_t* out_indices = nullptr;",
        "",
        "    // Probe CUB sizes",
        "    size_t sort_bytes = 0;",
        "    cub::DoubleBuffer<index_t> probe_keys(nullptr, nullptr);",
        "    cub::DoubleBuffer<value_t> probe_values(nullptr, nullptr);",
        "    cub::DeviceSegmentedSort::SortPairs(nullptr, sort_bytes, probe_keys, probe_values,",
        "                                        contract_products, contract_rows,",
        "                                        (index_t*)nullptr, (index_t*)nullptr, stream);",
        "",
        "    size_t scan_bytes = 0;",
        "    cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, (int32_t*)nullptr, (int32_t*)nullptr, contract_products, stream);",
        "",
        "    size_t scratch_bytes = sort_bytes > scan_bytes ? sort_bytes : scan_bytes;",
        "",
        "    // Massive allocation reduction: Dropped 6 heavy arrays",
        "    const size_t n = (size_t)contract_products;",
        "    const size_t pool_bytes = ",
        "        sizeof(index_t) * n +                  // keys_alt",
        "        sizeof(value_t) * n +                  // values_alt",
        "        sizeof(index_t) * (size_t)(contract_rows + 1) + // segments",
        "        sizeof(int32_t) * n +                  // head_flags",
        "        sizeof(int32_t) * n +                  // out_indices",
        "        256 + scratch_bytes;",
        "",
        "    void* contract_pool = nullptr;",
        "    cudaMallocAsync(&contract_pool, pool_bytes, stream);",
        "    char* base = (char*)contract_pool;",
        "    size_t offset = 0;",
        "",
        "    contract_keys_alt = (index_t*)(base + offset); offset += sizeof(index_t) * n;",
        "    contract_values_alt = (value_t*)(base + offset); offset += sizeof(value_t) * n;",
        "    contract_segments = (index_t*)(base + offset); offset += sizeof(index_t) * (size_t)(contract_rows + 1);",
        "    head_flags = (int32_t*)(base + offset); offset += sizeof(int32_t) * n;",
        "    out_indices = (int32_t*)(base + offset); offset += sizeof(int32_t) * n;",
        "    ",
        "    offset = ((offset + 255) / 256) * 256;",
        "    void* contract_scratch = (void*)(base + offset);",
        "",
        "    // 1. Build segment boundaries directly using your policy and transform",
        "    const index_t* contract_inner_offsets = $INNER_OFFSETS$;",
        "    thrust::transform(contract_policy,",
        "                      $REDUCED_OFFSETS$, $REDUCED_OFFSETS$ + (contract_rows + 1),",
        "                      contract_segments,",
        "                      [contract_inner_offsets] __device__ (index_t position) {",
        "                        return contract_inner_offsets[position];",
        "                      });",
        "",
        "    // Segmented Sort",
        "    cub::DoubleBuffer<index_t> d_keys($INNER_INDICES$, contract_keys_alt);",
        "    cub::DoubleBuffer<value_t> d_vals($TEMP_VALUES$, contract_values_alt);",
        "    cub::DeviceSegmentedSort::SortPairs(contract_scratch, scratch_bytes,",
        "                                        d_keys, d_vals,",
        "                                        contract_products, contract_rows,",
        "                                        contract_segments, contract_segments + 1, stream);",
        "",
        "    // Mark heads",
        "    int grid_products = (contract_products + threads_per_block - 1) / threads_per_block;",
        "    // NOTE: Call global kernels that are injected at the top of the file",
        "    mark_heads_kernel<<<grid_products, threads_per_block, 0, stream>>>(",
        "        d_keys.Current(), contract_segments, contract_rows, contract_products, head_flags);",
        "",
        "    // Inclusive Scan to generate 1-indexed group IDs",
        "    cub::DeviceScan::InclusiveSum(contract_scratch, scratch_bytes, head_flags, out_indices, contract_products, stream);",
        "",
        "    int32_t total_unique = 0;",
        "    cudaMemcpyAsync(&total_unique, out_indices + contract_products - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);",
        "    cudaStreamSynchronize(stream);",
        "    $OUT_LENGTH$ = total_unique;",
        "",
        "    // Output Allocation",
        "    cudaMallocAsync((void**)&$OUT_OFFSETS$, sizeof(index_t) * (contract_rows + 1), stream);",
        "    cudaMallocAsync((void**)&$OUT_INDICES$, sizeof(index_t) * (total_unique > 0 ? total_unique : 1), stream);",
        "    cudaMallocAsync((void**)&$OUT_VALUES$, sizeof(value_t) * (total_unique > 0 ? total_unique : 1), stream);",
        "    cudaMemsetAsync($OUT_VALUES$, 0, sizeof(value_t) * (total_unique > 0 ? total_unique : 1), stream);",
        "",
        "    // Extract Row Offsets",
        "    int grid_rows_k = (contract_rows + 1 + threads_per_block - 1) / threads_per_block;",
        "    extract_offsets_kernel<<<grid_rows_k, threads_per_block, 0, stream>>>(",
        "        contract_segments, out_indices, contract_rows, contract_products, total_unique, $OUT_OFFSETS$);",
        "",
        "    // Custom Segmented Sum (replaces thrust::reduce_by_key)",
        "    if (total_unique > 0) {",
        "        reduce_values_kernel<<<grid_products, threads_per_block, 0, stream>>>(",
        "            d_keys.Current(), d_vals.Current(), out_indices, contract_products, $OUT_INDICES$, $OUT_VALUES$);",
        "    }",
        "",
        "    // Cleanup",
        "    cudaFreeAsync($REDUCED_OFFSETS$, stream);",
        "    cudaFreeAsync($REDUCED_INDICES$, stream);",
        "    cudaFreeAsync($INNER_OFFSETS$, stream);",
        "    cudaFreeAsync($INNER_INDICES$, stream);",
        "    cudaFreeAsync($TEMP_VALUES$, stream);",
        "    cudaFreeAsync(contract_pool, stream);",
        "  }",
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
            {"$REDUCED_INDICES$", field(temp, temp.get_indices_field_name(reduced))},
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

    llir::lStmt StitchAndGenerateGPU::scatter_header_functions() {
        // This is an outrageous hack.
        // TODO: fix.
        return llir::RawStmt::make(R"(
template <typename index_t>
__global__ void mark_heads_kernel(
    const index_t* __restrict__ sorted_k,
    const index_t* __restrict__ contract_segments,
    const int32_t num_rows,
    const int32_t num_intermediate,
    int32_t* __restrict__ head_flags)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_intermediate) {
        int32_t low = 0;
        int32_t high = num_rows;
        while (low < high) {
            int32_t mid = (low + high + 1) / 2;
            if (contract_segments[mid] <= idx) { low = mid; } else { high = mid - 1; }
        }
        int32_t i = low;

        bool is_head = false;
        if (idx == contract_segments[i]) {
            is_head = true;
        } else if (sorted_k[idx] != sorted_k[idx - 1]) {
            is_head = true;
        }
        head_flags[idx] = is_head ? 1 : 0;
    }
}

template <typename index_t>
__global__ void extract_offsets_kernel(
    const index_t* __restrict__ contract_segments,
    const int32_t* __restrict__ out_indices, 
    const int32_t num_rows,
    const int32_t num_intermediate,
    const int32_t total_unique,
    index_t* __restrict__ Z_k_offsets)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= num_rows) {
        if (idx == num_rows) {
            Z_k_offsets[idx] = total_unique;
        } else {
            int32_t seg_start = contract_segments[idx];
            if (seg_start < num_intermediate) {
                Z_k_offsets[idx] = out_indices[seg_start] - 1;
            } else {
                Z_k_offsets[idx] = total_unique;
            }
        }
    }
}

template <typename index_t, typename value_t>
__global__ void reduce_values_kernel(
    const index_t* __restrict__ sorted_k,
    const value_t* __restrict__ sorted_val,
    const int32_t* __restrict__ out_indices,
    const int32_t num_intermediate,
    index_t* __restrict__ Z_k_indices,
    value_t* __restrict__ Z_values)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // NO EARLY RETURN! We flag valid threads to keep the warp synchronized
    bool valid = (idx < num_intermediate);

    int32_t my_out_id = -1;
    index_t my_k = 0;
    value_t my_val = 0;

    if (valid) {
        my_out_id = out_indices[idx] - 1;
        my_k = sorted_k[idx];
        my_val = sorted_val[idx];
    }

    // Unconditional warp-synchronous segmented scan
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int32_t other_id = __shfl_up_sync(0xffffffff, my_out_id, offset);
        value_t other_val = __shfl_up_sync(0xffffffff, my_val, offset);

        // Add only if both lanes belong to the same segment and are valid
        if (threadIdx.x % 32 >= offset && my_out_id == other_id && my_out_id != -1) {
            my_val += other_val;
        }
    }

    // Unconditional warp-synchronous read from the next lane
    int32_t next_id = __shfl_down_sync(0xffffffff, my_out_id, 1);

    if (valid) {
        bool is_last_in_warp = false;
        // Check lane 31, array boundary, OR segment boundary safely
        if (threadIdx.x % 32 == 31 || idx == num_intermediate - 1 || next_id != my_out_id) {
            is_last_in_warp = true;
        }

        if (is_last_in_warp) {
            atomicAdd(&Z_values[my_out_id], my_val);
            Z_k_indices[my_out_id] = my_k;
        }
    }
}
)");
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
        current_allocation_level = current_sparse_intersection;

        // CUB deduces its iterator types from the arguments even when d_temp_storage is null
        // and it only reports the temp-storage size, so the sizing queries below pass a typed
        // null pointer matching the buffers the real scans operate on.
        llir::lExpr null_scan_iterator =
            llir::Cast::make(llir::Generic_t::make("int32_t*"), llir::lConst::make(0));

        // The position map fields for this level, if any. They are read again at the next
        // sparse intersection, so they get a block of their own rather than a slice of the
        // per-level scratch that is freed on the way there.
        std::vector<std::pair<std::string, llir::lType>> pos_map_fields;
        if(compute_kernel_defined && operand_pos_map_struct_def.defined()) {
            for(auto &[tensor_name, tensor] : operand_tensors) {
                auto field = std::find_if(operand_pos_map_struct_def.as<llir::Struct_t>()->fields.begin(), operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end(),
                    [previous_sparse_intersection, &tensor](const std::pair<std::string, llir::lType> &field) {
                        return field.first == tensor.get_iterator_suffix(tensor.loop_index(previous_sparse_intersection));
                    });
                if(field != operand_pos_map_struct_def.as<llir::Struct_t>()->fields.end()) {
                    pos_map_fields.push_back(*field);
                }
            }
        }

        llir::lExpr size_expr = llir::lConst::make(0);
        if(compute_kernel_defined) {
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

        if(!pos_map_fields.empty()) {
            declare_operand_pos_map_once();

            llir::lExpr pos_map_length = result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection));
            std::string pos_map_block_name = "pos_map_block_" + get_all_loops_string(current_sparse_intersection);
            llir::lExpr pos_map_block_var = llir::lVar::make(llir::Generic_t::make("void*"), pos_map_block_name);
            main_func.body.emplace_back(
                llir::Declare::make(llir::Generic_t::make("void*"), pos_map_block_name, llir::lConst::make(0))
            );
            main_func.body.emplace_back(
                generate_single_memory_allocation_statement(
                    pos_map_block_var,
                    llir::Ptr_t::make(void_type),
                    llir::lVar::make(sizet_type, "sizeof(int32_t)") * llir::lConst::make((int)pos_map_fields.size()) * pos_map_length,
                    false
                )
            );
            long_lived_allocations.push_back(pos_map_block_var);

            std::string pos_map_offset_name = "pos_map_offset_" + get_all_loops_string(current_sparse_intersection);
            std::string pos_map_base_name = "pos_map_block_base_" + get_all_loops_string(current_sparse_intersection);
            main_func.body.emplace_back(
                llir::Declare::make(llir::Int_t::make(32), pos_map_offset_name, llir::lConst::make(0))
            );
            main_func.body.emplace_back(
                llir::Declare::make(llir::Generic_t::make("char*"), pos_map_base_name,
                    llir::Cast::make(llir::Generic_t::make("char*"), pos_map_block_var))
            );
            auto pos_map_offset_var = llir::lVar::make(llir::Int_t::make(32), pos_map_offset_name);
            auto pos_map_base_var = llir::lVar::make(llir::Generic_t::make("char*"), pos_map_base_name);

            for(const auto &field : pos_map_fields) {
                main_func.body.emplace_back(
                    llir::Store::make(
                        llir::lFieldAccess::make(llir::lVar::make(operand_pos_map_struct_def, get_operand_pos_map_var_name()), field.first),
                        llir::Cast::make(field.second, pos_map_base_var + pos_map_offset_var)
                    )
                );
                main_func.body.emplace_back(
                    llir::Accumulate::make(pos_map_offset_var, llir::lVar::make(sizet_type, "sizeof(int32_t)") * pos_map_length)
                );
            }
        }

        if(compute_kernel_defined) {
            main_func.body.emplace_back(
                llir::Declare::make(llir::Ptr_t::make(llir::Int_t::make(32)),
                                    get_work_offsets_var_name(current_sparse_intersection))
            );
            main_func.body.emplace_back(
                llir::Store::make(
                    work_offsets_var(),
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
        // Not sure if this is required or not?
        // main_func.body.emplace_back(
        //     llir::BaseExpr::make(
        //         llir::lFunctionCall::make("cudaStreamSynchronize", {cuda_stream_var})
        //     )
        // );

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
                    work_offsets_var(),
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
                    work_offsets_var() + llir::lConst::make(1),
                    work_offsets_var() + llir::lConst::make(1),
                    result_tensor.get_length_field(result_tensor.loop_index(previous_sparse_intersection)),
                    cuda_stream_var
                })
            )
        );
    }

    llir::lStmt StitchAndGenerateGPU::generate_free_statement(llir::lExpr address) {
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("cudaFreeAsync", {address, cuda_stream_var})
        );
    }

    llir::lStmt StitchAndGenerateGPU::generate_memory_free_statements(LoopNum sparse_intersection) {
        std::string mem_block_name = "mem_block_" + get_all_loops_string(sparse_intersection);
        return generate_free_statement(llir::lVar::make(llir::Generic_t::make("void*"), mem_block_name));
    }

    llir::lStmt StitchAndGenerateGPU::generate_total_work_from_offsets_statement(llir::lExpr index_expr) {
        return llir::BaseExpr::make(
            llir::lFunctionCall::make("cudaMemcpyAsync", {
                llir::lAddress::make(llir::lVar::make(llir::Int_t::make(32), "total_work")),
                work_offsets_var() + index_expr,
                llir::lVar::make(sizet_type, "sizeof(int32_t)"),
                device_to_host,
                cuda_stream_var
            })
        );
    }

} // namespace backend
} // namespace nacho
