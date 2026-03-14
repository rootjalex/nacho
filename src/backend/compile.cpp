#include "backend/compile.h"
#include "backend/partition.h"
#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Error.h"
#include "Format.h"
#include "Type.h"
#include "backend/tensor.h"
#include "backend/compute.h"
#include "Printer.h"
#include <map>

namespace nacho {
namespace backend {

    CINLowerer::CINLowerer(CIN cin, std::ostream &os) : cin(std::move(cin)), printer(os) {
        struct TensorVisitor : Visitor {
            std::map<std::string, TensorLowerer> &operand_tensors;
            TensorLowerer &result_tensor;
            TensorVisitor(std::map<std::string, TensorLowerer> &operand_tensors, TensorLowerer &result_tensor)
                : operand_tensors(operand_tensors), result_tensor(result_tensor) {}

            void add_tensor(std::string str, TensorType type) {
                TensorLowerer lowerer(str, type);
                operand_tensors[str] = lowerer;
            }

            void visit(const cTensor *node) override {
                add_tensor(node->name, node->type);
            }

            void visit(const Accumulate *node) override { 
                result_tensor = TensorLowerer(node->tensor, node->type, true);
                node->expr.accept(this); 
            }

            void visit(const Assign *node) override { 
                result_tensor = TensorLowerer(node->tensor, node->type, true);
                node->expr.accept(this); 
            }
        };

        TensorVisitor visitor(operand_tensors, result_tensor);
        this->cin.accept(&visitor);
    }

    std::vector<std::string> CINLowerer::get_loop_order() {
        std::vector<std::string> loop_order;
        struct ForallVisitor : Visitor {
            std::vector<std::string> &loop_order;
            ForallVisitor(std::vector<std::string> &loop_order) : loop_order(loop_order) {}

            void visit(const Forall *node) override {
                loop_order.push_back(node->idx);
                node->body.accept(this);
            }
        };
        ForallVisitor visitor(loop_order);
        cin.accept(&visitor);
        return loop_order;
    }

    std::vector<CIN> CINLowerer::get_forall_loops() {
        std::vector<CIN> forall_loops;
        struct ForallVisitor : Visitor {
            std::vector<CIN> &forall_loops;
            ForallVisitor(std::vector<CIN> &forall_loops) : forall_loops(forall_loops) {}

            void visit(const Forall *node) override {
                forall_loops.push_back(node);
                node->body.accept(this);
            }
        };
        ForallVisitor visitor(forall_loops);
        cin.accept(&visitor);
        return forall_loops;
    }

    void CINLowerer::lower_cin() {

        // All-Coordinate format uses a dedicated COO codegen path
        if (result_tensor.tensor_type.format.is_all_coordinate()) {
            lower_coo_cin();
            return;
        }

        std::vector<std::string> loop_order = get_loop_order();
        auto forall_loops = get_forall_loops();

        std::vector<int> sparse_intersection_levels = get_sparse_intersection_levels(cin);

        sparse_intersection_levels.insert(sparse_intersection_levels.begin(), -1);
        if(sparse_intersection_levels.back()!=(int)loop_order.size()-1) {
            sparse_intersection_levels.push_back(loop_order.size()-1);
        }

        this->lower_binary_search_function();
        this->lower_struct_definitions(sparse_intersection_levels[sparse_intersection_levels.size()-2]);

        for(int i=0; i< (int)sparse_intersection_levels.size()-1;i++) {
            int previous_sparse_intersection_level = sparse_intersection_levels[i];
            int current_sparse_intersection_level = sparse_intersection_levels[i+1];
            int next_sparse_intersection_level = i!=(int)sparse_intersection_levels.size()-2 ? sparse_intersection_levels[i+2] : forall_loops.size() ;

            auto previous_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + previous_sparse_intersection_level + 1);
            auto current_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + current_sparse_intersection_level + 1);

            // Generate work functions for partition kernel
            for(int level=0; level<=previous_sparse_intersection_level;level++) {
                printer.print(result_tensor.lower_work_function(previous_loop_order, level));
            }
            for(int level = previous_sparse_intersection_level + 1; level <= current_sparse_intersection_level; level++) {
                for (auto it : operand_tensors) {
                    printer.print(it.second.lower_work_function(current_loop_order, level));
                }
            }

            auto active_phase_tensors = get_active_tensors_for_level(current_sparse_intersection_level);
            PartitionKernelLowerer partition_lowerer(operand_tensors, result_tensor, active_phase_tensors, forall_loops, previous_sparse_intersection_level, current_sparse_intersection_level, next_sparse_intersection_level);

            llir::lType partition_struct_type = partition_lowerer.lower_partition_struct_definition();
            printer.print(partition_struct_type);
            printer.print(partition_lowerer.lower_partition_kernel());

            // Collect partition kernel info
            kernel_infos.push_back({
                .name = partition_lowerer.get_partition_function_name(),
                .template_args = {"index_t", "value_t"},
                .args = partition_lowerer.get_kernel_args(),
                .kind = KernelInfo::Partition,
                .phase = i
            });

            // Get the modified CIN
            CIN modified_cin = cin;

            if(i!=(int)sparse_intersection_levels.size()-2){
                modified_cin = get_phase_cin_for_sparse_intersection(current_sparse_intersection_level, cin);
                active_phase_tensors = get_active_tensors_for_level(next_sparse_intersection_level);
            }

            ComputeKernelLowerer compute_lowerer(operand_tensors, result_tensor, active_phase_tensors, forall_loops, modified_cin, previous_sparse_intersection_level, current_sparse_intersection_level, next_sparse_intersection_level);

            // Generate Precompute kernels
            bool has_precompute_kernel = result_tensor.tensor_type.format.get_prev_sparse_level(current_sparse_intersection_level+1) != -1;
            if(has_precompute_kernel) {
                printer.print(compute_lowerer.lower_precompute_kernel());

                // Collect precompute kernel info
                kernel_infos.push_back({
                    .name = compute_lowerer.get_precompute_function_name(),
                    .template_args = {"index_t", "value_t"},
                    .args = compute_lowerer.get_precompute_kernel_args(),
                    .kind = KernelInfo::Precompute,
                    .phase = i
                });
            }

            // Generate Compute kernel
            // We need to lower an extra work function for compute kernel if this not innermost sparse intersect
            // The target_dim value is fixed for this work function.
            if(i!=(int)sparse_intersection_levels.size()-2){
                for (auto it : operand_tensors) {
                    auto next_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + next_sparse_intersection_level + 1);
                    printer.print(it.second.lower_work_function(next_loop_order, current_sparse_intersection_level, true));
                }
            }
            printer.print(compute_lowerer.lower_compute_kernel());

            // Collect compute kernel info
            kernel_infos.push_back({
                .name = compute_lowerer.get_compute_function_name(),
                .template_args = {"index_t", "value_t"},
                .args = compute_lowerer.get_compute_kernel_args(),
                .kind = KernelInfo::Compute,
                .phase = i
            });

            // Collect struct info for this phase
            // Build count struct type using BaseKernelLowerer
            llir::lType counts_struct_type;
            if(has_precompute_kernel) {
                auto empty_map = std::map<std::string, TensorLowerer>();
                BaseKernelLowerer base_lowerer(operand_tensors, result_tensor, empty_map, forall_loops, previous_sparse_intersection_level, current_sparse_intersection_level, next_sparse_intersection_level);
                counts_struct_type = base_lowerer.lower_result_per_thread_count_struct();
            }

            phase_struct_infos.push_back({
                .partition_struct = partition_struct_type,
                .counts_struct = counts_struct_type,
                .previous_sparse_intersection_level = previous_sparse_intersection_level,
                .current_sparse_intersection_level = current_sparse_intersection_level,
                .next_sparse_intersection_level = next_sparse_intersection_level,
                .has_precompute_kernel = has_precompute_kernel
            });

        }

        // Generate host orchestration function
        lower_host_function();

    }

    // lower_struct_definitions loweres all the initial struct definitions for the program
    // this includes tensor struct definitions for both the operand and result tensors
    // This also result_per_thread_count structs and result_to_operand_pos_map structs
    // which might be required by the intermediate kernels.
    void CINLowerer::lower_struct_definitions(int last_sparse_intersection) {
        for (auto it : operand_tensors) {
            printer.print(it.second.lower_tensor_struct_definition());
            // printer.print(it.second.lower_tensor_index_definition());
        }
        printer.print(result_tensor.lower_tensor_struct_definition());


        // Use BaseKernelLowerer to lower the one time struct definitions of result_per_thread_count struct and result_to_operand_pos_map struct
        auto empty_map = std::map<std::string, TensorLowerer>();
        BaseKernelLowerer BaseLowerer(operand_tensors, result_tensor, empty_map, get_forall_loops(), -1, -1, -1);
        // Need to lower this struct only once
        if(!result_tensor.tensor_type.format.are_all_lvls_dense()) {
            printer.print(BaseLowerer.lower_result_per_thread_count_struct());
        }

        auto result_operand_pos_map = lower_result_to_operand_pos_map_struct(last_sparse_intersection);
        if(result_operand_pos_map.get() != nullptr) {
            printer.print(result_operand_pos_map);
        }
    }

    llir::lType CINLowerer::lower_result_to_operand_pos_map_struct(int last_sparse_intersection) {
        std::vector<std::string> generics = {"index_t"};
        std::vector<std::pair<std::string, llir::lType>> fields;
        auto forall_loops = get_forall_loops(); auto empty_map = std::map<std::string, TensorLowerer>();
        BaseKernelLowerer BaseLowerer(operand_tensors, result_tensor, empty_map, forall_loops, -1, -1, -1);
        for(int level=0; level<=last_sparse_intersection;level++) {
            if(level == -1 || level >= forall_loops.size()-1) {
                continue;
            }
            const Forall* forall = forall_loops[level].as<Forall>();
            std::string forall_idx = forall->idx;

            for(auto& [name, tensor] : operand_tensors) {
                if(BaseLowerer.has_result_to_operand_pos_field(forall, tensor)) {
                    fields.emplace_back(tensor.get_iterator_suffix(forall_idx), llir::Ptr_t::make(index_t));
                }
            }
        }
        if(fields.empty()) {
            return llir::lType();
        }

        return llir::Struct_t::make(BaseLowerer.get_result_to_operand_pos_map_struct_name(), std::move(fields),
                                 std::move(generics));
    }

    std::vector<int> CINLowerer::get_sparse_intersection_levels(CIN& cin) {
        struct Checker : public Visitor {
            int loop_level = -1;
            bool inside_sparse_intersection = false;
            bool found_sparse_intersection = false;
            std::vector<int> sparse_intersection_levels;

            void visit(const Intersect * node) override{
                //std::cout<<"Visiting intersect at loop level "<< loop_level << " , inside_sparse_intersection: "<< inside_sparse_intersection <<" "<<Seq(node)<<std::endl;
                if(inside_sparse_intersection){
                    node->a.accept(this);
                    node->b.accept(this);
                }

                // This means this is a sparse intersection
                if(node->is_sparse) {
                    inside_sparse_intersection = true;
                    //std::cout<<"Inside sparse intersection "<<(Seq)node<<std::endl;
                    if(node->a.get()->is_sparse && node->b.get()->is_sparse)
                    {
                        found_sparse_intersection = true;
                        inside_sparse_intersection = false;
                        return;
                    }
                    if(node->a.get()->is_sparse) {
                        node->b.accept(this);
                        inside_sparse_intersection = false;
                        node->a.accept(this);
                    } else {
                        node->a.accept(this);
                        inside_sparse_intersection = false;
                        node->b.accept(this);
                    }
                    inside_sparse_intersection = false;
                    return;
                }
                node->a.accept(this);
                node->b.accept(this);
                return;
            }

            void visit(const Index * node) override{
                // if we are inside a sparse intersection seq, 
                // then mark that we found a sparse intersection
                if(inside_sparse_intersection) {
                    found_sparse_intersection = true;
                }
            }


            void visit(const Forall * node) override{
                // if we already found a sparse intersection, 
                // that means this forall is inside a sparse intersection forall
                // hence cin is not innermost sparse
                loop_level++;
                found_sparse_intersection = false;
                //<<" Going inside seq "<<(Seq)node->seq<<std::endl;
                node->seq.accept(this);
                if(found_sparse_intersection) {
                    sparse_intersection_levels.push_back(loop_level);
                }
                found_sparse_intersection = false;
                node->body.accept(this);
            }
        };

        Checker checker;
         if (const Forall *forall = cin.as<Forall>()) {
            forall->accept(&checker);
        } else {
            internal_assert(false) << "Root node of CIN is not a Forall.";
        }
        return checker.sparse_intersection_levels;
    }

    CIN CINLowerer::get_phase_cin_for_sparse_intersection(int target_level, CIN& cin) {
        struct Modifier: public Mutator {
            int loop_level = -1;
            int target_level;
            Modifier(int target_level) : target_level(target_level) {}
            CIN visit(const Forall * node) override{
                loop_level++;
                if(loop_level < target_level) {
                    Seq seq = mutate(node->seq);
                    CIN body = mutate(node->body);
                    if (seq.same_as(node->seq) && body.same_as(node->body)) {
                        return node;
                    }
                    return Forall::make(node->idx, std::move(seq), std::move(body));
                } else {
                    Seq seq = mutate(node->seq);
                    CIN body = CalculateWork::make(std::move(node->body));
                    return Forall::make(node->idx, std::move(seq), std::move(body));
                }
            }
        };
        return Modifier(target_level).mutate(cin);
    }


    void CINLowerer::lower_binary_search_function() {
        std::vector<std::string> generics = {"index_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::device, llir::Function::inline_};

        std::vector<llir::Function::Argument> args;
        llir::lType ret_type;
        std::string name;
        llir::lStmt body;

        name = "binary_search";

        ret_type = index_t;


        args.emplace_back(
            llir::Function::Argument{.mutating = false, .type = llir::Ptr_t::make(index_t), .name = "arr"});
            args.emplace_back(
                llir::Function::Argument{.mutating = false, .type = index_t, .name = "target_value"});
            args.emplace_back(
                llir::Function::Argument{.mutating = true, .type = llir::Int_t::make(32), .name = "start_index"});
            args.emplace_back(
                llir::Function::Argument{.mutating = true, .type = llir::Int_t::make(32), .name = "end_index"});

        std::vector<llir::lStmt> stmts;
        
        llir::lExpr start_var = llir::lVar::make(index_t, "start_index");
        llir::lExpr mid_var = llir::lVar::make(index_t, "mid");
        llir::lExpr end_var = llir::lVar::make(index_t, "end_index");
        llir::lExpr arr_var = llir::lVar::make(llir::Ptr_t::make(index_t), "arr");
        llir::lExpr target_value_var = llir::lVar::make(index_t, "target_value");

        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                "mid",
                start_var + (((end_var - start_var) + 1) / 2)
            )
        );  
        std::vector<llir::lStmt> while_stmts;

        while_stmts.emplace_back(
            llir::Store::make(
                mid_var,
                start_var + (((end_var - start_var) + 1) / 2)
            )
        );

        while_stmts.emplace_back(llir::IfElse::make(
            arr_var[mid_var] <= target_value_var,
            llir::Store::make(start_var, mid_var),
            llir::Store::make(end_var, mid_var - 1)
        ));

        stmts.emplace_back(
            llir::While::make(
                start_var < end_var,
                llir::Sequence::make(std::move(while_stmts))
            )
        );

        stmts.emplace_back(
            llir::Store::make(
                mid_var,
                start_var + (((end_var - start_var) + 1) / 2)
            )
        );
        
        stmts.emplace_back(
            llir::Return::make(mid_var)
        );
        body = llir::Sequence::make(std::move(stmts));

        printer.print(llir::Function::make(std::move(generics), std::move(attributes), std::move(args), std::move(ret_type), name, std::move(body)));
    }

    std::map<std::string, TensorLowerer> CINLowerer::get_active_tensors_for_level(int level) {
            auto forall  = get_forall_loops()[level].as<Forall>();
            std::vector<Seq> locators = get_dense_locators(forall->seq);

            std::map<std::string, TensorLowerer> active_phase_tensors;

            // included tensors are the tensors which are included in the work
            // calculation. Non-included tensors are not co-iterated and instead looked up.
            std::map<std::string, TensorLowerer> excluded_tensors;
            for(const auto &loc : locators) {
                const auto *index = loc.as<Index>();
                if (!index){
                    internal_assert(false) << "Expected Index node in locator sequence: " << loc;
                }
                for(const auto &it : operand_tensors) {
                    if (it.second.tensor_name == index->tensor) {
                        excluded_tensors[it.second.tensor_name] = it.second;
                    }
                }
            }
            for(auto it : operand_tensors) {
                if(excluded_tensors.find(it.first) == excluded_tensors.end()) {
                    active_phase_tensors[it.first] = it.second;
                }
            }
            //std::cout<<" At level "<< level << " included tensors: "<< active_phase_tensors.size() << "\n";
            return active_phase_tensors;
    }


    void CINLowerer::lower_host_function() {
        std::vector<std::string> generics = {"index_t", "value_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::host};

        std::vector<llir::Function::Argument> func_args;
        llir::lType ret_type = llir::Generic_t::make("void");
        std::string name = result_tensor.tensor_name + "_compute";

        // Function arguments: const operand tensors + mutable result tensor
        for (const auto &it : operand_tensors) {
            func_args.emplace_back(llir::Function::Argument{
                .mutating = false,
                .type = llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                .name = it.second.tensor_name});
        }
        func_args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .by_reference = true,
            .type = llir::Generic_t::make(result_tensor.get_struct_name() + "<index_t, value_t>"),
            .name = result_tensor.tensor_name});

        std::vector<llir::lStmt> body_stmts;

        // Declare num_blocks and threads_per_block
        body_stmts.emplace_back(llir::Declare::make(
            index_t, "num_blocks", llir::lConst::make((int64_t)256)));
        body_stmts.emplace_back(llir::Declare::make(
            index_t, "threads_per_block", llir::lConst::make((int64_t)256)));
        body_stmts.emplace_back(llir::Declare::make(
            index_t, "num_threads",
            llir::lVar::make(index_t, "num_blocks") *
                llir::lVar::make(index_t, "threads_per_block")));
        body_stmts.emplace_back(llir::RawCode::make(
            "const cudaStream_t stream = cudaStreamPerThread;"));

        llir::lExpr num_blocks_var = llir::lVar::make(index_t, "num_blocks");
        llir::lExpr threads_per_block_var = llir::lVar::make(index_t, "threads_per_block");
        llir::lExpr num_threads_var = llir::lVar::make(index_t, "num_threads");
        llir::lExpr stream_var = llir::lVar::make(
            llir::Generic_t::make("cudaStream_t"), "stream");

        // For each phase, generate the host-side orchestration
        int num_phases = (int)phase_struct_infos.size();
        std::string prev_phase_outermost_nnz;  // tracks nnz from previous phase for T_work_offsets indexing
        int prev_phase_max_sparse = -1;  // highest sparse level allocated in previous phases
        for (int phase = 0; phase < num_phases; phase++) {
            const auto &phase_info = phase_struct_infos[phase];
            const llir::Struct_t *partition_struct = phase_info.partition_struct.as<llir::Struct_t>();
            std::vector<std::string> delayed_result_field_updates;

            body_stmts.emplace_back(llir::RawCode::make(
                "// ========== Phase " + std::to_string(phase) + " =========="));

            // 1. Compute total_work and per_thread_work
            if (phase_info.previous_sparse_intersection_level == -1) {
                // Phase 0: total_work = sum of operand nnz (length of innermost sparse dim)
                // Use the included tensors' work computation
                std::string total_work_expr;
                bool first = true;
                for (const auto &it : operand_tensors) {
                    // Find the work contribution - use the length of the sparse dim at current_sparse_intersection_level
                    auto forall = get_forall_loops()[phase_info.current_sparse_intersection_level].as<Forall>();
                    std::string idx = forall->idx;
                    if (it.second.tensor_type.format.level_exists(idx) &&
                        is_sparse_format(it.second.tensor_type.format.lvlfmt_of(idx))) {
                        if (!first) total_work_expr += " + ";
                        total_work_expr += it.second.tensor_name + "." +
                                          it.second.get_length_field_name(idx);
                        first = false;
                    }
                }
                // If result tensor has a dense dim at the current_sparse_intersection_level level,
                // we may need size from operand's nnz or similar
                if (first) {
                    // Fallback: sum all operand nnz-like fields
                    for (const auto &it : operand_tensors) {
                        auto &fmt = it.second.tensor_type.format;
                        if (!fmt.levels.empty()) {
                            int last = fmt.levels.size() - 1;
                            if (is_sparse_format(fmt.levels[last].format)) {
                                if (!first) total_work_expr += " + ";
                                total_work_expr += it.second.tensor_name + "." +
                                                  it.second.get_length_field_name(last);
                                first = false;
                            }
                        }
                    }
                }
                if (total_work_expr.empty()) {
                    total_work_expr = "0";
                }
                body_stmts.emplace_back(llir::RawCode::make(
                    "index_t total_work_" + std::to_string(phase) + " = " + total_work_expr + ";"));
            } else {
                // Phase N>0: total_work comes from prefix sum result of previous phase.
                // Read from the last valid entry (indexed by previous phase's output nnz).
                body_stmts.emplace_back(llir::RawCode::make(
                    "index_t total_work_" + std::to_string(phase) + ";"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaMemcpyAsync(&total_work_" + std::to_string(phase) +
                    ", &T_work_offsets_prefix[" + prev_phase_outermost_nnz +
                    "], sizeof(index_t), cudaMemcpyDeviceToHost, stream);"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaStreamSynchronize(stream);"));
            }

            std::string total_work_name = "total_work_" + std::to_string(phase);
            std::string ptw_name = "per_thread_work_" + std::to_string(phase);
            body_stmts.emplace_back(llir::RawCode::make(
                "index_t " + ptw_name + " = " + total_work_name + " / num_threads + 1;"));

            // 2. Allocate partition struct fields
            std::string partition_var = "partitions_" + std::to_string(phase);
            body_stmts.emplace_back(llir::RawCode::make(
                partition_struct->name + "<index_t> " + partition_var + ";"));
            for (const auto &field : partition_struct->fields) {
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaMallocAsync((void**)&" + partition_var + "." + field.first +
                    ", num_threads * sizeof(index_t), stream);"));
            }

            // 3. Launch partition kernel
            {
                std::vector<llir::lExpr> launch_args;
                // Build launch args matching the kernel signature
                for (const auto &ki : kernel_infos) {
                    if (ki.kind == KernelInfo::Partition && ki.phase == phase) {
                        for (const auto &arg : ki.args) {
                            if (arg.name == "partitions") {
                                launch_args.push_back(llir::lVar::make(
                                    llir::Generic_t::make(partition_struct->name + "<index_t>"),
                                    partition_var));
                            } else if (arg.name == "per_thread_work") {
                                launch_args.push_back(llir::lVar::make(index_t, ptw_name));
                            } else if (arg.name == "total_work") {
                                launch_args.push_back(llir::lVar::make(index_t, total_work_name));
                            } else {
                                launch_args.push_back(llir::lVar::make(
                                    arg.type, arg.name));
                            }
                        }
                        body_stmts.emplace_back(llir::KernelLaunch::make(
                            ki.name, ki.template_args,
                            num_blocks_var, threads_per_block_var,
                            std::move(launch_args),
                            llir::lExpr(),
                            stream_var));
                        break;
                    }
                }
            }

            // 4-6. Precompute + prefix sum (if has_precompute_kernel)
            std::string counts_var = "count_offsets_" + std::to_string(phase);
            if (phase_info.has_precompute_kernel) {
                const llir::Struct_t *counts_struct = phase_info.counts_struct.as<llir::Struct_t>();

                // 4. Allocate count struct fields
                body_stmts.emplace_back(llir::RawCode::make(
                    counts_struct->name + "<index_t> " + counts_var + ";"));
                for (const auto &field : counts_struct->fields) {
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaMallocAsync((void**)&" + counts_var + "." + field.first +
                        ", num_threads * sizeof(index_t), stream);"));
                }

                // 5. Launch precompute kernel
                for (const auto &ki : kernel_infos) {
                    if (ki.kind == KernelInfo::Precompute && ki.phase == phase) {
                        std::vector<llir::lExpr> launch_args;
                        for (const auto &arg : ki.args) {
                            if (arg.name == "partitions") {
                                launch_args.push_back(llir::lVar::make(
                                    llir::Generic_t::make(partition_struct->name + "<index_t>"),
                                    partition_var));
                            } else if (arg.name == "count_offsets") {
                                launch_args.push_back(llir::lVar::make(
                                    llir::Generic_t::make(counts_struct->name + "<index_t>"),
                                    counts_var));
                            } else if (arg.name == "per_thread_work") {
                                launch_args.push_back(llir::lVar::make(index_t, ptw_name));
                            } else {
                                launch_args.push_back(llir::lVar::make(
                                    arg.type, arg.name));
                            }
                        }
                        body_stmts.emplace_back(llir::KernelLaunch::make(
                            ki.name, ki.template_args,
                            num_blocks_var, threads_per_block_var,
                            std::move(launch_args),
                            llir::lExpr(),
                            stream_var));
                        break;
                    }
                }

                // 6. CUB prefix sum (two-pass pattern) for each count field
                for (const auto &field : counts_struct->fields) {
                    std::string prefix_var = counts_var + "_" + field.first + "_prefix";
                    body_stmts.emplace_back(llir::RawCode::make(
                        "index_t* " + prefix_var + ";"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaMallocAsync((void**)&" + prefix_var +
                        ", (num_threads + 1) * sizeof(index_t), stream);"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaMemsetAsync(" + prefix_var + ", 0, (num_threads + 1) * sizeof(index_t), stream);"));

                    // CUB two-pass
                    std::string temp_var = "d_temp_storage_" + std::to_string(phase) + "_" + field.first;
                    std::string temp_bytes = "temp_storage_bytes_" + std::to_string(phase) + "_" + field.first;
                    body_stmts.emplace_back(llir::RawCode::make(
                        "void* " + temp_var + " = nullptr;"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        "size_t " + temp_bytes + " = 0;"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cub::DeviceScan::InclusiveSum(" + temp_var + ", " + temp_bytes +
                        ", " + counts_var + "." + field.first +
                        ", " + prefix_var + " + 1, num_threads, stream);"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaMallocAsync(&" + temp_var + ", " + temp_bytes + ", stream);"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cub::DeviceScan::InclusiveSum(" + temp_var + ", " + temp_bytes +
                        ", " + counts_var + "." + field.first +
                        ", " + prefix_var + " + 1, num_threads, stream);"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaFreeAsync(" + temp_var + ", stream);"));

                    // Replace the count field with the prefix-summed version
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaFreeAsync(" + counts_var + "." + field.first + ", stream);"));
                    body_stmts.emplace_back(llir::RawCode::make(
                        counts_var + "." + field.first + " = " + prefix_var + ";"));
                }

                // 7. Read nnz from device
                // For each sparse dimension in the result, read the total count
                bool has_nnz_read = false;
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection_level; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        std::string nnz_name = "nnz_" + idx + "_" + std::to_string(phase);
                        std::string field_name = "dim_" + idx + "_count";
                        std::string prefix_var = counts_var + "_" + field_name + "_prefix";
                        body_stmts.emplace_back(llir::RawCode::make(
                            "index_t " + nnz_name + ";"));
                        body_stmts.emplace_back(llir::RawCode::make(
                            "cudaMemcpyAsync(&" + nnz_name + ", " + prefix_var +
                            " + num_threads, sizeof(index_t), cudaMemcpyDeviceToHost, stream);"));
                        has_nnz_read = true;
                    }
                }
                if (has_nnz_read) {
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaStreamSynchronize(stream);"));
                }

                // 8. Allocate output tensor arrays based on nnz
                // Allocate indices and values arrays for sparse dimensions
                // Skip levels already allocated in previous phases
                int alloc_start = prev_phase_max_sparse + 1;
                for (int lvl = alloc_start; lvl <= phase_info.current_sparse_intersection_level; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        std::string nnz_name = "nnz_" + idx + "_" + std::to_string(phase);
                        body_stmts.emplace_back(llir::RawCode::make(
                            "cudaMallocAsync((void**)&" + result_tensor.tensor_name + "." +
                            result_tensor.get_indices_field_name(idx) +
                            ", " + nnz_name + " * sizeof(index_t), stream);"));
                    }
                }
                // Allocate values if this is the last phase
                if (phase == num_phases - 1) {
                    // Find the innermost sparse dimension's nnz for values allocation
                    std::string values_nnz;
                    for (int lvl = phase_info.current_sparse_intersection_level; lvl >= 0; lvl--) {
                        auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                        if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                            values_nnz = "nnz_" + idx + "_" + std::to_string(phase);
                            break;
                        }
                    }
                    if (!values_nnz.empty()) {
                        body_stmts.emplace_back(llir::RawCode::make(
                            "cudaMallocAsync((void**)&" + result_tensor.tensor_name + ".values" +
                            ", " + values_nnz + " * sizeof(value_t), stream);"));
                    }
                }

                // Allocate offsets arrays for non-outermost sparse dimensions
                // (outermost sparse dim has no offsets field in the struct)
                for (int lvl = 1; lvl <= phase_info.current_sparse_intersection_level; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        // Offsets size depends on the parent dimension
                        std::string offsets_size;
                        auto parent_idx = result_tensor.tensor_type.format.levels[lvl - 1].index;
                        if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(parent_idx))) {
                            offsets_size = "nnz_" + parent_idx + "_" + std::to_string(phase) + " + 1";
                        } else {
                            offsets_size = result_tensor.tensor_name + "." +
                                          result_tensor.get_size_field_name(parent_idx) + " + 1";
                        }
                        body_stmts.emplace_back(llir::RawCode::make(
                            "cudaMallocAsync((void**)&" + result_tensor.tensor_name + "." +
                            result_tensor.get_offsets_field_name(idx) +
                            ", (" + offsets_size + ") * sizeof(index_t), stream);"));
                        body_stmts.emplace_back(llir::RawCode::make(
                            "cudaMemsetAsync(" + result_tensor.tensor_name + "." +
                            result_tensor.get_offsets_field_name(idx) +
                            ", 0, (" + offsets_size + ") * sizeof(index_t), stream);"));
                    }
                }

                // Defer result length/nnz updates until after compute launch.
                // Compute kernels in later phases may still need previous phase
                // length fields while traversing intermediate buffers.
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection_level; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        std::string nnz_name = "nnz_" + idx + "_" + std::to_string(phase);
                        delayed_result_field_updates.push_back(
                            result_tensor.tensor_name + "." +
                            result_tensor.get_length_field_name(idx) +
                            " = " + nnz_name + ";");
                    }
                }
                if (phase == num_phases - 1) {
                    // Set nnz to innermost sparse dim's count
                    for (int lvl = phase_info.current_sparse_intersection_level; lvl >= 0; lvl--) {
                        auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                        if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                            delayed_result_field_updates.push_back(
                                result_tensor.tensor_name + ".nnz = nnz_" +
                                idx + "_" + std::to_string(phase) + ";");
                            break;
                        }
                    }
                }
            }

            // For multi-phase expressions, compute the output nnz variable name
            // for the outermost sparse dim of this phase. This is used for
            // pos_map allocation and T_work_offsets sizing.
            bool is_last_phase = (phase == num_phases - 1);
            std::string phase_outermost_nnz;
            if (!is_last_phase) {
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection_level; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        phase_outermost_nnz = "nnz_" + idx + "_" + std::to_string(phase);
                        break;
                    }
                }
            }

            // Early return if this phase produced zero output elements
            // and there are more phases to come (prevents out-of-bounds reads).
            if (!is_last_phase && !phase_outermost_nnz.empty()) {
                // Emit: if (nnz == 0) { set remaining fields to 0, free intermediates, return; }
                std::string guard;
                guard += "if (" + phase_outermost_nnz + " == 0) {\n";
                // Set remaining output fields to zero
                for (int lvl = phase_info.current_sparse_intersection_level + 1;
                     lvl < (int)result_tensor.tensor_type.format.levels.size(); lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        guard += "  " + result_tensor.tensor_name + "." +
                                 result_tensor.get_length_field_name(idx) + " = 0;\n";
                    }
                }
                guard += "  " + result_tensor.tensor_name + ".nnz = 0;\n";
                // Free partition and count structs
                for (const auto &field : partition_struct->fields) {
                    guard += "  cudaFreeAsync(" + partition_var + "." + field.first + ", stream);\n";
                }
                if (phase_info.has_precompute_kernel) {
                    const llir::Struct_t *counts_struct = phase_info.counts_struct.as<llir::Struct_t>();
                    for (const auto &field : counts_struct->fields) {
                        guard += "  cudaFreeAsync(" + counts_var + "." + field.first + ", stream);\n";
                    }
                }
                guard += "  return;\n";
                guard += "}";
                body_stmts.emplace_back(llir::RawCode::make(guard));
            }

            // Allocate result_to_operand_pos_map if this is not the last phase.
            if (!is_last_phase && !phase_outermost_nnz.empty()) {
                auto forall_loops = get_forall_loops();
                auto empty_map = std::map<std::string, TensorLowerer>();
                BaseKernelLowerer base_lowerer(operand_tensors, result_tensor, empty_map,
                                               forall_loops, -1, -1, -1);
                std::string pos_map_struct = base_lowerer.get_result_to_operand_pos_map_struct_name();
                std::string pos_map_var = base_lowerer.get_result_to_operand_pos_map_var_name();

                body_stmts.emplace_back(llir::RawCode::make(
                    pos_map_struct + "<index_t> " + pos_map_var + ";"));
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection_level; lvl++) {
                    if (lvl >= (int)forall_loops.size() - 1) continue;
                    const Forall *forall = forall_loops[lvl].as<Forall>();
                    std::string forall_idx = forall->idx;
                    for (auto &[name, tensor] : operand_tensors) {
                        if (base_lowerer.has_result_to_operand_pos_field(forall, tensor)) {
                            std::string field = tensor.get_iterator_suffix(forall_idx);
                            body_stmts.emplace_back(llir::RawCode::make(
                                "cudaMallocAsync((void**)&" + pos_map_var + "." + field +
                                ", " + phase_outermost_nnz + " * sizeof(index_t), stream);"));
                        }
                    }
                }
            }

            // Allocate T_work_offsets if this is not the last phase.
            // Size is the output nnz of the outermost sparse dim (not total_work),
            // since the compute kernel only writes one entry per output element.
            if (!is_last_phase) {
                body_stmts.emplace_back(llir::RawCode::make(
                    "index_t* T_work_offsets;"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaMallocAsync((void**)&T_work_offsets, " + phase_outermost_nnz +
                    " * sizeof(index_t), stream);"));
            }

            // 9. Launch compute kernel
            for (const auto &ki : kernel_infos) {
                if (ki.kind == KernelInfo::Compute && ki.phase == phase) {
                    std::vector<llir::lExpr> launch_args;
                    for (const auto &arg : ki.args) {
                        if (arg.name == "partitions") {
                            launch_args.push_back(llir::lVar::make(
                                llir::Generic_t::make(partition_struct->name + "<index_t>"),
                                partition_var));
                        } else if (arg.name == "count_offsets") {
                            launch_args.push_back(llir::lVar::make(
                                llir::Generic_t::make("result_per_thread_count<index_t>"),
                                counts_var));
                        } else if (arg.name == "per_thread_work") {
                            launch_args.push_back(llir::lVar::make(index_t, ptw_name));
                        } else {
                            launch_args.push_back(llir::lVar::make(
                                arg.type, arg.name));
                        }
                    }
                    body_stmts.emplace_back(llir::KernelLaunch::make(
                        ki.name, ki.template_args,
                        num_blocks_var, threads_per_block_var,
                        std::move(launch_args),
                        llir::lExpr(),
                        stream_var));
                    break;
                }
            }

            // Prefix sum T_work_offsets if not the last phase.
            // Use the output nnz (not total_work) since only nnz entries are valid.
            if (!is_last_phase) {
                std::string tw_size = phase_outermost_nnz;
                std::string temp_var = "d_temp_tw_" + std::to_string(phase);
                std::string temp_bytes = "temp_bytes_tw_" + std::to_string(phase);
                body_stmts.emplace_back(llir::RawCode::make(
                    "index_t* T_work_offsets_prefix;"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaMallocAsync((void**)&T_work_offsets_prefix, (" +
                    tw_size + " + 1) * sizeof(index_t), stream);"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaMemsetAsync(T_work_offsets_prefix, 0, (" +
                    tw_size + " + 1) * sizeof(index_t), stream);"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "void* " + temp_var + " = nullptr;"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "size_t " + temp_bytes + " = 0;"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cub::DeviceScan::InclusiveSum(" + temp_var + ", " + temp_bytes +
                    ", T_work_offsets, T_work_offsets_prefix + 1, " + tw_size + ", stream);"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaMallocAsync(&" + temp_var + ", " + temp_bytes + ", stream);"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cub::DeviceScan::InclusiveSum(" + temp_var + ", " + temp_bytes +
                    ", T_work_offsets, T_work_offsets_prefix + 1, " + tw_size + ", stream);"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaFreeAsync(" + temp_var + ", stream);"));
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaFreeAsync(T_work_offsets, stream);"));
                // Reassign so next phase's kernels use the prefix sum result
                body_stmts.emplace_back(llir::RawCode::make(
                    "T_work_offsets = T_work_offsets_prefix;"));
            }

            for (const auto &stmt : delayed_result_field_updates) {
                body_stmts.emplace_back(llir::RawCode::make(stmt));
            }

            // 10. Free partition struct intermediates
            for (const auto &field : partition_struct->fields) {
                body_stmts.emplace_back(llir::RawCode::make(
                    "cudaFreeAsync(" + partition_var + "." + field.first + ", stream);"));
            }

            // Free count struct intermediates (if precompute exists and this is the last phase)
            // Count structs are consumed by compute kernel, so free after compute
            if (phase_info.has_precompute_kernel && is_last_phase) {
                const llir::Struct_t *counts_struct = phase_info.counts_struct.as<llir::Struct_t>();
                for (const auto &field : counts_struct->fields) {
                    body_stmts.emplace_back(llir::RawCode::make(
                        "cudaFreeAsync(" + counts_var + "." + field.first + ", stream);"));
                }
            }

            // Track outermost nnz for next phase's T_work_offsets indexing
            if (!phase_outermost_nnz.empty()) {
                prev_phase_outermost_nnz = phase_outermost_nnz;
            }
            prev_phase_max_sparse = phase_info.current_sparse_intersection_level;
        }

        llir::lStmt body = llir::Sequence::make(std::move(body_stmts));
        printer.print(llir::Function::make(std::move(generics), std::move(attributes),
                                           std::move(func_args), std::move(ret_type),
                                           name, std::move(body)));
    }

    void CINLowerer::lower_flat_wrapper(const std::string &op_name) {
        if (result_tensor.tensor_type.format.is_all_coordinate()) {
            lower_coo_flat_wrapper(op_name);
            return;
        }

        llir::lType value_t = llir::Generic_t::make("value_t");
        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<llir::Function::Attribute> attributes = {llir::Function::host};
        llir::lType ret_type = llir::Generic_t::make("void");

        std::vector<llir::Function::Argument> args;

        // For each operand: add all struct fields as flat args
        // All operand args are marked mutating to avoid const qualifiers,
        // since the operand structs store non-const pointers internally.
        for (const auto &[name, tensor] : operand_tensors) {
            llir::lType struct_type = tensor.lower_tensor_struct_definition();
            const auto *st = struct_type.as<llir::Struct_t>();
            for (const auto &[field_name, field_type] : st->fields) {
                args.push_back({
                    .mutating = true,
                    .type = field_type,
                    .name = name + "_" + field_name
                });
            }
        }

        // For result: add size fields as input
        for (size_t i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            args.push_back({
                .mutating = true,
                .type = index_t,
                .name = "result_" + result_tensor.get_size_field_name(idx)
            });
        }

        // Output references: nnz
        args.push_back({
            .mutating = true,
            .by_reference = true,
            .type = index_t,
            .name = "out_nnz"
        });

        // Output references: length for non-innermost sparse dims
        // (innermost sparse dim length == nnz, already covered by out_nnz)
        {
            // Find innermost sparse level
            int innermost_sparse = -1;
            for (int i = (int)result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
                auto idx = result_tensor.tensor_type.format.levels[i].index;
                if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                    innermost_sparse = i;
                    break;
                }
            }
            for (int i = 0; i < (int)result_tensor.tensor_type.format.levels.size(); i++) {
                auto idx = result_tensor.tensor_type.format.levels[i].index;
                if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx)) && i != innermost_sparse) {
                    args.push_back({
                        .mutating = true,
                        .by_reference = true,
                        .type = index_t,
                        .name = "out_" + result_tensor.get_length_field_name(idx)
                    });
                }
            }
        }

        // Output references: indices (and offsets for non-outermost) for each sparse dim
        for (int i = (int)result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                args.push_back({
                    .mutating = true,
                    .by_reference = true,
                    .type = llir::Ptr_t::make(index_t),
                    .name = "out_" + result_tensor.get_indices_field_name(idx)
                });
                if (i != 0) {
                    args.push_back({
                        .mutating = true,
                        .by_reference = true,
                        .type = llir::Ptr_t::make(index_t),
                        .name = "out_" + result_tensor.get_offsets_field_name(idx)
                    });
                }
            }
        }

        // Output reference: values
        args.push_back({
            .mutating = true,
            .by_reference = true,
            .type = llir::Ptr_t::make(value_t),
            .name = "out_values"
        });

        // Build function body
        std::vector<llir::lStmt> body_stmts;

        // Create and populate operand structs
        for (const auto &[name, tensor] : operand_tensors) {
            body_stmts.push_back(llir::RawCode::make(
                tensor.get_struct_name() + "<index_t, value_t> " + name + ";"));
            llir::lType struct_type = tensor.lower_tensor_struct_definition();
            const auto *st = struct_type.as<llir::Struct_t>();
            for (const auto &[field_name, field_type] : st->fields) {
                body_stmts.push_back(llir::RawCode::make(
                    name + "." + field_name + " = " + name + "_" + field_name + ";"));
            }
        }

        // Create result struct and populate size fields
        body_stmts.push_back(llir::RawCode::make(
            result_tensor.get_struct_name() + "<index_t, value_t> " +
            result_tensor.tensor_name + ";"));
        for (size_t i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            std::string size_field = result_tensor.get_size_field_name(idx);
            body_stmts.push_back(llir::RawCode::make(
                result_tensor.tensor_name + "." + size_field +
                " = result_" + size_field + ";"));
        }

        // Call the compute function
        std::string call = result_tensor.tensor_name + "_compute<index_t, value_t>(";
        bool first = true;
        for (const auto &[name, tensor] : operand_tensors) {
            if (!first) call += ", ";
            call += name;
            first = false;
        }
        call += ", " + result_tensor.tensor_name + ");";
        body_stmts.push_back(llir::RawCode::make(call));

        // Extract outputs
        body_stmts.push_back(llir::RawCode::make(
            "out_nnz = " + result_tensor.tensor_name + ".nnz;"));
        // Extract length for non-innermost sparse dims
        {
            int innermost_sparse = -1;
            for (int i = (int)result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
                auto idx = result_tensor.tensor_type.format.levels[i].index;
                if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                    innermost_sparse = i;
                    break;
                }
            }
            for (int i = 0; i < (int)result_tensor.tensor_type.format.levels.size(); i++) {
                auto idx = result_tensor.tensor_type.format.levels[i].index;
                if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx)) && i != innermost_sparse) {
                    body_stmts.push_back(llir::RawCode::make(
                        "out_" + result_tensor.get_length_field_name(idx) + " = " +
                        result_tensor.tensor_name + "." +
                        result_tensor.get_length_field_name(idx) + ";"));
                }
            }
        }
        for (int i = (int)result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                body_stmts.push_back(llir::RawCode::make(
                    "out_" + result_tensor.get_indices_field_name(idx) + " = " +
                    result_tensor.tensor_name + "." +
                    result_tensor.get_indices_field_name(idx) + ";"));
                if (i != 0) {
                    body_stmts.push_back(llir::RawCode::make(
                        "out_" + result_tensor.get_offsets_field_name(idx) + " = " +
                        result_tensor.tensor_name + "." +
                        result_tensor.get_offsets_field_name(idx) + ";"));
                }
            }
        }
        body_stmts.push_back(llir::RawCode::make(
            "out_values = " + result_tensor.tensor_name + ".values;"));

        llir::lStmt body = llir::Sequence::make(std::move(body_stmts));
        printer.print(llir::Function::make(std::move(generics), std::move(attributes),
                                           std::move(args), std::move(ret_type),
                                           op_name, std::move(body)));
    }

    // =========================================================================
    // COO (all-Coordinate format) codegen
    // =========================================================================

    void CINLowerer::lower_coo_cin() {
        // Emit struct definitions for operand and result tensors
        for (auto &it : operand_tensors) {
            printer.print(it.second.lower_tensor_struct_definition());
        }
        printer.print(result_tensor.lower_tensor_struct_definition());

        // COO count struct (single field: nnz_count)
        {
            std::vector<std::string> generics = {"index_t"};
            std::vector<std::pair<std::string, llir::lType>> fields;
            fields.emplace_back("nnz_count", llir::Ptr_t::make(index_t));
            printer.print(llir::Struct_t::make("result_per_thread_count",
                                               std::move(fields), std::move(generics)));
        }

        // COO partition struct (one position field per operand)
        {
            std::vector<std::string> generics = {"index_t"};
            std::vector<std::pair<std::string, llir::lType>> fields;
            for (const auto &it : operand_tensors) {
                fields.emplace_back(it.second.tensor_name + "_p",
                                    llir::Ptr_t::make(index_t));
            }
            std::string loop_suffix;
            for (const auto &lvl : result_tensor.tensor_type.format.levels) {
                loop_suffix += lvl.index;
            }
            printer.print(llir::Struct_t::make("partition_" + loop_suffix,
                                               std::move(fields), std::move(generics)));
        }

        lower_coo_compare_function();
        lower_coo_partition_kernel();
        lower_coo_precompute_kernel();
        lower_coo_compute_kernel();
        lower_coo_host_function();
    }

    void CINLowerer::lower_coo_compare_function() {
        // For each pair of operand tensors, generate a comparison function.
        // For 2-operand add, we need one compare function.
        // coo_compare(A, a_p, B, b_p) → -1, 0, +1

        auto operand_list = std::vector<std::pair<std::string, TensorLowerer>>(
            operand_tensors.begin(), operand_tensors.end());

        // For binary add we have exactly 2 operands
        internal_assert(operand_list.size() == 2)
            << "COO codegen currently supports binary add (2 operands)";

        auto &tensor_a = operand_list[0].second;
        auto &tensor_b = operand_list[1].second;

        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::device, llir::Function::inline_};

        std::vector<llir::Function::Argument> args;
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make(tensor_a.get_struct_name() + "<index_t, value_t>"),
            .name = tensor_a.tensor_name});
        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "a_p"});
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make(tensor_b.get_struct_name() + "<index_t, value_t>"),
            .name = tensor_b.tensor_name});
        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "b_p"});

        llir::lExpr a_var = llir::lVar::make(
            llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name);
        llir::lExpr b_var = llir::lVar::make(
            llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name);
        llir::lExpr a_p = llir::lVar::make(index_t, "a_p");
        llir::lExpr b_p = llir::lVar::make(index_t, "b_p");

        // Build comparison chain: for each dimension, compare coordinate
        std::vector<llir::lStmt> stmts;
        for (const auto &level : result_tensor.tensor_type.format.levels) {
            std::string idx = level.index;
            // A.dim_X_indices[a_p] vs B.dim_X_indices[b_p]
            llir::lExpr a_coord = llir::lFieldAccess::make(a_var, tensor_a.get_indices_field_name(idx))[a_p];
            llir::lExpr b_coord = llir::lFieldAccess::make(b_var, tensor_b.get_indices_field_name(idx))[b_p];

            stmts.emplace_back(llir::IfElse::make(
                a_coord < b_coord,
                llir::Return::make(llir::lConst::make((int64_t)-1)),
                llir::lStmt()));
            stmts.emplace_back(llir::IfElse::make(
                llir::lFieldAccess::make(a_var, tensor_a.get_indices_field_name(idx))[a_p] >
                    llir::lFieldAccess::make(b_var, tensor_b.get_indices_field_name(idx))[b_p],
                llir::Return::make(llir::lConst::make((int64_t)1)),
                llir::lStmt()));
        }
        stmts.emplace_back(llir::Return::make(llir::lConst::make((int64_t)0)));

        llir::lStmt body = llir::Sequence::make(std::move(stmts));
        printer.print(llir::Function::make(
            std::move(generics), std::move(attributes), std::move(args),
            index_t, "coo_compare", std::move(body)));
    }

    void CINLowerer::lower_coo_partition_kernel() {
        auto operand_list = std::vector<std::pair<std::string, TensorLowerer>>(
            operand_tensors.begin(), operand_tensors.end());
        auto &tensor_a = operand_list[0].second;
        auto &tensor_b = operand_list[1].second;

        std::string loop_suffix;
        for (const auto &lvl : result_tensor.tensor_type.format.levels) {
            loop_suffix += lvl.index;
        }

        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::global};

        std::vector<llir::Function::Argument> args;
        for (const auto &it : operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false,
                .type = llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                .name = it.second.tensor_name});
        }
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Generic_t::make("partition_" + loop_suffix + "<index_t>"),
            .name = "partitions"});
        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "per_thread_work"});
        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "total_work"});

        // Store kernel info for host function
        kernel_infos.push_back({
            .name = "partition_" + loop_suffix + "_kernel",
            .template_args = {"index_t", "value_t"},
            .args = args,
            .kind = KernelInfo::Partition,
            .phase = 0
        });

        llir::lExpr a_nnz = llir::lFieldAccess::make(
            llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name),
            "nnz");
        llir::lExpr b_nnz = llir::lFieldAccess::make(
            llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name),
            "nnz");

        std::vector<llir::lStmt> stmts;

        // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        stmts.emplace_back(llir::Declare::make(
            llir::Int_t::make(32), "thread_id",
            llir::lVar::make(index_t, "blockIdx.x") * llir::lVar::make(index_t, "blockDim.x") +
                llir::lVar::make(index_t, "threadIdx.x")));

        llir::lExpr tid = llir::lVar::make(llir::Int_t::make(32), "thread_id");

        // index_t diag = thread_id * per_thread_work;
        stmts.emplace_back(llir::Declare::make(
            index_t, "diag",
            tid * llir::lVar::make(index_t, "per_thread_work")));

        llir::lExpr diag = llir::lVar::make(index_t, "diag");

        // clamp diag
        stmts.emplace_back(llir::IfElse::make(
            diag > llir::lVar::make(index_t, "total_work"),
            llir::Store::make(diag, llir::lVar::make(index_t, "total_work")),
            llir::lStmt()));

        // a_top = min(diag, A.nnz)
        stmts.emplace_back(llir::Declare::make(
            index_t, "a_top",
            llir::lBinOp::make(llir::lBinOp::Min, diag, a_nnz)));
        // a_bot = max(0, diag - B.nnz)
        stmts.emplace_back(llir::Declare::make(
            index_t, "a_bot",
            llir::lBinOp::make(llir::lBinOp::Max, llir::lConst::make((int64_t)0),
                               diag - b_nnz)));

        llir::lExpr a_top = llir::lVar::make(index_t, "a_top");
        llir::lExpr a_bot = llir::lVar::make(index_t, "a_bot");

        // while (a_top > a_bot):
        //   a_mid = a_bot + (a_top - a_bot) / 2
        //   b_mid = diag - a_mid - 1
        //   if (coo_compare(A, a_mid, B, b_mid) > 0):
        //     a_top = a_mid
        //   else:
        //     a_bot = a_mid + 1
        {
            std::vector<llir::lStmt> while_stmts;
            llir::lExpr a_mid_expr = a_bot + (a_top - a_bot) / llir::lConst::make((int64_t)2);
            while_stmts.emplace_back(llir::Declare::make(index_t, "a_mid", a_mid_expr));
            llir::lExpr a_mid = llir::lVar::make(index_t, "a_mid");
            while_stmts.emplace_back(llir::Declare::make(index_t, "b_mid",
                diag - a_mid - llir::lConst::make((int64_t)1)));
            llir::lExpr b_mid = llir::lVar::make(index_t, "b_mid");

            // coo_compare call
            llir::lExpr cmp = llir::lFunctionCall::make("coo_compare<index_t, value_t>",
                {llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name),
                 a_mid,
                 llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name),
                 b_mid});

            while_stmts.emplace_back(llir::IfElse::make(
                cmp > llir::lConst::make((int64_t)0),
                llir::Store::make(a_top, a_mid),
                llir::Store::make(a_bot, a_mid + llir::lConst::make((int64_t)1))));

            stmts.emplace_back(llir::While::make(
                a_top > a_bot,
                llir::Sequence::make(std::move(while_stmts))));
        }

        // partitions.A_p[thread_id] = a_top;
        // partitions.B_p[thread_id] = diag - a_top;
        llir::lExpr partitions = llir::lVar::make(
            llir::Generic_t::make("partition_" + loop_suffix + "<index_t>"), "partitions");
        stmts.emplace_back(llir::Store::make(
            llir::lFieldAccess::make(partitions, tensor_a.tensor_name + "_p")[tid],
            a_top));
        stmts.emplace_back(llir::Store::make(
            llir::lFieldAccess::make(partitions, tensor_b.tensor_name + "_p")[tid],
            diag - a_top));

        stmts.emplace_back(llir::Return::make());

        llir::lStmt body = llir::Sequence::make(std::move(stmts));
        printer.print(llir::Function::make(
            std::move(generics), std::move(attributes), std::move(args),
            llir::Generic_t::make("void"),
            "partition_" + loop_suffix + "_kernel",
            std::move(body)));
    }

    void CINLowerer::lower_coo_precompute_kernel() {
        auto operand_list = std::vector<std::pair<std::string, TensorLowerer>>(
            operand_tensors.begin(), operand_tensors.end());
        auto &tensor_a = operand_list[0].second;
        auto &tensor_b = operand_list[1].second;

        std::string loop_suffix;
        for (const auto &lvl : result_tensor.tensor_type.format.levels) {
            loop_suffix += lvl.index;
        }

        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::global};

        std::vector<llir::Function::Argument> args;
        for (const auto &it : operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false,
                .type = llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                .name = it.second.tensor_name});
        }
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make("partition_" + loop_suffix + "<index_t>"),
            .name = "partitions"});
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Generic_t::make("result_per_thread_count<index_t>"),
            .name = "count_offsets"});
        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "per_thread_work"});

        kernel_infos.push_back({
            .name = "precompute_" + loop_suffix + "_kernel",
            .template_args = {"index_t", "value_t"},
            .args = args,
            .kind = KernelInfo::Precompute,
            .phase = 0
        });

        std::vector<llir::lStmt> stmts;

        // thread_id setup
        stmts.emplace_back(llir::Declare::make(
            llir::Int_t::make(32), "thread_id",
            llir::lVar::make(index_t, "blockIdx.x") * llir::lVar::make(index_t, "blockDim.x") +
                llir::lVar::make(index_t, "threadIdx.x")));
        stmts.emplace_back(llir::Declare::make(
            llir::Int_t::make(32), "max_thread_id",
            llir::lVar::make(index_t, "gridDim.x") * llir::lVar::make(index_t, "blockDim.x") -
                llir::lConst::make(1)));

        llir::lExpr tid = llir::lVar::make(llir::Int_t::make(32), "thread_id");
        llir::lExpr max_tid = llir::lVar::make(llir::Int_t::make(32), "max_thread_id");

        llir::lExpr partitions = llir::lVar::make(
            llir::Generic_t::make("partition_" + loop_suffix + "<index_t>"), "partitions");
        llir::lExpr a_nnz = llir::lFieldAccess::make(
            llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name), "nnz");
        llir::lExpr b_nnz = llir::lFieldAccess::make(
            llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name), "nnz");

        // Load partition boundaries
        stmts.emplace_back(llir::Declare::make(index_t, "start_a",
            llir::lFieldAccess::make(partitions, tensor_a.tensor_name + "_p")[tid]));
        stmts.emplace_back(llir::Declare::make(index_t, "end_a",
            llir::lSelect::make(tid < max_tid,
                llir::lFieldAccess::make(partitions, tensor_a.tensor_name + "_p")[tid + llir::lConst::make(1)],
                a_nnz)));
        stmts.emplace_back(llir::Declare::make(index_t, "start_b",
            llir::lFieldAccess::make(partitions, tensor_b.tensor_name + "_p")[tid]));
        stmts.emplace_back(llir::Declare::make(index_t, "end_b",
            llir::lSelect::make(tid < max_tid,
                llir::lFieldAccess::make(partitions, tensor_b.tensor_name + "_p")[tid + llir::lConst::make(1)],
                b_nnz)));

        llir::lExpr start_a = llir::lVar::make(index_t, "start_a");
        llir::lExpr end_a = llir::lVar::make(index_t, "end_a");
        llir::lExpr start_b = llir::lVar::make(index_t, "start_b");
        llir::lExpr end_b = llir::lVar::make(index_t, "end_b");

        // Merge loop counting output elements
        stmts.emplace_back(llir::Declare::make(index_t, "a_p", start_a));
        stmts.emplace_back(llir::Declare::make(index_t, "b_p", start_b));
        stmts.emplace_back(llir::Declare::make(index_t, "count", llir::lConst::make((int64_t)0)));

        llir::lExpr a_p = llir::lVar::make(index_t, "a_p");
        llir::lExpr b_p = llir::lVar::make(index_t, "b_p");
        llir::lExpr count = llir::lVar::make(index_t, "count");

        // Boundary overlap fix: if the merge-path split an equal pair
        // across threads, skip the B element that belongs to the previous
        // thread's overlap fixup.
        {
            llir::lExpr boundary_cmp = llir::lFunctionCall::make("coo_compare<index_t, value_t>",
                {llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name),
                 start_a - llir::lConst::make((int64_t)1),
                 llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name),
                 start_b});
            llir::lExpr cond =
                tid > llir::lConst::make((int64_t)0) &&
                start_a > llir::lConst::make((int64_t)0) &&
                start_b < end_b &&
                boundary_cmp == llir::lConst::make((int64_t)0);
            stmts.emplace_back(llir::IfElse::make(cond,
                llir::BaseExpr::make(llir::lIncrement::make(b_p)),
                llir::lStmt()));
        }

        {
            std::vector<llir::lStmt> while_stmts;

            llir::lExpr cmp = llir::lFunctionCall::make("coo_compare<index_t, value_t>",
                {llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name),
                 a_p,
                 llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name),
                 b_p});

            while_stmts.emplace_back(llir::Declare::make(index_t, "cmp", cmp));
            llir::lExpr cmp_var = llir::lVar::make(index_t, "cmp");

            while_stmts.emplace_back(llir::Accumulate::make(count, llir::lConst::make((int64_t)1)));
            while_stmts.emplace_back(llir::IfElse::make(
                cmp_var <= llir::lConst::make((int64_t)0),
                llir::BaseExpr::make(llir::lIncrement::make(a_p)),
                llir::lStmt()));
            while_stmts.emplace_back(llir::IfElse::make(
                cmp_var >= llir::lConst::make((int64_t)0),
                llir::BaseExpr::make(llir::lIncrement::make(b_p)),
                llir::lStmt()));

            stmts.emplace_back(llir::While::make(
                a_p < end_a && b_p < end_b,
                llir::Sequence::make(std::move(while_stmts))));
        }

        // Drain remaining
        stmts.emplace_back(llir::Accumulate::make(count,
            (end_a - a_p) + (end_b - b_p)));

        // Store count
        stmts.emplace_back(llir::Store::make(
            llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make("result_per_thread_count<index_t>"), "count_offsets"),
                "nnz_count")[tid],
            count));

        stmts.emplace_back(llir::Return::make());

        llir::lStmt body = llir::Sequence::make(std::move(stmts));
        printer.print(llir::Function::make(
            std::move(generics), std::move(attributes), std::move(args),
            llir::Generic_t::make("void"),
            "precompute_" + loop_suffix + "_kernel",
            std::move(body)));
    }

    void CINLowerer::lower_coo_compute_kernel() {
        auto operand_list = std::vector<std::pair<std::string, TensorLowerer>>(
            operand_tensors.begin(), operand_tensors.end());
        auto &tensor_a = operand_list[0].second;
        auto &tensor_b = operand_list[1].second;

        std::string loop_suffix;
        for (const auto &lvl : result_tensor.tensor_type.format.levels) {
            loop_suffix += lvl.index;
        }

        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::global};

        std::vector<llir::Function::Argument> args;
        for (const auto &it : operand_tensors) {
            args.emplace_back(llir::Function::Argument{
                .mutating = false,
                .type = llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                .name = it.second.tensor_name});
        }
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make("partition_" + loop_suffix + "<index_t>"),
            .name = "partitions"});
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make("result_per_thread_count<index_t>"),
            .name = "count_offsets"});
        args.emplace_back(llir::Function::Argument{
            .mutating = false, .type = index_t, .name = "per_thread_work"});
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Generic_t::make(result_tensor.get_struct_name() + "<index_t, value_t>"),
            .name = result_tensor.tensor_name});

        kernel_infos.push_back({
            .name = "compute_" + loop_suffix + "_kernel",
            .template_args = {"index_t", "value_t"},
            .args = args,
            .kind = KernelInfo::Compute,
            .phase = 0
        });

        std::vector<llir::lStmt> stmts;

        // thread_id setup
        stmts.emplace_back(llir::Declare::make(
            llir::Int_t::make(32), "thread_id",
            llir::lVar::make(index_t, "blockIdx.x") * llir::lVar::make(index_t, "blockDim.x") +
                llir::lVar::make(index_t, "threadIdx.x")));
        stmts.emplace_back(llir::Declare::make(
            llir::Int_t::make(32), "max_thread_id",
            llir::lVar::make(index_t, "gridDim.x") * llir::lVar::make(index_t, "blockDim.x") -
                llir::lConst::make(1)));

        llir::lExpr tid = llir::lVar::make(llir::Int_t::make(32), "thread_id");
        llir::lExpr max_tid = llir::lVar::make(llir::Int_t::make(32), "max_thread_id");

        llir::lExpr partitions = llir::lVar::make(
            llir::Generic_t::make("partition_" + loop_suffix + "<index_t>"), "partitions");
        llir::lExpr a_nnz = llir::lFieldAccess::make(
            llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name), "nnz");
        llir::lExpr b_nnz = llir::lFieldAccess::make(
            llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name), "nnz");

        // Load partition boundaries
        stmts.emplace_back(llir::Declare::make(index_t, "start_a",
            llir::lFieldAccess::make(partitions, tensor_a.tensor_name + "_p")[tid]));
        stmts.emplace_back(llir::Declare::make(index_t, "end_a",
            llir::lSelect::make(tid < max_tid,
                llir::lFieldAccess::make(partitions, tensor_a.tensor_name + "_p")[tid + llir::lConst::make(1)],
                a_nnz)));
        stmts.emplace_back(llir::Declare::make(index_t, "start_b",
            llir::lFieldAccess::make(partitions, tensor_b.tensor_name + "_p")[tid]));
        stmts.emplace_back(llir::Declare::make(index_t, "end_b",
            llir::lSelect::make(tid < max_tid,
                llir::lFieldAccess::make(partitions, tensor_b.tensor_name + "_p")[tid + llir::lConst::make(1)],
                b_nnz)));

        llir::lExpr start_a = llir::lVar::make(index_t, "start_a");
        llir::lExpr end_a = llir::lVar::make(index_t, "end_a");
        llir::lExpr start_b = llir::lVar::make(index_t, "start_b");
        llir::lExpr end_b = llir::lVar::make(index_t, "end_b");

        // Load output offset from prefix sum
        stmts.emplace_back(llir::Declare::make(index_t, "out_p",
            llir::lFieldAccess::make(
                llir::lVar::make(llir::Generic_t::make("result_per_thread_count<index_t>"), "count_offsets"),
                "nnz_count")[tid]));

        llir::lExpr out_p = llir::lVar::make(index_t, "out_p");

        stmts.emplace_back(llir::Declare::make(index_t, "a_p", start_a));
        stmts.emplace_back(llir::Declare::make(index_t, "b_p", start_b));

        llir::lExpr a_p = llir::lVar::make(index_t, "a_p");
        llir::lExpr b_p = llir::lVar::make(index_t, "b_p");

        llir::lExpr result_var = llir::lVar::make(
            llir::Generic_t::make(result_tensor.get_struct_name()), result_tensor.tensor_name);
        llir::lExpr a_var = llir::lVar::make(
            llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name);
        llir::lExpr b_var = llir::lVar::make(
            llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name);

        // Boundary overlap fix (start): skip B element that was split
        // from the previous thread's equal A element.
        {
            llir::lExpr boundary_cmp = llir::lFunctionCall::make("coo_compare<index_t, value_t>",
                {llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name),
                 start_a - llir::lConst::make((int64_t)1),
                 llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name),
                 start_b});
            llir::lExpr cond =
                tid > llir::lConst::make((int64_t)0) &&
                start_a > llir::lConst::make((int64_t)0) &&
                start_b < end_b &&
                boundary_cmp == llir::lConst::make((int64_t)0);
            stmts.emplace_back(llir::IfElse::make(cond,
                llir::BaseExpr::make(llir::lIncrement::make(b_p)),
                llir::lStmt()));
        }

        // Helper to build coordinate copy statements from a source tensor
        auto make_coord_copies = [&](TensorLowerer &src, llir::lExpr src_p) {
            std::vector<llir::lStmt> copies;
            for (const auto &level : result_tensor.tensor_type.format.levels) {
                std::string idx = level.index;
                llir::lExpr dst = llir::lFieldAccess::make(result_var, result_tensor.get_indices_field_name(idx))[out_p];
                llir::lExpr src_val = llir::lFieldAccess::make(
                    llir::lVar::make(llir::Generic_t::make(src.get_struct_name()), src.tensor_name),
                    src.get_indices_field_name(idx))[src_p];
                copies.emplace_back(llir::Store::make(dst, src_val));
            }
            return copies;
        };

        // Main merge loop
        {
            std::vector<llir::lStmt> while_stmts;

            llir::lExpr cmp = llir::lFunctionCall::make("coo_compare<index_t, value_t>",
                {llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name),
                 a_p,
                 llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name),
                 b_p});

            while_stmts.emplace_back(llir::Declare::make(index_t, "cmp", cmp));
            llir::lExpr cmp_var = llir::lVar::make(index_t, "cmp");

            // cmp < 0: A only
            {
                std::vector<llir::lStmt> a_only;
                auto copies = make_coord_copies(tensor_a, a_p);
                a_only.insert(a_only.end(), copies.begin(), copies.end());
                a_only.emplace_back(llir::Store::make(
                    llir::lFieldAccess::make(result_var, "values")[out_p],
                    llir::lFieldAccess::make(a_var, "values")[a_p]));
                a_only.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(a_p)));

                // cmp > 0: B only
                std::vector<llir::lStmt> b_or_both;
                {
                    std::vector<llir::lStmt> b_only;
                    auto b_copies = make_coord_copies(tensor_b, b_p);
                    b_only.insert(b_only.end(), b_copies.begin(), b_copies.end());
                    b_only.emplace_back(llir::Store::make(
                        llir::lFieldAccess::make(result_var, "values")[out_p],
                        llir::lFieldAccess::make(b_var, "values")[b_p]));
                    b_only.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(b_p)));

                    // cmp == 0: Both
                    std::vector<llir::lStmt> both;
                    auto both_copies = make_coord_copies(tensor_a, a_p);
                    both.insert(both.end(), both_copies.begin(), both_copies.end());
                    both.emplace_back(llir::Store::make(
                        llir::lFieldAccess::make(result_var, "values")[out_p],
                        llir::lFieldAccess::make(a_var, "values")[a_p] +
                            llir::lFieldAccess::make(b_var, "values")[b_p]));
                    both.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(a_p)));
                    both.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(b_p)));

                    b_or_both.emplace_back(llir::IfElse::make(
                        cmp_var > llir::lConst::make((int64_t)0),
                        llir::Sequence::make(std::move(b_only)),
                        llir::Sequence::make(std::move(both))));
                }

                while_stmts.emplace_back(llir::IfElse::make(
                    cmp_var < llir::lConst::make((int64_t)0),
                    llir::Sequence::make(std::move(a_only)),
                    llir::Sequence::make(std::move(b_or_both))));
            }

            while_stmts.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(out_p)));

            stmts.emplace_back(llir::While::make(
                a_p < end_a && b_p < end_b,
                llir::Sequence::make(std::move(while_stmts))));
        }

        // Drain remaining A elements
        {
            std::vector<llir::lStmt> drain_a;
            auto copies = make_coord_copies(tensor_a, a_p);
            drain_a.insert(drain_a.end(), copies.begin(), copies.end());
            drain_a.emplace_back(llir::Store::make(
                llir::lFieldAccess::make(result_var, "values")[out_p],
                llir::lFieldAccess::make(a_var, "values")[a_p]));
            drain_a.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(a_p)));
            drain_a.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(out_p)));

            stmts.emplace_back(llir::While::make(
                a_p < end_a,
                llir::Sequence::make(std::move(drain_a))));
        }

        // Drain remaining B elements
        {
            std::vector<llir::lStmt> drain_b;
            auto copies = make_coord_copies(tensor_b, b_p);
            drain_b.insert(drain_b.end(), copies.begin(), copies.end());
            drain_b.emplace_back(llir::Store::make(
                llir::lFieldAccess::make(result_var, "values")[out_p],
                llir::lFieldAccess::make(b_var, "values")[b_p]));
            drain_b.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(b_p)));
            drain_b.emplace_back(llir::BaseExpr::make(llir::lIncrement::make(out_p)));

            stmts.emplace_back(llir::While::make(
                b_p < end_b,
                llir::Sequence::make(std::move(drain_b))));
        }

        // Boundary overlap fix (end): if our last A element equals the next
        // thread's first B element, add B's value to our last output.
        {
            llir::lExpr boundary_cmp = llir::lFunctionCall::make("coo_compare<index_t, value_t>",
                {llir::lVar::make(llir::Generic_t::make(tensor_a.get_struct_name()), tensor_a.tensor_name),
                 end_a - llir::lConst::make((int64_t)1),
                 llir::lVar::make(llir::Generic_t::make(tensor_b.get_struct_name()), tensor_b.tensor_name),
                 end_b});
            llir::lExpr cond =
                tid < max_tid &&
                end_a > start_a &&
                end_b < b_nnz &&
                boundary_cmp == llir::lConst::make((int64_t)0);
            // Z.values[out_p - 1] += B.values[end_b]
            stmts.emplace_back(llir::IfElse::make(cond,
                llir::Accumulate::make(
                    llir::lFieldAccess::make(result_var, "values")[out_p - llir::lConst::make((int64_t)1)],
                    llir::lFieldAccess::make(b_var, "values")[end_b]),
                llir::lStmt()));
        }

        stmts.emplace_back(llir::Return::make());

        llir::lStmt body = llir::Sequence::make(std::move(stmts));
        printer.print(llir::Function::make(
            std::move(generics), std::move(attributes), std::move(args),
            llir::Generic_t::make("void"),
            "compute_" + loop_suffix + "_kernel",
            std::move(body)));
    }

    void CINLowerer::lower_coo_host_function() {
        std::string loop_suffix;
        for (const auto &lvl : result_tensor.tensor_type.format.levels) {
            loop_suffix += lvl.index;
        }

        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<llir::Function::Attribute> attributes = {llir::Function::host};
        llir::lType ret_type = llir::Generic_t::make("void");
        std::string name = result_tensor.tensor_name + "_compute";

        std::vector<llir::Function::Argument> func_args;
        for (const auto &it : operand_tensors) {
            func_args.emplace_back(llir::Function::Argument{
                .mutating = false,
                .type = llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                .name = it.second.tensor_name});
        }
        func_args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .by_reference = true,
            .type = llir::Generic_t::make(result_tensor.get_struct_name() + "<index_t, value_t>"),
            .name = result_tensor.tensor_name});

        auto operand_list = std::vector<std::pair<std::string, TensorLowerer>>(
            operand_tensors.begin(), operand_tensors.end());
        auto &tensor_a = operand_list[0].second;
        auto &tensor_b = operand_list[1].second;

        std::vector<llir::lStmt> body_stmts;

        // Boilerplate
        body_stmts.emplace_back(llir::Declare::make(index_t, "num_blocks", llir::lConst::make((int64_t)256)));
        body_stmts.emplace_back(llir::Declare::make(index_t, "threads_per_block", llir::lConst::make((int64_t)256)));
        body_stmts.emplace_back(llir::Declare::make(index_t, "num_threads",
            llir::lVar::make(index_t, "num_blocks") * llir::lVar::make(index_t, "threads_per_block")));
        body_stmts.emplace_back(llir::RawCode::make("const cudaStream_t stream = cudaStreamPerThread;"));

        llir::lExpr num_blocks_var = llir::lVar::make(index_t, "num_blocks");
        llir::lExpr tpb_var = llir::lVar::make(index_t, "threads_per_block");
        llir::lExpr num_threads_var = llir::lVar::make(index_t, "num_threads");
        llir::lExpr stream_var = llir::lVar::make(llir::Generic_t::make("cudaStream_t"), "stream");

        // total_work = A.nnz + B.nnz
        body_stmts.emplace_back(llir::RawCode::make(
            "index_t total_work_0 = " + tensor_a.tensor_name + ".nnz + " + tensor_b.tensor_name + ".nnz;"));
        body_stmts.emplace_back(llir::RawCode::make(
            "index_t per_thread_work_0 = total_work_0 / num_threads + 1;"));

        // Allocate partition struct
        std::string part_struct = "partition_" + loop_suffix;
        body_stmts.emplace_back(llir::RawCode::make(
            part_struct + "<index_t> partitions_0;"));
        for (const auto &it : operand_tensors) {
            body_stmts.emplace_back(llir::RawCode::make(
                "cudaMallocAsync((void**)&partitions_0." + it.second.tensor_name +
                "_p, num_threads * sizeof(index_t), stream);"));
        }

        // Launch partition kernel
        {
            std::vector<llir::lExpr> launch_args;
            for (const auto &it : operand_tensors) {
                launch_args.push_back(llir::lVar::make(
                    llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                    it.second.tensor_name));
            }
            launch_args.push_back(llir::lVar::make(
                llir::Generic_t::make(part_struct + "<index_t>"), "partitions_0"));
            launch_args.push_back(llir::lVar::make(index_t, "per_thread_work_0"));
            launch_args.push_back(llir::lVar::make(index_t, "total_work_0"));

            body_stmts.emplace_back(llir::KernelLaunch::make(
                "partition_" + loop_suffix + "_kernel", {"index_t", "value_t"},
                num_blocks_var, tpb_var, std::move(launch_args),
                llir::lExpr(), stream_var));
        }

        // Allocate count struct
        body_stmts.emplace_back(llir::RawCode::make(
            "result_per_thread_count<index_t> count_offsets_0;"));
        body_stmts.emplace_back(llir::RawCode::make(
            "cudaMallocAsync((void**)&count_offsets_0.nnz_count, num_threads * sizeof(index_t), stream);"));

        // Launch precompute kernel
        {
            std::vector<llir::lExpr> launch_args;
            for (const auto &it : operand_tensors) {
                launch_args.push_back(llir::lVar::make(
                    llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                    it.second.tensor_name));
            }
            launch_args.push_back(llir::lVar::make(
                llir::Generic_t::make(part_struct + "<index_t>"), "partitions_0"));
            launch_args.push_back(llir::lVar::make(
                llir::Generic_t::make("result_per_thread_count<index_t>"), "count_offsets_0"));
            launch_args.push_back(llir::lVar::make(index_t, "per_thread_work_0"));

            body_stmts.emplace_back(llir::KernelLaunch::make(
                "precompute_" + loop_suffix + "_kernel", {"index_t", "value_t"},
                num_blocks_var, tpb_var, std::move(launch_args),
                llir::lExpr(), stream_var));
        }

        // CUB prefix sum on nnz_count
        body_stmts.emplace_back(llir::RawCode::make("index_t* count_offsets_0_nnz_count_prefix;"));
        body_stmts.emplace_back(llir::RawCode::make(
            "cudaMallocAsync((void**)&count_offsets_0_nnz_count_prefix, (num_threads + 1) * sizeof(index_t), stream);"));
        body_stmts.emplace_back(llir::RawCode::make(
            "cudaMemsetAsync(count_offsets_0_nnz_count_prefix, 0, (num_threads + 1) * sizeof(index_t), stream);"));
        body_stmts.emplace_back(llir::RawCode::make("void* d_temp_storage_0_nnz_count = nullptr;"));
        body_stmts.emplace_back(llir::RawCode::make("size_t temp_storage_bytes_0_nnz_count = 0;"));
        body_stmts.emplace_back(llir::RawCode::make(
            "cub::DeviceScan::InclusiveSum(d_temp_storage_0_nnz_count, temp_storage_bytes_0_nnz_count, "
            "count_offsets_0.nnz_count, count_offsets_0_nnz_count_prefix + 1, num_threads, stream);"));
        body_stmts.emplace_back(llir::RawCode::make(
            "cudaMallocAsync(&d_temp_storage_0_nnz_count, temp_storage_bytes_0_nnz_count, stream);"));
        body_stmts.emplace_back(llir::RawCode::make(
            "cub::DeviceScan::InclusiveSum(d_temp_storage_0_nnz_count, temp_storage_bytes_0_nnz_count, "
            "count_offsets_0.nnz_count, count_offsets_0_nnz_count_prefix + 1, num_threads, stream);"));
        body_stmts.emplace_back(llir::RawCode::make("cudaFreeAsync(d_temp_storage_0_nnz_count, stream);"));
        body_stmts.emplace_back(llir::RawCode::make("cudaFreeAsync(count_offsets_0.nnz_count, stream);"));
        body_stmts.emplace_back(llir::RawCode::make("count_offsets_0.nnz_count = count_offsets_0_nnz_count_prefix;"));

        // Read output nnz
        body_stmts.emplace_back(llir::RawCode::make("index_t out_nnz;"));
        body_stmts.emplace_back(llir::RawCode::make(
            "cudaMemcpyAsync(&out_nnz, count_offsets_0_nnz_count_prefix + num_threads, "
            "sizeof(index_t), cudaMemcpyDeviceToHost, stream);"));
        body_stmts.emplace_back(llir::RawCode::make("cudaStreamSynchronize(stream);"));

        // Allocate output arrays: dim_X_indices for each dim + values
        for (const auto &level : result_tensor.tensor_type.format.levels) {
            body_stmts.emplace_back(llir::RawCode::make(
                "cudaMallocAsync((void**)&" + result_tensor.tensor_name + "." +
                result_tensor.get_indices_field_name(level.index) +
                ", out_nnz * sizeof(index_t), stream);"));
        }
        body_stmts.emplace_back(llir::RawCode::make(
            "cudaMallocAsync((void**)&" + result_tensor.tensor_name +
            ".values, out_nnz * sizeof(value_t), stream);"));

        // Launch compute kernel
        {
            std::vector<llir::lExpr> launch_args;
            for (const auto &it : operand_tensors) {
                launch_args.push_back(llir::lVar::make(
                    llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                    it.second.tensor_name));
            }
            launch_args.push_back(llir::lVar::make(
                llir::Generic_t::make(part_struct + "<index_t>"), "partitions_0"));
            launch_args.push_back(llir::lVar::make(
                llir::Generic_t::make("result_per_thread_count<index_t>"), "count_offsets_0"));
            launch_args.push_back(llir::lVar::make(index_t, "per_thread_work_0"));
            launch_args.push_back(llir::lVar::make(
                llir::Generic_t::make(result_tensor.get_struct_name() + "<index_t, value_t>"),
                result_tensor.tensor_name));

            body_stmts.emplace_back(llir::KernelLaunch::make(
                "compute_" + loop_suffix + "_kernel", {"index_t", "value_t"},
                num_blocks_var, tpb_var, std::move(launch_args),
                llir::lExpr(), stream_var));
        }

        // Update result tensor fields
        body_stmts.emplace_back(llir::RawCode::make(
            result_tensor.tensor_name + ".nnz = out_nnz;"));

        // Free intermediates
        for (const auto &it : operand_tensors) {
            body_stmts.emplace_back(llir::RawCode::make(
                "cudaFreeAsync(partitions_0." + it.second.tensor_name + "_p, stream);"));
        }
        body_stmts.emplace_back(llir::RawCode::make(
            "cudaFreeAsync(count_offsets_0.nnz_count, stream);"));

        llir::lStmt body = llir::Sequence::make(std::move(body_stmts));
        printer.print(llir::Function::make(
            std::move(generics), std::move(attributes), std::move(func_args),
            std::move(ret_type), name, std::move(body)));
    }

    void CINLowerer::lower_coo_flat_wrapper(const std::string &op_name) {
        llir::lType value_t = llir::Generic_t::make("value_t");
        std::vector<std::string> generics = {"index_t", "value_t"};
        std::vector<llir::Function::Attribute> attributes = {llir::Function::host};
        llir::lType ret_type = llir::Generic_t::make("void");

        std::vector<llir::Function::Argument> args;

        // For each operand: add all struct fields as flat args
        for (const auto &[name, tensor] : operand_tensors) {
            llir::lType struct_type = tensor.lower_tensor_struct_definition();
            const auto *st = struct_type.as<llir::Struct_t>();
            for (const auto &[field_name, field_type] : st->fields) {
                args.push_back({
                    .mutating = true,
                    .type = field_type,
                    .name = name + "_" + field_name
                });
            }
        }

        // Result size fields
        for (size_t i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            args.push_back({
                .mutating = true,
                .type = index_t,
                .name = "result_" + result_tensor.get_size_field_name(idx)
            });
        }

        // Output references: nnz
        args.push_back({
            .mutating = true, .by_reference = true, .type = index_t, .name = "out_nnz"});

        // Output references: dim_X_indices for each Coordinate dim
        for (int i = (int)result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            args.push_back({
                .mutating = true, .by_reference = true,
                .type = llir::Ptr_t::make(index_t),
                .name = "out_" + result_tensor.get_indices_field_name(idx)
            });
        }

        // Output reference: values
        args.push_back({
            .mutating = true, .by_reference = true,
            .type = llir::Ptr_t::make(value_t),
            .name = "out_values"
        });

        // Build function body
        std::vector<llir::lStmt> body_stmts;

        // Create and populate operand structs
        for (const auto &[name, tensor] : operand_tensors) {
            body_stmts.push_back(llir::RawCode::make(
                tensor.get_struct_name() + "<index_t, value_t> " + name + ";"));
            llir::lType struct_type = tensor.lower_tensor_struct_definition();
            const auto *st = struct_type.as<llir::Struct_t>();
            for (const auto &[field_name, field_type] : st->fields) {
                body_stmts.push_back(llir::RawCode::make(
                    name + "." + field_name + " = " + name + "_" + field_name + ";"));
            }
        }

        // Create result struct and populate size fields
        body_stmts.push_back(llir::RawCode::make(
            result_tensor.get_struct_name() + "<index_t, value_t> " +
            result_tensor.tensor_name + ";"));
        for (size_t i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            std::string size_field = result_tensor.get_size_field_name(idx);
            body_stmts.push_back(llir::RawCode::make(
                result_tensor.tensor_name + "." + size_field +
                " = result_" + size_field + ";"));
        }

        // Call compute function
        std::string call = result_tensor.tensor_name + "_compute<index_t, value_t>(";
        bool first = true;
        for (const auto &[name, tensor] : operand_tensors) {
            if (!first) call += ", ";
            call += name;
            first = false;
        }
        call += ", " + result_tensor.tensor_name + ");";
        body_stmts.push_back(llir::RawCode::make(call));

        // Extract outputs
        body_stmts.push_back(llir::RawCode::make(
            "out_nnz = " + result_tensor.tensor_name + ".nnz;"));
        for (int i = (int)result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            body_stmts.push_back(llir::RawCode::make(
                "out_" + result_tensor.get_indices_field_name(idx) + " = " +
                result_tensor.tensor_name + "." +
                result_tensor.get_indices_field_name(idx) + ";"));
        }
        body_stmts.push_back(llir::RawCode::make(
            "out_values = " + result_tensor.tensor_name + ".values;"));

        llir::lStmt body = llir::Sequence::make(std::move(body_stmts));
        printer.print(llir::Function::make(std::move(generics), std::move(attributes),
                                           std::move(args), std::move(ret_type),
                                           op_name, std::move(body)));
    }

}
} // namespace nacho
