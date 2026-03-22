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

    CINLowerer::CINLowerer(CIN cin, Printer &printer) : cin(std::move(cin)), printer(printer), reductionLoop(BEFORE_FIRST_LOOP) {
        loop_order = get_loop_order();
        struct TensorVisitor : Visitor {
            std::map<std::string, TensorLowerer> &operand_tensors;
            TensorLowerer &result_tensor;
            TensorLowerer &scatter_reduced_result_tensor;
            std::vector<std::string> &loop_order;
            LoopNum &reductionLoop;
            TensorVisitor(std::map<std::string, TensorLowerer> &operand_tensors, TensorLowerer &result_tensor, TensorLowerer &scatter_reduced_result_tensor, 
                std::vector<std::string> &loop_order, LoopNum &reductionLoop)
                : operand_tensors(operand_tensors), result_tensor(result_tensor), scatter_reduced_result_tensor(scatter_reduced_result_tensor), loop_order(loop_order), reductionLoop(reductionLoop) {}

            void add_tensor(std::string str, TensorType type) {
                TensorLowerer lowerer(str, type, loop_order);
                operand_tensors[str] = lowerer;
            }

            void visit(const cTensor *node) override {
                add_tensor(node->name, node->type);
            }

            void visit(const Accumulate *node) override {
                result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                reductionLoop = result_tensor.get_loop_num(node->accumulate_index);
                // scatter reduction
                if(reductionLoop < result_tensor.get_loop_num_for_last_sparse_level() ){
                    if(node->expr.as<cAdd>()) {
                        result_tensor = TensorLowerer(node->tensor+"_temp",  node->expr.as<cAdd>()->type, loop_order, true);
                        scatter_reduced_result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                    } else if(node->expr.as<cMul>()) {
                        result_tensor = TensorLowerer(node->tensor+"_temp",  node->expr.as<cMul>()->type, loop_order, true);
                        scatter_reduced_result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                    } else {
                        internal_assert(false) << "Expected Accumulate to have cAdd or cMul expr, inner sums are not yet supported";
                    }
                }
                node->expr.accept(this);
            }

            void visit(const Assign *node) override {
                result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                node->expr.accept(this);
            }
        };

        TensorVisitor visitor(operand_tensors, result_tensor, scatter_reduced_result_tensor, loop_order, reductionLoop);
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

    std::vector<CIN> CINLowerer::get_forall_list() {
        std::vector<CIN> forall_list;
        struct ForallVisitor : Visitor {
            std::vector<CIN> &forall_list;
            ForallVisitor(std::vector<CIN> &forall_list) : forall_list(forall_list) {}

            void visit(const Forall *node) override {
                forall_list.push_back(node);
                node->body.accept(this);
            }
        };
        ForallVisitor visitor(forall_list);
        cin.accept(&visitor);
        return forall_list;
    }

    void CINLowerer::lower_cin() {

        std::vector<std::string> loop_order = get_loop_order();
        auto forall_list = get_forall_list();

        std::vector<int> sparse_intersection_levels = get_all_sparse_intersection_levels(cin);

        sparse_intersection_levels.insert(sparse_intersection_levels.begin(), -1);
        if(sparse_intersection_levels.back()!=(int)loop_order.size()-1) {
            sparse_intersection_levels.push_back(loop_order.size()-1);
        }

        this->lower_binary_search_function();
        this->lower_struct_definitions(sparse_intersection_levels[sparse_intersection_levels.size()-2]);

        for(int i=0; i< (int)sparse_intersection_levels.size()-1;i++) {
            LoopNum previous_sparse_intersection(sparse_intersection_levels[i]);
            LoopNum current_sparse_intersection(sparse_intersection_levels[i+1]);
            LoopNum next_sparse_intersection(i!=(int)sparse_intersection_levels.size()-2 ? sparse_intersection_levels[i+2] : (int)forall_list.size());

            auto previous_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + previous_sparse_intersection.get() + 1);
            auto current_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + current_sparse_intersection.get() + 1);

            // Generate work functions for partition kernel
            for(int level=0; level<=previous_sparse_intersection.get();level++) {
                printer.print(result_tensor.lower_work_function(previous_loop_order, LoopNum(level)));
            }
            auto included_tensors = get_included_tensors_for_level(current_sparse_intersection);
            for(LoopNum level = previous_sparse_intersection + 1; level <= current_sparse_intersection; ++level) {
                for (auto it : included_tensors) {
                    printer.print(it.second.lower_work_function(current_loop_order, level));
                }
            }


            PartitionKernelLowerer partition_lowerer(operand_tensors, result_tensor, included_tensors, forall_list, previous_sparse_intersection, current_sparse_intersection, next_sparse_intersection, reductionLoop);

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
                modified_cin = get_modified_cin_for_sparse_intersection(current_sparse_intersection.get(), cin);
                included_tensors = get_included_tensors_for_level(next_sparse_intersection);
            }

            ComputeKernelLowerer compute_lowerer(operand_tensors, result_tensor, included_tensors, forall_list, modified_cin, previous_sparse_intersection, current_sparse_intersection, next_sparse_intersection, reductionLoop);

            // Generate Precompute kernels
            bool has_precompute = result_tensor.get_loop_num_for_prev_sparse_level(current_sparse_intersection+1) > BEFORE_FIRST_LOOP;
            if(has_precompute) {
                printer.print(compute_lowerer.lower_precompute_function());

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
                for (auto it : included_tensors) {
                    auto next_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + next_sparse_intersection.get() + 1);
                    printer.print(it.second.lower_work_function(next_loop_order, current_sparse_intersection, true));
                }
            }
            printer.print(compute_lowerer.lower_compute_function());

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
            if(has_precompute) {
                auto empty_map = std::map<std::string, TensorLowerer>();
                BaseKernelLowerer base_lowerer(operand_tensors, result_tensor, empty_map, forall_list, previous_sparse_intersection, current_sparse_intersection, next_sparse_intersection, reductionLoop);
                counts_struct_type = base_lowerer.lower_result_per_thread_count_struct();
            }

            phase_struct_infos.push_back({
                .partition_struct = partition_struct_type,
                .counts_struct = counts_struct_type,
                .previous_sparse_intersection = previous_sparse_intersection.get(),
                .current_sparse_intersection = current_sparse_intersection.get(),
                .next_sparse_intersection = next_sparse_intersection.get(),
                .has_precompute = has_precompute
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
        if(reductionLoop<result_tensor.get_loop_num_for_last_sparse_level()) {
            printer.print(scatter_reduced_result_tensor.lower_tensor_struct_definition());
        }


        // Use BaseKernelLowerer to lower the one time struct definitions of result_per_thread_count struct and result_to_operand_pos_map struct
        auto empty_map = std::map<std::string, TensorLowerer>();
        BaseKernelLowerer BaseLowerer(operand_tensors, result_tensor, empty_map, get_forall_list(), BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, reductionLoop);
        // Need to lower this struct only once
        if(!result_tensor.tensor_type.format.are_all_lvls_dense()) {
            printer.print(BaseLowerer.lower_result_per_thread_count_struct());
        }

        auto result_operand_pos_map = lower_result_pos_to_operand_pos_map_struct(last_sparse_intersection);
        if(result_operand_pos_map.get() != nullptr) {
            printer.print(result_operand_pos_map);
        }
    }

    llir::lType CINLowerer::lower_result_pos_to_operand_pos_map_struct(int last_sparse_intersection) {
        std::vector<std::string> generics = {"index_t"};
        std::vector<std::pair<std::string, llir::lType>> fields;
        auto forall_list = get_forall_list(); auto empty_map = std::map<std::string, TensorLowerer>();
        BaseKernelLowerer BaseLowerer(operand_tensors, result_tensor, empty_map, forall_list, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, reductionLoop);
        for(int level=0; level<=last_sparse_intersection;level++) {
            if(level >= (int)forall_list.size()-1) {
                continue;
            }
            const Forall* forall = forall_list[level].as<Forall>();
            std::string forall_idx = forall->idx;

            for(auto& [name, tensor] : operand_tensors) {
                if(BaseLowerer.exists_field_in_result_to_operand_pos_map(forall, tensor)) {
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

    std::vector<int> CINLowerer::get_all_sparse_intersection_levels(CIN& cin) {
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

    CIN CINLowerer::get_modified_cin_for_sparse_intersection(int target_level, CIN& cin) {
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
                llir::Function::Argument{.mutating = true, .type = index_t, .name = "start_index"});
            args.emplace_back(
                llir::Function::Argument{.mutating = true, .type = index_t, .name = "end_index"});

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

    std::map<std::string, TensorLowerer> CINLowerer::get_included_tensors_for_level(LoopNum loop_num) {
            auto forall  = get_forall_list()[loop_num.get()].as<Forall>();

            std::map<std::string, TensorLowerer> included_tensors;

            // TODO : Need to still verify if this is correct
            // For non-innermost sparse case, for dense level in the sparse intersection loop
            // which is the last loop for this phase there is nothing to actually read.
            // for inntermost sparse case we have non-zeros to read even if last level is dense
            // Just exclude dense tensors from work calculation for this case for now.
            // (Counting all memory loads (including indices probably gives a better idea for this))
            if(loop_num < LoopNum(loop_order.size()-1)) {
                for(auto it : operand_tensors) {
                    if(it.second.tensor_level_exists(loop_num) && it.second.is_sparse(loop_num)) {
                        included_tensors[it.first] = it.second;
                    }
                }
                return included_tensors;
            }

            std::vector<Seq> locators = get_dense_locators(forall->seq);

            // included tensors are the tensors which are included in the work
            // calculation. Non-included tensors are not co-iterated and instead looked up.
            std::map<std::string, TensorLowerer> excluded_tensors;
                
            for(const auto &it : operand_tensors) {
                if (!it.second.tensor_level_exists(loop_num)) {
                    excluded_tensors[it.second.tensor_name] = it.second;
                    continue;
                }
                for(const auto &loc : locators) {
                    const auto *index = loc.as<Index>();
                    if (!index){
                        internal_assert(false) << "Expected Index node in locator sequence: " << loc;
                    }
                    if (it.second.tensor_name == index->tensor) {
                        excluded_tensors[it.second.tensor_name] = it.second;
                    }
                }
            }
            for(auto it : operand_tensors) {
                if(excluded_tensors.find(it.first) == excluded_tensors.end()) {
                    included_tensors[it.first] = it.second;
                }
            }
            //std::cout<<" At level "<< level << " included tensors: "<< included_tensors.size() << "\n";
            return included_tensors;
    }


    void CINLowerer::lower_host_function() {
        std::vector<std::string> generics = {"index_t", "value_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::host};

        std::vector<llir::Function::Argument> func_args;
        llir::lType ret_type = llir::Generic_t::make("void");
        std::string name = result_tensor.tensor_name + "_compute";
        llir::lType value_t_type = llir::Generic_t::make("value_t");

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
        body_stmts.emplace_back(llir::Declare::make(
            llir::Generic_t::make("const cudaStream_t"), "stream",
            llir::lVar::make(llir::Generic_t::make("cudaStream_t"), "cudaStreamPerThread")));

        llir::lExpr num_blocks_var = llir::lVar::make(index_t, "num_blocks");
        llir::lExpr threads_per_block_var = llir::lVar::make(index_t, "threads_per_block");
        llir::lExpr num_threads_var = llir::lVar::make(index_t, "num_threads");
        llir::lExpr stream_var = llir::lVar::make(
            llir::Generic_t::make("cudaStream_t"), "stream");

        // Helper to build result tensor field access
        auto result_var = [&]() {
            return llir::lVar::make(
                llir::Generic_t::make(result_tensor.get_struct_name() + "<index_t, value_t>"),
                result_tensor.tensor_name);
        };

        // For each phase, generate the host-side orchestration
        int num_phases = (int)phase_struct_infos.size();
        std::string prev_phase_outermost_nnz;  // tracks nnz from previous phase for T_work_offsets indexing
        int prev_phase_max_sparse = -1;  // highest sparse level allocated in previous phases
        for (int phase = 0; phase < num_phases; phase++) {
            const auto &phase_info = phase_struct_infos[phase];
            const llir::Struct_t *partition_struct = phase_info.partition_struct.as<llir::Struct_t>();
            std::vector<llir::lStmt> delayed_result_field_updates;

            body_stmts.emplace_back(llir::Comment::make(
                "========== Phase " + std::to_string(phase) + " =========="));

            // 1. Compute total_work and per_thread_work
            std::string total_work_name = "total_work_" + std::to_string(phase);
            if (phase_info.previous_sparse_intersection == -1) {
                // Phase 0: total_work = sum of operand nnz (length of innermost sparse dim)
                llir::lExpr total_work_init;
                for (const auto &it : operand_tensors) {
                    auto forall = get_forall_list()[phase_info.current_sparse_intersection].as<Forall>();
                    std::string idx = forall->idx;
                    if (it.second.tensor_type.format.level_exists(idx) &&
                        is_sparse_format(it.second.tensor_type.format.lvlfmt_of(idx))) {
                        llir::lExpr field = llir::lFieldAccess::make(
                            llir::lVar::make(llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                                             it.second.tensor_name),
                            it.second.get_length_field_name(idx));
                        if (total_work_init.defined()) {
                            total_work_init = total_work_init + field;
                        } else {
                            total_work_init = field;
                        }
                    }
                }
                if (!total_work_init.defined()) {
                    // Fallback: sum all operand nnz-like fields
                    for (const auto &it : operand_tensors) {
                        auto &fmt = it.second.tensor_type.format;
                        if (!fmt.levels.empty()) {
                            int last = fmt.levels.size() - 1;
                            if (is_sparse_format(fmt.levels[last].format)) {
                                llir::lExpr field = llir::lFieldAccess::make(
                                    llir::lVar::make(llir::Generic_t::make(it.second.get_struct_name() + "<index_t, value_t>"),
                                                     it.second.tensor_name),
                                    it.second.get_length_field_name(fmt.levels[last].index));
                                if (total_work_init.defined()) {
                                    total_work_init = total_work_init + field;
                                } else {
                                    total_work_init = field;
                                }
                            }
                        }
                    }
                }
                if (!total_work_init.defined()) {
                    total_work_init = llir::lConst::make((int64_t)0);
                }
                body_stmts.emplace_back(llir::Declare::make(
                    index_t, total_work_name, total_work_init));
            } else {
                // Phase N>0: total_work comes from prefix sum result of previous phase.
                body_stmts.emplace_back(llir::Declare::make(index_t, total_work_name));
                llir::lExpr tw_var = llir::lVar::make(index_t, total_work_name);
                llir::lExpr prev_nnz_var = llir::lVar::make(index_t, prev_phase_outermost_nnz);
                llir::lExpr tw_prefix_var = llir::lVar::make(llir::Ptr_t::make(index_t), "T_work_offsets_prefix");
                body_stmts.emplace_back(llir::DeviceTransfer::make_memcpy(
                    llir::DeviceTransfer::D2H,
                    llir::lAddress::make(tw_var),
                    llir::lAddress::make(tw_prefix_var[prev_nnz_var]),
                    llir::lSizeOf::make(index_t),
                    stream_var, true));
            }

            std::string ptw_name = "per_thread_work_" + std::to_string(phase);
            body_stmts.emplace_back(llir::Declare::make(
                index_t, ptw_name,
                llir::lVar::make(index_t, total_work_name) / num_threads_var + 1));

            // 2. Declare partition and count struct variables
            std::string partition_var = "partitions_" + std::to_string(phase);
            body_stmts.emplace_back(llir::Declare::make(
                llir::Generic_t::make(partition_struct->name + "<index_t>"), partition_var));
            llir::lExpr partition_var_expr = llir::lVar::make(
                llir::Generic_t::make(partition_struct->name + "<index_t>"), partition_var);

            std::string counts_var = "count_offsets_" + std::to_string(phase);
            const llir::Struct_t *counts_struct = nullptr;
            llir::lExpr counts_var_expr;
            int num_count_fields = 0;
            if (phase_info.has_precompute) {
                counts_struct = phase_info.counts_struct.as<llir::Struct_t>();
                body_stmts.emplace_back(llir::Declare::make(
                    llir::Generic_t::make(counts_struct->name + "<index_t>"), counts_var));
                counts_var_expr = llir::lVar::make(
                    llir::Generic_t::make(counts_struct->name + "<index_t>"), counts_var);
                num_count_fields = (int)counts_struct->fields.size();
            }

            // 3. Slab allocation (single cudaMallocAsync for all per-thread temporaries)
            std::string slab_name = "slab_" + std::to_string(phase);
            std::string slab_base_name = "slab_base_" + std::to_string(phase);
            std::string cub_bytes_name = "cub_bytes_" + std::to_string(phase);
            std::string cub_scratch_name = "cub_scratch_" + std::to_string(phase);
            int num_partition_fields = (int)partition_struct->fields.size();

            // CubScratchQuery (only if has_precompute)
            if (phase_info.has_precompute) {
                body_stmts.emplace_back(llir::CubScratchQuery::make(
                    cub_bytes_name, num_threads_var + 1, stream_var));
            }

            // Build slab assignments: partition fields then count fields
            std::vector<std::pair<llir::lExpr, llir::lExpr>> slab_assignments;
            for (int i = 0; i < num_partition_fields; i++) {
                llir::lExpr target = llir::lFieldAccess::make(
                    partition_var_expr, partition_struct->fields[i].first);
                llir::lExpr offset;
                if (i == 0) offset = llir::lConst::make((int64_t)0);
                else if (i == 1) offset = num_threads_var;
                else offset = num_threads_var * i;
                slab_assignments.push_back({target, offset});
            }
            for (int i = 0; i < num_count_fields; i++) {
                llir::lExpr target = llir::lFieldAccess::make(
                    counts_var_expr, counts_struct->fields[i].first);
                llir::lExpr base = (num_partition_fields == 1)
                    ? num_threads_var
                    : num_threads_var * num_partition_fields;
                llir::lExpr offset;
                if (i == 0) offset = base;
                else if (i == 1) offset = base + (num_threads_var + 1);
                else offset = base + (num_threads_var + 1) * i;
                slab_assignments.push_back({target, offset});
            }

            // Compute total index_t elements in the slab
            llir::lExpr total_elements;
            {
                llir::lExpr part_total = (num_partition_fields == 1)
                    ? num_threads_var
                    : num_threads_var * num_partition_fields;
                if (num_count_fields == 0) {
                    total_elements = part_total;
                } else {
                    llir::lExpr count_total = (num_count_fields == 1)
                        ? (num_threads_var + 1)
                        : (num_threads_var + 1) * num_count_fields;
                    total_elements = part_total + count_total;
                }
            }

            body_stmts.emplace_back(llir::SlabAlloc::make(
                slab_name, slab_base_name, total_elements,
                phase_info.has_precompute ? cub_bytes_name : "",
                stream_var, std::move(slab_assignments),
                phase_info.has_precompute ? cub_scratch_name : ""));

            // Memset sentinel slot (count[N] = 0) for each count field
            for (int i = 0; i < num_count_fields; i++) {
                llir::lExpr count_field = llir::lFieldAccess::make(
                    counts_var_expr, counts_struct->fields[i].first);
                body_stmts.emplace_back(llir::DeviceTransfer::make_memset(
                    count_field + num_threads_var,
                    llir::lConst::make((int64_t)0),
                    llir::lSizeOf::make(index_t),
                    stream_var));
            }

            // 4. Launch partition kernel
            {
                std::vector<llir::lExpr> launch_args;
                for (const auto &ki : kernel_infos) {
                    if (ki.kind == KernelInfo::Partition && ki.phase == phase) {
                        for (const auto &arg : ki.args) {
                            if (arg.name == "partitions") {
                                launch_args.push_back(partition_var_expr);
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

            // 5-7. Precompute + in-place prefix sum + nnz read (if has_precompute)
            if (phase_info.has_precompute) {
                // 5. Launch precompute kernel
                for (const auto &ki : kernel_infos) {
                    if (ki.kind == KernelInfo::Precompute && ki.phase == phase) {
                        std::vector<llir::lExpr> launch_args;
                        for (const auto &arg : ki.args) {
                            if (arg.name == "partitions") {
                                launch_args.push_back(partition_var_expr);
                            } else if (arg.name == "count_offsets") {
                                launch_args.push_back(counts_var_expr);
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

                // 6. In-place ExclusiveSum over N+1 elements for each count field
                for (const auto &field : counts_struct->fields) {
                    body_stmts.emplace_back(llir::InPlacePrefixSum::make(
                        llir::lFieldAccess::make(counts_var_expr, field.first),
                        num_threads_var + 1,
                        stream_var,
                        llir::lVar::make(llir::Generic_t::make("void*"), cub_scratch_name),
                        llir::lVar::make(llir::Generic_t::make("size_t"), cub_bytes_name)));
                }

                // 7. Read nnz from count[N] (total after ExclusiveSum)
                bool has_nnz_read = false;
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        std::string nnz_name = "nnz_" + idx + "_" + std::to_string(phase);
                        std::string field_name = "dim_" + idx + "_count";
                        body_stmts.emplace_back(llir::Declare::make(index_t, nnz_name));
                        llir::lExpr nnz_var = llir::lVar::make(index_t, nnz_name);
                        llir::lExpr count_field = llir::lFieldAccess::make(counts_var_expr, field_name);
                        body_stmts.emplace_back(llir::DeviceTransfer::make_memcpy(
                            llir::DeviceTransfer::D2H,
                            llir::lAddress::make(nnz_var),
                            count_field + num_threads_var,
                            llir::lSizeOf::make(index_t),
                            stream_var, false));
                        has_nnz_read = true;
                    }
                }
                if (has_nnz_read) {
                    body_stmts.emplace_back(llir::BaseExpr::make(
                        llir::lFunctionCall::make("cudaStreamSynchronize", {stream_var})));
                }

                // 8. Allocate output tensor arrays based on nnz
                int alloc_start = prev_phase_max_sparse + 1;
                for (int lvl = alloc_start; lvl <= phase_info.current_sparse_intersection; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        std::string nnz_name = "nnz_" + idx + "_" + std::to_string(phase);
                        body_stmts.emplace_back(llir::DeviceAlloc::make(
                            llir::lFieldAccess::make(result_var(), result_tensor.get_indices_field_name(idx)),
                            llir::lVar::make(index_t, nnz_name) * llir::lSizeOf::make(index_t),
                            stream_var));
                    }
                }
                // Allocate values if this is the last phase
                if (phase == num_phases - 1) {
                    std::string values_nnz;
                    for (int lvl = phase_info.current_sparse_intersection; lvl >= 0; lvl--) {
                        auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                        if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                            values_nnz = "nnz_" + idx + "_" + std::to_string(phase);
                            break;
                        }
                    }
                    if (!values_nnz.empty()) {
                        body_stmts.emplace_back(llir::DeviceAlloc::make(
                            llir::lFieldAccess::make(result_var(), "values"),
                            llir::lVar::make(index_t, values_nnz) * llir::lSizeOf::make(value_t_type),
                            stream_var));
                    }
                }

                // Allocate offsets arrays for non-outermost sparse dimensions
                for (int lvl = 1; lvl <= phase_info.current_sparse_intersection; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        auto parent_idx = result_tensor.tensor_type.format.levels[lvl - 1].index;
                        llir::lExpr offsets_size;
                        if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(parent_idx))) {
                            offsets_size = llir::lVar::make(index_t,
                                "nnz_" + parent_idx + "_" + std::to_string(phase)) + 1;
                        } else {
                            offsets_size = llir::lFieldAccess::make(result_var(),
                                result_tensor.get_size_field_name(parent_idx)) + 1;
                        }
                        llir::lExpr offsets_field = llir::lFieldAccess::make(result_var(),
                            result_tensor.get_offsets_field_name(idx));
                        body_stmts.emplace_back(llir::DeviceAlloc::make(
                            offsets_field,
                            offsets_size * llir::lSizeOf::make(index_t),
                            stream_var));
                        body_stmts.emplace_back(llir::DeviceTransfer::make_memset(
                            llir::lFieldAccess::make(result_var(), result_tensor.get_offsets_field_name(idx)),
                            llir::lConst::make((int64_t)0),
                            offsets_size * llir::lSizeOf::make(index_t),
                            stream_var));
                    }
                }

                // Defer result length/nnz updates until after compute launch.
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        std::string nnz_name = "nnz_" + idx + "_" + std::to_string(phase);
                        delayed_result_field_updates.push_back(llir::Store::make(
                            llir::lFieldAccess::make(result_var(), result_tensor.get_length_field_name(idx)),
                            llir::lVar::make(index_t, nnz_name)));
                    }
                }
                if (phase == num_phases - 1) {
                    for (int lvl = phase_info.current_sparse_intersection; lvl >= 0; lvl--) {
                        auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                        if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                            delayed_result_field_updates.push_back(llir::Store::make(
                                llir::lFieldAccess::make(result_var(), "nnz"),
                                llir::lVar::make(index_t, "nnz_" + idx + "_" + std::to_string(phase))));
                            break;
                        }
                    }
                }
            }

            // For multi-phase expressions, compute the output nnz variable name
            bool is_last_phase = (phase == num_phases - 1);
            std::string phase_outermost_nnz;
            if (!is_last_phase) {
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection; lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        phase_outermost_nnz = "nnz_" + idx + "_" + std::to_string(phase);
                        break;
                    }
                }
            }

            // Early return if this phase produced zero output elements
            if (!is_last_phase && !phase_outermost_nnz.empty()) {
                llir::lExpr nnz_var = llir::lVar::make(index_t, phase_outermost_nnz);
                std::vector<llir::lStmt> guard_stmts;
                // Set remaining output fields to zero
                for (int lvl = phase_info.current_sparse_intersection + 1;
                     lvl < (int)result_tensor.tensor_type.format.levels.size(); lvl++) {
                    auto idx = result_tensor.tensor_type.format.levels[lvl].index;
                    if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                        guard_stmts.push_back(llir::Store::make(
                            llir::lFieldAccess::make(result_var(), result_tensor.get_length_field_name(idx)),
                            llir::lConst::make((int64_t)0)));
                    }
                }
                guard_stmts.push_back(llir::Store::make(
                    llir::lFieldAccess::make(result_var(), "nnz"),
                    llir::lConst::make((int64_t)0)));
                // Free slab instead of per-field frees
                guard_stmts.push_back(llir::DeviceFree::make(
                    llir::lVar::make(llir::Generic_t::make("void*"), slab_name),
                    stream_var));
                guard_stmts.push_back(llir::Return::make());
                body_stmts.emplace_back(llir::IfElse::make(
                    nnz_var == llir::lConst::make((int64_t)0),
                    llir::Sequence::make(std::move(guard_stmts)),
                    llir::lStmt()));
            }

            // Allocate result_to_operand_pos_map if this is not the last phase.
            if (!is_last_phase && !phase_outermost_nnz.empty()) {
                auto forall_list = get_forall_list();
                auto empty_map = std::map<std::string, TensorLowerer>();
                BaseKernelLowerer base_lowerer(operand_tensors, result_tensor, empty_map,
                                               forall_list, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, reductionLoop);
                std::string pos_map_struct = base_lowerer.get_result_to_operand_pos_map_struct_name();
                std::string pos_map_var = base_lowerer.get_result_to_operand_pos_map_var_name();

                body_stmts.emplace_back(llir::Declare::make(
                    llir::Generic_t::make(pos_map_struct + "<index_t>"), pos_map_var));
                llir::lExpr pos_map_var_expr = llir::lVar::make(
                    llir::Generic_t::make(pos_map_struct + "<index_t>"), pos_map_var);
                llir::lExpr nnz_var = llir::lVar::make(index_t, phase_outermost_nnz);
                for (int lvl = 0; lvl <= phase_info.current_sparse_intersection; lvl++) {
                    if (lvl >= (int)forall_list.size() - 1) continue;
                    const Forall *forall = forall_list[lvl].as<Forall>();
                    std::string forall_idx = forall->idx;
                    for (auto &[tname, tensor] : operand_tensors) {
                        if (base_lowerer.exists_field_in_result_to_operand_pos_map(forall, tensor)) {
                            std::string field = tensor.get_iterator_suffix(forall_idx);
                            body_stmts.emplace_back(llir::DeviceAlloc::make(
                                llir::lFieldAccess::make(pos_map_var_expr, field),
                                nnz_var * llir::lSizeOf::make(index_t),
                                stream_var));
                        }
                    }
                }
            }

            // Allocate T_work_offsets if this is not the last phase.
            if (!is_last_phase) {
                llir::lExpr nnz_var = llir::lVar::make(index_t, phase_outermost_nnz);
                body_stmts.emplace_back(llir::Declare::make(
                    llir::Ptr_t::make(index_t), "T_work_offsets"));
                body_stmts.emplace_back(llir::DeviceAlloc::make(
                    llir::lVar::make(llir::Ptr_t::make(index_t), "T_work_offsets"),
                    nnz_var * llir::lSizeOf::make(index_t),
                    stream_var));
            }

            // 9. Launch compute kernel
            for (const auto &ki : kernel_infos) {
                if (ki.kind == KernelInfo::Compute && ki.phase == phase) {
                    std::vector<llir::lExpr> launch_args;
                    for (const auto &arg : ki.args) {
                        if (arg.name == "partitions") {
                            launch_args.push_back(partition_var_expr);
                        } else if (arg.name == "count_offsets") {
                            llir::lExpr cv = llir::lVar::make(
                                llir::Generic_t::make("result_per_thread_count<index_t>"),
                                counts_var);
                            launch_args.push_back(cv);
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

            // Prefix sum T_work_offsets if not the last phase (unchanged — data-dependent size).
            if (!is_last_phase) {
                llir::lExpr tw_size_var = llir::lVar::make(index_t, phase_outermost_nnz);
                body_stmts.emplace_back(llir::Declare::make(
                    llir::Ptr_t::make(index_t), "T_work_offsets_prefix"));
                llir::lExpr tw_prefix_var = llir::lVar::make(llir::Ptr_t::make(index_t), "T_work_offsets_prefix");
                llir::lExpr tw_var = llir::lVar::make(llir::Ptr_t::make(index_t), "T_work_offsets");
                body_stmts.emplace_back(llir::DeviceAlloc::make(
                    tw_prefix_var,
                    (tw_size_var + 1) * llir::lSizeOf::make(index_t),
                    stream_var));
                body_stmts.emplace_back(llir::DeviceTransfer::make_memset(
                    llir::lVar::make(llir::Ptr_t::make(index_t), "T_work_offsets_prefix"),
                    llir::lConst::make((int64_t)0),
                    (tw_size_var + 1) * llir::lSizeOf::make(index_t),
                    stream_var));
                body_stmts.emplace_back(llir::PrefixSum::make(
                    tw_var,
                    tw_prefix_var,
                    tw_size_var,
                    stream_var,
                    "d_temp_tw_" + std::to_string(phase),
                    "temp_bytes_tw_" + std::to_string(phase)));
                body_stmts.emplace_back(llir::DeviceFree::make(tw_var, stream_var));
                // Reassign so next phase's kernels use the prefix sum result
                body_stmts.emplace_back(llir::Store::make(tw_var, tw_prefix_var));
            }

            for (const auto &stmt : delayed_result_field_updates) {
                body_stmts.emplace_back(stmt);
            }

            // 10. Free slab (single free replaces N per-field frees)
            body_stmts.emplace_back(llir::DeviceFree::make(
                llir::lVar::make(llir::Generic_t::make("void*"), slab_name),
                stream_var));

            // Track outermost nnz for next phase's T_work_offsets indexing
            if (!phase_outermost_nnz.empty()) {
                prev_phase_outermost_nnz = phase_outermost_nnz;
            }
            prev_phase_max_sparse = phase_info.current_sparse_intersection;
        }

        llir::lStmt body = llir::Sequence::make(std::move(body_stmts));
        printer.print(llir::Function::make(std::move(generics), std::move(attributes),
                                           std::move(func_args), std::move(ret_type),
                                           name, std::move(body)));
    }

    void CINLowerer::lower_flat_wrapper(const std::string &op_name) {
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
            llir::lType struct_type_def = tensor.lower_tensor_struct_definition();
            const auto *st = struct_type_def.as<llir::Struct_t>();
            llir::lType operand_struct_type = llir::Generic_t::make(
                tensor.get_struct_name() + "<index_t, value_t>");
            body_stmts.push_back(llir::Declare::make(operand_struct_type, name));
            llir::lExpr operand_var = llir::lVar::make(operand_struct_type, name);
            for (const auto &[field_name, field_type] : st->fields) {
                body_stmts.push_back(llir::Store::make(
                    llir::lFieldAccess::make(operand_var, field_name),
                    llir::lVar::make(field_type, name + "_" + field_name)));
            }
        }

        // Create result struct and populate size fields
        llir::lType result_struct_type = llir::Generic_t::make(
            result_tensor.get_struct_name() + "<index_t, value_t>");
        body_stmts.push_back(llir::Declare::make(
            result_struct_type, result_tensor.tensor_name));
        llir::lExpr result_var_expr = llir::lVar::make(result_struct_type, result_tensor.tensor_name);
        for (size_t i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            std::string size_field = result_tensor.get_size_field_name(idx);
            body_stmts.push_back(llir::Store::make(
                llir::lFieldAccess::make(result_var_expr, size_field),
                llir::lVar::make(index_t, "result_" + size_field)));
        }

        // Call the compute function
        {
            std::vector<llir::lExpr> call_args;
            for (const auto &[name, tensor] : operand_tensors) {
                call_args.push_back(llir::lVar::make(
                    llir::Generic_t::make(tensor.get_struct_name() + "<index_t, value_t>"), name));
            }
            call_args.push_back(result_var_expr);
            body_stmts.push_back(llir::BaseExpr::make(
                llir::lFunctionCall::make(
                    result_tensor.tensor_name + "_compute<index_t, value_t>", std::move(call_args))));
        }

        // Extract outputs
        body_stmts.push_back(llir::Store::make(
            llir::lVar::make(index_t, "out_nnz"),
            llir::lFieldAccess::make(result_var_expr, "nnz")));
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
                    body_stmts.push_back(llir::Store::make(
                        llir::lVar::make(index_t, "out_" + result_tensor.get_length_field_name(idx)),
                        llir::lFieldAccess::make(result_var_expr, result_tensor.get_length_field_name(idx))));
                }
            }
        }
        for (int i = (int)result_tensor.tensor_type.format.levels.size() - 1; i >= 0; i--) {
            auto idx = result_tensor.tensor_type.format.levels[i].index;
            if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(idx))) {
                body_stmts.push_back(llir::Store::make(
                    llir::lVar::make(llir::Ptr_t::make(index_t), "out_" + result_tensor.get_indices_field_name(idx)),
                    llir::lFieldAccess::make(result_var_expr, result_tensor.get_indices_field_name(idx))));
                if (i != 0) {
                    body_stmts.push_back(llir::Store::make(
                        llir::lVar::make(llir::Ptr_t::make(index_t), "out_" + result_tensor.get_offsets_field_name(idx)),
                        llir::lFieldAccess::make(result_var_expr, result_tensor.get_offsets_field_name(idx))));
                }
            }
        }
        body_stmts.push_back(llir::Store::make(
            llir::lVar::make(llir::Ptr_t::make(value_t), "out_values"),
            llir::lFieldAccess::make(result_var_expr, "values")));

        llir::lStmt body = llir::Sequence::make(std::move(body_stmts));
        printer.print(llir::Function::make(std::move(generics), std::move(attributes),
                                           std::move(args), std::move(ret_type),
                                           op_name, std::move(body)));
    }

}
} // namespace nacho
