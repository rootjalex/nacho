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

    CINLowerer::CINLowerer(CIN cin, std::ostream &os) : cin(std::move(cin)), printer(os), reductionLoops({}) {
        loop_order = get_loop_order();
        struct TensorVisitor : Visitor {
            std::map<std::string, TensorLowerer> &operand_tensors;
            TensorLowerer &result_tensor;
            TensorLowerer &reduced_result_tensor;
            std::vector<std::string> &loop_order;
            std::vector<LoopNum> &reductionLoops;
            bool &is_scatter_reduction;
            TensorVisitor(std::map<std::string, TensorLowerer> &operand_tensors, TensorLowerer &result_tensor, TensorLowerer &reduced_result_tensor, 
                std::vector<std::string> &loop_order, std::vector<LoopNum> &reductionLoops, bool &is_scatter_reduction)
                : operand_tensors(operand_tensors), result_tensor(result_tensor), reduced_result_tensor(reduced_result_tensor), loop_order(loop_order), reductionLoops(reductionLoops), is_scatter_reduction(is_scatter_reduction) {}

            void add_tensor(std::string str, TensorType type) {
                TensorLowerer lowerer(str, type, loop_order);
                operand_tensors[str] = lowerer;
            }

            void visit(const cTensor *node) override {
                add_tensor(node->name, node->type);
            }

            void visit(const Accumulate *node) override { 
                result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                for(const auto& idx : node->accumulate_indices) {
                    reductionLoops.push_back(result_tensor.get_loop_num(idx));
                }
                std::sort(reductionLoops.begin(), reductionLoops.end());

                is_scatter_reduction = false;
                for(auto loop = BEFORE_FIRST_LOOP+1; loop < result_tensor.end_loop_num(); ++loop) {
                    if(result_tensor.tensor_level_exists(loop) && result_tensor.is_sparse(loop) && reductionLoops.size()>0) {
                        int l = -1;
                        for(; l+1 < reductionLoops.size();l++) {
                            if(reductionLoops[l+1] > loop) {
                                break;
                            }
                        }
                        if(l>=0 && reductionLoops[l] != loop) {
                            is_scatter_reduction = true;
                            break;
                        }
                    }
                }
                if(node->expr.as<cAdd>()) {
                    result_tensor = TensorLowerer(node->tensor+"_temp",  node->expr.as<cAdd>()->type, loop_order, true);
                    reduced_result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                } else if(node->expr.as<cMul>()) {
                    result_tensor = TensorLowerer(node->tensor+"_temp",  node->expr.as<cMul>()->type, loop_order, true);
                    reduced_result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                } else {
                    internal_assert(false) << "Expected Accumulate to have cAdd or cMul expr";
                }
                node->expr.accept(this); 
            }

            void visit(const Assign *node) override { 
                result_tensor = TensorLowerer(node->tensor, node->type, loop_order, true);
                node->expr.accept(this); 
            }
        };

        TensorVisitor visitor(operand_tensors, result_tensor, reduced_result_tensor, loop_order, reductionLoops, is_scatter_reduction);
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
        auto forall_list = get_forall_list();

        std::vector<LoopNum> sparse_intersection_levels = get_all_sparse_intersection_levels(cin);

        sparse_intersection_levels.insert(sparse_intersection_levels.begin(), BEFORE_FIRST_LOOP);
        if(sparse_intersection_levels.back()!=LoopNum(loop_order.size()-1)) {
            sparse_intersection_levels.push_back(LoopNum(loop_order.size()-1));
        }

        this->lower_binary_search_function(true);
        this->lower_binary_search_function(false);
        this->lower_struct_definitions(sparse_intersection_levels[sparse_intersection_levels.size()-2]);

        for(int i=0; i< sparse_intersection_levels.size()-1;i++) {
            LoopNum previous_sparse_intersection = sparse_intersection_levels[i];
            LoopNum current_sparse_intersection = sparse_intersection_levels[i+1];
            LoopNum next_sparse_intersection = i!=sparse_intersection_levels.size()-2 ? sparse_intersection_levels[i+2] : LoopNum(forall_list.size());

            auto previous_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + previous_sparse_intersection.get() + 1);
            auto current_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + current_sparse_intersection.get() + 1);

            // Generate work functions for partition kernel
            for(LoopNum loop=BEFORE_FIRST_LOOP+1; loop<=previous_sparse_intersection;++loop) {
                printer.print(result_tensor.lower_work_function(previous_loop_order, loop));
            }
            auto included_tensors = get_included_tensors_for_level(current_sparse_intersection);
            for(LoopNum level = previous_sparse_intersection + 1; level <= current_sparse_intersection; ++level) {
                for (auto it : included_tensors) {
                    printer.print(it.second.lower_work_function(current_loop_order, level));
                }
            }


            PartitionKernelLowerer partition_lowerer(operand_tensors, result_tensor, included_tensors, forall_list, previous_sparse_intersection, current_sparse_intersection, next_sparse_intersection, reductionLoops, is_scatter_reduction, reduced_result_tensor);

            printer.print(partition_lowerer.lower_partition_struct_definition());
            printer.print(partition_lowerer.lower_partition_kernel());

            // Get the modified CIN 
            CIN modified_cin = cin;
            
            if(i!=sparse_intersection_levels.size()-2){
                modified_cin = get_modified_cin_for_sparse_intersection(current_sparse_intersection, cin);
                included_tensors = get_included_tensors_for_level(next_sparse_intersection);
            }

            ComputeKernelLowerer compute_lowerer(operand_tensors, result_tensor, included_tensors, forall_list, modified_cin, previous_sparse_intersection, current_sparse_intersection, next_sparse_intersection, reductionLoops, is_scatter_reduction, reduced_result_tensor);
            
            // Generate Precompute kernels
            TensorLowerer & output_write_tensor = 
                i == sparse_intersection_levels.size()-2 && reductionLoops.size() > 0 &&  !is_scatter_reduction 
                    ? reduced_result_tensor : result_tensor;
            if(output_write_tensor.get_loop_num_for_prev_sparse_level(current_sparse_intersection+1) > BEFORE_FIRST_LOOP) {
                printer.print(compute_lowerer.lower_precompute_function());
            }

            // Generate Compute kernel
            // We need to lower an extra work function for compute kernel if this not innermost sparse intersect
            // The target_dim value is fixed for this work function.
            if(i!=sparse_intersection_levels.size()-2){
                for (auto it : included_tensors) {
                    auto next_loop_order = std::vector<std::string>(loop_order.begin(), loop_order.begin() + next_sparse_intersection.get() + 1);
                    printer.print(it.second.lower_work_function(next_loop_order, current_sparse_intersection, true));
                }
            }
            printer.print(compute_lowerer.lower_compute_function());

        }
        
    }

    // lower_struct_definitions loweres all the initial struct definitions for the program
    // this includes tensor struct definitions for both the operand and result tensors
    // This also result_per_thread_count structs and result_to_operand_pos_map structs
    // which might be required by the intermediate kernels.
    void CINLowerer::lower_struct_definitions(LoopNum last_sparse_intersection) {
        for (auto it : operand_tensors) {
            printer.print(it.second.lower_tensor_struct_definition());
            // printer.print(it.second.lower_tensor_index_definition());
        }
        printer.print(result_tensor.lower_tensor_struct_definition());
        if(is_scatter_reduction) {
            printer.print(reduced_result_tensor.lower_tensor_struct_definition());
        }


        // Use BaseKernelLowerer to lower the one time struct definitions of result_per_thread_count struct and result_to_operand_pos_map struct
        auto empty_map = std::map<std::string, TensorLowerer>();
        BaseKernelLowerer BaseLowerer(operand_tensors, result_tensor, empty_map, get_forall_list(), BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, reductionLoops, is_scatter_reduction, reduced_result_tensor);
        // Need to lower this struct only once
        if(!result_tensor.are_all_lvls_dense()) {
            printer.print(BaseLowerer.lower_result_per_thread_count_struct());
        }

        auto result_operand_pos_map = lower_result_pos_to_operand_pos_map_struct(last_sparse_intersection);
        if(result_operand_pos_map.get() != nullptr) {
            printer.print(result_operand_pos_map);
        }
    }

    llir::lType CINLowerer::lower_result_pos_to_operand_pos_map_struct(LoopNum last_sparse_intersection) {
        std::vector<std::string> generics = {"index_t"};
        std::vector<std::pair<std::string, llir::lType>> fields;
        auto forall_list = get_forall_list(); auto empty_map = std::map<std::string, TensorLowerer>();
        BaseKernelLowerer BaseLowerer(operand_tensors, result_tensor, empty_map, forall_list, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, BEFORE_FIRST_LOOP, reductionLoops, is_scatter_reduction, reduced_result_tensor);
        for(LoopNum loop=BEFORE_FIRST_LOOP+1; loop<=last_sparse_intersection;++loop) {
            if(loop >= LoopNum(forall_list.size()-1)) {
                continue;
            }
            const Forall* forall = forall_list[loop.get()].as<Forall>();
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

    std::vector<LoopNum> CINLowerer::get_all_sparse_intersection_levels(CIN& cin) {
        struct Checker : public Visitor {
            LoopNum loop_num = BEFORE_FIRST_LOOP;
            bool inside_sparse_intersection = false;
            bool found_sparse_intersection = false;
            std::vector<LoopNum> sparse_intersection_levels;

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
                ++loop_num;
                found_sparse_intersection = false;

                // level should be considered separately as a sparse intersection only if it is a unique level.
                // TODO : not sure if the unique requirment is entirely correct, need to investigate further.
                // TODO : if changed also change in RemoveDenseLocators
                if(node->seq.get()->is_unique) {
                    //<<" Going inside seq "<<(Seq)node->seq<<std::endl;
                    node->seq.accept(this);
                    if(found_sparse_intersection) {
                        sparse_intersection_levels.push_back(loop_num);
                    }
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

    CIN CINLowerer::get_modified_cin_for_sparse_intersection(LoopNum target_loop, CIN& cin) {
        struct Modifier: public Mutator {
            LoopNum loop_num = BEFORE_FIRST_LOOP;
            LoopNum target_loop;
            Modifier(LoopNum target_loop) : target_loop(target_loop) {}
            CIN visit(const Forall * node) override{
                ++loop_num;
                if(loop_num < target_loop) {
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
        return Modifier(target_loop).mutate(cin);
    }


    void CINLowerer::lower_binary_search_function(bool is_upper_bound) {
        std::vector<std::string> generics = {"index_t"};

        std::vector<llir::Function::Attribute> attributes = {
            llir::Function::device, llir::Function::inline_};

        std::vector<llir::Function::Argument> args;
        llir::lType ret_type;
        std::string name;
        llir::lStmt body;

        name = "binary_search_" + (std::string)(is_upper_bound ? "ub" : "lb");

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
        llir::lExpr mid_expr = start_var + (((end_var - start_var) + 1) / 2);

        stmts.emplace_back(
            llir::Declare::make(
                index_t,
                "mid",
                mid_expr
            )
        );  
        std::vector<llir::lStmt> while_stmts;

        while_stmts.emplace_back(
            llir::Store::make(
                mid_var,
                mid_expr
            )
        );

        if(is_upper_bound) {
            while_stmts.emplace_back(llir::IfElse::make(
                arr_var[mid_var] <= target_value_var,
                llir::Store::make(start_var, mid_var),
                llir::Store::make(end_var, mid_var - 1)
            ));
        } else {
            while_stmts.emplace_back(
                llir::IfElse::make(
                    arr_var[mid_var] < target_value_var,
                    llir::Store::make(start_var, mid_var),
                    llir::IfElse::make(
                    arr_var[mid_var] > target_value_var,
                    llir::Store::make(end_var, mid_var - 1),
                    llir::Sequence::make({
                        llir::IfElse::make(start_var == mid_var-1, llir::Return::make(mid_var), nullptr),
                        llir::Store::make(end_var, mid_var),
                    })
            )));
        }

        stmts.emplace_back(
            llir::While::make(
                start_var < end_var,
                llir::Sequence::make(std::move(while_stmts))
            )
        );

        stmts.emplace_back(
            llir::Store::make(
                mid_var,
                mid_expr
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


}
} // namespace nacho
