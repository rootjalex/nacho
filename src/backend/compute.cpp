#include "backend/compute.h"
#include "IRFwdDecl.h"
#include "Simplify.h"
#include "llir/Function.h"
#include "llir/LLIR.h"

#include <functional>

namespace nacho {
namespace backend {

void ComputeKernelLowerer::add_partition_assignments(
    std::vector<llir::lStmt> &stmts) {
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const llir::lType i32 = llir::Int_t::make(32);
    stmts.emplace_back(llir::Declare::make(
        i32, "thread_id",
        llir::lVar::make(index_t, "blockIdx.x") * llir::lVar::make(index_t, "blockDim.x") + llir::lVar::make(index_t, "threadIdx.x")
    ));

    llir::lExpr thread_id_var = llir::lVar::make(i32, "thread_id");

    // TODO: make this const!
    stmts.emplace_back(
        llir::Declare::make(i32, "max_thread_id",
                            llir::lVar::make(index_t, "gridDim.x") *
                                    llir::lVar::make(index_t, "blockDim.x") -
                                llir::lConst::make(1)));

    llir::lExpr max_thread_id_var = llir::lVar::make(i32, "max_thread_id");

    CIN loop = cin;

    llir::lExpr partitions_var = llir::lVar::make(
        llir::Generic_t::make(get_partition_struct_name() + "<index_t>"),
        "partitions");

    auto add_single_partition_load = [&](TensorLowerer &tensor, TensorLevelNum level,
                                         llir::lExpr maximum_iterator) {
        // TODO: make const!
        // index_t start_i = partitions.i[thread_id];
        llir::lExpr partitions_field =
            llir::lFieldAccess::make(partitions_var, tensor.get_iterator_suffix(level));
        stmts.emplace_back(llir::Declare::make(
            index_t, tensor.get_start_name(level),
            partitions_field[thread_id_var]));
        // index_t end_i = (thread_id < max_thread_id) ? partitions.i[thread_id
        // + 1] : iterator_max;
        llir::lExpr next_partition = partitions_field[thread_id_var + 1];
        llir::lExpr guarded_end =
            llir::lSelect::make(thread_id_var < max_thread_id_var,
                                next_partition, maximum_iterator);
        stmts.emplace_back(
            llir::Declare::make(index_t, tensor.get_end_name(level), guarded_end));
        return llir::lVar::make(index_t, tensor.get_start_name(level));
    };

    auto get_max_iterator = [&](const Index *idx) {
        TensorLowerer tlow = get_tensor(idx->tensor);
        return get_partition_initializer_expr_for_boundary_cases(tlow.tensor_level_to_loop_num(TensorLevelNum(idx->level)), tlow, true);
    };

    LoopNum loop_level = BEFORE_FIRST_LOOP + 1;
    while (const auto *forall = loop.as<Forall>()) {
        std::vector<llir::lExpr> iter_vars;
        if(loop_level<= previous_sparse_intersection) {
            if(result_tensor.tensor_level_exists(loop_level)) {
                auto idx = result_tensor.get_index_sequence(result_tensor.loop_num_to_tensor_level(loop_level));
                llir::lExpr iter_var = add_single_partition_load(result_tensor, TensorLevelNum(result_tensor.loop_num_to_tensor_level(loop_level)), get_max_iterator(idx.as<Index>()));
                iter_vars.emplace_back(std::move(iter_var));
            }
        } else {
            auto [iters, locs, has_universe_iter] = partition_iterators_locators(forall->seq);

            if (has_universe_iter) {
                // Iterating over the universe!
                const Index * idx;
                // get any index either from iterators or locators
                // just need to get the (logical) size of the level
                internal_assert(!locs.empty() || !iters.empty());
                if(!locs.empty()) {
                    idx = locs[0].as<Index>();
                    internal_assert(idx) << locs[0];
                } else {
                    idx = iters[0].as<Index>();
                    internal_assert(idx) << iters[0];
                }
                auto tlow = get_tensor(idx->tensor);
                // TODO: Dedup this with the get_partition_initializer_expr_for_boundary_cases function
                // issue is that its currently using a random tensor whose cooresponding level
                // may not be dense. Probably can use result tensor here?
                llir::lExpr max_iter = tlow.get_size_field(TensorLevelNum(idx->level)) 
                                         - (loop_level == current_sparse_intersection? 1 : 0);
                                         
                stmts.emplace_back(
                    llir::Declare::make(
                        index_t, "start_" + forall->idx,
                        llir::lFieldAccess::make(partitions_var, forall->idx)[thread_id_var]
                    ));
                stmts.emplace_back(
                    llir::Declare::make(
                        index_t, "end_" + forall->idx,
                        llir::lSelect::make(thread_id_var < max_thread_id_var,
                                            llir::lFieldAccess::make(partitions_var, forall->idx)[thread_id_var + 1],
                                            max_iter)
                    ));
                llir::lExpr iter_var = llir::lVar::make(index_t, "start_" + forall->idx);
                iter_vars.emplace_back(std::move(iter_var));
            } else {
                bool has_dense_iter = false;
                // For each iterator, construct
                for (const auto &iter : iters) {
                    const Index *idx = iter.as<Index>();
                    internal_assert(idx) << iter;
                    TensorLowerer tlow = get_tensor(idx->tensor);
                    has_dense_iter |= !tlow.is_sparse(TensorLevelNum(idx->level)); 
                    llir::lExpr max_iter = get_max_iterator(idx);
                    llir::lExpr iter_var =
                        add_single_partition_load(tlow, TensorLevelNum(idx->level), max_iter);
                    iter_vars.emplace_back(std::move(iter_var));
                }
            }
        }

        loop = forall->body;
        ++loop_level;

        // Last iterators need to be bumped forward by one!
        if (!loop.as<Forall>()) {
            for (const auto &var : iter_vars) {
                // iter++
                stmts.push_back(
                    llir::BaseExpr::make(llir::lIncrement::make(var)));
            }
        }
    }
    return;
}

llir::lStmt ComputeKernelLowerer::
    lower_precompute_function() {
    std::vector<std::string> generics = {"index_t", "value_t"};

    std::vector<llir::Function::Attribute> attributes = {
        llir::Function::global};

    std::vector<llir::Function::Argument> args;
    llir::lType ret_type;
    std::string name;
    llir::lStmt body;

    name = get_precompute_function_name();

    ret_type = llir::Generic_t::make("void");

    for (auto tensor : operand_tensors) {
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make(tensor.second.get_struct_name() +
                                          "<index_t, value_t>"),
            .name = tensor.second.tensor_name});
    }

    // Add a partitions_{loops} argument type.
    args.emplace_back(llir::Function::Argument{
        .mutating = false,
        .type =
            llir::Generic_t::make(get_partition_struct_name() + "<index_t>"),
        .name = "partitions"});

    if(!output_write_tensor.are_all_lvls_dense())
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Generic_t::make(get_counts_struct_name() + "<index_t>"),
            .name = "count_offsets"});

    args.emplace_back(llir::Function::Argument{
        .mutating = false, .type = index_t, .name = "per_thread_work"});

    if(previous_sparse_intersection != BEFORE_FIRST_LOOP)
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make(result_tensor.get_struct_name() +
                                        "<index_t, value_t>"),
            .name = result_tensor.tensor_name});

    bool need_operand_pos_map_arg = false;
    for(LoopNum i = BEFORE_FIRST_LOOP+1; i<=previous_sparse_intersection; ++i) {
        for(auto it: operand_tensors) {
            if(exists_field_in_result_to_operand_pos_map(forall_list[i.get()].as<Forall>(), it.second)){
                need_operand_pos_map_arg = true;
                break;
            }
        }
    }

    if(need_operand_pos_map_arg) {
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Generic_t::make(get_result_to_operand_pos_map_struct_name() + "<index_t>"),
            .name = get_result_to_operand_pos_map_var_name()
        });
    }

    std::vector<llir::lStmt> stmts;

    // Add common initialization statements
    add_partition_assignments(stmts);

    // index_t count = thread_id * per_thread_work;
    stmts.emplace_back(llir::Declare::make(
        llir::Generic_t::make("index_t"), "count",
            llir::lVar::make(index_t, "thread_id") * llir::lVar::make(index_t, "per_thread_work")));

    // Declare local count variables to be used to calculat the offsets into
    // result. These will be stored in count_offsets at the end.
    for (LoopNum i = BEFORE_FIRST_LOOP+1; i <= current_sparse_intersection; ++i) {
        auto index = output_write_tensor.loop_name(i);
        if (output_write_tensor.tensor_level_exists(i) && output_write_tensor.is_sparse(i)) {
            stmts.emplace_back(llir::Declare::make(
                index_t, "count_" + index, llir::lConst::make((int64_t)0)));
        }
    }

    std::set<Seq, SeqLessThan> defined; // initially empty
    internal_assert(cin.defined());
    stmts.push_back(lower_loop(cin, defined, /*is_precompute*/ true, BEFORE_FIRST_LOOP+1, nullptr));

    for (LoopNum i = BEFORE_FIRST_LOOP+1; i <= current_sparse_intersection; ++i) {
        auto index = output_write_tensor.loop_name(i);
        if (output_write_tensor.tensor_level_exists(i) && output_write_tensor.is_sparse(i)) {
            stmts.emplace_back(llir::Store::make(
                llir::lArrayAccess::make(
                    llir::lFieldAccess::make(
                        llir::lVar::make(
                            llir::Generic_t::make(get_counts_struct_name()),
                            "count_offsets"),
                        get_counts_field_name(index)),
                    llir::lVar::make(index_t, "thread_id")),
                llir::lVar::make(index_t, "count_" + index)));
        }
    }

    stmts.emplace_back(llir::Return::make());
    body = llir::Sequence::make(std::move(stmts));

    return llir::Function::make(std::move(generics), std::move(attributes),
                                std::move(args), std::move(ret_type), name,
                                std::move(body));
}

llir::lStmt ComputeKernelLowerer::
    lower_compute_function() {
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    std::vector<std::string> generics = {"index_t", "value_t"};

    std::vector<llir::Function::Attribute> attributes = {
        llir::Function::global};

    std::vector<llir::Function::Argument> args;
    llir::lType ret_type;
    std::string name;
    llir::lStmt body;

    name = get_compute_function_name();

    ret_type = llir::Generic_t::make("void");

    for (auto tensor : operand_tensors) {
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make(tensor.second.get_struct_name() +
                                          "<index_t, value_t>"),
            .name = tensor.second.tensor_name});
    }

    // Add partition argument.
    args.emplace_back(llir::Function::Argument{
        .mutating = false,
        .type =
            llir::Generic_t::make(get_partition_struct_name() + "<index_t>"),
        .name = "partitions"});

    if(!output_write_tensor.are_all_lvls_dense())
        args.emplace_back(llir::Function::Argument{
            .mutating = false,
            .type = llir::Generic_t::make(get_counts_struct_name() + "<index_t>"),
            .name = "count_offsets"});

    args.emplace_back(llir::Function::Argument{
        .mutating = false, .type = index_t, .name = "per_thread_work"});

    args.emplace_back(llir::Function::Argument{
        .mutating = true,
        .type = llir::Generic_t::make(result_tensor.get_struct_name() +
                                      "<index_t, value_t>"),
        .name = result_tensor.tensor_name});

    if(next_sparse_intersection != LoopNum(forall_list.size())) {
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Ptr_t::make(index_t),
            .name = "T_work_offsets"});
    }

    bool need_operand_pos_map_arg = false;
    LoopNum loop_num = next_sparse_intersection==LoopNum(forall_list.size()) ? previous_sparse_intersection: current_sparse_intersection;
    for(LoopNum i = BEFORE_FIRST_LOOP+1; i <= loop_num; ++i) {
        for(auto it: operand_tensors) {
            if(exists_field_in_result_to_operand_pos_map(forall_list[i.get()].as<Forall>(), it.second)){
                need_operand_pos_map_arg = true;
                break;
            }
        }
    }

    if(need_operand_pos_map_arg) {
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Generic_t::make(get_result_to_operand_pos_map_struct_name() + "<index_t>"),
            .name = get_result_to_operand_pos_map_var_name()
        });
    }

    std::vector<llir::lStmt> stmts;

    // Add common initialization statements
    add_partition_assignments(stmts);

    for (LoopNum i = BEFORE_FIRST_LOOP+1; i <= current_sparse_intersection; ++i) {
        auto index = output_write_tensor.loop_name(i);
        if (output_write_tensor.tensor_level_exists(i) && output_write_tensor.is_sparse(i)) {
            stmts.emplace_back(llir::Declare::make(
                index_t, "offset_" + index,
                llir::lArrayAccess::make(
                    llir::lFieldAccess::make(
                        llir::lVar::make(
                            llir::Generic_t::make(get_counts_struct_name()),
                            "count_offsets"),
                        get_counts_field_name(index)),
                    llir::lVar::make(index_t, "thread_id"))
            ));
        }
    }

    std::set<Seq, SeqLessThan> defined; // initially empty
    internal_assert(cin.defined());
    stmts.push_back(lower_loop(cin, defined, /*is_precompute*/ false, BEFORE_FIRST_LOOP+1, nullptr));

    stmts.emplace_back(llir::Return::make());
    body = llir::Sequence::make(std::move(stmts));

    return llir::Function::make(std::move(generics), std::move(attributes),
                                std::move(args), std::move(ret_type), name,
                                std::move(body));
}


std::map<TensorLevelNum,llir::lExpr> get_iter_vars_result(TensorLowerer &tlower, TensorLevelNum end_level) {
        std::map<TensorLevelNum,llir::lExpr> iter_vars;
        for (TensorLevelNum level = BEFORE_FIRST_LEVEL+1; level < end_level; ++level) {
            std::string iter_name = tlower.tensor_level_name(level);
            if(tlower.is_sparse(level)) {
                iter_name = "offset_" + tlower.tensor_level_name(level);
            }
            iter_vars[level] = llir::lVar::make(
                llir::Generic_t::make("index_t"),
                iter_name
            );
        }
        return iter_vars;
};

llir::lStmt ComputeKernelLowerer::lower_assign_statement(
      CIN assign, bool is_precompute) {
    
    if(is_precompute) {
        return llir::lStmt();
    }

    struct Converter : public Visitor {
        llir::lExpr result;
        std::map<std::string, TensorLowerer>& operand_tensors;
        Converter(std::map<std::string, TensorLowerer>& operand_tensors)
            : operand_tensors(operand_tensors) {}

        void visit(const cAdd *node) override {
            node->a.accept(this);
            auto a = std::move(result);
            node->b.accept(this);
            auto b = std::move(result);
            result = a + b;
        }

        void visit(const cMul *node) override {
            node->a.accept(this);
            auto a = std::move(result);
            node->b.accept(this);
            auto b = std::move(result);
            result = a * b;
        }


        void visit(const cTensor *node) override {
            auto get_iter_vars_operands = [&](TensorLowerer &tlower, TensorLevelNum end_level) {
                std::map<TensorLevelNum, llir::lExpr> iter_vars;
                for (TensorLevelNum level = BEFORE_FIRST_LEVEL+1; level < end_level; ++level) {
                    std::string iter_name = tlower.get_iter_name(level);
                    iter_vars[level] = llir::lVar::make(
                        llir::Generic_t::make("index_t"),
                        iter_name
                    );
                }
                return iter_vars;
            };

            result =  llir::lArrayAccess::make(
                llir::lFieldAccess::make(
                    llir::lVar::make(
                        llir::Generic_t::make(operand_tensors[node->name].get_struct_name()),
                        node->name
                    ),
                    operand_tensors[node->name].get_values_field_name()
                ),
                operand_tensors[node->name].get_level_indexing_expression(
                    operand_tensors[node->name].end_tensor_level(),
                    false,
                    get_iter_vars_operands(
                        operand_tensors[node->name],
                        operand_tensors[node->name].end_tensor_level()
                    )
                )
            );
        };
    };

    if(assign.as<Assign>()) {
        const Assign* assign_stmt = assign.as<Assign>();
        auto converter = Converter(operand_tensors);
        assign_stmt->expr.accept(&converter);
        return llir::Store::make(
            output_write_tensor.get_values_field()[
                output_write_tensor.get_level_indexing_expression(
                    output_write_tensor.end_tensor_level(),
                    false,
                    get_iter_vars_result(
                        output_write_tensor,
                        output_write_tensor.end_tensor_level()
                    )
                )],
            converter.result
        );

    } else if(assign.as<Accumulate>()) {
        const Accumulate* accumulate_stmt = assign.as<Accumulate>();
        auto converter = Converter(operand_tensors);
        accumulate_stmt->expr.accept(&converter);

        // Append reduction case
        if(!is_scatter_reduction) {
            // Append case optimization: When reduction is there on one of the innermost loop. Can optimize by not doing an accumulate add directly
            // instead write to a temporary variable and atomic add that once in a loop that is present in the result.
            if(reduction_loops.back() + 1 == output_write_tensor.end_loop_num()) {
                return llir::Accumulate::make(
                    llir::lVar::make(value_t, "value"),
                    converter.result
                );
            } else {
                return llir::BaseExpr::make(
                llir::lFunctionCall::make(
                    "atomicAdd",
                    {
                        llir::lAddress::make(output_write_tensor.get_values_field()[
                            output_write_tensor.get_level_indexing_expression(
                                output_write_tensor.end_tensor_level(),
                                false,
                                get_iter_vars_result(
                                    output_write_tensor,
                                    output_write_tensor.end_tensor_level()
                                )
                            )]),
                        converter.result
                    }
                )
            );
            }
        } else {
            return llir::Store::make(
            output_write_tensor.get_values_field()[
                output_write_tensor.get_level_indexing_expression(
                    output_write_tensor.end_tensor_level(),
                    false,
                    get_iter_vars_result(
                        output_write_tensor,
                        output_write_tensor.end_tensor_level()
                    )
                )],
            converter.result
            );
        }

    } else if(assign.as<CalculateWork>()) {
        llir::lExpr work_expr = llir::lConst::make((int64_t)0);
        auto get_work_expr = [&](TensorLowerer& Tensor) {
            std::vector<llir::lExpr> work_args;
            work_args.emplace_back(
                llir::lVar::make(llir::Generic_t::make(Tensor.get_struct_name()), Tensor.tensor_name)
            );
            for(LoopNum j=BEFORE_FIRST_LOOP+1;j<current_sparse_intersection;++j) {
                if(Tensor.tensor_level_exists(j))
                    work_args.emplace_back(Tensor.get_iter_var(Tensor.loop_num_to_tensor_level(j), index_t));
            }

            if(Tensor.tensor_level_exists(current_sparse_intersection) && Tensor.is_sparse(current_sparse_intersection)) {
                work_args.emplace_back(Tensor.get_iter_var(Tensor.loop_num_to_tensor_level(current_sparse_intersection), index_t));
            } else {
                work_args.emplace_back(llir::lVar::make(index_t, Tensor.loop_name(current_sparse_intersection)));
            }

            // pass broadcast sizes for all dimensions after current forall which are not present in the tensor
            for(LoopNum j=current_sparse_intersection+1;j<=next_sparse_intersection;++j){
                if(!Tensor.tensor_level_exists(j)) {
                    auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [&](const auto& op_tensor) {
                        return op_tensor.second.tensor_level_exists(j);
                    });
                    internal_assert(it != operand_tensors.end()) << "Expected operand tensor to exist";

                    work_args.emplace_back(it->second.get_size_field(it->second.loop_num_to_tensor_level(j)));
                }
            }

            std::string forall_idx = forall_list[current_sparse_intersection.get()].as<Forall>()->idx;
            return llir::lFunctionCall::make(Tensor.get_work_function_name(get_all_loops_string(next_sparse_intersection),forall_idx),work_args);
        };

        // Finds the tensors which need to be included in the work calculation
        struct WorkTensorGetter : Visitor {
            std::vector<TensorLowerer> tensors;
            std::map<std::string, TensorLowerer>& included_tensors;

            WorkTensorGetter(std::map<std::string, TensorLowerer>& included_tensors)
                : included_tensors(included_tensors) {}

            void visit(const cTensor *node) override {
                if (included_tensors.find(node->name) != included_tensors.end()) {
                    tensors.emplace_back(included_tensors.at(node->name));
                }
            }
        };


        WorkTensorGetter work_tensor_getter(included_tensors);
        const CalculateWork* calculate_work_stmt = assign.as<CalculateWork>();
        calculate_work_stmt->body.accept(&work_tensor_getter);

        for(auto it: work_tensor_getter.tensors) {
            work_expr = work_expr + get_work_expr(it);
        }
        internal_assert(result_tensor.tensor_level_exists(current_sparse_intersection)) << "Current sparse intersection " << current_sparse_intersection.get() << " does not exist in tensor " << result_tensor.tensor_name;
        return llir::Store::make(
            llir::lVar::make(llir::Generic_t::make("index_t"), "T_work_offsets")[
                result_tensor.get_level_indexing_expression(
                    result_tensor.loop_num_to_tensor_level(current_sparse_intersection)+1,
                    true,
                    get_iter_vars_result(
                        result_tensor,
                        result_tensor.loop_num_to_tensor_level(current_sparse_intersection)+1
                    )
                )
            ],
            work_expr
        );  
    } else {
        internal_assert(false) << "Expected Assign or Accumulate in lower_assign_statement: " << assign;
    }

    return llir::lStmt();
}

llir::lStmt ComputeKernelLowerer::lower_loop(
    CIN loop, const std::set<Seq, SeqLessThan> &defined,
    bool is_precompute, LoopNum loop_num, llir::lExpr offset_write_cond) {
    // std::cout<<"Lowering loop "<<loop<<std::endl;
    const Forall *forall = loop.as<Forall>();
    internal_assert(forall) << "Expected Forall in lower_loop: " << loop;

    bool is_loop_before_prev_intersection = loop_num <= previous_sparse_intersection;

    // Two optimizations:
    // 1. The intersection/union of dense iterators is a single dense iterator.
    // 2. Remove dense iterators when intersected with a sparse iterator.
    // TODO: the Universes inserted here breaks a lot of assumptions,
    // And breaks partitioning. Need to fix this earlier in the pipeline.
    // auto [seq, locators] = remove_locators(forall->seq);
    // TODO: handle locators!
    const auto &seq = forall->seq;

    // Check if seq is in lattices, if it is, use that lattice, otherwise
    // build it.

    auto it = lattices.find(seq);
    if (it == lattices.end()) {
        it = lattices.emplace(seq, Lattice::build(seq)).first;
    }

    const Lattice &lattice = it->second;

    std::vector<llir::lStmt> stmts;

    llir::lType index_t = llir::Generic_t::make("index_t");

    auto [forall_iters, _, has_universe_iter] =
        partition_iterators_locators(forall->seq);

    // std::cout<<forall->seq<<" iters: "<<std::endl;
    // for(const auto& iter: forall_iters) {
    //     std::cout<<iter<<", ";
    // }
    // std::cout<<std::endl;
    
    if(is_loop_before_prev_intersection) {
        // If this loop is before the previous intersection, then we are going to iterate only over the result tensor index
        forall_iters = std::vector<Seq>{result_tensor.get_index_sequence(result_tensor.loop_num_to_tensor_level(loop_num))};
        has_universe_iter = false; // universe iterator is not needed for loops before the previous intersection since we are only iterating over the result tensor index
    }

    using lExprTuple = std::tuple<llir::lExpr, llir::lExpr, llir::lExpr>;
    std::map<Seq, lExprTuple, SeqLessThan> imap;

    auto get_start = [&](TensorLevelNum level, TensorLowerer& tlower) {
        if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
            // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
            auto loop_num = tlower.tensor_level_to_loop_num(level);
            return result_tensor.get_start(result_tensor.loop_num_to_tensor_level(loop_num));
        } else {
            return tlower.get_start(level);
        }
    };

    auto get_end = [&](TensorLevelNum level, TensorLowerer& tlower) {
        if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
            // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
            auto loop_num = tlower.tensor_level_to_loop_num(level);
            return result_tensor.get_end(result_tensor.loop_num_to_tensor_level(loop_num));
        } else {
            return tlower.get_end(level);
        }
    };

    auto get_iter = [&](TensorLevelNum level, TensorLowerer& tlower) {
        if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
            // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
            auto loop_num = tlower.tensor_level_to_loop_num(level);
            return result_tensor.get_iter(result_tensor.loop_num_to_tensor_level(loop_num));
        } else {
            return tlower.get_iter(level);
        }
    };

    auto get_stop = [&](TensorLevelNum level, TensorLowerer& tlower) {
        if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
            // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
            auto loop_num = tlower.tensor_level_to_loop_num(level);
            return result_tensor.get_stop(result_tensor.loop_num_to_tensor_level(loop_num));
        } else {
            return tlower.get_stop(level);
        }
    };

    auto get_seg_end = [&](TensorLevelNum level, TensorLowerer& tlower) {
        if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
            // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
            auto loop_num = tlower.tensor_level_to_loop_num(level);
            return result_tensor.is_sparse(level) && !result_tensor.is_unique(loop_num) 
                ? result_tensor.get_seg_end(result_tensor.loop_num_to_tensor_level(loop_num)) 
                : result_tensor.get_iter(result_tensor.loop_num_to_tensor_level(loop_num));
        } else {
            return tlower.is_sparse(level) && !tlower.is_unique(level) 
                ? tlower.get_seg_end(level) 
                : tlower.get_iter(level);
        }
    };

    // gets all the iterator initialze conditions for a particular loop 'loop_num"
    // This condition is needed to initialze an iterator in a lower loop in a tensor, 
    // if 'loop_num' is not a level in that tensor
    // 
    std::function<llir::lExpr(LoopNum, bool)> get_all_iterator_conditions =
        [&](LoopNum loop_num, bool end) -> llir::lExpr {
            llir::lExpr cond;
            const Forall * forall = forall_list[loop_num.get()].as<Forall>();
            std::string loop_idx = forall->idx;
            auto [iters, _, has_universe_iter] = partition_iterators_locators(forall->seq);
            if(is_loop_before_prev_intersection) {
                iters = std::vector<Seq>{result_tensor.get_index_sequence(result_tensor.loop_num_to_tensor_level(loop_num))};
                has_universe_iter = false;
            }
            // if any dense iter is present condition is just if that iter is at start/end
            if(has_universe_iter || std::any_of(iters.begin(), iters.end(), [&](const Seq& iter){return !get_tensor(iter.as<Index>()->tensor).is_sparse(TensorLevelNum(iter.as<Index>()->level));})) {
                return llir::lVar::make(index_t, "iter_" + loop_idx) ==
                    (end ? llir::lVar::make(index_t, "end_" + loop_idx) : llir::lVar::make(index_t, "start_" + loop_idx));
            }

            // all iters sparse
            // in this case condition is 
            // for start - if all of the sparse iters are at start as we store next iters if
            // // the iter does not lie on partition boundary.
            // for end - if any of the end iters are at end, as this means parition boundary is already reached.
            // // iter to first reach its end would only reach it at parition boundary. as other iters are guaranteed 
            // // to be ahead of boundaries.
            for(const auto& iter: iters) {
                auto idx = iter.as<Index>();
                TensorLowerer tlower = get_tensor(idx->tensor);
                llir::lExpr val = get_iter(TensorLevelNum(idx->level), tlower);
                llir::lExpr extrema = end ? get_end(TensorLevelNum(idx->level), tlower) : get_start(TensorLevelNum(idx->level), tlower);
                if(cond.defined()) {
                    // We use || here as if any of the iterator is at start position
                    cond = end ? cond || (val == extrema) : cond && (val == extrema);
                } else {
                    cond = val == extrema;
                }
            }
            return cond;
        };

    // Insert prologue (initiate iterators!).
    if (has_universe_iter) {
        // Iterating the universe!
        llir::lExpr start_cond;
        llir::lExpr stop_cond;
        for(LoopNum ln = BEFORE_FIRST_LOOP+1; ln < loop_num; ++ln) {
            if(start_cond.defined()) {
                start_cond = start_cond && get_all_iterator_conditions(ln, false);
                stop_cond = stop_cond && get_all_iterator_conditions(ln, true);
            } else {
                start_cond = get_all_iterator_conditions(ln, false);
                stop_cond = get_all_iterator_conditions(ln, true);
            }
        }

        auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [&](const auto& op_tensor) {
            return op_tensor.second.tensor_level_exists(loop_num);
        });
        internal_assert(it != operand_tensors.end()) << "No tensor found for loop num: " << loop_num;

        stmts.push_back(llir::Declare::make(
            index_t, "iter_" + forall->idx,
            start_cond.defined() 
                ? llir::lSelect::make(std::move(start_cond), llir::lVar::make(index_t, "start_" + forall->idx), llir::lConst::make((int32_t)0))
                : llir::lVar::make(index_t, "start_" + forall->idx))) ;
        stmts.push_back(llir::Declare::make(
            index_t, "stop_" + forall->idx,
            stop_cond.defined() 
                ? llir::lSelect::make(std::move(stop_cond), llir::lVar::make(index_t, "end_" + forall->idx), it->second.get_size_field(it->second.loop_num_to_tensor_level(loop_num))-1)
                : llir::lVar::make(index_t, "end_" + forall->idx)));
        llir::lExpr offset_write_sub_cond = llir::lVar::make(index_t, "iter_" + forall->idx) < llir::lVar::make(index_t, "stop_" + forall->idx);
        offset_write_cond = offset_write_cond.defined() ? offset_write_cond || offset_write_sub_cond : offset_write_sub_cond;
    }
    // add a default mapping for Universe iterator (TODO: very hacky (using Empty) handle this better)
    imap[Empty::make(false)] = {llir::lVar::make(index_t, "iter_" + forall->idx), llir::lVar::make(index_t, "stop_" + forall->idx), llir::lVar::make(index_t, "iter_" + forall->idx)};

    for (const auto &i : forall_iters) {
        const Index *idx = i.as<Index>();
        internal_assert(idx) << i;

        TensorLowerer tlower = get_tensor(idx->tensor);

        llir::lExpr pidx = get_start(TensorLevelNum(idx->level), tlower);
        llir::lExpr pend = get_end(TensorLevelNum(idx->level), tlower);

        llir::lExpr start_value;
        llir::lExpr stop_value;
        if (loop_num == BEFORE_FIRST_LOOP + 1) {
            // Fully iterate the partition.
            start_value = pidx;
            stop_value = pend;
        } else {
            std::function<llir::lExpr(LoopNum, bool)> get_condition =
                [&](LoopNum loop_num, bool end) -> llir::lExpr {
                internal_assert(loop_num > BEFORE_FIRST_LOOP) << loop_num;
                llir::lExpr cond;
                if(tlower.tensor_level_exists(loop_num)){
                    auto level = tlower.loop_num_to_tensor_level(loop_num);
                    llir::lExpr val = get_iter(level, tlower);
                    llir::lExpr extrema = end ? get_end(level, tlower) : get_start(level, tlower);

                    cond = val == extrema;
                } else {
                    cond = get_all_iterator_conditions(loop_num, end);
                }
                if (loop_num == BEFORE_FIRST_LOOP + 1) {
                    return cond;
                } else {
                    llir::lExpr rec = get_condition(loop_num - 1, end);
                    return cond && rec;
                }
            };

            std::map<TensorLevelNum, llir::lExpr> pos_vars_start, pos_vars_end;
            for(TensorLevelNum level = BEFORE_FIRST_LEVEL + 1; level < TensorLevelNum(idx->level); ++level) {
                pos_vars_start[level] = tlower.get_iter(level);
                pos_vars_end[level] = tlower.is_sparse(level) && !tlower.is_unique(level) ? tlower.get_seg_end(level) : tlower.get_iter(level);
            }

            // For the start value, if all previous iterators are at their
            // respective starts, then use this start, otherwise get the
            // iterator from the data structure!
            llir::lExpr start_cond = get_condition(tlower.tensor_level_to_loop_num(TensorLevelNum(idx->level))-1, false);
            start_value = llir::lSelect::make(
                std::move(start_cond), pidx,
                tlower.get_bound(TensorLevelNum(idx->level), /*upper_bound=*/false, pos_vars_start));

            // For the stop, if all previous iterators are at their respective
            // ends, then use this stop, otherwise get the iterator from the
            // data structure!
            llir::lExpr stop_cond = get_condition(tlower.tensor_level_to_loop_num(TensorLevelNum(idx->level))-1, true);
            llir::lExpr bound =
                tlower.get_bound(TensorLevelNum(idx->level), /*upper_bound=*/true, pos_vars_end);
            stop_value = llir::lSelect::make(std::move(stop_cond), pend,
                                             std::move(bound));
        }

        stmts.push_back(llir::Declare::make(
            index_t, tlower.get_iter_name(TensorLevelNum(idx->level)), start_value));
        // This is const.
        stmts.push_back(llir::Declare::make(
            index_t, tlower.get_stop_name(TensorLevelNum(idx->level)), stop_value));


        lExprTuple p = {get_iter(TensorLevelNum(idx->level), tlower), get_stop(TensorLevelNum(idx->level), tlower), get_seg_end(TensorLevelNum(idx->level), tlower)};
        llir::lExpr offset_write_sub_cond = std::get<2>(p) < std::get<1>(p);
        offset_write_cond = offset_write_cond.defined() ? offset_write_cond || offset_write_sub_cond : offset_write_sub_cond;
        imap[i] = std::move(p);
    }

    // Append case optimization: When reduction is there on one of the innermost loop. Can optimize by not doing an accumulate add directly
    // instead write to a temporary variable and atomic add that once in a loop that is present in the result.
    if(next_sparse_intersection.get() == forall_list.size() && reduction_loops.size() > 0 && !is_scatter_reduction && reduction_loops.back() + 1 == output_write_tensor.end_loop_num() && !is_precompute) {
        LoopNum innermost_non_reduce_loop = BEFORE_FIRST_LOOP;
        for(auto it = output_write_tensor.end_loop_num()-1; it != BEFORE_FIRST_LOOP; --it) {
            if(std::find(reduction_loops.begin(), reduction_loops.end(), it) == reduction_loops.end()) {
                innermost_non_reduce_loop = it;
                break;
            }
        }
        if(loop_num == innermost_non_reduce_loop + 1) {
            stmts.push_back(llir::Declare::make(
                llir::Generic_t::make("value_t"), "value",
                llir::lConst::make((int64_t)0)
            ));
        }
    }

    auto make_loop = [&](const Seq &s) {
        auto [is, _, has_universe_iter] = partition_iterators_locators(s);

        if(is_loop_before_prev_intersection) {
             internal_assert(result_tensor.tensor_level_exists(loop_num)) << "Loop num " << loop_num.get() << " does not exist in result tensor";
            // If this loop is before the previous intersection, then we are going to iterate only over the result tensor index
            is = std::vector<Seq>{result_tensor.get_index_sequence(result_tensor.loop_num_to_tensor_level(loop_num))};
            has_universe_iter = false; // universe iterator is not needed for loops before the previous intersection since we are only iterating over the result tensor index
        }

        std::vector<lExprTuple> iters;
        iters.reserve(is.size() + (has_universe_iter ? 1 : 0));

        
        // the condition for offset write eg - Z.dim_j_offsets[offset_i + 1] = offset_j;

        llir::lExpr cond;
        if(has_universe_iter) {
            const auto & miter = imap.find(Empty::make(false));
            internal_assert(miter != imap.end()) << "Expected universe iterator to be present in imap";
            iters.push_back(miter->second);
            cond = std::get<0>(miter->second) <= std::get<1>(miter->second);
        }
        for (const auto &i : is) {
            const auto &miter = imap.find(i);
            internal_assert(miter != imap.end()) << i;
            iters.push_back(miter->second);
            
            if (cond.defined()) {
                cond = cond && std::get<0>(miter->second) <= std::get<1>(miter->second);
            } else {
                cond = std::get<0>(miter->second) <= std::get<1>(miter->second);
            }
        }

        internal_assert(cond.defined()) << s;

        auto get_body_epilogue_stmt = [&](const LoopNum loop_num) {
            // epilogue statement only required for loops >= previous_sparse_intersection_loops
            if(loop_num < previous_sparse_intersection) {
                return llir::lStmt();
            }

            if(!output_write_tensor.tensor_level_exists(loop_num)) {
                return llir::lStmt();
            }
            std::string idx = forall_list[loop_num.get()].as<Forall>()->idx;

            std::vector<llir::lStmt> stmts;

            auto current_level = output_write_tensor.loop_num_to_tensor_level(loop_num);
            auto next_level = TensorLevelNum(current_level.get() + 1);
            // If this is not the innermost loop need to wrap the increment statement under a condition
            // also, add the offset calculation statement here
            if(output_write_tensor.tensor_level_to_loop_num(next_level) != current_sparse_intersection+1 && output_write_tensor.is_sparse(next_level) && output_write_tensor.is_compressed(next_level)) {
                // offset calculation statement here
                // eg :- result.dim_j_offsets[offset_i + 1] = offset_j
                TensorLevelNum level = output_write_tensor.loop_num_to_tensor_level(loop_num+1);
                if(!is_precompute) {
                        stmts.push_back(llir::Store::make(
                        output_write_tensor.get_offsets_field(next_level)[
                            output_write_tensor.get_level_indexing_expression(
                                level, true, 
                                get_iter_vars_result(
                                    output_write_tensor,
                                    level)
                                )],
                        llir::lVar::make(index_t, "offset_" + output_write_tensor.tensor_level_name(next_level))));
                }
            }

            if(loop_num > previous_sparse_intersection && !is_precompute) {
                auto offset_var = llir::lVar::make(index_t, "offset_" + idx);
                if(output_write_tensor.is_sparse(idx) && output_write_tensor.is_unique(idx)) {
                    stmts.push_back(llir::Store::make(
                        output_write_tensor.get_indices_field(idx)[offset_var],
                        llir::lVar::make(index_t, idx)));

                    // also add the index assign statement for any non-unique sparse levels above this level
                    for(TensorLevelNum level  = current_level -1; level > BEFORE_FIRST_LEVEL ; --level) {
                        if(output_write_tensor.is_sparse(level) && !output_write_tensor.is_unique(level)) {
                            std::string level_idx = forall_list[output_write_tensor.tensor_level_to_loop_num(level).get()].as<Forall>()->idx;
                            auto offset_var = llir::lVar::make(index_t, "offset_" + level_idx);
                            stmts.push_back(llir::Store::make(
                                output_write_tensor.get_indices_field(output_write_tensor.tensor_level_name(level))[offset_var],
                                llir::lVar::make(index_t, level_idx)));
                        } else {
                            break;
                        }
                    }
                }
            }

            if (output_write_tensor.is_sparse(idx) && output_write_tensor.is_unique(idx)) {
                // Count this loop iteration.
                stmts.push_back(llir::BaseExpr::make(
                    llir::lIncrement::make(llir::lVar::make(
                        index_t,
                        (is_precompute ? "count_" : "offset_") + idx))));
                // also increment the offset for any non-unique sparse levels above this level
                for(TensorLevelNum level  = current_level -1; level > BEFORE_FIRST_LEVEL ; --level) {
                    if(output_write_tensor.is_sparse(level) && !output_write_tensor.is_unique(level)) {
                        std::string level_idx = forall_list[output_write_tensor.tensor_level_to_loop_num(level).get()].as<Forall>()->idx;
                        stmts.push_back(llir::BaseExpr::make(
                            llir::lIncrement::make(llir::lVar::make(
                                index_t,
                                (is_precompute ? "count_" : "offset_") + level_idx))));
                    } else {
                        break;
                    }
                }
            }


            if(stmts.size()>0) {
                return output_write_tensor.tensor_level_to_loop_num(next_level) != current_sparse_intersection+1 ? llir::IfElse::make(offset_write_cond, llir::Sequence::make(stmts), nullptr) : llir::Sequence::make(stmts);
            } else {
                return llir::lStmt();
            }
        };

        
        auto make_assign_operand_pos_map = [&](llir::lExpr index, std::set<Seq, SeqLessThan>& as) {
            // only assign for new loops, indexes for loops <= previous_sparse_intersection has already been calculated
            if(loop_num <= previous_sparse_intersection) {
                return llir::lStmt();
            }

            std::vector<llir::lStmt> stmts;
            auto offset_var = llir::lVar::make(index_t, "offset_" + forall->idx);
            if (result_tensor.tensor_level_exists(forall->idx)){
                // need to populate result_to_op_map for when this is not the innermost sparse intersection
                if(next_sparse_intersection!= LoopNum(forall_list.size())) {
                    const Forall * current_loop = forall_list[loop_num.get()].as<Forall>();
                    for(auto it: operand_tensors) {
                        if(exists_field_in_result_to_operand_pos_map(current_loop, it.second)){
                            internal_assert(it.second.tensor_level_exists(loop_num)) << "Tensor level does not exist" << loop_num << " in tensor " << it.second.tensor_name;
                            auto level = it.second.loop_num_to_tensor_level(loop_num);
                            Seq temp = it.second.get_index_sequence(level);
                            if (as.count(temp) != 0) {
                                stmts.push_back(
                                    llir::Store::make(
                                        map_result_pos_to_operand_pos(current_loop, it.second, offset_var, index),
                                        llir::lVar::make(index_t, it.second.get_iter_name(level))
                                    )
                                );
                            } else {
                                // this tensor is not defined currently so store -1
                                stmts.push_back(
                                    llir::Store::make(
                                        map_result_pos_to_operand_pos(current_loop, it.second, offset_var, index),
                                        llir::lConst::make(-1)
                                    )
                                );
                            }
                        }
                    }
                }
            } else {
                //internal_assert(false) << "TODO: Handle reduction case";
                return llir::lStmt();
            }
            if(stmts.size() == 0) {
                return llir::lStmt();
            } else {
                return llir::Sequence::make(std::move(stmts));
            }
        };

        auto make_body = [&](const Seq &a) {
            // Do NOT change this to partition_iterators_locators, body
            // needs all defined.
            auto as = indexes(a);
            auto new_def = defined;
            for (const auto &term : as) {
                new_def.insert(term);
            }
            CIN cin = simplify(new_def, forall->body);
            if (!cin.defined()) {
                for (const auto &d : new_def) {
                    std::cout << "def: " << d << "\n";
                }
            }

            internal_assert(cin.defined()) << (CIN)forall;


            llir::lStmt body;
            if (cin.as<Forall>()) {
                body = lower_loop(cin, new_def, is_precompute, loop_num + 1, offset_write_cond);
            } else {
                body = lower_assign_statement(cin, is_precompute);
            }

            // Store the index value in the result
            llir::lStmt assign_operand_pos_map_stmt;
        
            if(!is_precompute && previous_sparse_intersection < loop_num){
                assign_operand_pos_map_stmt = make_assign_operand_pos_map(llir::lVar::make(index_t, forall->idx), new_def);
            }

            if(assign_operand_pos_map_stmt.defined()){
                body =  body.defined() ? llir::Sequence::make(
                    {assign_operand_pos_map_stmt, std::move(body)}) : std::move(assign_operand_pos_map_stmt);
            }

            auto epilogue_stmt = get_body_epilogue_stmt(loop_num);
            if (body.defined() && epilogue_stmt.defined()) {
                body = llir::Sequence::make({std::move(body), std::move(epilogue_stmt)});
            } else if (epilogue_stmt.defined()) {
                body = std::move(epilogue_stmt);
            }

            return body;
        };

        // required only for multiple iterators case
        auto make_if_else_condition = [&](const Seq &a) {
            // auto as = indexes(a);
            auto [as, _, has_universe_iter] = partition_iterators_locators(a);
            // only dense or universe left. Condition is always true
            // as coord will always exist
            if(as.size() == 0) {
                return llir::lConst::make(true);
            }
            llir::lExpr cond;
            llir::lExpr var = llir::lVar::make(index_t, forall->idx);
            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                llir::lExpr idx_expr = llir::lVar::make(
                    index_t, get_tensor(idx->tensor)
                                 .get_coord_name(TensorLevelNum(idx->level)));
                llir::lExpr sub = var == idx_expr;

                if (cond.defined()) {
                    cond = cond && sub;
                } else {
                    cond = sub;
                }
            }
            internal_assert(cond.defined()) << a;
            return cond;
        };

        auto make_evals = [&](const Seq &a) {
            std::vector<llir::lStmt> stmts;
            // auto as = indexes(a);
            //std::cout<<"make evals : "<<a<<std::endl;
            auto [as, ls, has_universe_iter] = partition_iterators_locators(a);
            if(is_loop_before_prev_intersection) {
                auto level = result_tensor.loop_num_to_tensor_level(loop_num);
                stmts.push_back(llir::IfElse::make(
                    result_tensor.get_iter(level) >= (result_tensor.is_sparse(level)? result_tensor.get_length_field(level) : result_tensor.get_size_field(level)),
                    llir::Break::make(),
                    nullptr
                ));

                for(const auto &term: as) {
                     const Index *idx = term.as<Index>();
                    internal_assert(idx) << term;
                    TensorLowerer tlower = get_tensor(idx->tensor);
                    std::map<TensorLevelNum, llir::lExpr> pos_vars;
                    for(TensorLevelNum level = BEFORE_FIRST_LEVEL + 1; level < TensorLevelNum(idx->level); ++level) {
                        pos_vars[level] = tlower.get_iter(level);
                    }
                    stmts.push_back(
                        llir::Declare::make(
                            index_t,
                            tlower.get_iter_name(TensorLevelNum(idx->level)),
                            llir::lSelect::make(
                                map_result_pos_to_operand_pos(forall, tlower,result_tensor.get_iter(TensorLevelNum(idx->level))) != llir::lConst::make(-1),
                                map_result_pos_to_operand_pos(forall, tlower,result_tensor.get_iter(TensorLevelNum(idx->level))),
                                tlower.get_bound(TensorLevelNum(idx->level), true, pos_vars) + 1
                            )
                        )
                    );
                }
            }

            llir::lExpr value;
            bool has_dense_iter = false;
            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                TensorLowerer tlower = get_tensor(idx->tensor);
                auto level = TensorLevelNum(idx->level);
                stmts.push_back(tlower.make_eval(level, index_t, loop_num == current_sparse_intersection));
                if (!value.defined()) {
                    value = tlower.get_coord_var(level, index_t);
                } else {
                    value = llir::lBinOp::make(
                        llir::lBinOp::Min,
                        tlower.get_coord_var(level, index_t),
                        std::move(value));
                }
                has_dense_iter = has_dense_iter || !tlower.is_sparse(TensorLevelNum(idx->level));
            }
            internal_assert(value.defined() == !as.empty()) << a;

            if (has_universe_iter){
                internal_assert(imap.find(Empty::make(false)) != imap.end()) << "Expected universe iterator to be present in imap";
                value = std::get<0>(imap.find(Empty::make(false))->second);
            }

            // not iterating over the universe, evaluate index in universe.
            stmts.push_back(llir::Declare::make(index_t, forall->idx,
                                                std::move(value)));
            
            value = llir::lVar::make(index_t, forall->idx);

            if(loop_num<current_sparse_intersection) {
                auto it = std::find_if(operand_tensors.begin(), operand_tensors.end(), [&](const auto& op_tensor) {
                        return op_tensor.second.tensor_level_exists(loop_num);
                    });
                internal_assert(it!=operand_tensors.end());
                stmts.push_back(llir::IfElse::make(
                    value==it->second.get_size_field(forall->idx),
                    llir::Break::make(),
                    nullptr
                ));
            }

            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                TensorLowerer tlower = get_tensor(idx->tensor);
                auto level = TensorLevelNum(idx->level);
                if(tlower.is_sparse(level) && !tlower.is_unique(level))
                    stmts.push_back(tlower.make_seg_end(level, value));
            }

            // Add locators.
            if(!has_universe_iter && !ls.empty() && !has_dense_iter) {
                const Index *idx = ls[0].as<Index>();
                internal_assert(idx && !idx->is_sparse) << ls[0];
                TensorLowerer tlower = get_tensor(idx->tensor);
                // This is necesary for reads
                // later. Hopefully, copy propagation is good on this generated
                // code.
                stmts.push_back(llir::Declare::make(
                    index_t, tlower.get_iter_name(TensorLevelNum(idx->level)), value));
            }
            return stmts;
        };

        auto make_incs = [&](const Seq &a, std::vector<llir::lStmt> &stmts) {
            // auto as = indexes(a);
            auto [as, _, has_universe_iter] = partition_iterators_locators(a);
            if(is_loop_before_prev_intersection) {
                internal_assert(result_tensor.tensor_level_exists(loop_num)) << "Loop num " << loop_num.get() << " does not exist in result tensor";
                // If this loop is before the previous intersection, then we are going to iterate only over the result tensor index
                as = std::vector<Seq>{result_tensor.get_index_sequence(result_tensor.loop_num_to_tensor_level(loop_num))};
                has_universe_iter = false; // universe iterator is not needed for loops before the previous intersection since we are only iterating over the result tensor index
            }
            // if its a single iterator always increment
            if(as.size()==1) {
                const Index *idx = as[0].as<Index>();
                internal_assert(idx) << as[0];
                TensorLowerer tlower = get_tensor(idx->tensor);
                if(tlower.is_sparse(TensorLevelNum(idx->level)) && !tlower.is_unique(TensorLevelNum(idx->level))) {
                    stmts.push_back(tlower.make_seg_inc(TensorLevelNum(idx->level), index_t));
                }
                stmts.push_back(llir::Accumulate::make(tlower.get_iter(TensorLevelNum(idx->level)) , llir::lConst::make(1)));
            } else {
                for (const auto &term : as) {
                    const Index *idx = term.as<Index>();
                    internal_assert(idx) << term;
                    TensorLowerer tlower = get_tensor(idx->tensor);
                    if(tlower.is_sparse(TensorLevelNum(idx->level)) && !tlower.is_unique(TensorLevelNum(idx->level))) {
                        stmts.push_back(tlower.make_seg_inc(TensorLevelNum(idx->level), index_t));
                    }
                    stmts.push_back(tlower.make_inc(TensorLevelNum(idx->level), index_t));
                }
            }
            if(has_universe_iter) {
                stmts.push_back(llir::Accumulate::make(llir::lVar::make(index_t, "iter_"+forall->idx), llir::lConst::make(1)));
            }
            // TODO: what if as is empty??
        };
        
        if ((iters.size() == 1)) {
            // For loop (dense)!
            static const llir::lExpr _1 = llir::lConst::make(1);
            const llir::lVar *start = std::get<0>(iters[0]).as<llir::lVar>();

            // Evaluate (prologue)
            auto stmts = make_evals(s);

            llir::lStmt body = make_body(s);

            if (const auto *as_seq = body.as<llir::Sequence>()) {
                stmts.insert(stmts.end(), as_seq->stmts.begin(),
                            as_seq->stmts.end());
            } else {
                stmts.push_back(std::move(body));
            }

            make_incs(s, stmts);

            body = llir::Sequence::make(std::move(stmts));
            

            // TODO: load index if sparse iterator
            return llir::For::make(start->name, std::move(cond), llir::lExpr(),
                            std::move(body));
        } else {
            internal_assert(iters.size() > 1) << "Expected multiple iterators";
            // While loop
            auto cases = lattice.sub_points(s);

            std::function<llir::lStmt(size_t)> build_ifelse =
                [&](size_t count) -> llir::lStmt {
                if (count >= cases.size()) {
                    return llir::lStmt();
                }
                llir::lStmt b = make_body(cases[count]);
                auto [iters, _, has_universe_iter] = partition_iterators_locators(cases[count]);
                if (iters.empty() && has_universe_iter) {
                    // Iterating only over the universe
                    return b;
                }
                if(b.defined()) {
                    auto ifelse = llir::IfElse::make(make_if_else_condition(cases[count]), b,
                                        build_ifelse(count + 1));
                    return ifelse;
                } else {
                    return build_ifelse(count + 1);
                }
            };

            // Prologue (evaluate)
            auto stmts = make_evals(s);

            auto if_else = build_ifelse(0); 
            if(!if_else.defined()) {
                return llir::lStmt();
            }
            // Execute
            stmts.push_back(if_else);
            // Epilogue (incrments!)
            make_incs(s, stmts);

            auto body = llir::Sequence::make(std::move(stmts));

            return llir::While::make(std::move(cond), std::move(body));
        }
    };
    if(is_loop_before_prev_intersection ) {
        stmts.push_back(make_loop(forall->seq));
    } else {
        const auto tsort = lattice.topological_order();

        for (const auto &s : tsort) {
            // std::cout << "Building loop for: " << s << std::endl;
            auto loop = make_loop(s);
            if(loop.defined()) {
                stmts.push_back(std::move(loop));
            }
        }
    }
    // Append case optimization: When reduction is there on one of the innermost loop. Can optimize by not doing an accumulate add directly
    // instead write to a temporary variable and atomic add that once in a loop that is present in the result.
    if(next_sparse_intersection.get() == forall_list.size() && reduction_loops.size() > 0 && !is_scatter_reduction && reduction_loops.back() + 1 == output_write_tensor.end_loop_num() && !is_precompute) {
        LoopNum innermost_non_reduce_loop=BEFORE_FIRST_LOOP;
        for(auto it = output_write_tensor.end_loop_num()-1; it != BEFORE_FIRST_LOOP; --it) {
            if(std::find(reduction_loops.begin(), reduction_loops.end(), it) == reduction_loops.end()) {
                innermost_non_reduce_loop = it;
                break;
            }
        }
        if(loop_num == innermost_non_reduce_loop+1) {
                // need to add the final store for reduction variable here since reduction variable is not stored in the loop body when there are sparse intersections
            stmts.push_back(
                llir::BaseExpr::make(
                llir::lFunctionCall::make(
                    "atomicAdd",
                    {
                        llir::lAddress::make(output_write_tensor.get_values_field()[
                            output_write_tensor.get_level_indexing_expression(
                                output_write_tensor.end_tensor_level(),
                                false,
                                get_iter_vars_result(
                                    output_write_tensor,
                                    output_write_tensor.end_tensor_level()
                                )
                            )]),
                        llir::lVar::make(value_t, "value")
                    }
                )
            ));
        }
    }
    if (!stmts.empty()) {
        return llir::Sequence::make(std::move(stmts));
    }
    return llir::lStmt();
}

} // namespace backend

} // namespace nacho
