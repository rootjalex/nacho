#include "backend/compute.h"
#include "IRFwdDecl.h"
#include "Simplify.h"
#include "llir/Function.h"
#include "llir/LLIR.h"

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
        if(!tlow.is_sparse(TensorLevelNum(idx->level))) {
            return tlow.get_length_field(TensorLevelNum(idx->level)) - 1;
        }
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
            auto [iters, locs] = partition_iterators_locators(forall->seq);

            if (iters.empty()) {
                // Iterating over the universe!
                // Need to get the maximum of the universe from one of the locators.
                internal_assert(!locs.empty()) << forall->seq;
                const Index *idx = locs[0].as<Index>();
                internal_assert(idx) << locs[0];
                // Use `size` for dense, `length` for sparse.
                internal_assert(!idx->is_sparse) << locs[0];
                llir::lExpr max_iter = get_max_iterator(idx);
                auto tlow = get_tensor(idx->tensor);
                llir::lExpr iter_var =
                    add_single_partition_load(tlow, TensorLevelNum(idx->level), max_iter);
                iter_vars.emplace_back(std::move(iter_var));
            } else {
                // For each iterator, construct
                for (const auto &iter : iters) {
                    const Index *idx = iter.as<Index>();
                    internal_assert(idx) << iter;
                    TensorLowerer tlow = get_tensor(idx->tensor);
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
        auto index = result_tensor.loop_name(i);
        if (result_tensor.is_sparse(i)) {
            stmts.emplace_back(llir::Declare::make(
                index_t, "count_" + index, llir::lConst::make((int64_t)0)));
        }
    }

    std::set<Seq, SeqLessThan> defined; // initially empty
    internal_assert(cin.defined());
    stmts.push_back(lower_loop(cin, defined, /*is_precompute*/ true, BEFORE_FIRST_LOOP+1));

    for (LoopNum i = BEFORE_FIRST_LOOP+1; i <= current_sparse_intersection; ++i) {
        auto index = result_tensor.loop_name(i);
        if (result_tensor.is_sparse(i)) {
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
        auto index = result_tensor.loop_name(i);
        if (result_tensor.is_sparse(i)) {
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
    stmts.push_back(lower_loop(cin, defined, /*is_precompute*/ false, BEFORE_FIRST_LOOP+1));

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


    if(assign.as<Assign>()) {
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
                    for (TensorLevelNum level = BEFORE_FIRST_LEVEL+1; level <= end_level; ++level) {
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

        const Assign* assign_stmt = assign.as<Assign>();
        auto converter = Converter(operand_tensors);
        assign_stmt->expr.accept(&converter);
        return llir::Store::make(
            llir::lArrayAccess::make(
                llir::lFieldAccess::make(
                    llir::lVar::make(
                        llir::Generic_t::make(result_tensor.get_struct_name()),
                        result_tensor.tensor_name
                    ),
                    result_tensor.get_values_field_name()
                ),
                result_tensor.get_level_indexing_expression(
                    result_tensor.end_tensor_level(),
                    false,
                    get_iter_vars_result(
                        result_tensor,
                        result_tensor.end_tensor_level()
                    )
                )
            ),
            converter.result
        );

    } else if(assign.as<Accumulate>()) {
        internal_assert(false) << "TODO: Generation for Accumulate/Reductions not supported.";
    } else if(assign.as<CalculateWork>()) {
        llir::lExpr work_expr = llir::lConst::make((int64_t)0);
        auto get_work_expr = [&](TensorLowerer& Tensor) {
            std::vector<llir::lExpr> work_args;
            work_args.emplace_back(
                llir::lVar::make(llir::Generic_t::make(Tensor.get_struct_name()), Tensor.tensor_name)
            );
            for(LoopNum j=BEFORE_FIRST_LOOP+1;j<=current_sparse_intersection;++j) {
                if(Tensor.tensor_level_exists(j))
                    work_args.emplace_back(Tensor.get_iter_var(Tensor.loop_num_to_tensor_level(j), llir::Generic_t::make("index_t")));
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
                    false,
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
    bool is_precompute, LoopNum loop_num) {

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

    auto [forall_iters, _] =
        partition_iterators_locators(forall->seq);

    if(is_loop_before_prev_intersection) {
        internal_assert(result_tensor.tensor_level_exists(loop_num)) << "Yet to suport reductions above any sparse intersections";
        // If this loop is before the previous intersection, then we are going to iterate only over the result tensor index
        forall_iters = result_tensor.is_sparse(forall->idx) ? std::vector<Seq>{result_tensor.get_index_sequence(result_tensor.loop_num_to_tensor_level(loop_num))} : std::vector<Seq>{};
    }

    using lExprPair = std::pair<llir::lExpr, llir::lExpr>;
    std::map<Seq, lExprPair, SeqLessThan> imap;

    // Insert prologue (initiate iterators!).
    if (forall_iters.empty()) {
        // Iterating the universe!
        stmts.push_back(llir::Declare::make(
            index_t, forall->idx,
            llir::lVar::make(index_t, "start_" + forall->idx)));
    }


    for (const auto &i : forall_iters) {
        const Index *idx = i.as<Index>();
        internal_assert(idx) << i;

        TensorLowerer tlower = get_tensor(idx->tensor);

        auto get_start = [&](TensorLevelNum level) {
            if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
                // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
                return result_tensor.get_start(level);
            } else {
                return tlower.get_start(level);
            }
            
        };

        auto get_end = [&](TensorLevelNum level) {
            if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
                // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
                return result_tensor.get_end(level);
            } else {
                return tlower.get_end(level);
            }
        };

        auto get_iter = [&](TensorLevelNum level) {
            if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
                // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
                return result_tensor.get_iter(level);
            } else {
                return tlower.get_iter(level);
            }
        };

        auto get_stop = [&](TensorLevelNum level) {
            if(tlower.tensor_level_to_loop_num(level) <= previous_sparse_intersection) {
                // For levels before the previous intersection, we only iterate over the result tensor index, so the start is just the iterator variable of the result tensor.
                return result_tensor.get_stop(level);
            } else {
                return tlower.get_stop(level);
            }
        };

        llir::lExpr pidx = get_start(TensorLevelNum(idx->level));
        llir::lExpr pend = get_end(TensorLevelNum(idx->level));

        llir::lExpr start_value;
        llir::lExpr stop_value;
        if (idx->level == 0) {
            // Fully iterate the partition.
            start_value = pidx;
            stop_value = pend;
        } else {
            std::function<llir::lExpr(TensorLevelNum, bool)> get_condition =
                [&](TensorLevelNum level, bool end) -> llir::lExpr {
                internal_assert(level > BEFORE_FIRST_LEVEL) << level;
                
                llir::lExpr val = get_iter(level);
                llir::lExpr extrema = end ? get_end(level) : get_start(level);

                llir::lExpr cond = val == extrema;
                if (level == BEFORE_FIRST_LEVEL + 1) {
                    return cond;
                } else {
                    llir::lExpr rec = get_condition(level - 1, end);
                    return cond && rec;
                }
            };

            std::map<TensorLevelNum, llir::lExpr> pos_vars;
            for(TensorLevelNum level = BEFORE_FIRST_LEVEL + 1; level < TensorLevelNum(idx->level); ++level) {
                pos_vars[level] = tlower.get_iter(level);
            }

            // For the start value, if all previous iterators are at their
            // respective starts, then use this start, otherwise get the
            // iterator from the data structure!
            llir::lExpr start_cond = get_condition(TensorLevelNum(idx->level - 1), false);
            start_value = llir::lSelect::make(
                std::move(start_cond), pidx,
                tlower.get_bound(TensorLevelNum(idx->level), /*upper_bound=*/false, pos_vars));

            // For the stop, if all previous iterators are at their respective
            // ends, then use this stop, otherwise get the iterator from the
            // data structure!
            llir::lExpr stop_cond = get_condition(TensorLevelNum(idx->level - 1), true);
            llir::lExpr bound =
                tlower.get_bound(TensorLevelNum(idx->level), /*upper_bound=*/true, pos_vars);
            stop_value = llir::lSelect::make(std::move(stop_cond), pend,
                                             std::move(bound));
        }

        stmts.push_back(llir::Declare::make(
            index_t, tlower.get_iter_name(TensorLevelNum(idx->level)), start_value));
        // This is const.
        stmts.push_back(llir::Declare::make(
            index_t, tlower.get_stop_name(TensorLevelNum(idx->level)), stop_value));


        lExprPair p = {get_iter(TensorLevelNum(idx->level)), get_stop(TensorLevelNum(idx->level))};
        imap[i] = std::move(p);
    }

    // if(result_tensor.is_sparse(forall->idx) && result_tensor.tensor_type.format.get_level_order(forall->idx) > 0) {
    //     stmts.push_back(llir::Declare::make(
    //     llir::Generic_t::make("bool"), "atleast_one_iter_" + forall->idx, atleast_one_iter_cond));
    // }


    auto make_loop = [&](const Seq &s) {
        auto [is, _] = partition_iterators_locators(s);

        if(is_loop_before_prev_intersection) {
             internal_assert(result_tensor.tensor_level_exists(loop_num)) << "Yet to support reductions above any sparse intersections";
            // If this loop is before the previous intersection, then we are going to iterate only over the result tensor index
            is = result_tensor.is_sparse(forall->idx) ? std::vector<Seq>{result_tensor.get_index_sequence(result_tensor.loop_num_to_tensor_level(loop_num))} : std::vector<Seq>{};
        }

        std::vector<lExprPair> iters;
        iters.reserve(is.size());

        llir::lExpr cond;
        // the condition for offset write eg - Z.dim_j_offsets[offset_i + 1] = offset_j;
        llir::lExpr offset_write_cond;
        for (const auto &i : is) {
            const auto &miter = imap.find(i);
            internal_assert(miter != imap.end()) << i;
            iters.push_back(miter->second);
            if (cond.defined()) {
                cond = cond && miter->second.first <= miter->second.second;
                offset_write_cond = offset_write_cond && (miter->second.first != miter->second.second);
            } else {
                cond = miter->second.first <= miter->second.second;
                offset_write_cond = (miter->second.first != miter->second.second);
            }
        }
        if (is.empty()) {
            // Iterating over the universe.
            cond = llir::lVar::make(index_t, forall->idx) <=
                   llir::lVar::make(index_t, "end_" + forall->idx);
            offset_write_cond = llir::lVar::make(index_t, forall->idx) !=
                                llir::lVar::make(index_t, "end_" + forall->idx);
        }
        internal_assert(cond.defined()) << s;

        auto get_body_epilogue_stmt = [&](const CIN nextCin, Seq seq, const Forall *forall) {

            // epilogue statement only required for loops >= previous_sparse_intersection_loops
            if(loop_num < previous_sparse_intersection) {
                return llir::lStmt();
            }

            llir::lStmt stmt;

            // // Get the condition for the particular (tensor,level) this level iterates
            // // till the extrema i.e stop_var != end_var
            // auto get_condition_stop_eq_extrema = [&](const Index *idx) -> llir::lExpr {
            //         internal_assert(idx) << idx;
            //         TensorLowerer tlower(idx->tensor, idx->type);
            //         llir::lExpr stop_var = llir::lVar::make(
            //             index_t, tlower.get_stop_name(idx->level));
            //         llir::lExpr end_var = llir::lVar::make(
            //             index_t, tlower.get_end_name(idx->level));
            //         llir::lExpr cond = stop_var != end_var;
            //         return cond;
            // };

            if (result_tensor.is_sparse(forall->idx)) {
                // Count this loop iteration.
                stmt = llir::BaseExpr::make(
                    llir::lIncrement::make(llir::lVar::make(
                        index_t,
                        (is_precompute ? "count_" : "offset_") + forall->idx)));
            }

            auto nextForall = nextCin.as<Forall>();
            // If this is not the innermost loop need to wrap the increment statement under a condition
            // also, add the offset calculation statement here
            if(nextForall && result_tensor.is_sparse(nextForall->idx)) {
                // offset calculation statement here
                // eg :- result.dim_j_offsets[offset_i + 1] = offset_j
                TensorLevelNum level = result_tensor.loop_num_to_tensor_level(loop_num+1);
                llir::lStmt store_stmt;
                if(!is_precompute) {
                        store_stmt = llir::Store::make(
                        result_tensor.get_offsets_field(nextForall->idx)[
                            result_tensor.get_level_indexing_expression(
                                level, true, 
                                get_iter_vars_result(
                                    result_tensor,  
                                    level)
                                )],
                        llir::lVar::make(index_t, "offset_" + nextForall->idx));
                }

                // nextForall is >previous sparse intersect because of start condition
                auto [idxs, _] = partition_iterators_locators(forall->seq);
                internal_assert(idxs.size()>0) << forall->seq;
                // llir::lExpr cond;
            
                // for (int i = 0; i < idxs.size(); i++) {
                //     llir::lExpr idx_cond = get_condition_stop_eq_extrema(idxs[i].as<Index>());
                //     cond = i == 0 ? idx_cond : cond && idx_cond;
                // }

                offset_write_cond = offset_write_cond || (llir::lVar::make(index_t, "thread_id") == llir::lVar::make(index_t, "max_thread_id"));

                if(store_stmt.defined() && stmt.defined()) {
                    stmt = llir::IfElse::make(offset_write_cond, llir::Sequence::make({std::move(store_stmt),std::move(stmt)}), nullptr);
                } else if(store_stmt.defined()) {
                    stmt = llir::IfElse::make(offset_write_cond, std::move(store_stmt), nullptr);
                } else if(stmt.defined()) {
                    stmt = llir::IfElse::make(offset_write_cond, std::move(stmt), nullptr);
                }
            }
            return stmt;
        };

        
        auto make_assign_indices = [&](llir::lExpr index, std::set<Seq, SeqLessThan>& as) {
            // only assign for new loops, indexes for loops <= previous_sparse_intersection has already been calculated
            if(loop_num <= previous_sparse_intersection) {
                return llir::lStmt();
            }

            std::vector<llir::lStmt> stmts;
            auto offset_var = llir::lVar::make(index_t, "offset_" + forall->idx);
            if (result_tensor.tensor_level_exists(forall->idx)){
                if(result_tensor.is_sparse(forall->idx)) {
                    stmts.push_back(llir::Store::make(
                        result_tensor.get_indices_field(forall->idx)[offset_var],
                        index));
                }

                // need to populate result_to_op_map for when this is not the innermost sparse intersection
                if(next_sparse_intersection!= LoopNum(forall_list.size())) {
                    const Forall * current_sparse_intersection_loop = forall_list[current_sparse_intersection.get()].as<Forall>();
                    for(auto it: operand_tensors) {
                        if(exists_field_in_result_to_operand_pos_map(current_sparse_intersection_loop, it.second)){
                            internal_assert(it.second.tensor_level_exists(loop_num)) << "Tensor level does not exist";
                            auto level = it.second.loop_num_to_tensor_level(loop_num);
                            Seq temp = it.second.get_index_sequence(level);
                            if (as.count(temp) != 0) {
                                stmts.push_back(
                                    llir::Store::make(
                                        map_result_pos_to_operand_pos(current_sparse_intersection_loop, it.second, offset_var, index),
                                        llir::lVar::make(index_t, it.second.get_iter_name(level))
                                    )
                                );
                            } else {
                                // this tensor is not defined currently so store -1
                                stmts.push_back(
                                    llir::Store::make(
                                        map_result_pos_to_operand_pos(current_sparse_intersection_loop, it.second, offset_var, index),
                                        llir::lConst::make(-1)
                                    )
                                );
                            }
                        }
                    }
                }
            } else {
                internal_assert(false) << "TODO: Handle reduction case";
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
            internal_assert(cin.defined()) << forall->body;


            llir::lStmt body;
            if (cin.as<Forall>()) {
                body = lower_loop(cin, new_def, is_precompute, loop_num + 1);
            } else {
                body = lower_assign_statement(cin, is_precompute);
            }

            // Store the index value in the result
            llir::lStmt assign_indices_stmt;
        
            if(!is_precompute && previous_sparse_intersection < loop_num){
                assign_indices_stmt = make_assign_indices(llir::lVar::make(index_t, forall->idx), new_def);
            }

            if(assign_indices_stmt.defined()){
                body = llir::Sequence::make(
                    {assign_indices_stmt, std::move(body)});
            }

            auto epilogue_stmt = get_body_epilogue_stmt(cin, a, forall);
            assert(body.defined() || epilogue_stmt.defined());
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
            auto [as, _] = partition_iterators_locators(a);
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
            auto [as, ls] = partition_iterators_locators(a);
            if(is_loop_before_prev_intersection) {
                for(const auto &term: as) {
                     const Index *idx = term.as<Index>();
                    internal_assert(idx) << term;
                    TensorLowerer tlower = get_tensor(idx->tensor);
                    stmts.push_back(
                        llir::Declare::make(
                            index_t,
                            tlower.get_iter_name(TensorLevelNum(idx->level)),
                            llir::lSelect::make(
                                map_result_pos_to_operand_pos(forall, tlower,result_tensor.get_iter(TensorLevelNum(idx->level))) != llir::lConst::make(-1),
                                map_result_pos_to_operand_pos(forall, tlower,result_tensor.get_iter(TensorLevelNum(idx->level))),
                                tlower.get_size_field(TensorLevelNum(idx->level))
                            )
                        )
                    );
                }
            }
            // TODO: different eval for locators!!
            llir::lExpr value;
            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                TensorLowerer tlower = get_tensor(idx->tensor);
                stmts.push_back(tlower.make_eval(TensorLevelNum(idx->level), index_t, is_loop_before_prev_intersection));
                if (!value.defined()) {
                    value = tlower.get_coord_var(TensorLevelNum(idx->level), index_t);
                } else {
                    value = llir::lBinOp::make(
                        llir::lBinOp::Min,
                        tlower.get_coord_var(TensorLevelNum(idx->level), index_t),
                        std::move(value));
                }
            }
            internal_assert(value.defined() == !as.empty()) << a;

            if (!as.empty()) {
                // not iterating over the universe, evaluate index in universe.
                stmts.push_back(llir::Declare::make(index_t, forall->idx,
                                                    std::move(value)));
            }

            // Add locators.
            value = llir::lVar::make(index_t, forall->idx);
            for (const auto &loc : ls) {
                const Index *idx = loc.as<Index>();
                internal_assert(idx && !idx->is_sparse) << loc;
                TensorLowerer tlower = get_tensor(idx->tensor);
                // Iterators are the dense thing, this is necesary for reads
                // later. Hopefully, copy propagation is good on this generated
                // code.
                stmts.push_back(llir::Declare::make(
                    index_t, tlower.get_iter_name(TensorLevelNum(idx->level)), value));
            }

            return stmts;
        };

        auto make_incs = [&](const Seq &a, std::vector<llir::lStmt> &stmts) {
            // auto as = indexes(a);
            auto [as, _] = partition_iterators_locators(a);

            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                TensorLowerer tlower = get_tensor(idx->tensor);
                stmts.push_back(tlower.make_inc(TensorLevelNum(idx->level), index_t));
            }
            // TODO: what if as is empty??
        };
        
        if(is_loop_before_prev_intersection) {
            static const llir::lExpr _1 = llir::lConst::make(1);
            std::string name = result_tensor.is_sparse(forall->idx) ? iters[0].first.as<llir::lVar>()->name : forall->idx;
            // Evaluate (prologue)
            auto stmts = make_evals(s);

            llir::lStmt body = make_body(s);

            if (const auto *as_seq = body.as<llir::Sequence>()) {
                stmts.insert(stmts.end(), as_seq->stmts.begin(),
                            as_seq->stmts.end());
            } else {
                stmts.push_back(std::move(body));
            }

            body = llir::Sequence::make(std::move(stmts));

            // TODO: load index if sparse iterator
            return llir::For::make(name, std::move(cond), _1,
                                std::move(body));

        } else {
            if (iters.size() == 0) {
                // For loop (dense)!
                static const llir::lExpr _1 = llir::lConst::make(1);

                // Evaluate (prologue)
                auto stmts = make_evals(s);

                llir::lStmt body = make_body(s);

                if (const auto *as_seq = body.as<llir::Sequence>()) {
                    stmts.insert(stmts.end(), as_seq->stmts.begin(),
                                as_seq->stmts.end());
                } else {
                    stmts.push_back(std::move(body));
                }

                body = llir::Sequence::make(std::move(stmts));

                // TODO: load index if sparse iterator
                return llir::For::make(forall->idx, std::move(cond), _1,
                                    std::move(body));
            } else if (iters.size() == 1) {
                // internal_error << "TODO: handle sparse for loop optimization";
                // For loop (sparse)!
                const llir::lVar *start = iters[0].first.as<llir::lVar>();
                
                static const llir::lExpr _1 = llir::lConst::make(1);

                auto index =  is[0].as<Index>();
                if(!index) {
                    internal_assert(false) << "Expected Index: " << is[0];
                }
                TensorLowerer tlower = get_tensor(index->tensor);

                llir::lStmt body; 
                
                // Store the index value in the result
                if(!is_precompute) {
                    body = llir::Declare::make(index_t, forall->idx, tlower.get_coord(TensorLevelNum(index->level), index_t));
                }

                if( body.defined()) {
                    body = llir::Sequence::make({std::move(body), make_body(s)});
                } else {
                    body = make_body(s);
                }
                
                // TODO: load index if sparse iterator
                return llir::For::make(start->name, std::move(cond), _1,
                                    std::move(body));
            } else {
                // While loop
                auto cases = lattice.sub_points(s);

                std::function<llir::lStmt(size_t)> build_ifelse =
                    [&](size_t count) -> llir::lStmt {
                    if (count >= cases.size()) {
                        return llir::lStmt();
                    }
                    llir::lStmt b = make_body(cases[count]);
                    internal_assert(b.defined());
                    if (cases.size() == 1 &&
                        std::get<0>(partition_iterators_locators(cases[0]))
                            .empty()) {
                        // Iterating over the universe
                        return b;
                    }
                    auto ifelse = llir::IfElse::make(make_if_else_condition(cases[count]), b,
                                            build_ifelse(count + 1));
                    return ifelse;
                };

                // Prologue (evaluate)
                auto stmts = make_evals(s);

                // Execute
                stmts.push_back(build_ifelse(0));
                // Epilogue (incrments!)
                make_incs(s, stmts);

                auto body = llir::Sequence::make(std::move(stmts));

                return llir::While::make(std::move(cond), std::move(body));
            }
        }
    };
    if(is_loop_before_prev_intersection ) {
        stmts.push_back(make_loop(forall->seq));
    } else {
        const auto tsort = lattice.topological_order();

        for (const auto &s : tsort) {
            stmts.push_back(make_loop(s));
        }
    }

    return llir::Sequence::make(std::move(stmts));
}

} // namespace backend

} // namespace nacho
