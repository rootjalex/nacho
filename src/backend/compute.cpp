#include "backend/compute.h"
#include "IRFwdDecl.h"
#include "Simplify.h"
#include "llir/Function.h"
#include "llir/LLIR.h"

namespace nacho {
namespace backend {

llir::lType ComputeFunctionLowerer::lower_result_per_thread_count_struct() {
    llir::lType index_t = llir::Generic_t::make("index_t");
    std::vector<std::string> generics = {"index_t"};
    std::vector<std::pair<std::string, llir::lType>> fields;

    for (int i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
        auto index = result_tensor.tensor_type.format.levels[i].index;
        if (is_sparse_format(
                result_tensor.tensor_type.format.lvlfmt_of(index))) {
            fields.emplace_back(get_counts_field_name(index),
                                llir::Ptr_t::make(index_t));
        }
    }

    return llir::Struct_t::make(get_counts_struct_name(), std::move(fields),
                                std::move(generics));
}

void ComputeFunctionLowerer::add_partition_assignments(
    std::vector<llir::lStmt> &stmts) {
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const llir::lType i32 = llir::Int_t::make(32);
    stmts.emplace_back(llir::Declare::make(
        i32, "thread_id",
        llir::lBinOp::make(
            llir::lBinOp::Add,
            llir::lBinOp::make(llir::lBinOp::Mul,
                               llir::lVar::make(index_t, "blockIdx.x"),
                               llir::lVar::make(index_t, "blockDim.x")),
            llir::lVar::make(index_t, "threadIdx.x"))));

    llir::lExpr thread_id_var = llir::lVar::make(i32, "thread_id");

    // TODO: make this const!
    stmts.emplace_back(
        llir::Declare::make(i32, "max_thread_id",
                            llir::lVar::make(index_t, "gridDim.x") *
                                    llir::lVar::make(index_t, "blockDim.x") -
                                llir::lConst::make(1)));

    llir::lExpr max_thread_id_var = llir::lVar::make(i32, "max_thread_id");

    CIN loop = cin;

    const std::string partitions_name = get_partition_struct_name();
    llir::lExpr partitions_var = llir::lVar::make(
        llir::Generic_t::make(get_partition_struct_name() + "<index_t>"),
        partitions_name);

    auto add_single_partition_load = [&](const std::string &suffix,
                                         llir::lExpr maximum_iterator) {
        // TODO: make const!
        // index_t start_i = partitions.i[thread_id];
        llir::lExpr partitions_field =
            llir::lFieldAccess::make(partitions_var, suffix);
        stmts.emplace_back(llir::Declare::make(
            index_t, get_start_name(suffix),
            llir::lArrayAccess::make(partitions_field, thread_id_var)));
        // index_t end_i = (thread_id < max_thread_id) ? partitions.i[thread_id
        // + 1] : iterator_max;
        llir::lExpr next_partition = llir::lArrayAccess::make(
            partitions_field, thread_id_var + llir::lConst::make(1));
        llir::lExpr guarded_end =
            llir::lSelect::make(thread_id_var < max_thread_id_var,
                                next_partition, maximum_iterator);
        stmts.emplace_back(
            llir::Declare::make(index_t, get_end_name(suffix), guarded_end));
        return llir::lVar::make(index_t, get_start_name(suffix));
    };

    auto print_list = [](std::ostream &os, const std::vector<Seq> &seqs) {
        bool first = true;
        os << "{";
        for (const auto &s : seqs) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << s;
        }
        os << "}";
    };

    auto get_max_iterator = [&](const Index *idx) {
        TensorLowerer tlow(idx->tensor, idx->type);
        // TODO: fix this type hack.
        llir::lExpr tensor = llir::lVar::make(
            llir::Generic_t::make(tlow.get_struct_name()), tlow.tensor_name);
        std::string field_name = idx->is_sparse
                                     ? tlow.get_length_field_name(idx->level)
                                     : tlow.get_size_field_name(idx->level);
        return llir::lFieldAccess::make(tensor, field_name) -
               llir::lConst::make(1);
    };

    while (const auto *forall = loop.as<Forall>()) {
        auto [iters, locs] = partition_iterators_locators(forall->seq);

        std::vector<llir::lExpr> iter_vars;

        if (iters.empty()) {
            // Iterating over the universe!
            // Need to get the maximum of the universe from one of the locators.
            internal_assert(!locs.empty()) << forall->seq;
            const Index *idx = locs[0].as<Index>();
            internal_assert(idx) << locs[0];
            // Use `size` for dense, `length` for sparse.
            internal_assert(!idx->is_sparse) << locs[0];
            llir::lExpr max_iter = get_max_iterator(idx);
            llir::lExpr iter_var =
                add_single_partition_load(forall->idx, max_iter);
            iter_vars.emplace_back(std::move(iter_var));
        } else {
            // For each iterator, construct
            for (const auto &iter : iters) {
                const Index *idx = iter.as<Index>();
                internal_assert(idx) << iter;
                TensorLowerer tlow(idx->tensor, idx->type);
                llir::lExpr max_iter = get_max_iterator(idx);
                std::string suffix = tlow.get_iterator_suffix(idx->level);
                llir::lExpr iter_var =
                    add_single_partition_load(suffix, max_iter);
                iter_vars.emplace_back(std::move(iter_var));
            }
        }

        loop = forall->body;

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

llir::lStmt ComputeFunctionLowerer::
    lower_precompute_function_for_innermost_sparse_intersection() {
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
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

    std::vector<llir::lStmt> stmts;

    // Add common initialization statements
    add_partition_assignments(stmts);

    // index_t count = thread_id * per_thread_work;
    stmts.emplace_back(llir::Declare::make(
        llir::Generic_t::make("index_t"), "count",
        llir::lBinOp::make(llir::lBinOp::Mul,
                           llir::lVar::make(index_t, "thread_id"),
                           llir::lVar::make(index_t, "per_thread_work"))));

    // Declare local count variables to be used to calculat the offsets into
    // result. These will be stored in count_offsets at the end.
    for (int i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
        auto index = result_tensor.tensor_type.format.levels[i].index;
        if (is_sparse_format(
                result_tensor.tensor_type.format.lvlfmt_of(index))) {
            stmts.emplace_back(llir::Declare::make(
                index_t, "count_" + index, llir::lConst::make((int64_t)0)));
        }
    }

    std::set<Seq, SeqLessThan> defined; // initially empty
    internal_assert(cin.defined());
    stmts.push_back(lower_loop(cin, defined, /*is_precompute*/ true));

    for (int i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
        auto index = result_tensor.tensor_type.format.levels[i].index;
        if (is_sparse_format(
                result_tensor.tensor_type.format.lvlfmt_of(index))) {
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

llir::lStmt ComputeFunctionLowerer::
    lower_compute_function_for_innermost_sparse_intersection() {
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
        .type = llir::Ptr_t::make(llir::Generic_t::make(
            result_tensor.get_struct_name() + "<index_t, value_t>")),
        .name = "result"});

    std::vector<llir::lStmt> stmts;

    // Add common initialization statements
    add_partition_assignments(stmts);

    for (int i = 0; i < result_tensor.tensor_type.format.levels.size(); i++) {
        auto index = result_tensor.tensor_type.format.levels[i].index;
        if (is_sparse_format(
                result_tensor.tensor_type.format.lvlfmt_of(index))) {
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
    stmts.push_back(lower_loop(cin, defined, /*is_precompute*/ false));

    stmts.emplace_back(llir::Return::make());
    body = llir::Sequence::make(std::move(stmts));

    return llir::Function::make(std::move(generics), std::move(attributes),
                                std::move(args), std::move(ret_type), name,
                                std::move(body));
}


std::vector<llir::lExpr> get_iter_vars_result(TensorLowerer &tlower) {
        std::vector<llir::lExpr> iter_vars;
        for (int level = 0; level < tlower.tensor_type.format.levels.size(); level++) {
            auto level_info = tlower.tensor_type.format.levels[level];
            std::string iter_name = level_info.index;
            if(is_sparse_format(level_info.format)) {
                iter_name = "offset_" + level_info.index;
            }
            iter_vars.push_back(llir::lVar::make(
                llir::Generic_t::make("index_t"),
                iter_name
            ));
        }
        return iter_vars;
};

llir::lStmt ComputeFunctionLowerer::lower_assign_statement(
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
            auto get_iter_vars_operands = [&](TensorLowerer &tlower) {
                std::vector<llir::lExpr> iter_vars;
                for (int level = 0; level < tlower.tensor_type.format.levels.size(); level++) {
                    std::string iter_name = tlower.get_iter_name(level);
                    iter_vars.push_back(llir::lVar::make(
                        llir::Generic_t::make("index_t"),
                        iter_name
                    ));
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
                operand_tensors[node->name].get_offset_expression_for_next_sparse(
                    operand_tensors[node->name].tensor_type.format.get_last_sparse_level(),
                    operand_tensors[node->name].tensor_type.format.levels.size() - 1,
                    false, true,
                    get_iter_vars_operands(operand_tensors[node->name])
                )
            );
        };
    };

    if(assign.as<Assign>()) {
        const Assign* assign_stmt = assign.as<Assign>();
        auto converter = Converter(operand_tensors);
        assign_stmt->expr.accept(&converter);
        return llir::Store::make(
            llir::lArrayAccess::make(
                llir::lFieldAccess::make(
                    llir::lVar::make(
                        llir::Generic_t::make(result_tensor.get_struct_name()),
                        "result"
                    ),
                    result_tensor.get_values_field_name()
                ),
                result_tensor.get_offset_expression_for_next_sparse(
                    result_tensor.tensor_type.format.get_last_sparse_level(),
                    result_tensor.tensor_type.format.levels.size() - 1,
                    false, true,
                    get_iter_vars_result(result_tensor)
                )
            ),
            converter.result
        );

    } else if(assign.as<Accumulate>()) {
        internal_assert(false) << "TODO: Generation for Accumulate/Reductions not supported.";
    } else {
        internal_assert(false) << "Expected Assign or Accumulate in lower_assign_statement: " << assign;
    }

    return llir::lStmt();
}

llir::lStmt ComputeFunctionLowerer::lower_loop(
    CIN loop, const std::set<Seq, SeqLessThan> &defined, bool is_precompute) {

    const Forall *forall = loop.as<Forall>();
    internal_assert(forall) << "Expected Forall in lower_loop: " << loop;

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

    auto idxs = indexes(seq);

    using lExprPair = std::pair<llir::lExpr, llir::lExpr>;
    std::map<Seq, lExprPair, SeqLessThan> imap;

    // Insert prologue (initiate iterators!).
    for (const auto &i : idxs) {
        const Index *idx = i.as<Index>();
        internal_assert(idx) << i;

        TensorLowerer tlower(idx->tensor, idx->type);

        auto get_idx = [&](int level) {
            return llir::lVar::make(index_t, tlower.get_idx_name(level));
        };

        auto get_end = [&](int level) {
            return llir::lVar::make(index_t, tlower.get_end_name(level));
        };

        auto get_iter = [&](int level) {
            return llir::lVar::make(index_t, tlower.get_iter_name(level));
        };

        auto get_stop = [&](int level) {
            return llir::lVar::make(index_t, tlower.get_stop_name(level));
        };

        llir::lExpr pidx = get_idx(idx->level);
        llir::lExpr pend = get_end(idx->level);

        llir::lExpr start_value;
        llir::lExpr stop_value;
        if (idx->level == 0) {
            // Fully iterate the partition.
            start_value = pidx;
            stop_value = pend;
        } else {
            std::function<llir::lExpr(int, bool)> get_condition =
                [&](int level, bool end) -> llir::lExpr {
                internal_assert(level >= 0) << level;
                llir::lExpr val = get_iter(level);
                llir::lExpr extrema = end ? get_end(level) : get_idx(level);
                llir::lExpr cond = val == extrema;
                if (level == 0) {
                    return cond;
                } else {
                    llir::lExpr rec = get_condition(level - 1, end);
                    return cond && rec;
                }
            };
            // For the start value, if all previous iterators are at their
            // respective starts, then use this start, otherwise get the
            // iterator from the data structure!
            llir::lExpr start_cond = get_condition(idx->level - 1, false);
            start_value = llir::lSelect::make(
                std::move(start_cond), pidx,
                tlower.get_bound(idx->level, index_t, /*upper_bound=*/false));

            // For the stop, if all previous iterators are at their respective
            // ends, then use this stop, otherwise get the iterator from the
            // data structure!
            llir::lExpr stop_cond = get_condition(idx->level - 1, true);
            stop_value = llir::lSelect::make(
                std::move(stop_cond), pend,
                tlower.get_bound(idx->level, index_t, /*upper_bound=*/true));
        }

        stmts.push_back(llir::Declare::make(
            index_t, tlower.get_iter_name(idx->level), start_value));
        // This is const.
        stmts.push_back(llir::Declare::make(
            index_t, tlower.get_stop_name(idx->level), stop_value));

        lExprPair p = {get_iter(idx->level), get_stop(idx->level)};
        imap[i] = std::move(p);
    }

    auto make_loop = [&](const Seq &s) {
        auto is = indexes(s);

        std::vector<lExprPair> iters;
        iters.reserve(is.size());

        // TODO: locator optimization?

        llir::lExpr cond;
        for (const auto &i : is) {
            const auto &miter = imap.find(i);
            internal_assert(miter != imap.end()) << i;
            iters.push_back(miter->second);
            if (cond.defined()) {
                cond = cond && miter->second.first <= miter->second.second;
            } else {
                cond = miter->second.first <= miter->second.second;
            }
        }
        
        // used only for compute kernel, adds the epilogue for offset calculation 
        // in result.
        auto make_offset_calculation = [&](const Forall* forall) {
            if (is_sparse_format(result_tensor.tensor_type.format.lvlfmt_of(forall->idx))) {
                int level = result_tensor.tensor_type.format.get_level_order(forall->idx);
                auto store_stmt = llir::Store::make(
                    llir::lArrayAccess::make(
                        llir::lFieldAccess::make(
                            llir::lVar::make(
                                llir::Generic_t::make(result_tensor.get_struct_name()),
                                "result"
                            ),
                            result_tensor.get_offsets_field_name(forall->idx)
                        ),
                        result_tensor.get_offset_expression_for_next_sparse(
                            result_tensor.tensor_type.format.get_prev_sparse_level(level), level-1,
                            true, true, get_iter_vars_result(result_tensor)
                        )
                    ),
                    llir::lVar::make(index_t, "offset_" + forall->idx)
                );

                auto get_condition = [&](const Index *idx) -> llir::lExpr {
                    internal_assert(idx) << idx;
                    TensorLowerer tlower(idx->tensor, idx->type);
                    llir::lExpr stop_var = llir::lVar::make(index_t, tlower.get_stop_name(idx->level));
                    llir::lExpr extrema = tlower.get_bound(idx->level, index_t, /*upper_bound=*/true);
                    llir::lExpr cond = stop_var == extrema;
                    return cond;
                };

                auto idxs = indexes(forall->seq);
                llir::lExpr cond;
                for (int i=0; i < idxs.size(); i++) {
                    llir::lExpr idx_cond = get_condition(idxs[i].as<Index>());
                    cond = i==0 ? idx_cond : cond && idx_cond;
                }

                return llir::IfElse::make(
                    cond,
                    std::move(store_stmt),
                    nullptr
                );
            }
            return llir::lStmt();
        };

        auto make_body = [&](const Seq &a) {
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
            if(cin.as<Forall>()) {
                body = lower_loop(cin, new_def, is_precompute);
                auto offset_stmt =  make_offset_calculation(cin.as<Forall>());
                if(offset_stmt.defined()) {
                    body = llir::Sequence::make({std::move(body), std::move(offset_stmt)});
                }
            } else {
                body = lower_assign_statement(cin, is_precompute);
            }

            if (seq.get()->is_sparse) {
                    // Count this loop iteration.
                    llir::lStmt inc = llir::BaseExpr::make(llir::lIncrement::make(
                        llir::lVar::make(index_t, (is_precompute? "count_": "offset_") + forall->idx)));
                    if (body.defined()) {
                        body =
                            llir::Sequence::make({std::move(body), std::move(inc)});
                    } else {
                        body = std::move(inc);
                    }
            }
            return body;
        };

        auto make_condition = [&](const Seq &a) {
            auto as = indexes(a);
            llir::lExpr cond;
            llir::lExpr var = llir::lVar::make(index_t, forall->idx);
            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                llir::lExpr idx_expr = llir::lVar::make(
                    index_t, TensorLowerer(idx->tensor, idx->type)
                                 .get_coord_name(idx->level));
                llir::lExpr sub = var == idx_expr;

                if (cond.defined()) {
                    cond = cond && sub;
                } else {
                    cond = sub;
                }
            }
            return cond;
        };

        auto make_evals = [&](const Seq &a) {
            auto as = indexes(a);
            std::vector<llir::lStmt> stmts;
            llir::lExpr value;
            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                TensorLowerer tlower(idx->tensor, idx->type);
                stmts.push_back(tlower.make_eval(idx->level, index_t));
                if (!value.defined()) {
                    value = tlower.get_coord_var(idx->level, index_t);
                } else {
                    value = llir::lBinOp::make(
                        llir::lBinOp::Min,
                        tlower.get_coord_var(idx->level, index_t),
                        std::move(value));
                }
            }
            stmts.push_back(
                llir::Declare::make(index_t, forall->idx, std::move(value)));
            return stmts;
        };

        auto make_incs = [&](const Seq &a, std::vector<llir::lStmt> &stmts) {
            auto as = indexes(a);

            for (const auto &term : as) {
                const Index *idx = term.as<Index>();
                internal_assert(idx) << term;
                TensorLowerer tlower(idx->tensor, idx->type);
                stmts.push_back(tlower.make_inc(idx->level, index_t));
            }
        };

        auto make_assign_indices = [&](llir::lExpr index) {
            if(result_tensor.tensor_type.format.level_exists(forall->idx) && 
               is_sparse_format(
                   result_tensor.tensor_type.format.lvlfmt_of(forall->idx))) {
                return llir::Store::make(
                    llir::lArrayAccess::make(
                        llir::lFieldAccess::make(
                            llir::lVar::make(
                                llir::Generic_t::make(result_tensor.get_struct_name()), 
                                "result"),
                            result_tensor.get_indices_field_name(forall->idx)
                            ),
                            llir::lVar::make(index_t, "offset_" + forall->idx)
                        ),
                        index
                    );
            }
            return llir::lStmt();
        };

        

        if (iters.size() == 1) {
            // For loop!
            const llir::lVar *start = iters[0].first.as<llir::lVar>();
            
            static const llir::lExpr _1 = llir::lConst::make(1);

            auto index =  is[0].as<Index>();
            if(!index) {
                internal_assert(false) << "Expected Index: " << is[0];
            }
            TensorLowerer tlower(index->tensor, index->type);

            llir::lStmt body; 
            
            // Store the index value in the result
            if(!is_precompute) {
                body = llir::Declare::make(index_t, forall->idx, tlower.get_coord(index->level, index_t));
                body =  llir::Sequence::make({std::move(body), make_assign_indices(
                    llir::lVar::make(index_t, forall->idx)
                )});
            }

            if( body.defined()) {
                body = llir::Sequence::make({std::move(body), make_body(is[0])});
            } else {
                body = make_body(is[0]);
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
                auto ifelse = llir::IfElse::make(make_condition(cases[count]), b,
                                          build_ifelse(count + 1));
                return ifelse;
            };

            // Prologue (evaluate)
            auto stmts = make_evals(s);

            // Store the index value in the result
            if(!is_precompute){
                auto stmt = make_assign_indices(llir::lVar::make(index_t, forall->idx));
                if(stmt.defined()) {
                    stmts.push_back(std::move(stmt));
                }
            }
            // Execute
            stmts.push_back(build_ifelse(0));
            // Epilogue (incrments!)
            make_incs(s, stmts);

            auto body = llir::Sequence::make(std::move(stmts));

            return llir::While::make(std::move(cond), std::move(body));
        }
    };

    const auto tsort = lattice.topological_order();

    for (const auto &s : tsort) {
        stmts.push_back(make_loop(s));
    }

    return llir::Sequence::make(std::move(stmts));
}
} // namespace backend

} // namespace nacho
