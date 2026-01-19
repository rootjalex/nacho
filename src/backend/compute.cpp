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

void ComputeFunctionLowerer::add_common_function_body_for_initialization(
    std::vector<llir::lStmt> &stmts) {
    llir::lType index_t = llir::Generic_t::make("index_t");
    llir::lType value_t = llir::Generic_t::make("value_t");
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    stmts.emplace_back(llir::Declare::make(
        llir::Int_t::make(32), "thread_id",
        llir::lBinOp::make(
            llir::lBinOp::Add,
            llir::lBinOp::make(llir::lBinOp::Mul,
                               llir::lVar::make(index_t, "blockIdx.x"),
                               llir::lVar::make(index_t, "blockDim.x")),
            llir::lVar::make(index_t, "threadIdx.x"))));

    llir::lExpr thread_id_var = llir::lVar::make(index_t, "thread_id");

    // Generates the following for all tensors X index combinations
    // index_t start_B_i = partitions_B.n_p[thread_id];
    // index_t end_B_i = B.dim_i_length - 1;
    // if (thread_id < ((gridDim.x * blockDim.x) - 1)) {
    //     end_B_i = partitions_B.i_p[thread_id + 1];
    // }
    // TODO: end_B calculation might need some changing when non lowermost
    // intersections (see spgemm.cu)
    for (auto it : operand_tensors) {
        auto tensor = it.second;
        for (int j = 0; j < tensor.tensor_type.format.levels.size(); j++) {
            auto level = tensor.tensor_type.format.levels[j];

            stmts.emplace_back(llir::Declare::make(
                index_t, get_iterator_name(tensor, level.index),
                llir::lArrayAccess::make(
                    llir::lFieldAccess::make(
                        llir::lVar::make(
                            llir::Ptr_t::make(llir::Generic_t::make(
                                tensor.get_index_struct_name())),
                            "partitions_" + tensor.tensor_name),
                        level.index +
                            (is_sparse_format(level.format) ? "_p" : "")),
                    thread_id_var)));
            stmts.emplace_back(llir::Declare::make(
                index_t, get_end_iterator_name(tensor, level.index),
                llir::lBinOp::make(
                    llir::lBinOp::Sub,
                    llir::lFieldAccess::make(
                        llir::lVar::make(
                            llir::Ptr_t::make(llir::Generic_t::make(
                                tensor.get_struct_name())),
                            tensor.tensor_name),
                        is_sparse_format(level.format)
                            ? tensor.get_length_field_name(level.index)
                            : tensor.get_size_field_name(level.index)),
                    llir::lConst::make((int64_t)1))));
            stmts.emplace_back(llir::IfElse::make(
                llir::lBinOp::make(
                    llir::lBinOp::Lt, thread_id_var,
                    llir::lBinOp::make(
                        llir::lBinOp::Sub,
                        llir::lBinOp::make(
                            llir::lBinOp::Mul,
                            llir::lVar::make(index_t, "gridDim.x"),
                            llir::lVar::make(index_t, "blockDim.x")),
                        llir::lConst::make((int64_t)1))),
                llir::Store::make(
                    llir::lVar::make(
                        index_t, get_end_iterator_name(tensor, level.index)),
                    llir::lArrayAccess::make(
                        llir::lFieldAccess::make(
                            llir::lVar::make(
                                llir::Ptr_t::make(llir::Generic_t::make(
                                    tensor.get_index_struct_name())),
                                "partitions_" + tensor.tensor_name),
                            level.index +
                                (is_sparse_format(level.format) ? "_p" : "")),
                        llir::lBinOp::make(llir::lBinOp::Add, thread_id_var,
                                           llir::lConst::make((int64_t)1)))),
                nullptr));
        }
    }

    for (auto it : operand_tensors) {
        auto last_forall = forall_list[forall_list.size() - 1].as<Forall>();
        if (it.second.tensor_type.format.level_exists(last_forall->idx)) {
            stmts.emplace_back(
                llir::BaseExpr::make(llir::lIncrement::make(llir::lVar::make(
                    index_t, get_iterator_name(it.second, last_forall->idx)))));
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
    for (auto tensor : operand_tensors) {
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Ptr_t::make(llir::Generic_t::make(
                tensor.second.get_index_struct_name() + "<index_t, value_t>")),
            .name = "partitions_" + tensor.second.tensor_name});
    }

    args.emplace_back(llir::Function::Argument{
        .mutating = true,
        .type = llir::Generic_t::make(get_counts_struct_name() + "<index_t>"),
        .name = "count_offsets"});

    args.emplace_back(llir::Function::Argument{
        .mutating = false, .type = index_t, .name = "per_thread_work"});

    std::vector<llir::lStmt> stmts;

    // Add common initialization statements
    add_common_function_body_for_initialization(stmts);

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
    for (auto tensor : operand_tensors) {
        args.emplace_back(llir::Function::Argument{
            .mutating = true,
            .type = llir::Ptr_t::make(llir::Generic_t::make(
                tensor.second.get_index_struct_name() + "<index_t, value_t>")),
            .name = "partitions_" + tensor.second.tensor_name});
    }

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
    add_common_function_body_for_initialization(stmts);

    stmts.emplace_back(llir::Return::make());
    body = llir::Sequence::make(std::move(stmts));

    return llir::Function::make(std::move(generics), std::move(attributes),
                                std::move(args), std::move(ret_type), name,
                                std::move(body));
}

std::vector<Seq> indexes(const Seq &seq) {
    struct GetIndexes : public Visitor {
        std::vector<Seq> indexes;
        std::set<Seq, SeqLessThan> visited;
        void visit(const Index *node) override {
            if (!visited.count(node)) {
                indexes.push_back(node);
                visited.insert(node);
            }
        }
    };
    GetIndexes getter;
    seq.accept(&getter);
    return getter.indexes;
}

llir::lStmt ComputeFunctionLowerer::lower_loop(
    CIN loop, const std::set<Seq, SeqLessThan> &defined, bool is_precompute) {

    const Forall *forall = loop.as<Forall>();
    if (!forall) {
        // Must be a write, do nothing if this is precompute, convert to llir if
        // Assign.
        const Assign *assign = loop.as<Assign>();
        internal_assert(assign) << "Expected Assign in lower_loop: " << loop;
        if (is_precompute) {
            return llir::lStmt();
        }
        internal_error << "TODO: handle Accumulate in lowering compute loop.\n";
    }
    internal_assert(forall) << "Expected Forall in lower_loop: " << loop;
    const Seq &seq = forall->seq;

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

        lExprPair p = {
            llir::lVar::make(index_t, tlower.get_iter_name(idx->level)),
            llir::lVar::make(index_t, tlower.get_stop_name(idx->level))};
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
            llir::lStmt body = lower_loop(cin, new_def, is_precompute);
            if (is_precompute && seq.get()->is_sparse) {
                // Count this loop iteration.
                llir::lStmt inc = llir::BaseExpr::make(llir::lIncrement::make(
                    llir::lVar::make(index_t, "count_" + forall->idx)));
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

        if (iters.size() == 1) {
            // For loop!
            const llir::lVar *start = iters[0].first.as<llir::lVar>();
            static const llir::lExpr _1 = llir::lConst::make(1);
            auto body = make_body(is[0]);
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
                auto b = make_body(cases[count]);
                internal_assert(b.defined());
                return llir::IfElse::make(make_condition(cases[count]), b,
                                          build_ifelse(count + 1));
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
    };

    const auto tsort = lattice.topological_order();

    for (const auto &s : tsort) {
        stmts.push_back(make_loop(s));
    }

    return llir::Sequence::make(std::move(stmts));
}
} // namespace backend

} // namespace nacho
