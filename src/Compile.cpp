#include "Compile.h"

#include "CIN.h"
#include "Error.h"
#include "Frontend.h"
#include "Mutator.h"
#include "Printer.h"

namespace nacho {

namespace {

bool contains_inner_sum(const Expr &expr) {
    struct Checker : public Visitor {
        bool found = false;

        void visit(const Sum *node) override { found = true; }
    };

    Checker checker;

    if (const Sum *sum = expr.as<Sum>()) {
        sum->a.accept(&checker);
    } else {
        expr.accept(&checker);
    }

    return checker.found;
}

// Drop broadcasts and sums
struct Converter : public Visitor {
    cExpr cexpr;

    template <typename cT, typename T>
    void visit_binop(const T *node) {
        cexpr = cExpr();
        node->a.accept(this);
        auto a = std::move(cexpr);
        node->b.accept(this);
        auto b = std::move(cexpr);
        cexpr = cT::make(std::move(a), std::move(b));
    }

    void visit(const Add *node) { visit_binop<cAdd>(node); }

    void visit(const Mul *node) { visit_binop<cMul>(node); }

    void visit(const Tensor *node) {
        cexpr = cTensor::make(node->type, node->name);
    }
};

cExpr from_expr(const Expr &expr) {
    Converter converter;
    expr.accept(&converter);
    return converter.cexpr;
}

} // namespace

CIN compile_to_cin(const Expr &expr, std::string out) {
    TensorType out_type = expr.type();

    internal_assert(out_type.format.bc_levels.empty())
        << "TODO: support explicit loop ordering for: " << expr
        << " with format: " << out_type.format;

    // TODO: support reorder loops if it doesn't break dependencies.

    // TODO: need to support inner_sums for generality.
    // Will use Where statements.
    internal_assert(!contains_inner_sum(expr));

    CIN cin;
    std::vector<Level> levels;

    if (const Sum *sum = expr.as<Sum>()) {
        cin = Accumulate::make(out, out_type, sum->index, from_expr(sum->a));
        // Include sum loop.
        levels = sum->a.type().format.levels;
    } else {
        cin = Assign::make(out, out_type, from_expr(expr));
        levels = out_type.format.levels;
    }

    std::vector<std::string> index_list;

    for (const auto &level : levels) {
        index_list.push_back(level.index);
    }

    for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
        cin =
            Forall::make(it->index, build_seq(it->index, index_list, expr), std::move(cin));
    }
    return cin;
}

} // namespace nacho