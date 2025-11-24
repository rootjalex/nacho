#include "Compile.h"

#include "CIN.h"
#include "Error.h"
#include "Frontend.h"
#include "Printer.h"

namespace nacho {

namespace {

bool contains_inner_sum(const Expr &expr) {
    struct Checker : public Visitor {
        bool found = false;

        void visit(const Sum *node) { found = true; }
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

struct BuildSeq : public Visitor {
    Seq seq;

    const std::string &index;

    BuildSeq(const std::string &index) : index(index) {}

    template <typename S, typename T>
    void visit_binop(const T *node) {
        seq = Seq();
        node->a.accept(this);
        auto a = std::move(seq);
        node->b.accept(this);
        auto b = std::move(seq);
        seq = S::make(std::move(a), std::move(b));
    }

    void visit(const Add *node) { visit_binop<Union>(node); }

    void visit(const Bc *node) {
        if (index == node->index) {
            seq = Universe::make(index);
        } else {
            node->a.accept(this);
        }
    }

    void visit(const Mul *node) { visit_binop<Intersect>(node); }

    void visit(const Sum *node) {
        internal_assert(index != node->index)
            << "Cannot get seq for index: " << index << " from " << Expr(node);
        node->a.accept(this);
    }

    void visit(const Tensor *node) {
        // Find index of `index` in levels.
        const auto &levels = node->type.format.levels;

        auto it =
            std::find_if(levels.begin(), levels.end(),
                         [&](const Level &lvl) { return lvl.index == index; });
        internal_assert(it != levels.end())
            << "Index: " << index << " not found in: " << Expr(node);
        size_t level = std::distance(levels.begin(), it);

        seq = Index::make(node->name, node->type.format, level);
    }
};

Seq build_seq(const std::string &index, const Expr &expr) {
    BuildSeq builder(index);
    expr.accept(&builder);
    return builder.seq;
}

CIN handle_sums(const std::string &out, const TensorType &type,
                const Expr &expr) {
    // Any non-innermost sum gets turned into a scalar temporary
    // static size_t count = 0;

    if (!contains_inner_sum(expr)) {
        if (const Sum *sum = expr.as<Sum>()) {
            CIN cin = Accumulate::make(out, type, from_expr(sum->a));
            return Forall::make(sum->index, build_seq(sum->index, sum->a),
                                std::move(cin));
        } else {
            return Assign::make(out, type, from_expr(expr));
        }
    }

    internal_error << "TODO: handle inner sums with Where statements!" << expr;
    return CIN();
}

} // namespace

CIN compile_to_cin(const Expr &expr, std::string out) {
    TensorType out_type = expr.type();

    internal_assert(out_type.format.bc_levels.empty())
        << "TODO: support explicit loop ordering for: " << expr
        << " with format: " << out_type.format;

    // TODO: support reorder loops if it doesn't break dependencies.

    CIN cin = handle_sums(out, out_type, expr);
    for (auto it = out_type.format.levels.rbegin();
         it != out_type.format.levels.rend(); ++it) {
        cin =
            Forall::make(it->index, build_seq(it->index, expr), std::move(cin));
    }
    return cin;
}

} // namespace nacho