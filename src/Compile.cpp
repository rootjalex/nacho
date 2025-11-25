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

    // Default Sum behavior is fine

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

bool is_dense(const Seq &seq) {
    struct IsDense : public Visitor {
        bool dense = false;

        void visit(const Index *node) override {
            dense =
                node->format.levels[node->level].format == LevelFormat::Dense;
        }

        void visit(const Intersect *node) override {
            node->a.accept(this);
            if (!dense) {
                return;
            }
            // implicit &&
            node->b.accept(this);
        }

        void visit(const Union *node) override {
            node->a.accept(this);
            if (dense) {
                return;
            }
            // implicit ||
            node->b.accept(this);
        }

        void visit(const Universe *) override { dense = true; }
    };

    IsDense checker;
    seq.accept(&checker);
    return checker.dense;
}

// TODO: move this to an Optimize.cpp?
Seq simplify_seq(const Seq &seq) {
    struct Simplify : public Mutator {
        Seq visit(const Intersect *node) override {
            Seq a = mutate(node->a);
            Seq b = mutate(node->b);

            // a ∩ U = a
            if (b.is<Universe>() || is_dense(b)) {
                return a;
            }

            // U ∩ b = b
            if (a.is<Universe>() || is_dense(a)) {
                return b;
            }

            if (a.same_as(node->a) && b.same_as(node->b)) {
                return node;
            }
            return Intersect::make(std::move(a), std::move(b));
        }
    };
    return Simplify().mutate(seq);
}

Seq build_seq(const std::string &index, const Expr &expr) {
    BuildSeq builder(index);
    expr.accept(&builder);
    // TODO: break this out into a simplify or optimize pass?
    return simplify_seq(builder.seq);
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
        cin = Accumulate::make(out, out_type, from_expr(sum->a));
        // Include sum loop.
        levels = sum->a.type().format.levels;
    } else {
        cin = Assign::make(out, out_type, from_expr(expr));
        levels = out_type.format.levels;
    }

    for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
        cin =
            Forall::make(it->index, build_seq(it->index, expr), std::move(cin));
    }
    return cin;
}

} // namespace nacho