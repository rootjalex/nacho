#include "Compile.h"

#include "CIN.h"
#include "Error.h"
#include "Frontend.h"
#include "Mutator.h"
#include "Printer.h"

#include <algorithm>
#include <iterator>

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
        cexpr = cT::make(std::move(a), std::move(b), node->type);
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

// Identify special COO (Coordinate Format) patterns in the expression.
// For expressions like elementwise COO addition and multiplication,
// we consider all the levels of the COO tensor as a single merged level.
struct OptimizeCOO : public Visitor {
    bool is_optimizable = false;
    Expr optimized_expr;
    int num_tensors = 0;
    std::vector<TensorIndex> sum_indices;
    TensorIndex merged_index;

    template <typename T>
    void visit_binop(const T *node) {
       is_optimizable = false;
       node->a.accept(this);
       bool a_coo = is_optimizable;
       Expr a_optimized = optimized_expr;
       is_optimizable = false;
       node->b.accept(this);
       bool b_coo = is_optimizable;
       Expr b_optimized = optimized_expr;

       is_optimizable = a_coo && b_coo && num_tensors<=2;
       if(is_optimizable) {
           optimized_expr = T::make(a_optimized, b_optimized);
       }
    }

    void visit(const Add *node) { visit_binop<Add>(node); }
    void visit(const Mul *node) { visit_binop<Mul>(node); }

    void visit(const Sum *node) {
        is_optimizable = false;
        sum_indices.push_back(node->index);
        node->a.accept(this);
        if(is_optimizable) {
            if(optimized_expr.as<Sum>()) {
                return;
            }
            optimized_expr = Sum::make(merged_index, optimized_expr);
        }
    }


    void visit(const Tensor *node) override {
        num_tensors++;
        // Check if the tensor is in COO format
        if (node->type.format.is_coo()) {
            // If expression has a reduction, the COO optimization is applicable
            // only if all the individual indices in the merged index are reduced.

            if(sum_indices.size() > 0) {
                for(const auto& idx : node->type.format.levels) {
                    if(std::find(sum_indices.begin(), sum_indices.end(), idx.index) == sum_indices.end()) {
                        is_optimizable = false;
                        return;
                    }
                }
            }

            is_optimizable = true;

            TensorIndex merged_index;

            std::transform(
                node->type.format.levels.begin(), node->type.format.levels.end(),
                std::back_inserter(merged_index.indices),
                [](const Level &lvl) {return lvl.index.str();}
            );

            TensorType coo_type = TensorType(
                Format::ordered({
                    Level(merged_index, LevelFormat::MergedCoordinate),
                }),
                node->type.dtype
            );
            this->merged_index = std::move(merged_index);
            optimized_expr = Tensor::make(coo_type, node->name);
        } else {
            is_optimizable = false;
        }
    }
};

Expr optimize_coo(const Expr &expr) {
    OptimizeCOO optimizer;
    expr.accept(&optimizer);
    if (optimizer.is_optimizable) {
        return optimizer.optimized_expr;
    } else {
        return expr;
    }
}

} // namespace

CIN compile_to_cin(const Expr &expr, std::string out) {
    Expr optimized_expr = optimize_coo(expr);

    TensorType out_type = optimized_expr.type();

    internal_assert(out_type.format.bc_levels.empty())
        << "TODO: support explicit loop ordering for: " << expr
        << " with format: " << out_type.format;

    // TODO: support reorder loops if it doesn't break dependencies.

    // TODO: need to support inner_sums for generality.
    // Will use Where statements.
    // internal_assert(!contains_inner_sum(expr));


    CIN cin;
    std::vector<Level> levels;

    if (const Sum *sum = optimized_expr.as<Sum>()) {
        std::vector<TensorIndex> accumulate_indices;
        accumulate_indices.push_back(sum->index);
        while(sum->a.as<Sum>()) {
            sum = sum->a.as<Sum>();
            accumulate_indices.push_back(sum->index);
        }
        cin = Accumulate::make(out, out_type, accumulate_indices, from_expr(sum->a));
        // Include sum loop.
        levels = sum->a.type().format.levels;
    } else {
        cin = Assign::make(out, out_type, from_expr(optimized_expr));
        levels = out_type.format.levels;
    }

    std::vector<TensorIndex> index_list;

    for (const auto &level : levels) {
        index_list.push_back(level.index);
    }


    for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
        cin =
            Forall::make(it->index, build_seq(it->index, index_list, optimized_expr), std::move(cin));
    }
    return cin;
}

} // namespace nacho