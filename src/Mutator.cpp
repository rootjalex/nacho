#include "Mutator.h"

#include "CIN.h"
#include "Frontend.h"
#include "Seq.h"

namespace nacho {

Expr Mutator::mutate(const Expr &expr) {
    return expr.defined() ? expr.get()->mutate_Expr(this) : Expr();
}

Expr Mutator::visit(const Add *node) { return mutate_binop<Expr>(node); }

Expr Mutator::visit(const Bc *node) {
    Expr a = mutate(node->a);
    if (a.same_as(node->a)) {
        return node;
    }
    return Bc::make(node->index, std::move(a));
}

Expr Mutator::visit(const Mul *node) { return mutate_binop<Expr>(node); }

Expr Mutator::visit(const Sum *node) {
    Expr a = mutate(node->a);
    if (a.same_as(node->a)) {
        return node;
    }
    return Sum::make(node->index, std::move(a));
}

Expr Mutator::visit(const Tensor *node) { return node; }

Seq Mutator::mutate(const Seq &seq) {
    return seq.defined() ? seq.get()->mutate_Seq(this) : Seq();
}

Seq Mutator::visit(const Empty *node) { return node; }

Seq Mutator::visit(const Index *node) { return node; }

Seq Mutator::visit(const Intersect *node) { return mutate_binop<Seq>(node); }

Seq Mutator::visit(const Union *node) { return mutate_binop<Seq>(node); }

Seq Mutator::visit(const Universe *node) { return node; }

cExpr Mutator::mutate(const cExpr &cexpr) {
    return cexpr.defined() ? cexpr.get()->mutate_cExpr(this) : cExpr();
}

cExpr Mutator::visit(const cAdd *node) { return mutate_binop_cexpr<cExpr>(node); }

cExpr Mutator::visit(const cMul *node) { return mutate_binop_cexpr<cExpr>(node); }

cExpr Mutator::visit(const cTensor *node) { return node; }

CIN Mutator::mutate(const CIN &cin) {
    return cin.defined() ? cin.get()->mutate_CIN(this) : CIN();
}

CIN Mutator::visit(const Accumulate *node) {
    cExpr expr = mutate(node->expr);
    if (expr.same_as(node->expr)) {
        return node;
    }
    return Accumulate::make(node->tensor, node->type, node->accumulate_indices, std::move(expr));
}

CIN Mutator::visit(const Assign *node) {
    cExpr expr = mutate(node->expr);
    if (expr.same_as(node->expr)) {
        return node;
    }
    return Assign::make(node->tensor, node->type, std::move(expr));
}

CIN Mutator::visit(const Forall *node) {
    Seq seq = mutate(node->seq);
    CIN body = mutate(node->body);
    if (seq.same_as(node->seq) && body.same_as(node->body)) {
        return node;
    }
    return Forall::make(node->idx, std::move(seq), std::move(body));
}

CIN Mutator::visit(const Sequence *node) {
    std::vector<CIN> stmts;
    stmts.reserve(node->stmts.size());
    bool changed = false;
    for (const auto &stmt : node->stmts) {
        stmts.push_back(mutate(stmt));
        changed = changed || !stmts.back().same_as(stmt);
    }
    if (!changed) {
        return node;
    }
    return Sequence::make(std::move(stmts));
}

CIN Mutator::visit(const Where *node) {
    CIN producer = mutate(node->producer);
    CIN consumer = mutate(node->consumer);
    if (producer.same_as(node->producer) && consumer.same_as(node->consumer)) {
        return node;
    }
    return Where::make(node->temp, node->temp_type, std::move(producer),
                       std::move(consumer));
}

CIN Mutator::visit(const CalculateWork *node) {
    CIN body = mutate(node->body);
    if (body.same_as(node->body)) {
        return node;
    }
    return CalculateWork::make(std::move(body));
}

} // namespace nacho
