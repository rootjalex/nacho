#include "Visitor.h"

#include "CIN.h"
#include "Frontend.h"
#include "Seq.h"

namespace nacho {

void Visitor::visit(const Add *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const Bc *node) { node->a.accept(this); }

void Visitor::visit(const Mul *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const Sum *node) { node->a.accept(this); }

void Visitor::visit(const Tensor *node) { (void)node; }

void Visitor::visit(const Index *node) { (void)node; }

void Visitor::visit(const Intersect *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const Union *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const Universe *node) { (void)node; }

void Visitor::visit(const cAdd *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const cMul *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const cTensor *node) { (void)node; }

void Visitor::visit(const Accumulate *node) { node->expr.accept(this); }

void Visitor::visit(const Assign *node) { node->expr.accept(this); }

void Visitor::visit(const Forall *node) {
    node->seq.accept(this);
    node->body.accept(this);
}

void Visitor::visit(const Sequence *node) {
    for (const auto &stmt : node->stmts) {
        stmt.accept(this);
    }
}

void Visitor::visit(const Where *node) {
    node->producer.accept(this);
    node->consumer.accept(this);
}

} // namespace nacho
