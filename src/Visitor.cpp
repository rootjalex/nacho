#include "Visitor.h"

#include "Frontend.h"

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

} // namespace nacho
