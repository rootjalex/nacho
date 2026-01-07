#include "Visitor.h"

#include "CIN.h"
#include "Frontend.h"
#include "Seq.h"

#include "llir/LLIR.h"

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

void Visitor::visit(const llir::Generic_t *node) { (void)node; }

void Visitor::visit(const llir::Int_t *node) { (void)node; }

void Visitor::visit(const llir::Float_t *node) { (void)node; }

void Visitor::visit(const llir::Ptr_t *node) { node->type.accept(this); }

void Visitor::visit(const llir::Tuple_t *node) {
    for (const auto &t : node->types) {
        t.accept(this);
    }
}

void Visitor::visit(const llir::lBinOp *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const llir::lConst *node) { (void)node; }

void Visitor::visit(const llir::lBuild *node) {
    for (const auto &e : node->values) {
        e.accept(this);
    }
}

void Visitor::visit(const llir::lVar *node) { (void)node; }

void Visitor::visit(const llir::Declare *node) { node->init.accept(this); }

void Visitor::visit(const llir::IfElse *node) {
    node->cond.accept(this);
    node->then_case.accept(this);
    if (node->else_case.defined()) {
        node->else_case.accept(this);
    }
}

void Visitor::visit(const llir::Return *node) { node->expr.accept(this); }

void Visitor::visit(const llir::Sequence *node) {
    for (const auto &stmt : node->stmts) {
        stmt.accept(this);
    }
}

void Visitor::visit(const llir::Store *node) {
    if (node->index.defined()) {
        node->index.accept(this);
    }
    node->expr.accept(this);
}

void Visitor::visit(const llir::While *node) {
    node->cond.accept(this);
    node->body.accept(this);
}

} // namespace nacho
