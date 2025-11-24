#pragma once

#include "IRFwdDecl.h"

namespace nacho {

struct Visitor {
    // Expr
    virtual void visit(const Add *);
    virtual void visit(const Bc *);
    virtual void visit(const Mul *);
    virtual void visit(const Sum *);
    virtual void visit(const Tensor *);

    // Seq
    virtual void visit(const Index *);
    virtual void visit(const Intersect *);
    virtual void visit(const Union *);
    virtual void visit(const Universe *);

    // cExpr
    virtual void visit(const cAdd *);
    virtual void visit(const cMul *);
    virtual void visit(const cTensor *);

    // CIN
    virtual void visit(const Accumulate *);
    virtual void visit(const Assign *);
    virtual void visit(const Forall *);
    virtual void visit(const Sequence *);
    virtual void visit(const Where *);
};

#define RESTRICT_VISITOR(IRNODE)                                               \
    void visit(const IRNODE *) final {                                         \
        internal_error << "Restricted Visitor class does not handle: "         \
                       << typeid(IRNODE).name();                               \
    }

} // namespace nacho
