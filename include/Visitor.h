#pragma once

#include "IRFwdDecl.h"

namespace nacho {

struct Visitor {
    virtual void visit(const Add *);
    virtual void visit(const Bc *);
    virtual void visit(const Mul *);
    virtual void visit(const Sum *);
    virtual void visit(const Tensor *);
};

#define RESTRICT_VISITOR(IRNODE)                                               \
    void visit(const IRNODE *) final {                                         \
        internal_error << "Restricted Visitor class does not handle: "         \
                       << typeid(IRNODE).name();                               \
    }

} // namespace nacho
