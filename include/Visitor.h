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
    virtual void visit(const Empty *);
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

    // LLIR
    virtual void visit(const llir::Generic_t *);
    virtual void visit(const llir::Int_t *);
    virtual void visit(const llir::Float_t *);
    virtual void visit(const llir::Ptr_t *);
    virtual void visit(const llir::Tuple_t *);
    virtual void visit(const llir::Struct_t *);
    virtual void visit(const llir::lBinOp *);
    virtual void visit(const llir::lConst *);
    virtual void visit(const llir::lBuild *);
    virtual void visit(const llir::lArrayAccess *);
    virtual void visit(const llir::lFieldAccess *);
    virtual void visit(const llir::lPtrAccess *);
    virtual void visit(const llir::lVar *);
    virtual void visit(const llir::lFunctionCall *);
    virtual void visit(const llir::lIncrement *);
    virtual void visit(const llir::Declare *);
    virtual void visit(const llir::IfElse *);
    virtual void visit(const llir::Return *);
    virtual void visit(const llir::Sequence *);
    virtual void visit(const llir::Store *);
    virtual void visit(const llir::While *);
    virtual void visit(const llir::Function *node);
    virtual void visit(const llir::BaseExpr *node);
    virtual void visit(const llir::Break *node);
    virtual void visit(const llir::For *node);
};

#define RESTRICT_VISITOR(IRNODE)                                               \
    void visit(const IRNODE *) final {                                         \
        internal_error << "Restricted Visitor class does not handle: "         \
                       << typeid(IRNODE).name();                               \
    }

} // namespace nacho
