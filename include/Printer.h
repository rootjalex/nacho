#pragma once

#include "Visitor.h"

#include <iostream>

namespace nacho {

std::ostream &operator<<(std::ostream &os, const TensorIndex &idx);
std::ostream &operator<<(std::ostream &os, const Level &lvl);
std::ostream &operator<<(std::ostream &os, const Format &format);

std::ostream &operator<<(std::ostream &os, const Expr &expr);

std::ostream &operator<<(std::ostream &os, const Seq &seq);
std::ostream &operator<<(std::ostream &os, const cExpr &cexpr);
std::ostream &operator<<(std::ostream &os, const CIN &cin);

std::ostream &operator<<(std::ostream &os, const llir::lExpr &lexpr);
std::ostream &operator<<(std::ostream &os, const llir::lStmt &lstmt);
std::ostream &operator<<(std::ostream &os, const llir::lType &ltype);

struct Printer : public Visitor {
    explicit Printer(std::ostream &os) : os(os) {}

    virtual void print(const Expr &);
    virtual void print_no_parens(const Expr &);
    virtual void visit(const Add *) override;
    virtual void visit(const Bc *) override;
    virtual void visit(const Mul *) override;
    virtual void visit(const Sum *) override;
    virtual void print_tensor(const std::string &name, const TensorType &type);
    virtual void visit(const Tensor *) override;

    virtual void print(const Seq &);
    virtual void print_no_parens(const Seq &);
    virtual void visit(const Empty *) override;
    virtual void visit(const Index *) override;
    virtual void visit(const Intersect *) override;
    virtual void visit(const Union *) override;
    virtual void visit(const Universe *) override;

    virtual void print(const cExpr &);
    virtual void print_no_parens(const cExpr &);
    virtual void visit(const cAdd *) override;
    virtual void visit(const cMul *) override;
    virtual void visit(const cTensor *) override;

    virtual void print(const CIN &);
    virtual void visit(const Accumulate *) override;
    virtual void visit(const Assign *) override;
    virtual void visit(const Forall *) override;
    virtual void visit(const Sequence *) override;
    virtual void visit(const Where *) override;
    virtual void visit(const CalculateWork *) override;

    virtual void print(const llir::lType &);
    virtual void visit(const llir::Generic_t *) override;
    virtual void visit(const llir::Int_t *) override;
    virtual void visit(const llir::Float_t *) override;
    virtual void visit(const llir::Ptr_t *) override;
    virtual void visit(const llir::Tuple_t *) override;
    virtual void visit(const llir::Struct_t *) override;
    virtual void visit(const llir::Bool_t *) override;

    virtual void print(const llir::lExpr &);
    virtual void print_no_parens(const llir::lExpr &);
    virtual void visit(const llir::lOp *) override;
    virtual void visit(const llir::lBinOp *) override;
    virtual void visit(const llir::lConst *) override;
    virtual void visit(const llir::lBuild *) override;
    virtual void visit(const llir::lSelect *) override;
    virtual void visit(const llir::lArrayAccess *) override;
    virtual void visit(const llir::lFieldAccess *) override;
    virtual void visit(const llir::lPtrAccess *) override;
    virtual void visit(const llir::lVar *) override;
    virtual void visit(const llir::lFunctionCall *) override;
    virtual void visit(const llir::lIncrement *) override;
    virtual void visit(const llir::lAddress *) override;
    virtual void visit(const llir::RawExpr *) override;
    virtual void visit(const llir::lLambda *) override;
    virtual void visit(const llir::Cast *) override;

    virtual void print(const llir::lStmt &);
    virtual void visit(const llir::Declare *) override;
    virtual void visit(const llir::IfElse *) override;
    virtual void visit(const llir::Return *) override;
    virtual void visit(const llir::Sequence *) override;
    virtual void visit(const llir::Store *) override;
    virtual void visit(const llir::While *) override;
    virtual void visit(const llir::Function *) override;
    virtual void visit(const llir::BaseExpr *) override;
    virtual void visit(const llir::Break *) override;
    virtual void visit(const llir::For *) override;
    virtual void visit(const llir::Accumulate *) override;
    virtual void visit(const llir::RawStmt *) override;

    void indent() { indent_count++; }
    void dedent() { indent_count--; }

  private:
    /** The stream on which we're outputting */
    std::ostream &os;

    // false -> print parens on open()/close()
    bool implicit_parens = true;

    void open() const {
        if (!implicit_parens) {
            os << "(";
        }
    }

    void close() const {
        if (!implicit_parens) {
            os << ")";
        }
    }

    size_t indent_count = 0;

    virtual void print_indent() {
        for (size_t i = 0; i < indent_count; i++) {
            os << "  "; // two spaces
        }
    }

    virtual void end_line() { os << "\n"; }
};

} // namespace nacho
