#pragma once

#include "Visitor.h"

#include <iostream>

namespace nacho {

std::ostream &operator<<(std::ostream &os, const Expr &expr);

struct Printer : public Visitor {
    explicit Printer(std::ostream &os) : os(os) {}

    virtual void print(const Expr &);
    virtual void print_no_parens(const Expr &);
    virtual void visit(const Add *) override;
    virtual void visit(const Bc *) override;
    virtual void visit(const Mul *) override;
    virtual void visit(const Sum *) override;
    virtual void visit(const Tensor *) override;

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

    // TODO: indentation for statements
};

} // namespace nacho
