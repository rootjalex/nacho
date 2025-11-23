#include "Printer.h"

#include "Frontend.h"
#include "Scope.h"

namespace nacho {

std::ostream &operator<<(std::ostream &os, const Expr &expr) {
    if (expr.defined()) {
        Printer printer(os);
        printer.print_no_parens(expr);
    } else {
        os << "(undef-expr)";
    }
    return os;
}

void Printer::print(const Expr &expr) {
    ScopedValue<bool> old(implicit_parens, false);
    expr.accept(this);
}

void Printer::print_no_parens(const Expr &expr) {
    ScopedValue<bool> old(implicit_parens, true);
    expr.accept(this);
}

void Printer::visit(const Add *node) {
    open();
    print(node->a);
    os << " + ";
    print(node->b);
    close();
}

void Printer::visit(const Bc *node) {
    os << "bc<" << node->index << ">(";
    print_no_parens(node->a);
    os << ")";
}

void Printer::visit(const Mul *node) {
    open();
    print(node->a);
    os << " + ";
    print(node->b);
    close();
}

void Printer::visit(const Sum *node) {
    os << "sum<" << node->index << ">(";
    print_no_parens(node->a);
    os << ")";
}

void Printer::visit(const Tensor *node) {
    os << node->name;
    if (!node->type.format.levels.empty()) {
        os << "[";
        bool first = true;
        for (const auto &lvl : node->type.format.levels) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << lvl.index;
        }
        os << "]";
    }
}

} // namespace nacho
