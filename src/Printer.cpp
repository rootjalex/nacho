#include "Printer.h"

#include "CIN.h"
#include "Frontend.h"
#include "Scope.h"

namespace nacho {

std::ostream &operator<<(std::ostream &os, const Level &lvl) {
    os << lvl.index << " : ";
    switch (lvl.format) {
    case LevelFormat::Dense: {
        os << "D";
        break;
    }
    case LevelFormat::Compressed: {
        os << "C";
        break;
    }
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const Format &format) {
    os << "Format{";

    // Ordered levels come first
    if (!format.levels.empty()) {
        os << "[";
        for (size_t i = 0; i < format.levels.size(); ++i) {
            os << format.levels[i];
            if (i + 1 < format.levels.size())
                os << ", ";
        }
        os << "]";
    }

    // Unordered levels
    if (!format.bc_levels.empty()) {
        if (!format.levels.empty())
            os << ", "; // separator only if needed

        os << "{";
        bool first = true;
        for (const auto &lvl : format.bc_levels) {
            if (!first)
                os << ", ";
            os << lvl;
            first = false;
        }
        os << "}";
    }

    // If both empty, show explicitly
    if (format.levels.empty() && format.bc_levels.empty()) {
        os << "empty";
    }

    os << "}";

    return os;
}

std::ostream &operator<<(std::ostream &os, const Expr &expr) {
    if (expr.defined()) {
        Printer printer(os);
        printer.print_no_parens(expr);
    } else {
        os << "(undef-expr)";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const Seq &seq) {
    if (seq.defined()) {
        Printer printer(os);
        printer.print_no_parens(seq);
    } else {
        os << "(undef-seq)";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const cExpr &cexpr) {
    if (cexpr.defined()) {
        Printer printer(os);
        printer.print_no_parens(cexpr);
    } else {
        os << "(undef-cexpr)";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const CIN &cin) {
    if (cin.defined()) {
        Printer printer(os);
        printer.print(cin);
    } else {
        os << "(undef-cin)";
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
    os << " * ";
    print(node->b);
    close();
}

void Printer::visit(const Sum *node) {
    os << "sum<" << node->index << ">(";
    print_no_parens(node->a);
    os << ")";
}

void Printer::print_tensor(const std::string &name, const TensorType &type) {
    os << name;
    if (!type.format.levels.empty()) {
        os << "[";
        bool first = true;
        for (const auto &lvl : type.format.levels) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << lvl.index;
        }
        os << "]";
    }
}

void Printer::visit(const Tensor *node) {
    print_tensor(node->name, node->type);
}

void Printer::print(const Seq &seq) {
    ScopedValue<bool> old(implicit_parens, false);
    seq.accept(this);
}

void Printer::print_no_parens(const Seq &seq) {
    ScopedValue<bool> old(implicit_parens, true);
    seq.accept(this);
}

void Printer::visit(const Index *node) {
    os << node->format.levels[node->level].index;
    os << "_" << node->tensor;
    // Print dependency
    if (node->level != 0) {
        os << "[";
        os << node->format.levels[node->level - 1].index;
        os << "]";
    }
}

void Printer::visit(const Intersect *node) {
    open();
    print(node->a);
    os << " ∩ ";
    print(node->b);
    close();
}

void Printer::visit(const Union *node) {
    open();
    print(node->a);
    os << " ∪ ";
    print(node->b);
    close();
}

void Printer::visit(const Universe *node) { os << "U_{" << node->idx << "}"; }

void Printer::print(const cExpr &cexpr) {
    ScopedValue<bool> old(implicit_parens, false);
    cexpr.accept(this);
}

void Printer::print_no_parens(const cExpr &cexpr) {
    ScopedValue<bool> old(implicit_parens, true);
    cexpr.accept(this);
}

void Printer::visit(const cAdd *node) {
    open();
    print(node->a);
    os << " + ";
    print(node->b);
    close();
}

void Printer::visit(const cMul *node) {
    open();
    print(node->a);
    os << " * ";
    print(node->b);
    close();
}

void Printer::visit(const cTensor *node) {
    print_tensor(node->name, node->type);
}

void Printer::print(const CIN &cin) { cin.accept(this); }

void Printer::visit(const Accumulate *node) {
    print_indent();
    print_tensor(node->tensor, node->type);
    os << " += ";
    print_no_parens(node->expr);
    end_line();
}

void Printer::visit(const Assign *node) {
    print_indent();
    print_tensor(node->tensor, node->type);
    os << " = ";
    print_no_parens(node->expr);
    end_line();
}

void Printer::visit(const Forall *node) {
    print_indent();
    os << "forall " << node->idx << " in ";
    print_no_parens(node->seq);
    os << "\n";
    indent();
    print(node->body);
    dedent();
}

void Printer::visit(const Sequence *node) {
    for (const auto &stmt : node->stmts) {
        print(stmt);
    }
}

void Printer::visit(const Where *node) {
    print_indent();
    os << "let " << node->temp << "\n";
    indent();
    print(node->producer);
    dedent();
    print_indent();
    os << "in\n";
    indent();
    print(node->consumer);
    dedent();
}

} // namespace nacho
