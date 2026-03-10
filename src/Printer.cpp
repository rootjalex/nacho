#include "Printer.h"

#include "CIN.h"
#include "Frontend.h"
#include "Scope.h"

#include "llir/LLIR.h"

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

std::ostream &operator<<(std::ostream &os, const llir::lExpr &lexpr) {
    if (lexpr.defined()) {
        Printer printer(os);
        printer.print_no_parens(lexpr);
    } else {
        os << "(undef-lexpr)";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const llir::lStmt &lstmt) {
    if (lstmt.defined()) {
        Printer printer(os);
        printer.print(lstmt);
    } else {
        os << "(undef-lstmt)";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, llir::lType &ltype) {
    if (ltype.defined()) {
        Printer printer(os);
        printer.print(ltype);
    } else {
        os << "(undef-ltype)";
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

void Printer::visit(const Empty *node) {
    if (node->is_sparse) {
        os << "E";
    } else {
        os << "F";
    }
}

void Printer::visit(const Index *node) {
    os << node->type.format.levels[node->level].index;
    os << "_" << node->tensor;
    // Print dependency
    if (node->level != 0) {
        os << "[";
        os << node->type.format.levels[node->level - 1].index;
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

void Printer::visit(const CalculateWork *node) {
    print_indent();
    os << "calculate_work_stmt\n";
    indent();
    print(node->body);
    dedent();
}

void Printer::print(const llir::lType &ltype) { ltype.accept(this); }

void Printer::visit(const llir::Generic_t *node) { os << node->name; }

void Printer::visit(const llir::Int_t *node) {
    os << "int" << std::to_string(node->bits) << "_t";
}

void Printer::visit(const llir::Float_t *node) {
    switch (node->bits) {
    case 16: {
        os << "__half";
        break;
    }
    case 32: {
        os << "float";
        break;
    }
    case 64: {
        os << "double";
        break;
    }
    default: {
        internal_error << "Unhandled bits for float_t: " << node->bits;
    }
    }
}

void Printer::visit(const llir::Ptr_t *node) {
    node->type.accept(this);
    os << "*";
}

void Printer::visit(const llir::Tuple_t *node) {
    os << "tuple<";
    for (size_t i = 0, e = node->types.size(); i < e; i++) {
        if (i > 0) {
            os << ", ";
        }
        print(node->types[i]);
    }
    os << ">";
}

void Printer::visit(const llir::Struct_t *node) {
    if (!node->generics.empty()) {
        os << "template<";
        bool first = true;
        for (const auto &g : node->generics) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << "typename " << g;
        }
        os << ">\n";
    }
    os << "struct " << node->name << " {\n";
    indent();
    for (const auto &field : node->fields) {
        print_indent();
        print(field.second);
        os << " " << field.first << ";\n";
    }
    dedent();
    os << "};\n";
}

void Printer::print(const llir::lExpr &lexpr) {
    ScopedValue<bool> old(implicit_parens, false);
    lexpr.accept(this);
}

void Printer::print_no_parens(const llir::lExpr &lexpr) {
    ScopedValue<bool> old(implicit_parens, true);
    lexpr.accept(this);
}

bool is_infix_op(const llir::lBinOp::Op op) {
    switch (op) {
    case llir::lBinOp::And:
    case llir::lBinOp::Add:
    case llir::lBinOp::Div:
    case llir::lBinOp::Eq:
    case llir::lBinOp::Neq:
    case llir::lBinOp::Leq:
    case llir::lBinOp::Lt:
    case llir::lBinOp::Mul:
    case llir::lBinOp::Or:
    case llir::lBinOp::Sub:
        return true;
    default: {
        return false;
    }
    }
}

std::string get_op_string(const llir::lOp::Op op) {
    switch (op) {
    case llir::lOp::Not: {
        return "!";
        break;
    }
    default: {
        internal_error << "Unknown lOp::Op in printing.";
    }
    }
}

std::string get_op_string(const llir::lBinOp::Op op) {
    switch (op) {
    case llir::lBinOp::And: {
        return " && ";
        break;
    }
    case llir::lBinOp::Add: {
        return " + ";
        break;
    }
    case llir::lBinOp::Div: {
        return " / ";
        break;
    }
    case llir::lBinOp::Eq: {
        return " == ";
        break;
    }
    case llir::lBinOp::Neq: {
        return " != ";
        break;
    }
    case llir::lBinOp::Leq: {
        return " <= ";
        break;
    }
    case llir::lBinOp::Lt: {
        return " < ";
        break;
    }
    case llir::lBinOp::Min: {
        return "min";
        break;
    }
    case llir::lBinOp::Max: {
        return "max";
        break;
    }
    case llir::lBinOp::Mul: {
        return " * ";
        break;
    }
    case llir::lBinOp::Or: {
        return " || ";
        break;
    }
    case llir::lBinOp::Sub: {
        return " - ";
        break;
    }
    default: {
        internal_error << "Unknown lBinOp::Op in printing.";
    }
    }
}

void Printer::visit(const llir::lOp *node) {
    os << get_op_string(node->op);
    print_no_parens(node->a);
}

void Printer::visit(const llir::lBinOp *node) {
    if (is_infix_op(node->op)) {
        open();
        print(node->a);
        os << get_op_string(node->op);
        print(node->b);
        close();
    } else if (node->op == llir::lBinOp::Load) {
        print(node->a);
        os << "[";
        print_no_parens(node->b);
        os << "]";
    } else {
        // prefix notation
        os << get_op_string(node->op) << "(";
        print_no_parens(node->a);
        os << ", ";
        print_no_parens(node->b);
        os << ")";
    }
}

void Printer::visit(const llir::lConst *node) {
    // Standard C++ printing of variant options.
    std::visit([&](auto &&v) { os << v; }, node->value);
}

void Printer::visit(const llir::lBuild *node) {
    // TODO: fix this to match C++!
    open();
    print(node->type);
    os << "(";
    for (size_t i = 0, e = node->values.size(); i < e; i++) {
        if (i > 0) {
            os << ", ";
        }
        print_no_parens(node->values[i]);
    }
    os << ")";
    close();
}

void Printer::visit(const llir::lSelect *node) {
    // TODO: fix this to match C++!
    open();
    print(node->cond);
    os << " ? ";
    print(node->tval);
    os << " : ";
    print(node->fval);
    close();
}

void Printer::visit(const llir::lFieldAccess *node) {
    print(node->object);
    os << "." << node->field;
}

void Printer::visit(const llir::lArrayAccess *node) {
    node->array.accept(this);
    os << "[";
    node->index.accept(this);
    os << "]";
}

void Printer::visit(const llir::lPtrAccess *node) {
    node->ptr.accept(this);
    os << "->";
    node->index.accept(this);
}

void Printer::visit(const llir::lVar *node) { os << node->name; }

void Printer::visit(const llir::lFunctionCall *node) {
    os << node->function_name << "(";
    for (size_t i = 0, e = node->args.size(); i < e; i++) {
        if (i > 0) {
            os << ", ";
        }
        print_no_parens(node->args[i]);
    }
    os << ")";
}

void Printer::visit(const llir::lIncrement *node) {
    print_no_parens(node->var);
    os << "++";
}

void Printer::print(const llir::lStmt &lstmt) { lstmt.accept(this); }

void Printer::visit(const llir::Declare *node) {
    print_indent();
    print(node->type);
    os << " ";
    os << node->name;
    if(node->init.defined()){
        os << " = ";
        print_no_parens(node->init);
    }
    os << ";\n";
}

void Printer::visit(const llir::IfElse *node) {
    print_indent();
    os << "if (";
    print_no_parens(node->cond);
    os << ") {\n";
    indent();
    print(node->then_case);
    dedent();
    print_indent();
    os << "}";
    if (node->else_case.defined()) {
        os << " else {\n";
        indent();
        print(node->else_case);
        dedent();
        print_indent();
        os << "}";
    }
    os << "\n";
}

void Printer::visit(const llir::Return *node) {
    print_indent();
    os << "return";
    if (node->expr.defined()) {
        os << " ";
        print_no_parens(node->expr);
    }
    os << ";\n";
}

void Printer::visit(const llir::Sequence *node) {
    for (const auto &stmt : node->stmts) {
        print(stmt);
    }
}

void Printer::visit(const llir::Store *node) {
    print_indent();
    print_no_parens(node->var);
    os << " = ";
    print_no_parens(node->value);
    os << ";\n";
}

void Printer::visit(const llir::While *node) {
    print_indent();
    os << "while (";
    print_no_parens(node->cond);
    os << ") {\n";
    indent();
    print(node->body);
    dedent();
    print_indent();
    os << "}\n";
}

void Printer::visit(const llir::BaseExpr *node) { 
    print_indent();
    print(node->expr); 

    os << ";\n";
}

void Printer::visit(const llir::Break *node) {
    print_indent();
    os << "break";
    os << ";\n";
}

void Printer::visit(const llir::For *node) {
    print_indent();
    os << "for (";
    if (node->type.defined()) {
        internal_assert(node->init.defined());
        print(node->type);
        os << " ";
        os << node->name;
        os << " = ";
        print_no_parens(node->init);
    }
    os << "; ";
    print_no_parens(node->cond);
    os << "; ";
    os << node->name << " += ";
    print_no_parens(node->inc);
    os << ") {\n";
    indent();
    print(node->body);
    dedent();
    print_indent();
    os << "}\n";
}

void Printer::visit(const llir::Accumulate *node) {
    print_indent();
    print_no_parens(node->var);
    os << " += ";
    print(node->value);
    os << ";\n";
}

} // namespace nacho
