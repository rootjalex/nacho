#include "llir/LLIR.h"

#include "Error.h"
#include "Printer.h"

#include <algorithm>

namespace nacho {
namespace llir {

lType Generic_t::make(std::string name) {
    internal_assert(!name.empty()) << "Cannot make Generic with empty name.";
    Generic_t *node = new Generic_t;
    node->name = std::move(name);
    return node;
}

lType Float_t::make(uint8_t bits) {
    internal_assert(bits <= 64 && bits % 8 == 0) << "Cannot make float_t of bad bitwidth: " << bits;
    Float_t *node = new Float_t;
    node->bits = bits;
    return node;
}

lType Int_t::make(uint8_t bits) {
    internal_assert(bits <= 64 && bits % 8 == 0) << "Cannot make int_t of bad bitwidth: " << bits;
    Int_t *node = new Int_t;
    node->bits = bits;
    return node;
}

lType Ptr_t::make(lType type) {
    internal_assert(type.defined()) << "Cannot make ptr_t of undefined type.";
    Ptr_t *node = new Ptr_t;
    node->type = std::move(type);
    return node;
}

lExpr lBinOp::make(lBinOp::Op op, lExpr a, lExpr b) {
    internal_assert(a.defined() && b.defined())
        << "lBinOp of undefined: " << a << " + " << b;
    lBinOp *node = new lBinOp;
    node->op = op;
    // TODO: TYPE INFERENCE
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

lExpr operator+(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Add, std::move(a), std::move(b));
}

lExpr operator-(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Sub, std::move(a), std::move(b));
}

lExpr operator*(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Mul, std::move(a), std::move(b));
}

lExpr operator/(lExpr a, lExpr b){
    return lBinOp::make(lBinOp::Div, std::move(a), std::move(b));
}

lExpr operator<(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Lt, std::move(a), std::move(b));
}

lExpr operator<=(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Leq, std::move(a), std::move(b));
}

lExpr lConst::make(std::variant<int64_t, uint64_t, double> value) {
    lConst *node = new lConst;
    // TODO: TYPE
    node->value = std::move(value);
    return node;
}

lExpr lLoad::make(lExpr var, lExpr idx) {
    internal_assert(var.defined())
        << "lLoad from undefined";
    internal_assert(idx.defined())
        << "lLoad with undefined index.";
    lLoad *node = new lLoad;
    // TODO: TYPE INFERENCE
    node->var = std::move(var);
    node->idx = std::move(idx);
    return node;
}

lExpr lVar::make(lType type, std::string name) {
    internal_assert(type.defined()) << "Cannot make lVar with undef type.";
    internal_assert(!name.empty()) << "Cannot make lVar with empty name.";
    lVar *node = new lVar;
    node->type = std::move(type);
    node->name = std::move(name);
    return node;
}

lStmt Declare::make(lType type, std::string name, lExpr init) {
    internal_assert(!name.empty()) << "Cannot make Declare with empty name.";
    internal_assert(init.defined())
        << "Cannot make Declare with empty init: " << name;
    Declare *node = new Declare;
    node->type = std::move(type);
    node->name = std::move(name);
    node->init = std::move(init);
    return node;
}

lStmt IfElse::make(lExpr cond, lStmt then_case, lStmt else_case) {
    internal_assert(cond.defined())
        << "Cannot make IfElse with empty condition";
    internal_assert(then_case.defined())
        << "Cannot make IfElse with then_case";
    IfElse *node = new IfElse;
    node->cond = std::move(cond);
    node->then_case = std::move(then_case);
    node->else_case = std::move(else_case);
    return node;
}

lStmt Return::make() {
    // Use the same pointer for all void returns
    static Return *node = new Return;
    return node;
}

lStmt Return::make(lExpr expr) {
    internal_assert(expr.defined()) << "Undefined expr in Return::make()";
    Return *node = new Return;
    node->expr = std::move(expr);
    return node;
}

lStmt Sequence::make(std::vector<lStmt> stmts) {
    internal_assert(!stmts.empty()) << "Sequence with no stmts";

    Sequence *node = new Sequence;
    node->stmts = std::move(stmts);
    return node;
}

lStmt Store::make(std::string name, lExpr expr) {
    return Store::make(std::move(name), lExpr(), std::move(expr));
}

lStmt Store::make(std::string name, lExpr index, lExpr expr) {
    internal_assert(!name.empty()) << "Cannot make Store with empty name.";
    internal_assert(expr.defined())
        << "Cannot make Store with empty value: " << name;
    Store *node = new Store;
    node->name = std::move(name);
    node->index = std::move(index);
    node->expr = std::move(expr);
    return node;
}

lStmt While::make(lExpr cond, lStmt body) {
    internal_assert(cond.defined())
        << "Cannot make While with empty condition";
    internal_assert(body.defined())
        << "Cannot make While with body";
    While *node = new While;
    node->cond = std::move(cond);
    node->body = std::move(body);
    return node;
}

} // namespace llir
} // namespace nacho
