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

lType Tuple_t::make(std::vector<lType> types) {
    internal_assert(!types.empty()) << "Cannot make empty tuple type.";
    for (const auto &t : types) {
        internal_assert(t.defined())
            << "Cannot make tuple with empty element type.";
    }
    Tuple_t *node = new Tuple_t;
    node->types = std::move(types);
    return node;
}

lType Struct_t::make(std::string name, std::vector<std::pair<std::string, lType>> fields, std::vector<std::string> generics) {
    internal_assert(!name.empty()) << "Cannot make Struct with empty name.";
    internal_assert(!fields.empty()) << "Cannot make Struct with empty fields.";
    for (const auto &field : fields) {
        internal_assert(field.second.defined())
            << "Cannot make Struct with undefined field type: " << field.first;
        internal_assert(!field.first.empty())
            << "Cannot make Struct with empty field name.";
    }
    Struct_t *node = new Struct_t;
    node->name = std::move(name);
    node->fields = std::move(fields);
    node->generics = std::move(generics);
    return node;
}

lExpr lBinOp::make(lBinOp::Op op, lExpr a, lExpr b) {
    internal_assert(a.defined() && b.defined())
        << "lBinOp of undefined: " << a << " op " << b;
    lBinOp *node = new lBinOp;
    node->op = op;
    // TODO: TYPE INFERENCE BASED ON OP
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

lExpr lExpr::operator[](const lExpr &idx) {
    return lBinOp::make(lBinOp::Load, *this, idx);
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

lExpr operator>(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Lt, std::move(b), std::move(a));
}

lExpr operator<=(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Leq, std::move(a), std::move(b));
}

lExpr operator>=(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Leq, std::move(b), std::move(a));
}

lExpr operator==(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::Eq, std::move(a), std::move(b));
}

lExpr operator&&(lExpr a, lExpr b) {
    return lBinOp::make(lBinOp::And, std::move(a), std::move(b));
}

lExpr lConst::make(std::variant<int64_t, uint64_t, double, bool> value) {
    lConst *node = new lConst;
    // TODO: TYPE
    node->value = std::move(value);
    return node;
}

lExpr lBuild::make(lType type, std::vector<lExpr> values) {
    internal_assert(type.defined()) << "lBuild with undefined type";
    for (const auto &v : values) {
        internal_assert(v.defined())
            << "Cannot make lBuild with empty element.";
    }
    lBuild *node = new lBuild;
    node->type = std::move(type);
    node->values = std::move(values);
    return node;
}

lExpr lSelect::make(lExpr cond, lExpr tval, lExpr fval) {
    internal_assert(cond.defined()) << "lSelect with undefined cond";
    internal_assert(tval.defined()) << "lSelect with undefined tval";
    internal_assert(fval.defined()) << "lSelect with undefined fval";

    lSelect *node = new lSelect;
    // TODO: define lType::type()
    // TODO: assert tval and fval have equal types.
    // node->type = tval.type();
    node->cond = std::move(cond);
    node->tval = std::move(tval);
    node->fval = std::move(fval);
    return node;
}

lExpr lArrayAccess::make(lExpr array, lExpr index) {
    internal_assert(array.defined() && index.defined())
        << "lArrayAccess of undefined: " << array << " [ " << index << " ]";
    lArrayAccess *node = new lArrayAccess;
    node->array = std::move(array);
    node->index = std::move(index);
    return node;
}

lExpr lFieldAccess::make(lExpr object, std::string field) {
    internal_assert(object.defined() && !field.empty())
        << "lFieldAccess of undefined: " << object << " . " << field;
    lFieldAccess *node = new lFieldAccess;
    node->object = std::move(object);
    node->field = std::move(field);
    return node;
}

lExpr lPtrAccess::make(lExpr ptr, lExpr index) {
    internal_assert(ptr.defined() && index.defined())
        << "lPtrAccess of undefined: " << ptr << " [ " << index << " ]";
    lPtrAccess *node = new lPtrAccess;
    node->ptr = std::move(ptr);
    node->index = std::move(index);
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

lExpr lFunctionCall::make(std::string function_name, std::vector<lExpr> args) {
    internal_assert(!function_name.empty()) << "Cannot make lFunctionCall with empty function name.";
    for (const auto &arg : args) {
        internal_assert(arg.defined())
            << "Cannot make lFunctionCall with undefined argument.";
    }
    lFunctionCall *node = new lFunctionCall;
    node->function_name = std::move(function_name);
    node->args = std::move(args);
    return node;
}

lExpr lIncrement::make(lExpr var) {
    internal_assert(var.defined()) << "Undefined var in lIncrement::make()";
    lIncrement *node = new lIncrement;
    node->var = std::move(var);
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
    Return *node = new Return;
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

lStmt Store::make(lExpr var, lExpr value) {
    internal_assert(var.defined())
        << "Cannot make Store with undefined var";
    internal_assert(value.defined())
        << "Cannot make Store with undefined value";
    Store *node = new Store;
    node->var = std::move(var);
    node->value = std::move(value);
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

lStmt BaseExpr::make(lExpr expr) {
    internal_assert(expr.defined()) << "Undefined expr in BaseExpr::make()";
    BaseExpr *node = new BaseExpr;
    node->expr = std::move(expr);
    return node;
}

lStmt Break::make() {
    Break *node = new Break;
    return node;
}

lStmt For::make(lType type, std::string name, lExpr init, lExpr cond, lExpr inc,
                lStmt body) {
    internal_assert(type.defined()) << "Undefined type in For::make()";
    internal_assert(!name.empty()) << "Empty name in For::make()";
    internal_assert(init.defined()) << "Undefined init in For::make()";
    internal_assert(cond.defined()) << "Undefined cond in For::make()";
    internal_assert(inc.defined()) << "Undefined inc in For::make()";
    internal_assert(body.defined()) << "Undefined body in For::make()";

    For *node = new For;
    node->type = std::move(type);
    node->name = std::move(name);
    node->init = std::move(init);
    node->cond = std::move(cond);
    node->inc = std::move(inc);
    node->body = std::move(body);
    return node;
}

lStmt For::make(std::string name, lExpr cond, lExpr inc, lStmt body) {
    internal_assert(!name.empty()) << "Empty name in For::make()";
    internal_assert(cond.defined()) << "Undefined cond in For::make()";
    internal_assert(inc.defined()) << "Undefined inc in For::make()";
    internal_assert(body.defined()) << "Undefined body in For::make()";

    For *node = new For;
    node->type = lType(); // undefined!
    node->name = std::move(name);
    node->init = lExpr(); // undefined!
    node->cond = std::move(cond);
    node->inc = std::move(inc);
    node->body = std::move(body);
    return node;
}

lStmt Accumulate::make(lExpr var, lExpr value) {
    internal_assert(var.defined())
        << "Cannot make Accumulate with undefined var";
    internal_assert(value.defined())
        << "Cannot make Accumulate with undefined value";
    Accumulate *node = new Accumulate;
    node->var = std::move(var);
    node->value = std::move(value);
    return node;
}

} // namespace llir
} // namespace nacho
