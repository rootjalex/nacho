#pragma once

#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Visitor.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace nacho {
namespace llir {

struct lType;
struct lExpr;
struct lStmt;

// Types (for declarations)
// TODO: make this complete
enum class lTypeEnum { Generic_t, Float_t, Int_t, Ptr_t, Tuple_t, Struct_t };

// Expressions
enum class lExprEnum {
    lOp,
    lBinOp,
    lConst,
    lVar,
    lBuild,
    lSelect,
    lArrayAccess,
    lFieldAccess,
    lPtrAccess,
    lFunctionCall,
    lIncrement,
    lAddress,
};

// Statements
enum class lStmtEnum {
    Declare,
    IfElse,
    Return,
    Sequence,
    Store,
    While,
    Function,
    BaseExpr,
    Break,
    For,
    Accumulate
};

using IRlTypeNode = IRNode<lType, lTypeEnum>;

using IRlExprNode = IRNode<lExpr, lExprEnum>;

using IRlStmtNode = IRNode<lStmt, lStmtEnum>;

// This is necessary to get mutate() to work properly...
struct BaselTypeNode : public IRlTypeNode {
    BaselTypeNode(lTypeEnum t) : IRlTypeNode(t) {}
    // virtual lStmt mutate_lStmt(Mutator *m) const = 0;
};

template <typename T>
struct lTypeNode : public BaselTypeNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    // lType mutate_lType(Mutator *m) const override;
    lTypeNode() : BaselTypeNode(T::node_type) {}
    ~lTypeNode() override = default;
};

struct lType : public IRHandle<IRlTypeNode> {
    /** Make an undefined lType */
    lType() = default;

    /** Make a lType from a concrete lType node pointer (e.g. Int_t) */
    lType(const IRlTypeNode *n) : IRHandle<IRlTypeNode>(n) {}

    /** Override get() to return a BaselTypeNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaselTypeNode *get() const { return (const BaselTypeNode *)ptr; }

    // TODO: implement copy/move semantics!
};

// template <typename T>
// lType lTypeNode<T>::mutate_lType(Mutator *m) const {
//     return m->visit((const T *)this);
// }

struct BaselExprNode : public IRlExprNode {
    BaselExprNode(lExprEnum t) : IRlExprNode(t) {}
    // virtual lExpr mutate_lExpr(Mutator *m) const = 0;
    lType type;
};

template <typename T>
struct lExprNode : public BaselExprNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    // lExpr mutate_lExpr(Mutator *m) const override;
    lExprNode() : BaselExprNode(T::node_type) {}
    ~lExprNode() override = default;
};

struct lExpr : public IRHandle<IRlExprNode> {
    /** Make an undefined lExpr */
    lExpr() = default;

    /** Make a lExpr from a concrete lExpr node pointer (e.g. lBinOp) */
    lExpr(const IRlExprNode *n) : IRHandle<IRlExprNode>(n) {}

    /** Override get() to return a BaselExprNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaselExprNode *get() const { return (const BaselExprNode *)ptr; }

    // TODO: implement copy/move semantics!

    // Overloads a[b] operation.
    lExpr operator[](const lExpr &idx);
};

// template <typename T>
// lExpr lExprNode<T>::mutate_lExpr(Mutator *m) const {
//     return m->visit((const T *)this);
// }

struct BaselStmtNode : public IRlStmtNode {
    BaselStmtNode(lStmtEnum t) : IRlStmtNode(t) {}
    // virtual lStmt mutate_lStmt(Mutator *m) const = 0;
};


template <typename T>
struct lStmtNode : public BaselStmtNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    // lStmt mutate_lStmt(Mutator *m) const override;
    lStmtNode() : BaselStmtNode(T::node_type) {}
    ~lStmtNode() override = default;
};

struct lStmt : public IRHandle<IRlStmtNode> {
    /** Make an undefined lStmt */
    lStmt() = default;

    /** Make an lStmt from a concrete lStmt node pointer (e.g. While) */
    lStmt(const IRlStmtNode *n) : IRHandle<IRlStmtNode>(n) {}

    /** Override get() to return a BaselStmtNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaselStmtNode *get() const { return (const BaselStmtNode *)ptr; }

    // TODO: implement copy/move semantics!
};

// template <typename T>
// lStmt lStmtNode<T>::mutate_lStmt(Mutator *m) const {
//     return m->visit((const T *)this);
// }

struct Generic_t : lTypeNode<Generic_t> {
    std::string name;

    static lType make(std::string name);

    static const lTypeEnum node_type = lTypeEnum::Generic_t;
};

struct Float_t : lTypeNode<Float_t> {
    uint8_t bits;

    static lType make(uint8_t bits);

    static const lTypeEnum node_type = lTypeEnum::Float_t;
};

struct Int_t : lTypeNode<Int_t> {
    uint8_t bits;

    static lType make(uint8_t bits);

    static const lTypeEnum node_type = lTypeEnum::Int_t;
};

struct Ptr_t : lTypeNode<Ptr_t> {
    lType type;

    static lType make(lType type);

    static const lTypeEnum node_type = lTypeEnum::Ptr_t;
};

struct Tuple_t : lTypeNode<Tuple_t> {
    std::vector<lType> types;

    static lType make(std::vector<lType> types);

    static const lTypeEnum node_type = lTypeEnum::Tuple_t;
};

struct Struct_t : lTypeNode<Struct_t> {
    std::string name;
    std::vector<std::string> generics;
    std::vector<std::pair<std::string, lType>> fields;

    static lType make(std::string name,
                      std::vector<std::pair<std::string, lType>> fields,
                      std::vector<std::string> generics);

    static const lTypeEnum node_type = lTypeEnum::Struct_t;
};


struct lOp : lExprNode<lOp> {
    enum Op {
        Not
    };
    Op op;
    lExpr a;

    static lExpr make(Op op, lExpr a);

    static const lExprEnum node_type = lExprEnum::lOp;
};

struct lBinOp : lExprNode<lBinOp> {
    enum Op {
        And,
        Add,
        Div,
        Eq,
        Neq,
        Leq,
        Load,
        Lt,
        Min,
        Max,
        Mul,
        Or,
        Sub,
        OrEq,
    };
    Op op;
    lExpr a, b;

    static lExpr make(Op op, lExpr a, lExpr b);

    static const lExprEnum node_type = lExprEnum::lBinOp;
};

// Convenient overloading
// TODO: do the rest!
lExpr operator+(lExpr a, lExpr b);
lExpr operator-(lExpr a, lExpr b);
lExpr operator*(lExpr a, lExpr b);
lExpr operator+(lExpr a, int32_t b);
lExpr operator-(lExpr a, int32_t b);
lExpr operator*(lExpr a, int32_t b);
lExpr operator/(lExpr a, int32_t b);
lExpr operator/(lExpr a, lExpr b);
lExpr operator<(lExpr a, lExpr b);
lExpr operator>(lExpr a, lExpr b);
lExpr operator<=(lExpr a, lExpr b);
lExpr operator>=(lExpr a, lExpr b);
lExpr operator==(lExpr a, lExpr b);
lExpr operator&&(lExpr a, lExpr b);
lExpr operator!=(lExpr a, lExpr b);
lExpr operator||(lExpr a, lExpr b);

struct lConst : lExprNode<lConst> {
    std::variant<int64_t, uint64_t, double, bool> value;

    static lExpr make(std::variant<int64_t, uint64_t, double, bool> value);

    static const lExprEnum node_type = lExprEnum::lConst;
};

struct lBuild : lExprNode<lBuild> {
    std::vector<lExpr> values;

    static lExpr make(lType type, std::vector<lExpr> values);

    static const lExprEnum node_type = lExprEnum::lBuild;
};

struct lSelect : lExprNode<lSelect> {
    // (cond ? tval : fval)
    lExpr cond, tval, fval;

    static lExpr make(lExpr cond, lExpr tval, lExpr fval);

    static const lExprEnum node_type = lExprEnum::lSelect;
};

struct lArrayAccess : lExprNode<lArrayAccess> {
    lExpr array;
    lExpr index;

    static lExpr make(lExpr array, lExpr index);

    static const lExprEnum node_type = lExprEnum::lArrayAccess;
};

struct lFieldAccess : lExprNode<lFieldAccess> {
    lExpr object;
    std::string field;

    static lExpr make(lExpr object, std::string field);

    static const lExprEnum node_type = lExprEnum::lFieldAccess;
};

struct lPtrAccess : lExprNode<lPtrAccess> {
    lExpr ptr;
    lExpr index;

    static lExpr make(lExpr ptr, lExpr index);

    static const lExprEnum node_type = lExprEnum::lPtrAccess;
};

struct lVar : lExprNode<lVar> {
    std::string name;

    static lExpr make(lType type, std::string name);

    static const lExprEnum node_type = lExprEnum::lVar;
};

struct lFunctionCall : lExprNode<lFunctionCall> {
    std::string function_name;
    std::vector<lExpr> args;

    static lExpr make(std::string function_name, std::vector<lExpr> args);

    static const lExprEnum node_type = lExprEnum::lFunctionCall;
};

struct lIncrement : lExprNode<lIncrement> {
    lExpr var;

    static lExpr make(lExpr var);

    static const lExprEnum node_type = lExprEnum::lIncrement;
};

struct lAddress : lExprNode<lAddress> {
    lExpr var;

    static lExpr make(lExpr var);

    static const lExprEnum node_type = lExprEnum::lAddress;
};

struct Declare : lStmtNode<Declare> {
    lType type;
    std::string name;
    lExpr init;

    static lStmt make(lType type, std::string name, lExpr init);
    static lStmt make(lType type, std::string name);

    static const lStmtEnum node_type = lStmtEnum::Declare;
};

struct IfElse : lStmtNode<IfElse> {
    lExpr cond;
    lStmt then_case;
    lStmt else_case; // poosibly empty.

    static lStmt make(lExpr cond, lStmt then_case, lStmt else_case);

    static const lStmtEnum node_type = lStmtEnum::IfElse;
};

struct Return : lStmtNode<Return> {
    lExpr expr; // possibly empty

    static lStmt make();
    static lStmt make(lExpr expr);

    static const lStmtEnum node_type = lStmtEnum::Return;
};

struct Break : lStmtNode<Break> {
    static lStmt make();

    static const lStmtEnum node_type = lStmtEnum::Break;
};

struct Sequence : lStmtNode<Sequence> {
    std::vector<lStmt> stmts;

    static lStmt make(std::vector<lStmt> stmts);

    static const lStmtEnum node_type = lStmtEnum::Sequence;
};

struct Store : lStmtNode<Store> {
    lExpr var;
    lExpr value;

    static lStmt make(lExpr var, lExpr value);

    static const lStmtEnum node_type = lStmtEnum::Store;
};

struct While : lStmtNode<While> {
    lExpr cond;
    lStmt body;

    static lStmt make(lExpr cond, lStmt body);

    static const lStmtEnum node_type = lStmtEnum::While;
};

struct BaseExpr : lStmtNode<BaseExpr> {
    lExpr expr;
    static lStmt make(lExpr expr);

    static const lStmtEnum node_type = lStmtEnum::BaseExpr;
};

struct For : lStmtNode<For> {
    // for ((`type` `name` = `init`)? ; `cond`; `name` += `inc`) { `body`
    // }
    lType type;
    std::string name;
    lExpr init;
    lExpr cond;
    lExpr inc;
    lStmt body;

    static lStmt make(lType type, std::string name, lExpr init, lExpr cond,
                      lExpr inc, lStmt body);
    static lStmt make(std::string name, lExpr cond, lExpr inc, lStmt body);

    static const lStmtEnum node_type = lStmtEnum::For;
};

struct Accumulate : lStmtNode<Accumulate> {
    // var += value;
    lExpr var;
    lExpr value;

    static lStmt make(lExpr var, lExpr value);

    static const lStmtEnum node_type = lStmtEnum::Accumulate;
};

} // namespace llir

template <>
inline RefCount &
ref_count<llir::IRlTypeNode>(const llir::IRlTypeNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<llir::IRlTypeNode>(const llir::IRlTypeNode *t) {
    delete t;
}

template <>
inline RefCount &
ref_count<llir::IRlExprNode>(const llir::IRlExprNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<llir::IRlExprNode>(const llir::IRlExprNode *t) {
    delete t;
}

template <>
inline RefCount &
ref_count<llir::IRlStmtNode>(const llir::IRlStmtNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<llir::IRlStmtNode>(const llir::IRlStmtNode *t) {
    delete t;
}

} // namespace nacho
