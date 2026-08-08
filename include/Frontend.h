#pragma once

#include "Format.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Type.h"
#include "Visitor.h"

namespace nacho {

struct Expr;

enum class ExprEnum {
    Add,
    Bc,
    Mul,
    Sum,
    Tensor,
};

using IRExprNode = IRNode<Expr, ExprEnum>;

/** This is necessary to get mutate() to work properly...
 *  They all contain their types (e.g. Int(32), Float(32))
 */
struct BaseExprNode : public IRExprNode {
    BaseExprNode(ExprEnum t) : IRExprNode(t) {}
    virtual Expr mutate_Expr(Mutator *m) const = 0;
    TensorType type;
};

template <typename T>
struct ExprNode : public BaseExprNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    Expr mutate_Expr(Mutator *m) const override;
    ExprNode() : BaseExprNode(T::node_type) {}
    ~ExprNode() override = default;
};

struct Expr : public IRHandle<IRExprNode> {
    /** Make an undefined expr */
    Expr() = default;

    /** Make an expr from a concrete expr node pointer (e.g. Add) */
    Expr(const IRExprNode *n) : IRHandle<IRExprNode>(n) {}

    /** Override get() to return a BaseExprNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseExprNode *get() const { return (const BaseExprNode *)ptr; }

    // TODO: implement copy/move semantics!

    TensorType type() const { return get()->type; }
};

template <>
inline RefCount &ref_count<IRExprNode>(const IRExprNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<IRExprNode>(const IRExprNode *t) {
    delete t;
}

template <typename T>
Expr ExprNode<T>::mutate_Expr(Mutator *m) const {
    return m->visit((const T *)this);
}

struct Add : ExprNode<Add> {
    Expr a, b;

    static Expr make(Expr a, Expr b);

    static const ExprEnum node_type = ExprEnum::Add;
};

// Broadcast
struct Bc : ExprNode<Bc> {
    TensorIndex index;
    Expr a;

    static Expr make(TensorIndex index, Expr a);

    static const ExprEnum node_type = ExprEnum::Bc;
};

struct Mul : ExprNode<Mul> {
    Expr a, b;

    static Expr make(Expr a, Expr b);

    static const ExprEnum node_type = ExprEnum::Mul;
};

struct Sum : ExprNode<Sum> {
    TensorIndex index;
    Expr a;

    static Expr make(TensorIndex index, Expr a);
    static Expr make(std::string index, Expr a);

    static const ExprEnum node_type = ExprEnum::Sum;
};

struct Tensor : ExprNode<Tensor> {
    std::string name;

    static Expr make(TensorType type, std::string name);

    static const ExprEnum node_type = ExprEnum::Tensor;
};

Expr operator+(Expr a, Expr b);
Expr operator*(Expr a, Expr b);
Expr bc(std::string idx, Expr a);
Expr sum(std::string idx, Expr a);

} // namespace nacho
