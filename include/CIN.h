#pragma once

#include "Format.h"
#include "Frontend.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Seq.h"
#include "Type.h"
#include "Visitor.h"

namespace nacho {

struct CIN;
struct cExpr;

// Scalar math
enum class cExprEnum {
    cAdd,
    cMul,
    cTensor,
};

// Concrete Index Notation
enum class CINEnum {
    Accumulate,
    Assign,
    Forall,
    Sequence,
    Where,
    CalculateWork,
};

using IRcExprNode = IRNode<cExpr, cExprEnum>;

using IRCINNode = IRNode<CIN, CINEnum>;

// This is necessary to get mutate() to work properly...
struct BasecExprNode : public IRcExprNode {
    BasecExprNode(cExprEnum t) : IRcExprNode(t) {}
    virtual cExpr mutate_cExpr(Mutator *m) const = 0;
};

struct BaseCINNode : public IRCINNode {
    BaseCINNode(CINEnum t) : IRCINNode(t) {}
    virtual CIN mutate_CIN(Mutator *m) const = 0;
};

template <typename T>
struct cExprNode : public BasecExprNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    cExpr mutate_cExpr(Mutator *m) const override;
    cExprNode() : BasecExprNode(T::node_type) {}
    ~cExprNode() override = default;
};

template <typename T>
struct CINNode : public BaseCINNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    CIN mutate_CIN(Mutator *m) const override;
    CINNode() : BaseCINNode(T::node_type) {}
    ~CINNode() override = default;
};

struct cExpr : public IRHandle<IRcExprNode> {
    /** Make an undefined cExpr */
    cExpr() = default;

    /** Make a cExpr from a concrete cExpr node pointer (e.g. Add) */
    cExpr(const IRcExprNode *n) : IRHandle<IRcExprNode>(n) {}

    /** Override get() to return a BasecExprNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BasecExprNode *get() const { return (const BasecExprNode *)ptr; }

    // TODO: implement copy/move semantics!
};

template <>
inline RefCount &ref_count<IRcExprNode>(const IRcExprNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<IRcExprNode>(const IRcExprNode *t) {
    delete t;
}

template <typename T>
cExpr cExprNode<T>::mutate_cExpr(Mutator *m) const {
    return m->visit((const T *)this);
}

struct CIN : public IRHandle<IRCINNode> {
    /** Make an undefined CIN */
    CIN() = default;

    /** Make an CIN from a concrete CIN node pointer (e.g. Forall) */
    CIN(const IRCINNode *n) : IRHandle<IRCINNode>(n) {}

    /** Override get() to return a BaseCINNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseCINNode *get() const { return (const BaseCINNode *)ptr; }

    // TODO: implement copy/move semantics!
};

template <>
inline RefCount &ref_count<IRCINNode>(const IRCINNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<IRCINNode>(const IRCINNode *t) {
    delete t;
}

template <typename T>
CIN CINNode<T>::mutate_CIN(Mutator *m) const {
    return m->visit((const T *)this);
}

// Scalar math
struct cAdd : cExprNode<cAdd> {
    cExpr a, b;
    TensorType type;
    
    static cExpr make(cExpr a, cExpr b, TensorType type);

    static const cExprEnum node_type = cExprEnum::cAdd;
};

struct cMul : cExprNode<cMul> {
    cExpr a, b;
    TensorType type;

    static cExpr make(cExpr a, cExpr b, TensorType type);

    static const cExprEnum node_type = cExprEnum::cMul;
};

struct cTensor : cExprNode<cTensor> {
    TensorType type;
    std::string name;

    static cExpr make(TensorType type, std::string name);

    static const cExprEnum node_type = cExprEnum::cTensor;
};

// Concrete Index Notation
struct Accumulate : CINNode<Accumulate> {
    std::string tensor;
    TensorType type;
    std::vector<TensorIndex> accumulate_indices; // the indices we are accumulating over.
    cExpr expr;

    static CIN make(std::string tensor, TensorType type, std::vector<TensorIndex> accumulate_indices, cExpr expr);

    static const CINEnum node_type = CINEnum::Accumulate;
};

struct Assign : CINNode<Assign> {
    std::string tensor;
    TensorType type;
    cExpr expr; // no sums or broadcasts!

    static CIN make(std::string tensor, TensorType type, cExpr expr);

    static const CINEnum node_type = CINEnum::Assign;
};

struct Forall : CINNode<Forall> {
    TensorIndex idx;
    Seq seq;
    CIN body;

    static CIN make(TensorIndex idx, Seq seq, CIN body);

    static const CINEnum node_type = CINEnum::Forall;
};

struct Sequence : CINNode<Sequence> {
    std::vector<CIN> stmts;

    static CIN make(std::vector<CIN> stmts);

    static const CINEnum node_type = CINEnum::Sequence;
};

struct Where : CINNode<Where> {
    std::string temp;
    TensorType temp_type;
    CIN producer; // writes to temp
    CIN consumer; // reads from temp.

    static CIN make(std::string temp, TensorType temp_type, CIN consumer,
                    CIN producer);

    static const CINEnum node_type = CINEnum::Where;
};

// CalculateWork is used as intermediate node, when we have multiple sparse intersections. 
// This node is placed belore the current sparse intersection that is being evaluated on
// The orginal body is just embeded inside a caculate work node.
struct CalculateWork :  CINNode<CalculateWork> {
    CIN body;
    static CIN make(CIN body);
    static const CINEnum node_type = CINEnum::CalculateWork;
};

} // namespace nacho
