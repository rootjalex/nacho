#pragma once

#include "Format.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Type.h"
#include "Visitor.h"
#include "Error.h"
#include "Printer.h"
#include "IRFwdDecl.h"
#include "Frontend.h"


namespace nacho {

struct Seq;

enum class SeqEnum {
    Empty,
    Index,
    Intersect,
    Union,
    Universe,
};

using IRSeqNode = IRNode<Seq, SeqEnum>;

// This is necessary to get mutate() to work properly...
struct BaseSeqNode : public IRSeqNode {
    BaseSeqNode(SeqEnum t) : IRSeqNode(t) {}
    virtual Seq mutate_Seq(Mutator *m) const = 0;

    bool is_sparse = false;
    bool is_unique = false;
    bool is_compressed = false;
    bool is_singleton = false;
};

template <typename T>
struct SeqNode : public BaseSeqNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    Seq mutate_Seq(Mutator *m) const override;
    SeqNode() : BaseSeqNode(T::node_type) {}
    ~SeqNode() override = default;
};

struct Seq : public IRHandle<IRSeqNode> {
    /** Make an undefined Seq */
    Seq() = default;

    /** Make an Seq from a concrete Seq node pointer (e.g. Add) */
    Seq(const IRSeqNode *n) : IRHandle<IRSeqNode>(n) {}

    /** Override get() to return a BaseSeqNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseSeqNode *get() const { return (const BaseSeqNode *)ptr; }

    // TODO: implement copy/move semantics!
};

template <>
inline RefCount &ref_count<IRSeqNode>(const IRSeqNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<IRSeqNode>(const IRSeqNode *t) {
    delete t;
}

template <typename T>
Seq SeqNode<T>::mutate_Seq(Mutator *m) const {
    return m->visit((const T *)this);
}

// Only used in lattice construction (simplification)
struct Empty : SeqNode<Empty> {
    // is_sparse -> Empty, !is_sparse -> Full
    static Seq make(bool is_sparse);

    static const SeqEnum node_type = SeqEnum::Empty;
};

// TODO : Probably need to dedup with TensorIndex as logically
// both are the same thing
struct Index : SeqNode<Index> {
    std::string tensor;
    TensorType type;
    size_t level;

    static Seq make(std::string tensor, TensorType type, size_t level);

    static const SeqEnum node_type = SeqEnum::Index;
};

struct Intersect : SeqNode<Intersect> {
    Seq a, b;

    static Seq make(Seq a, Seq b);

    static const SeqEnum node_type = SeqEnum::Intersect;
};

struct Union : SeqNode<Union> {
    Seq a, b;

    static Seq make(Seq a, Seq b);

    static const SeqEnum node_type = SeqEnum::Union;
};

struct Universe : SeqNode<Universe> {
    TensorIndex idx;

    // List of tensors on which this universe represents a broadcast over
    // stored the name, type and the last level before this universe level
    // for the tensor.
    std::vector<std::tuple<std::string, TensorType, size_t>> tensors;

    static Seq make(TensorIndex idx, std::vector<std::tuple<std::string, TensorType, size_t>> tensors);

    static const SeqEnum node_type = SeqEnum::Universe;
};


std::vector<std::string> get_tensors_in_seq(const Seq &seq);


Seq build_seq(const TensorIndex &index, std::vector<TensorIndex> &index_list, const Expr &expr);

bool is_dense(const Seq &seq);

} // namespace nacho
