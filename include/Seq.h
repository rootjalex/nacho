#pragma once

#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Visitor.h"

#include "Format.h"
#include "Type.h"

namespace nacho {

struct Seq;

enum class SeqEnum {
    Index,
    Intersect,
    Union,
    Universe,
};

using IRSeqNode = IRNode<Seq, SeqEnum>;

// This is necessary to get mutate() to work properly...
struct BaseSeqNode : public IRSeqNode {
    BaseSeqNode(SeqEnum t) : IRSeqNode(t) {}
    // virtual Seq mutate_Seq(Mutator *m) const = 0;
};

template <typename T>
struct SeqNode : public BaseSeqNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    // Seq mutate_Seq(Mutator *m) const override;
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

// template <typename T>
// Seq SeqNode<T>::mutate_Seq(Mutator *m) const {
//     return m->visit((const T *)this);
// }

struct Index : SeqNode<Index> {
    std::string tensor;
    Format format;
    size_t level;

    static Seq make(std::string tensor, Format format, size_t level);

    static const SeqEnum node_type = SeqEnum::Index;
};

struct Intersect : SeqNode<Intersect> {
    Seq a, b;

    static Seq make(Seq a, Seq b);

    static const SeqEnum node_type = SeqEnum::Intersect;
};

struct Union : SeqNode<Intersect> {
    Seq a, b;

    static Seq make(Seq a, Seq b);

    static const SeqEnum node_type = SeqEnum::Union;
};

struct Universe : SeqNode<Universe> {
    std::string idx;

    static Seq make(std::string idx);

    static const SeqEnum node_type = SeqEnum::Universe;
};

} // namespace nacho
