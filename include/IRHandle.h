// IRHandle.h
#pragma once

#include "IntrusivePtr.h"

#include "Visitor.h"

#include <cassert>

namespace nacho {

/** IR nodes are passed around opaque handles to them. This is a
   base class for those handles. It manages the reference count,
   and dispatches visitors. */
template <typename IRNode>
struct IRHandle : public IntrusivePtr<const IRNode> {
    IRHandle() = default;

    IRHandle(const IRNode *p) : IntrusivePtr<const IRNode>(p) {}

    /** Dispatch to the correct visitor method for this node. E.g. if
     * this node is actually an Add node, then this will call
     * Visitor::visit(const Add *) */
    void accept(Visitor *v) const {
        assert(this->ptr);
        this->ptr->accept(v);
    }

    // Downcast this ir node to its actual type (e.g. Add, or Select). This
    // returns nullptr if the node is not of the requested type. This is similar
    // to `llvm::dyn_cast_or_null` and C++'s `dynamic_cast`. For example,
    //
    //      if (const Add *add = node->as<Add>()) {
    //        // This is an add node.
    //      }
    //
    template <typename T>
    const T *as() const {
        if (this->ptr && node_type() == T::node_type) {
            return (const T *)this->ptr;
        }
        return nullptr;
    }

    // Returns whether this IR is one of type `Ts`. For example,
    //
    //      // Executes if op is one of these two types.
    //      if (op.is<ir::Lambda, ir::Function>()) { ... }
    //
    template <typename... Ts>
    bool is() const {
        if (this->ptr == nullptr)
            return false;
        if (((node_type() == Ts::node_type) || ...)) {
            return true;
        }
        return false;
    }

    inline typename IRNode::TypeEnum node_type() const {
        return this->ptr->node_type_;
    }
};

} // namespace nacho
