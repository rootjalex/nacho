#include "Seq.h"

#include "Error.h"
#include "Printer.h"

#include <algorithm>

namespace nacho {

Seq Index::make(std::string tensor, TensorType type, size_t level) {
    internal_assert(!tensor.empty()) << "Index with empty tensor";
    internal_assert(level < type.format.levels.size())
        << "Cannot make Index of " << tensor << " with format: " << type.format
        << " of level: " << level;
    Index *node = new Index;
    node->tensor = std::move(tensor);
    node->type = std::move(type);
    node->level = level;

    node->is_sparse = is_sparse_format(node->type.format.lvlfmt_of(
        node->type.format.levels[level].index));

    return node;
}

Seq Intersect::make(Seq a, Seq b) {
    internal_assert(a.defined() && b.defined())
        << "Intersection of undefined: " << a << " \\cap " << b;
    Intersect *node = new Intersect;
    node->a = std::move(a);
    node->b = std::move(b);

    // an intersection sequence is sparse if either a or b are sparse
    node->is_sparse = node->a.get()->is_sparse || node->b.get()->is_sparse;

    return node;
}

Seq Union::make(Seq a, Seq b) {
    internal_assert(a.defined() && b.defined())
        << "Union undefined: " << a << " \\cup " << b;
    Union *node = new Union;
    node->a = std::move(a);
    node->b = std::move(b);

    // a union sequence is sparse if both a and b are sparse
    node->is_sparse = node->a.get()->is_sparse && node->b.get()->is_sparse;

    return node;
}

Seq Universe::make(std::string idx) {
    internal_assert(!idx.empty()) << "Universe with empty idx";
    Universe *node = new Universe;
    node->idx = std::move(idx);

    // a universe sequence is not sparse
    node->is_sparse = false;
    return node;
}

} // namespace nacho
