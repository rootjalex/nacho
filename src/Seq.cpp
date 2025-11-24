#include "Seq.h"

#include "Error.h"
#include "Printer.h"

#include <algorithm>

namespace nacho {

Seq Index::make(std::string tensor, Format format, size_t level) {
    internal_assert(!tensor.empty()) << "Index with empty tensor";
    internal_assert(level < format.levels.size())
        << "Cannot make Index of " << tensor << " with format: " << format
        << " of level: " << level;
    Index *node = new Index;
    node->tensor = std::move(tensor);
    node->format = std::move(format);
    node->level = level;
    return node;
}

Seq Intersect::make(Seq a, Seq b) {
    internal_assert(a.defined() && b.defined())
        << "Intersection of undefined: " << a << " \\cap " << b;
    Intersect *node = new Intersect;
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

Seq Union::make(Seq a, Seq b) {
    internal_assert(a.defined() && b.defined())
        << "Union undefined: " << a << " \\cup " << b;
    Union *node = new Union;
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

Seq Universe::make(std::string idx) {
    internal_assert(!idx.empty()) << "Universe with empty idx";
    Universe *node = new Universe;
    node->idx = std::move(idx);
    return node;
}

} // namespace nacho
