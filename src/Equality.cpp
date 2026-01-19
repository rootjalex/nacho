#include "Equality.h"

namespace nacho {

enum class Cmp { Less, Equals, Greater };

namespace {

// Works for e.g., string, int, etc.
template <typename T>
Cmp compare_prim(const T &t0, const T &t1) {
    if (t0 == t1) {
        return Cmp::Equals;
    } else if (t0 < t1) {
        return Cmp::Less;
    } else {
        return Cmp::Greater;
    }
}

Cmp compare(const Seq &a, const Seq &b) {
    if (Cmp c = compare_prim(a.node_type(), b.node_type()); c != Cmp::Equals) {
        return c;
    }

    // Must be the same node type.

    switch (a.node_type()) {
    case SeqEnum::Empty: {
        const Empty *a_emp = a.as<Empty>();
        const Empty *b_emp = b.as<Empty>();
        return compare_prim(a_emp->is_sparse, b_emp->is_sparse);
    }
    case SeqEnum::Index: {
        const Index *a_idx = a.as<Index>();
        const Index *b_idx = b.as<Index>();

        if (Cmp c = compare_prim(a_idx->tensor, b_idx->tensor);
            c != Cmp::Equals) {
            return c;
        }

        // Skip tensor type comparison for now, assume name has a unique type.
        /*
        if (Cmp c = compare(a_idx->type, b_idx->type); c != Cmp::Equals) {
            return c;
        }
        */

        return compare_prim(a_idx->level, b_idx->level);
    }
    case SeqEnum::Intersect: {
        const Intersect *a_int = a.as<Intersect>();
        const Intersect *b_int = b.as<Intersect>();

        if (Cmp c = compare(a_int->a, b_int->a); c != Cmp::Equals) {
            return c;
        }

        return compare(a_int->b, b_int->b);
    }
    case SeqEnum::Union: {
        const Union *a_uni = a.as<Union>();
        const Union *b_uni = b.as<Union>();

        if (Cmp c = compare(a_uni->a, b_uni->a); c != Cmp::Equals) {
            return c;
        }

        return compare(a_uni->b, b_uni->b);
    }
    case SeqEnum::Universe: {
        const Universe *a_uni = a.as<Universe>();
        const Universe *b_uni = b.as<Universe>();
        return compare_prim(a_uni->idx, b_uni->idx);
    }
    }
}

} // namespace

bool equals(const Seq &a, const Seq &b) { return compare(a, b) == Cmp::Equals; }

bool SeqLessThan::operator()(const Seq &a, const Seq &b) const {
    return compare(a, b) == Cmp::Less;
}

} // namespace nacho
