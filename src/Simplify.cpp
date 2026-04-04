#include "Simplify.h"

#include "Mutator.h"
#include "backend/tensor.h"

#include <algorithm>

namespace nacho {

namespace {

struct SimplifySeq : public Mutator {
    Seq visit(const Intersect *node) override {
        // Recursively mutate children
        Seq rec = Mutator::visit(node);
        node = rec.as<Intersect>();
        internal_assert(node) << rec;

        // Intersection rules do *not* need to treat
        // empty/full as different cases.

        if (node->a.as<Empty>()) {
            return node->a; // empty intersect b = empty
        }

        if (node->b.as<Empty>()) {
            return node->b; // a intersect empty = empty
        }
        return node;
    }

    Seq visit(const Union *node) override {
        // Recursively mutate children
        Seq rec = Mutator::visit(node);
        node = rec.as<Union>();
        internal_assert(node) << rec;

        // Union rules *do* need to treat
        // empty/full as different cases.

        if (const Empty *a = node->a.as<Empty>()) {
            if (a->is_sparse) {
                return node->b; // empty union b = b
            } else {
                return node->a; // full union b = full
            }
        }

        if (const Empty *b = node->b.as<Empty>()) {
            if (b->is_sparse) {
                return node->a; // a union empty = empty
            } else {
                return node->b; // a union full = full
            }
        }
        return node;
    }
};

struct RemoveAndSimplify : public SimplifySeq {
    using SimplifySeq::visit;

    const Seq &remove;
    Seq repl;

    RemoveAndSimplify(const Seq &remove, Seq repl)
        : remove(remove), repl(std::move(repl)) {}

    Seq mutate(const Seq &seq) override {
        if (equals(seq, remove)) {
            return repl;
        }
        return Mutator::mutate(seq);
    }
};

struct Simplify : public SimplifySeq {
    using SimplifySeq::visit;

    Simplify(const std::set<Seq, SeqLessThan> &defined) : defined(defined) {}

    Seq visit(const Index *node) override {
        // std::cout<<"Visiting Index: "<<(Seq)node<<std::endl;
        for (size_t level = 0; level < node->level; level++) {
            Seq temp = Index::make(node->tensor, node->type, level);
            if (defined.count(temp) == 0) {
                // std::cout<<"Index "<<(Seq)temp<<" is not defined\n";
                return Empty::make(true);
            }
        }
        // All parents are defined.
        return node;
    }

    Seq visit(const Universe *node) override {
        // std::cout<<"Visiting Universe: "<<node<<std::endl;
        bool atleast_one_tensor_in_universe_defined = false;
        for (const auto &[name, type, level] : node->tensors) {
            bool tensor_defined = true;
            for(int i = 0; i <= level; i++) {
                Seq temp = Index::make(name, type, i);
                if (defined.count(temp) == 0) {
                    tensor_defined = false;
                    break;
                }
            }
            if(tensor_defined) {
                atleast_one_tensor_in_universe_defined = true;
                break;
            }
        }

        if (!atleast_one_tensor_in_universe_defined) {
            return Empty::make(true);
        }
        // Parent is defined.
        return node;
    }

    CIN visit(const Forall *node) override {
        Seq seq = mutate(node->seq);

        if (seq.is<Empty>()) {
            std::cout << node->seq << " simplified to empty\n";
            return CIN();
        }

        auto temp = defined;
        // For scoping
        auto here = indexes(seq);
        for (const auto &idx : here) {
            defined.insert(idx);
        }

        CIN body = mutate(node->body);

        // Reset `defined`
        defined = temp;

        if (!body.defined()) {
            std::cout << node->body << " simplified to empty\n";
            return body; // propagate undef.
        }

        if (seq.same_as(node->seq) && body.same_as(node->body)) {
            return node;
        }
        return Forall::make(node->idx, std::move(seq), std::move(body));
    }

    template <typename T>
    CIN handle_write(const T *node) {
        cExpr expr = mutate(node->expr);
        if (!expr.defined()) {
            return CIN();
        } else if (expr.same_as(node->expr)) {
            return node;
        }
        return T::make(node->tensor, node->type, std::move(expr));
    }

    CIN visit(const Accumulate *node) override { 
        cExpr expr = mutate(node->expr);
        if (!expr.defined()) {
            return CIN();
        } else if (expr.same_as(node->expr)) {
            return node;
        }
        return Accumulate::make(node->tensor, node->type, node->accumulate_index, std::move(expr));
    }

    CIN visit(const Assign *node) override { 
        cExpr expr = mutate(node->expr);
        if (!expr.defined()) {
            return CIN();
        } else if (expr.same_as(node->expr)) {
            return node;
        }
        return Assign::make(node->tensor, node->type, std::move(expr)); 
    }

    cExpr visit(const cTensor *node) override {
        // std::cout << "Visiting cTensor: " << node << std::endl;
        for (size_t level = 0; level < node->type.format.levels.size();
             level++) {
            Seq temp = Index::make(node->name, node->type, level);
            if (defined.count(temp) == 0) {
                // std::cout << "Index " << temp << " is not defined\n";
                return cExpr(); // used as 0
            }
        }
        // All parents are defined.
        return node;
    }

    template <typename T>
    cExpr handle_dedup(const T *node, cExpr a, cExpr b) {
        if (a.same_as(node->a) && b.same_as(node->b)) {
            return node;
        }
        return T::make(std::move(a), std::move(b), std::move(node->type));
    }

    cExpr visit(const cAdd *node) override {
        cExpr a = mutate(node->a);
        cExpr b = mutate(node->b);
        if (!a.defined()) {
            return b; // 0 + b = b
        } else if (!b.defined()) {
            return a; // a + 0 = a
        }
        return handle_dedup(node, std::move(a), std::move(b));
    }

    cExpr visit(const cMul *node) override {
        cExpr a = mutate(node->a);
        cExpr b = mutate(node->b);
        if (!a.defined()) {
            return a; // 0 * b = 0
        } else if (!b.defined()) {
            return b; // a * 0 = 0
        }
        return handle_dedup(node, std::move(a), std::move(b));
    }

  private:
    std::set<Seq, SeqLessThan> defined;
};

// The intersection/union of dense iterators is a single dense iterator.
struct RemoveDenseCoiteration : public Mutator {
    std::vector<Seq> locators;

    template <typename T>
    Seq handle(const T *node) {
        // Recurse on the children first.
        Seq rec = Mutator::visit(node);
        if (node->is_sparse) {
            return rec;
        }
        // Is a dense op.

        node = rec.as<T>();
        internal_assert(node) << rec;
        const Seq &a = node->a;
        const Seq &b = node->b;

        const Index *ia = a.as<Index>();
        const Index *ib = b.as<Index>();

        bool removed_a = false;
        bool removed_b = false;

        // If a is a dense iterator and b is dense,
        // then locate into a.
        if (ia && !ia->is_sparse && !b.get()->is_sparse) {
            locators.push_back(a);
            removed_a = true;
        }

        // Likewise, if b is a dense iterator and a is dense,
        // then locate into b.
        if (ib && !ib->is_sparse && !a.get()->is_sparse) {
            locators.push_back(b);
            removed_b = true;
        }

        // If we removed both, return a Universe.
        if (removed_a && removed_b) {
            std::vector<std::tuple<std::string, TensorType, size_t>> tensors;
            tensors.emplace_back(ia->tensor, ia->type, ia->level - 1);
            tensors.emplace_back(ib->tensor, ib->type, ib->level - 1);
            return Universe::make(ia->type.format.levels[ia->level].index, std::move(tensors));
        } else if (removed_a) {
            return b;
        } else if (removed_b) {
            return a;
        }

        // Also perform Universe simplification
        if (a.is<Universe>() && b.is<Universe>()) {
            std::vector<std::tuple<std::string, TensorType, size_t>> tensors;
            for (const auto &[name, type, level] : a.as<Universe>()->tensors) {
                tensors.emplace_back(name, type, level);
            }
            for (const auto &[name, type, level] : b.as<Universe>()->tensors) {
                tensors.emplace_back(name, type, level);
            }
            return Universe::make(a.as<Universe>()->idx, std::move(tensors));
        }

        // Nothing to be done.
        return rec;
    }

    Seq visit(const Intersect *node) override { return handle(node); }

    Seq visit(const Union *node) override { return handle(node); }
};

// Remove dense iterators when intersected with a sparse "unique" iterator.
// TODO: If changed from sparse unique to something else, then may also require change in
// get_all_sparse_intersection_levels
struct RemoveDenseLocators : public SimplifySeq {
    using SimplifySeq::visit;

    std::vector<Seq> &locators;
    bool in_sparse_intersection = false;

    RemoveDenseLocators(std::vector<Seq> &locators) : locators(locators) {}

    Seq visit(const Index *node) override {
        if (in_sparse_intersection && !node->is_sparse) {
            locators.push_back(node);
            // make this sparse so it doesn't propagate fullness up!
            return Empty::make(true);
        }
        return node;
    }

    Seq visit(const Intersect *node) override {
        if (in_sparse_intersection) {
            // Already removing dense iterators, keep going.
            return Mutator::visit(node);
        }
        bool a_sparse = node->a.get()->is_sparse;
        bool a_unique = node->a.get()->is_unique;
        bool b_sparse = node->b.get()->is_sparse;
        bool b_unique = node->b.get()->is_unique;

        Seq a, b;

        if (a_sparse && !b_sparse) {
            // b is dense under sparse shadow
            a = mutate(node->a);
            if(a_unique)
                in_sparse_intersection = true;
            b = mutate(node->b);
            if(a_unique)
                in_sparse_intersection = false;
        } else if (b_sparse && !a_sparse) {
            // a is dense under sparse shadow
            if(b_unique)
                in_sparse_intersection = true;
            a = mutate(node->a);
            if(b_unique)
                in_sparse_intersection = false;
            b = mutate(node->b);
        } else {
            // both sparse or both dense: normal recursion
            a = mutate(node->a);
            b = mutate(node->b);
        }

        // Simplifications
        if (a.is<Empty>() || a.is<Universe>()) {
            return b;
        } else if (b.is<Empty>() || b.is<Universe>()) {
            return a;
        }

        if (a.same_as(node->a) && b.same_as(node->b)) {
            return node;
        }

        return Intersect::make(a, b);
    }
};

} // namespace

CIN simplify(const std::set<Seq, SeqLessThan> &defined, const CIN &cin) {
    // std::cout << "Simplifying CIN: " << cin << "\n";
    return Simplify(defined).mutate(cin);
}

Seq remove_and_simplify(const Seq &orig, const Seq &remove) {
    Seq repl = Empty::make(remove.get()->is_sparse);
    RemoveAndSimplify mutator(remove, std::move(repl));
    return mutator.mutate(orig);
}

// returns iterators, locators, and whether there is a universe iterator
std::tuple<std::vector<Seq>, std::vector<Seq>, bool>
partition_iterators_locators(const Seq &seq) {
    // First remove dense coiteration (turns into Universe with locators)
    RemoveDenseCoiteration rm_dense;
    Seq ret = rm_dense.mutate(seq);
    // std::cout << "After removing dense coiteration: " << ret << "\n";
    std::vector<Seq> locators = std::move(rm_dense.locators);
    // Then remove any dense iterators that are intersected with sparse
    // iterators.
    RemoveDenseLocators rm_locators(locators);
    ret = rm_locators.mutate(ret);
    // std::cout << "After removing dense locators: " << ret << "\n";

    // Get any iterators that are left.
    std::vector<Seq> iterators = indexes(ret);


    // if sequence is dense and there is no dense iterator
    // a universe iterator is required.
    if(!seq.get()->is_sparse) {
        if(!std::any_of(iterators.begin(), iterators.end(), [](const Seq& s) {
            return !s.get()->is_sparse;
        })) {
            return {iterators, locators, true};
        }
    }

    return {iterators, locators, false};
}

std::vector<Seq> get_dense_locators(const Seq &seq) {
    std::vector<Seq> locators;
    RemoveDenseLocators rm_locators(locators);
    rm_locators.mutate(seq);
    return locators;
}

std::vector<Seq> indexes(const Seq &seq) {
    struct GetIndexes : public Visitor {
        std::vector<Seq> indexes;
        std::set<Seq, SeqLessThan> visited;
        void visit(const Index *node) override {
            if (!visited.count(node)) {
                indexes.push_back(node);
                visited.insert(node);
            }
        }
    };
    GetIndexes getter;
    seq.accept(&getter);
    return getter.indexes;
}

} // namespace nacho
