#include "Simplify.h"

#include "Mutator.h"

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
        for (size_t level = 0; level < node->level; level++) {
            Seq temp = Index::make(node->tensor, node->type, level);
            if (defined.count(temp) == 0) {
                return Empty::make(node->is_sparse);
            }
        }
        // All parents are defined.
        return node;
    }

    // TODO: dedup with indexes() in compute.cpp
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

    CIN visit(const Accumulate *node) override { return handle_write(node); }

    CIN visit(const Assign *node) override { return handle_write(node); }

    cExpr visit(const cTensor *node) override {
        for (size_t level = 0; level < node->type.format.levels.size();
             level++) {
            Seq temp = Index::make(node->name, node->type, level);
            if (defined.count(temp) == 0) {
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
        return T::make(std::move(a), std::move(b));
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

} // namespace

CIN simplify(const std::set<Seq, SeqLessThan> &defined, const CIN &cin) {
    return Simplify(defined).mutate(cin);
}

Seq remove_and_simplify(const Seq &orig, const Seq &remove) {
    Seq repl = Empty::make(remove.get()->is_sparse);
    RemoveAndSimplify mutator(remove, std::move(repl));
    return mutator.mutate(orig);
}

} // namespace nacho
