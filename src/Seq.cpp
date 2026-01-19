#include "Seq.h"

#include <algorithm>

namespace nacho {

Seq Empty::make(bool is_sparse) {
    // TODO: statically allocate two and return base on is_sparse
    Empty *node = new Empty;
    node->is_sparse = is_sparse;
    return node;
}

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


Seq simplify_seq(const Seq &seq) {
    struct Simplify : public Mutator {
        Seq visit(const Intersect *node) override {
            Seq a = mutate(node->a);
            Seq b = mutate(node->b);

            // a ∩ U = a
            if (b.is<Universe>() || is_dense(b)) {
                return a;
            }

            // U ∩ b = b
            if (a.is<Universe>() || is_dense(a)) {
                return b;
            }

            if (a.same_as(node->a) && b.same_as(node->b)) {
                return node;
            }
            return Intersect::make(std::move(a), std::move(b));
        }
    };
    return Simplify().mutate(seq);
}


struct BuildSeq : public Visitor {
    Seq seq;

    const std::string &index;

    BuildSeq(const std::string &index) : index(index) {}

    template <typename S, typename T>
    void visit_binop(const T *node) {
        seq = Seq();
        node->a.accept(this);
        auto a = std::move(seq);
        node->b.accept(this);
        auto b = std::move(seq);
        seq = S::make(std::move(a), std::move(b));
    }

    void visit(const Add *node) { visit_binop<Union>(node); }

    void visit(const Bc *node) {
        if (index == node->index) {
            seq = Universe::make(index);
        } else {
            node->a.accept(this);
        }
    }

    void visit(const Mul *node) { visit_binop<Intersect>(node); }

    // Default Sum behavior is fine

    void visit(const Tensor *node) {
        // Find index of `index` in levels.
        const auto &levels = node->type.format.levels;

        auto it =
            std::find_if(levels.begin(), levels.end(),
                         [&](const Level &lvl) { return lvl.index == index; });
        internal_assert(it != levels.end())
            << "Index: " << index << " not found in: " << Expr(node);
        size_t level = std::distance(levels.begin(), it);

        seq = Index::make(node->name, node->type, level);
    }
};

bool is_dense(const Seq &seq) {
    struct IsDense : public Visitor {
        bool dense = false;

        void visit(const Index *node) override {
            dense =
                node->type.format.levels[node->level].format == LevelFormat::Dense;
        }

        void visit(const Intersect *node) override {
            node->a.accept(this);
            if (!dense) {
                return;
            }
            // implicit &&
            node->b.accept(this);
        }

        void visit(const Union *node) override {
            node->a.accept(this);
            if (dense) {
                return;
            }
            // implicit ||
            node->b.accept(this);
        }

        void visit(const Universe *) override { dense = true; }
    };

    IsDense checker;
    seq.accept(&checker);
    return checker.dense;
}


Seq build_seq(const std::string &index, const Expr &expr) {
    BuildSeq builder(index);
    expr.accept(&builder);
    // TODO: break this out into a simplify or optimize pass?
    //return simplify_seq(builder.seq);
    return builder.seq;
}



std::vector<std::string> get_tensors_in_seq(const Seq &seq) {
    struct TensorCollector : public Visitor {
        std::vector<std::string> tensors;

        void visit(const Index *node) override {
            if (std::find(tensors.begin(), tensors.end(), node->tensor) ==
                tensors.end()) {
                tensors.push_back(node->tensor);
            }
        }
    };

    TensorCollector collector;
    seq.accept(&collector);
    return collector.tensors;
}

} // namespace nacho

