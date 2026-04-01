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
    node->is_unique = node->is_sparse && is_unique_format(node->type.format.lvlfmt_of(
        node->type.format.levels[level].index));
    node->is_compressed = node->is_sparse && is_compressed_format(node->type.format.lvlfmt_of(
        node->type.format.levels[level].index));
    node->is_singleton = node->is_sparse && is_singleton_format(node->type.format.lvlfmt_of(
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
    node->is_compressed = node->a.get()->is_compressed && node->b.get()->is_compressed;
    node->is_singleton = node->a.get()->is_singleton || node->b.get()->is_singleton;
    node->is_unique = node->a.get()->is_unique || node->b.get()->is_unique;

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
    node->is_compressed = node->a.get()->is_compressed || node->b.get()->is_compressed;
    node->is_singleton = node->a.get()->is_singleton && node->b.get()->is_singleton;
    node->is_unique = node->a.get()->is_unique && node->b.get()->is_unique;
    
    return node;
}

Seq Universe::make(std::string idx, std::vector<std::tuple<std::string, TensorType, size_t>> tensors) {
    internal_assert(!idx.empty()) << "Universe with empty idx";
    internal_assert(!tensors.empty()) << "Universe should represent a broadcast on atleast 1 tensor";
    Universe *node = new Universe;
    node->idx = std::move(idx);
    node->tensors = std::move(tensors);

    // a universe sequence is not sparse
    node->is_sparse = false;
    return node;
}


struct BuildSeq : public Visitor {
    Seq seq;

    const std::string &index;
    std::vector<std::string> &index_list;
    bool is_under_bc = false;
    std::vector<std::tuple<std::string, TensorType, size_t>>  bc_tensors;

    BuildSeq(const std::string &index, std::vector<std::string> &index_list)
        : index(index), index_list(index_list) {}

    template <typename S, typename T>
    void visit_binop(const T *node) {
        if(is_under_bc) {
            node->a.accept(this);
            node->b.accept(this);
            return;
        }
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
            is_under_bc = true;
            node->a.accept(this);
            seq = Universe::make(index, bc_tensors);
            is_under_bc = false;
        } else {
            node->a.accept(this);
        }
    }

    void visit(const Mul *node) { visit_binop<Intersect>(node); }

    // Default Sum behavior is fine

    void visit(const Tensor *node) {
        if(is_under_bc) {
            auto it = std::find_if(index_list.begin(), index_list.end(),
                         [&](const std::string &idx) { return idx == index; });
            internal_assert(it != index_list.end())
            << "Index: " << index << " not found";
            int loop_num = std::distance(index_list.begin(), it);
            // Find the last level before this universe level for this tensor.
            std::string prev_level_idx;
            for(int prev_level = loop_num - 1; prev_level >= 0; prev_level--) {
                if(node->type.format.level_exists(index_list[prev_level])) {
                    prev_level_idx = index_list[prev_level];
                    break;
                }
            }
            if(prev_level_idx.empty()) {
                bc_tensors.push_back(std::make_tuple(node->name, node->type, -1));
            } else {
                const auto &levels = node->type.format.levels;
                auto it =
                std::find_if(levels.begin(), levels.end(),
                         [&](const Level &lvl) { return lvl.index == prev_level_idx; });
                internal_assert(it != levels.end())
                    << "Index: " << prev_level_idx << " not found";
                size_t level = std::distance(levels.begin(), it);
                bc_tensors.push_back(std::make_tuple(node->name, node->type, level));
            }
            return;
        }

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


Seq build_seq(const std::string &index, std::vector<std::string> &index_list, const Expr &expr) {
    BuildSeq builder(index, index_list);
    expr.accept(&builder);
    // TODO: break this out into a simplify or optimize pass?
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

