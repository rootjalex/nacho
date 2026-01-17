#include "Format.h"

#include "Error.h"
#include "Printer.h"

#include <unordered_map>

namespace nacho {

// Map index -> position for fast lookup
OrderMap index_order_map(const std::vector<Level> &levels) {
    OrderMap pos;
    for (size_t i = 0; i < levels.size(); ++i) {
        pos[levels[i].index] = i;
    }
    return pos;
}
namespace {



std::set<std::string> extract_indices(const std::set<Level> &S) {
    std::set<std::string> out;
    for (auto &lvl : S)
        out.insert(lvl.index);
    return out;
}

// Formats are incompatible if their index sets are not equal *or*
// if their ordered indices produce discordant traversals.
bool compatible_formats(const Format &a, const Format &b) {
    auto a_order = index_order_map(a.levels);
    auto b_order = index_order_map(b.levels);

    // Ordering violation check
    auto violates_order = [&](const std::vector<Level> &order,
                              const OrderMap &L0_order,
                              const OrderMap &L1_order) {
        // Find any pair (x, y) s.t. x is before y in L0 but x is after y in L1.
        for (size_t i = 0; i + 1 < order.size(); ++i) {
            const auto &lhs = order[i].index;
            for (size_t j = i + 1; j < order.size(); ++j) {
                const auto &rhs = order[j].index;

                // Only meaningful if both indices also appear in L1
                auto x = L1_order.find(lhs);
                auto y = L1_order.find(rhs);
                if (x == L1_order.end() || y == L1_order.end())
                    continue;

                // x appears after y in L1 ordering
                if (x->second > y->second)
                    return true;
            }
        }
        return false;
    };

    // A ordering must match B ordering
    if (violates_order(a.levels, a_order, b_order))
        return false;

    // B ordering must match A ordering
    if (violates_order(b.levels, b_order, a_order))
        return false;

    // Check indices match up (requires broadcasting to have happened already).
    auto a_all = extract_indices(a.get_all_levels());
    auto b_all = extract_indices(b.get_all_levels());

    if (a_all != b_all)
        return false;

    return true;
}

Format
make_format(const Format &a, const Format &b,
            std::function<LevelFormat(LevelFormat, LevelFormat)> combine) {
    internal_assert(compatible_formats(a, b))
        << "Incompatible formats: " << a << " and " << b;

    auto A = index_order_map(a.levels);
    auto B = index_order_map(b.levels);

    auto all = extract_indices(a.get_all_levels());

    // Topological merge of ordering constraints
    std::set<std::string> remaining(all.begin(), all.end());
    std::vector<std::string> merged;
    merged.reserve(all.size());
    std::set<std::string> broadcasts;

    auto precedes = [&](const std::string &x, const std::string &y) {
        bool inA = A.count(x) && A.count(y) && A.at(x) < A.at(y);
        bool inB = B.count(x) && B.count(y) && B.at(x) < B.at(y);
        return inA || inB;
    };

    while (!remaining.empty()) {
        bool progress = false;

        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
            const std::string &idx = *it;

            bool has_pred = false;
            for (auto &other : remaining) {
                if (other == idx)
                    continue;
                if (precedes(other, idx)) {
                    has_pred = true;
                    break;
                }
            }

            if (!has_pred) {
                // if idx is broadcasted in A and B then it should be put in
                // broadcasts, not merged.
                if (a.is_bc_lvl(idx) && b.is_bc_lvl(idx)) {
                    broadcasts.insert(idx);
                } else {
                    merged.push_back(idx);
                }
                remaining.erase(it);
                progress = true;
                break;
            }
        }

        internal_assert(progress)
            << "Cycle while merging formats " << a << " and " << b;
    }

    // Build final ordered levels
    Format out;
    out.levels.reserve(merged.size());

    for (auto &idx : merged) {
        LevelFormat fa = a.lvlfmt_of(idx);
        LevelFormat fb = b.lvlfmt_of(idx);
        out.levels.push_back(Level{idx, combine(fa, fb)});
    }

    for (auto &idx : broadcasts) {
        LevelFormat fa = a.lvlfmt_of(idx);
        LevelFormat fb = b.lvlfmt_of(idx);
        out.bc_levels.insert(Level{idx, combine(fa, fb)});
    }

    return out;
}

} // namespace

int Format::get_prev_sparse_level(int curr_level) const {
    for (int i = curr_level - 1; i >= 0; --i) {
        if (is_sparse_format(levels[i].format)) {
            return i;
        }
    }
    return -1;
}

int Format::get_next_sparse_level(int curr_level) const {
    for (size_t i = curr_level + 1; i < levels.size(); ++i) {
        if (is_sparse_format(levels[i].format)) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(levels.size());
}

int Format::get_last_sparse_level() const {
    for (int i = static_cast<int>(levels.size()) - 1; i >= 0; --i) {
        if (is_sparse_format(levels[i].format)) {
            return i;
        }
    }
    return -1;
}

int Format::get_level_order(const std::string &idx) const {
    for (size_t i = 0; i < levels.size(); ++i) {
        if (levels[i].index == idx) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

LevelFormat Format::lvlfmt_of(const std::string &idx) const {
    auto all = get_all_levels();
    auto it = std::find_if(all.begin(), all.end(),
                           [&](const Level &lv) { return lv.index == idx; });
    internal_assert(it != all.end());
    return it->format;
}

bool Format::level_exists(const std::string &idx) const {
    return std::any_of(levels.begin(), levels.end(),
                       [&](const Level &lvl) { return lvl.index == idx; });
}

bool Format::is_bc_lvl(const std::string &idx) const {
    return std::any_of(bc_levels.begin(), bc_levels.end(),
                       [&](const Level &lvl) { return lvl.index == idx; });
}

Format add_formats(const Format &a, const Format &b) {
    internal_assert(compatible_formats(a, b))
        << "Incompatible formats: " << a << " + " << b;

    // Now build a format z based on the following rules:
    // Any ordering in a or b must exist in the ordered `levels` of z
    // Any index that is dense in *either* a or b must be dense in z
    // Any index that is sparse in *both* a and b must be sparse in z.

    return make_format(a, b, [](LevelFormat fa, LevelFormat fb) {
        // Dense if either is dense
        if (fa == LevelFormat::Dense || fb == LevelFormat::Dense)
            return LevelFormat::Dense;
        return LevelFormat::Compressed;
    });
}

Format mul_formats(const Format &a, const Format &b) {
    internal_assert(compatible_formats(a, b))
        << "Incompatible formats: " << a << " * " << b;

    // Now build a format z based on the following rules:
    // Any ordering in a or b must exist in the ordered `levels` of z
    // Any index that is sparse in *either* a or b must be sparse in z
    // Any index that is dense in *both* a and b must be dense in z.

    return make_format(a, b, [](LevelFormat fa, LevelFormat fb) {
        // Sparse if either is sparse
        if (fa == LevelFormat::Compressed || fb == LevelFormat::Compressed)
            return LevelFormat::Compressed;
        return LevelFormat::Dense;
    });
}

} // namespace nacho
