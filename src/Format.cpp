#include "Format.h"

#include <algorithm>
#include <functional>

#include "Error.h"
#include "Printer.h"

#include <unordered_map>

namespace nacho {

namespace {

std::string level_code(const Level &level) {
    switch (level.format) {
        case LevelFormat::Dense:                 return "d";
        case LevelFormat::Compressed_unique:     return "cu";
        case LevelFormat::Compressed_non_unique: return "cn";
        case LevelFormat::Singleton_unique:      return "su";
        case LevelFormat::Singleton_non_unique:  return "sn";
        // A merged level stands for several dimensions at once, so the code carries how
        // many: a flattened 2-D COO ("m2") is a different layout from a 3-D one ("m3").
        case LevelFormat::MergedCoordinate:      return "m" + std::to_string(level.index.indices.size());
    }
    internal_assert(false) << "Unhandled LevelFormat " << static_cast<int>(level.format);
    return "";
}

// Signature of the flattened form compile_to_cin() rewrites a COO tensor into.
std::string coo_merged_signature(const Format &format) {
    return "m" + std::to_string(format.levels.size());
}

// Declared layout names, keyed by format_signature().
std::unordered_map<std::string, std::string> &named_format_registry() {
    static std::unordered_map<std::string, std::string> registry;
    return registry;
}

} // namespace

std::string format_signature(const Format &format) {
    std::string signature;
    for (const auto &level : format.levels) {
        if (!signature.empty()) {
            signature += "_";
        }
        signature += level_code(level);
    }
    for (const auto &level : format.bc_levels) {
        signature += "_bc";
        signature += level_code(level);
    }
    return signature;
}

namespace {

void register_format_name(const std::string &signature, const std::string &layout_name) {
    auto &registry = named_format_registry();
    auto [entry, inserted] = registry.emplace(signature, layout_name);
    internal_assert(inserted || entry->second == layout_name)
        << "Format signature '" << signature << "' is already named '" << entry->second
        << "', cannot also name it '" << layout_name << "'";
}

} // namespace

Format Format::named(std::string layout_name) const {
    internal_assert(!layout_name.empty()) << "Cannot name a format with an empty string";

    register_format_name(format_signature(*this), layout_name);
    // compile_to_cin() flattens COO tensors to a single merged level. The flattened form
    // has the same buffers, so it keeps the same name.
    if (is_coo()) {
        register_format_name(coo_merged_signature(*this), layout_name);
    }

    Format copy = *this;
    copy.name = std::move(layout_name);
    return copy;
}

const std::string &format_name(const Format &format) {
    if (!format.name.empty()) {
        return format.name;
    }

    const std::string signature = format_signature(format);
    const auto &registry = named_format_registry();
    auto entry = registry.find(signature);
    internal_assert(entry != registry.end())
        << "No name declared for format signature '" << signature << "'. Add .named(\"...\") "
        << "to a Format with this level structure so the bindings can name its Python class.";
    return entry->second;
}

// Map index -> position for fast lookup
OrderMap index_order_map(const std::vector<Level> &levels) {
    OrderMap pos;
    for (size_t i = 0; i < levels.size(); ++i) {
        pos[levels[i].index] = i;
    }
    return pos;
}
namespace {



std::set<TensorIndex> extract_indices(const std::set<Level> &S) {
    std::set<TensorIndex> out;
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
    std::set<TensorIndex> remaining(all.begin(), all.end());
    std::vector<TensorIndex> merged;
    merged.reserve(all.size());
    std::set<TensorIndex> broadcasts;

    auto precedes = [&](const TensorIndex &x, const TensorIndex &y) {
        bool inA = A.count(x) && A.count(y) && A.at(x) < A.at(y);
        bool inB = B.count(x) && B.count(y) && B.at(x) < B.at(y);
        return inA || inB;
    };

    while (!remaining.empty()) {
        bool progress = false;

        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
            const TensorIndex &idx = *it;

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

TensorLevelNum Format::get_prev_sparse_level(TensorLevelNum curr_level) const {
    for (TensorLevelNum i = curr_level - 1; i > BEFORE_FIRST_LEVEL; --i) {
        if (is_sparse_format(levels[i.get()].format)) {
            return TensorLevelNum(i);
        }
    }
    return BEFORE_FIRST_LEVEL;
}

TensorLevelNum Format::get_next_sparse_level(TensorLevelNum curr_level) const {
    for (TensorLevelNum i = curr_level + 1; i < TensorLevelNum(static_cast<int>(levels.size())); ++i) {
        if (is_sparse_format(levels[i.get()].format)) {
            return i;
        }
    }
    return TensorLevelNum(static_cast<int>(levels.size()));
}

TensorLevelNum Format::get_last_sparse_level() const {
    for (int i = static_cast<int>(levels.size()) - 1; i >= 0; --i) {
        if (is_sparse_format(levels[i].format)) {
            return TensorLevelNum(i);
        }
    }
    return BEFORE_FIRST_LEVEL;
}

TensorLevelNum Format::get_level_order(const TensorIndex &idx) const {
    for (size_t i = 0; i < levels.size(); ++i) {
        if (levels[i].index == idx) {
            return TensorLevelNum(static_cast<int>(i));
        }
    }
    return BEFORE_FIRST_LEVEL;
}

LevelFormat Format::lvlfmt_of(const TensorIndex &idx) const {
    auto all = get_all_levels();
    auto it = std::find_if(all.begin(), all.end(),
                           [&](const Level &lv) { return lv.index == idx; });
    internal_assert(it != all.end());
    return it->format;
}

bool Format::is_sparse(const TensorIndex &idx) const {
    return is_sparse_format(lvlfmt_of(idx));
}

bool Format::level_exists(const TensorIndex &idx) const {
    return std::any_of(levels.begin(), levels.end(),
                       [&](const Level &lvl) { return lvl.index == idx; });
}

bool Format::is_bc_lvl(const TensorIndex &idx) const {
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
        // MergedFormat Special Handling
        if(is_merged_format(fa) || is_merged_format(fb)) {
            internal_assert(is_merged_format(fa) && is_merged_format(fb)) << "Merged format is only compatible with itself: " << static_cast<int>(fa) << " vs " << static_cast<int>(fb);
            return LevelFormat::MergedCoordinate;
        }

        // Dense if either is dense
        if (!is_sparse_format(fa) || !is_sparse_format(fb))
            return LevelFormat::Dense;
        else {
            // TODO: This may not be completely right
            if(is_singleton_format(fa) && is_singleton_format(fb)) {
                if(is_unique_format(fa) && is_unique_format(fb)) {
                    return LevelFormat::Singleton_unique;
                } else if(!is_unique_format(fa) && !is_unique_format(fb)) {
                    return LevelFormat::Singleton_non_unique;
                } else {
                    return LevelFormat::Singleton_non_unique;
                }
            } else if(is_compressed_format(fa) && is_compressed_format(fb)) {
                if(is_unique_format(fa) && is_unique_format(fb)) {
                    return LevelFormat::Compressed_unique;
                } else if(!is_unique_format(fa) && !is_unique_format(fb)) {
                    return LevelFormat::Compressed_non_unique;
                } else {
                    return LevelFormat::Compressed_non_unique;
                }
            } else {
                if(is_unique_format(fa) && is_unique_format(fb)) {
                    return LevelFormat::Compressed_unique;
                } else if(!is_unique_format(fa) && !is_unique_format(fb)) {
                    return LevelFormat::Compressed_non_unique;
                } else {
                    return LevelFormat::Compressed_non_unique;
                }
            }

        }
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
        // MergedFormat Special Handling
        if(is_merged_format(fa) || is_merged_format(fb)) {
            internal_assert(is_merged_format(fa) && is_merged_format(fb)) << "Merged format is only compatible with itself: " << static_cast<int>(fa) << " vs " << static_cast<int>(fb);
            return LevelFormat::MergedCoordinate;
        }

        // Sparse if either is sparse
        if (is_sparse_format(fa) || is_sparse_format(fb)) {
            if(!is_sparse_format(fa)){
                return fb;
            }
            if(!is_sparse_format(fb)){
                return fa;
            }
            // TODO: This may not be completely right
            if(is_singleton_format(fa) && is_singleton_format(fb)) {
                if(is_unique_format(fa) && is_unique_format(fb)) {
                    return LevelFormat::Singleton_unique;
                } else if(!is_unique_format(fa) && !is_unique_format(fb)) {
                    return LevelFormat::Singleton_non_unique;
                } else {
                    return LevelFormat::Singleton_unique;
                }
            } else if(is_compressed_format(fa) && is_compressed_format(fb)) {
                if(is_unique_format(fa) && is_unique_format(fb)) {
                    return LevelFormat::Compressed_unique;
                } else if(!is_unique_format(fa) && !is_unique_format(fb)) {
                    return LevelFormat::Compressed_non_unique;
                } else {
                    return LevelFormat::Compressed_unique;
                }
            } else {
                if(is_unique_format(fa) && is_unique_format(fb)) {
                    return LevelFormat::Singleton_unique;
                } else if(!is_unique_format(fa) && !is_unique_format(fb)) {
                    return LevelFormat::Singleton_non_unique;
                } else {
                    return LevelFormat::Singleton_unique;
                }
            }
        }
        return LevelFormat::Dense;
    });
}

bool Format::are_all_lvls_dense() const {
    return std::all_of(levels.begin(), levels.end(),
                       [](const Level &lvl) { return lvl.format == LevelFormat::Dense; });
}

} // namespace nacho
