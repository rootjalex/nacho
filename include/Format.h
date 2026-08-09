#pragma once

#include <cassert>
#include <numeric>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "Error.h"

namespace nacho {

enum class LevelFormat {
    Dense,
    Compressed_non_unique,
    Compressed_unique,
    Singleton_non_unique,
    Singleton_unique,
    MergedCoordinate,
};

class TensorLevelNum {
        public:
        int value;
        explicit TensorLevelNum(int v) : value(v) {}
        int get() const { return value; }
        bool operator<(const TensorLevelNum& other) const {
            return value < other.value;
        }
        bool operator>(const TensorLevelNum& other) const {
            return value > other.value;
        }
        bool operator<=(const TensorLevelNum& other) const {
            return value <= other.value;
        }
        bool operator>=(const TensorLevelNum& other) const {
            return value >= other.value;
        }
        bool operator==(const TensorLevelNum& other) const {
            return value == other.value;
        }
        bool operator!=(const TensorLevelNum& other) const {
            return value != other.value;
        }
        TensorLevelNum operator+(int other) const {
            return TensorLevelNum(value + other);
        }

        TensorLevelNum operator-(int other) const {
            return TensorLevelNum(value - other);
        }

        TensorLevelNum& operator++() {
            ++value;
            return *this;
        }
        TensorLevelNum& operator--() {
            --value;
            return *this;
        }
};

inline TensorLevelNum min(const TensorLevelNum& a, const TensorLevelNum& b) {
            if (b.value < a.value) {
                return b;
            }
            return a;
}



inline std::ostream &operator<<(std::ostream &os, const TensorLevelNum &lvl) {
    return os << lvl.value;
}

inline static const TensorLevelNum BEFORE_FIRST_LEVEL{-1};

inline bool is_sparse_format(const LevelFormat lvl_fmt) {
    return lvl_fmt != LevelFormat::Dense;
}

inline bool is_compressed_format(const LevelFormat lvl_fmt) {
    internal_assert(is_sparse_format(lvl_fmt)) << "Level format " << static_cast<int>(lvl_fmt) << " is not a sparse format.";
    return lvl_fmt == LevelFormat::Compressed_unique || lvl_fmt == LevelFormat::Compressed_non_unique;
}

inline bool is_singleton_format(const LevelFormat lvl_fmt) {
    internal_assert(is_sparse_format(lvl_fmt)) << "Level format " << static_cast<int>(lvl_fmt) << " is not a sparse format.";
    return lvl_fmt == LevelFormat::Singleton_unique || lvl_fmt == LevelFormat::Singleton_non_unique;
}   

inline bool is_unique_format(const LevelFormat lvl_fmt) {
    internal_assert(is_sparse_format(lvl_fmt)) << "Level format " << static_cast<int>(lvl_fmt) << " is not a sparse format.";
    return lvl_fmt == LevelFormat::Compressed_unique || lvl_fmt == LevelFormat::Singleton_unique || lvl_fmt == LevelFormat::MergedCoordinate;
}

inline bool is_merged_format(const LevelFormat lvl_fmt) {
    return lvl_fmt == LevelFormat::MergedCoordinate;
}



// A Tensor index represents the index associated with a specific dimension in a tensor
// It can be either a single index or a merged index (for merged levels)
// eg: in a tensor index expression A_ijk , i and j and k are individual Tensor index
// representing different dimensions. 
// As a special case for COO format, sometimes multiple indices could be merged 
// into single tensor index. This is to enable optimization where all the levels in 
// a COO are considered as a flat single level.
// TODO : Probably need to dedup with Index(Seq.h) as logically
// both are the same thing
struct TensorIndex {
    std::vector<std::string> indices;

    TensorIndex() = default;
    explicit TensorIndex(std::string s) : indices{std::move(s)} {}

    bool is_merged_index() const {
        return indices.size() > 1;
    }

    bool empty() const {
        return indices.empty();
    }

    std::string str(size_t i) const {
       assert(i < indices.size());
       return indices[i];
    }

    std::string str() const {
       return std::accumulate(indices.begin(), indices.end(), std::string(""));
    }

    // Needed for std::set
    bool operator<(const TensorIndex &other) const {
        return str() < other.str();
    }

    // Needed for std::set_difference
    bool operator==(const TensorIndex &other) const {
        return str() == other.str();
    }
};

} // namespace nacho

namespace std {
template <>
struct hash<nacho::TensorIndex> {
    size_t operator()(const nacho::TensorIndex &idx) const noexcept {
        return std::hash<std::string>()(idx.str());
    }
};
} // namespace std

namespace nacho {

// A Level represents a specific level in a tensor format hierarchy.
// It is a TensorIndex with a concrete Level format which
// defines the layout and organization of the tensor data at that level.
struct Level {
    TensorIndex index;
    LevelFormat format;

    Level() = default;
    Level(TensorIndex index, LevelFormat format) : index(std::move(index)), format(format) {}
    Level(std::string idx, LevelFormat format) : index(TensorIndex(std::move(idx))), format(format) {}

    // Needed for std::set
    bool operator<(const Level &other) const {
        return index < other.index ||
               (index == other.index && format < other.format);
    }

    // Needed for std::set_difference
    bool operator==(const Level &other) const {
        return index == other.index && format == other.format;
    }
};
using OrderMap = std::unordered_map<TensorIndex, size_t>;
OrderMap index_order_map(const std::vector<Level> &levels);

// Format represents the layout and organization data for a specific tensor.
struct Format {
    std::vector<Level> levels;
    std::set<Level> bc_levels;

    // User-facing name for this layout, e.g. "CSR". Empty unless named() was called.
    // Used to name the Python class the bindings generate for tensors of this format.
    std::string name;

    // Factories
    static Format ordered(std::vector<Level> lvls) {
        return Format{.levels = std::move(lvls)};
    }
    static Format unordered(std::set<Level> lvls) {
        return Format{.bc_levels = std::move(lvls)};
    }

    // Names this layout and records signature -> name globally, so that formats
    // *derived* from this one (an expression's result format, built fresh by
    // add_formats/mul_formats) can recover the same name from their level structure.
    Format named(std::string layout_name) const;

    bool is_coo() const {
        if (levels.size() < 2)
            return false;
        if(levels[0].format != LevelFormat::Compressed_non_unique)
            return false;
        if(levels[levels.size() - 1].format != LevelFormat::Singleton_unique)
            return false;
        for(int i = 1; i < levels.size() - 1; i++) {
            if(levels[i].format != LevelFormat::Singleton_non_unique)
                return false;
        }
        return true;
    }

    std::set<Level> get_all_levels() const {
        if (levels.empty()) {
            return bc_levels;
        } else {
            std::set<Level> copy = bc_levels;
            copy.insert(levels.cbegin(), levels.cend());
            return copy;
        }
    }
    TensorLevelNum get_prev_sparse_level(TensorLevelNum curr_level) const;
    TensorLevelNum get_next_sparse_level(TensorLevelNum curr_level) const;
    TensorLevelNum get_last_sparse_level() const;
    TensorLevelNum get_level_order(const TensorIndex &idx) const;
    TensorLevelNum get_end_level() const {
        return TensorLevelNum(static_cast<int>(levels.size()));
    }

    LevelFormat lvlfmt_of(const TensorIndex &idx) const;
    bool is_sparse(const TensorIndex &idx) const;
    bool level_exists(const TensorIndex &idx) const;
    bool is_bc_lvl(const TensorIndex &idx) const;
    bool are_all_lvls_dense() const;

};

// Format inference.
Format add_formats(const Format &a, const Format &b);
Format mul_formats(const Format &a, const Format &b);

// A structural key for a layout: one code per level in order, joined by '_'
// (d, cu, cn, su, sn, m). CSR is "d_cu", DCSR "cu_cu", 2D COO "cn_su".
// Independent of index names, so a_ij and b_jk in CSR share a signature.
std::string format_signature(const Format &format);

// The name declared for this layout via named(), directly or by signature.
// Fails with a diagnostic naming the signature if the layout was never named.
const std::string &format_name(const Format &format);

} // namespace nacho
