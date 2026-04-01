#pragma once

#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace nacho {

enum class LevelFormat {
    Dense,
    Compressed_non_unique,
    Compressed_unique,
    Singleton_non_unique,
    Singleton_unique,
    // TODO: Coordinate, Hash, etc (Change below function too when adding more sparse types)
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
    return lvl_fmt == LevelFormat::Compressed_unique || lvl_fmt == LevelFormat::Compressed_non_unique;
}

inline bool is_singleton_format(const LevelFormat lvl_fmt) {
    return lvl_fmt == LevelFormat::Singleton_unique || lvl_fmt == LevelFormat::Singleton_non_unique;
}   

inline bool is_unique_format(const LevelFormat lvl_fmt) {
    return lvl_fmt == LevelFormat::Compressed_unique || lvl_fmt == LevelFormat::Singleton_unique;
}

struct Level {
    std::string index;
    LevelFormat format;

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
using OrderMap = std::unordered_map<std::string, size_t>;
OrderMap index_order_map(const std::vector<Level> &levels);
struct Format {
    std::vector<Level> levels;
    std::set<Level> bc_levels;

    // Factories
    static Format ordered(std::vector<Level> lvls) {
        return Format{.levels = std::move(lvls)};
    }
    static Format unordered(std::set<Level> lvls) {
        return Format{.bc_levels = std::move(lvls)};
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
    TensorLevelNum get_level_order(const std::string &idx) const;
    TensorLevelNum get_end_level() const {
        return TensorLevelNum(static_cast<int>(levels.size()));
    }

    LevelFormat lvlfmt_of(const std::string &idx) const;
    bool is_sparse(const std::string &idx) const;
    bool level_exists(const std::string &idx) const;
    bool is_bc_lvl(const std::string &idx) const;
    bool are_all_lvls_dense() const;

};

// Format inference.
Format add_formats(const Format &a, const Format &b);
Format mul_formats(const Format &a, const Format &b);

} // namespace nacho
