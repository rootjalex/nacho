#pragma once

#include <set>
#include <string>
#include <vector>

namespace nacho {

enum class LevelFormat {
    Dense,
    Compressed,
    // TODO: Coordinate, Hash, etc (Change below function too when adding more sparse types)
};

inline bool is_sparse_format(const LevelFormat lvl_fmt) {
    return lvl_fmt == LevelFormat::Compressed;
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
    int get_prev_sparse_level(int curr_level) const;   
    int get_next_sparse_level(int curr_level) const;
    int get_last_sparse_level() const;
    int get_level_order(const std::string &idx) const;

    LevelFormat lvlfmt_of(const std::string &idx) const;
    bool is_sparse(const std::string &idx) const;
    bool level_exists(const std::string &idx) const;
    bool is_bc_lvl(const std::string &idx) const;
};

// Format inference.
Format add_formats(const Format &a, const Format &b);
Format mul_formats(const Format &a, const Format &b);

} // namespace nacho
