#pragma once

#include <set>
#include <string>
#include <vector>

namespace nacho {

enum class LevelFormat {
    Dense,
    Compressed,
    // TODO: Coordinate, Hash, etc
};

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

struct Format {
    std::vector<Level> levels;
    std::set<Level> unordered_levels;

    // Factories
    static Format ordered(std::vector<Level> lvls) {
        return Format{.levels = std::move(lvls)};
    }
    static Format unordered(std::set<Level> lvls) {
        return Format{.unordered_levels = std::move(lvls)};
    }

    std::set<Level> get_all_levels() const {
        if (levels.empty()) {
            return unordered_levels;
        } else {
            std::set<Level> copy = unordered_levels;
            copy.insert(levels.cbegin(), levels.cend());
            return copy;
        }
    }
};

} // namespace nacho
