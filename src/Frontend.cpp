#include "Frontend.h"

#include "Error.h"

#include <algorithm>

namespace nacho {

bool compatible_and_broadcast(Expr &a, Expr &b) {
    if (a.type().dtype != b.type().dtype) {
        return false;
    }

    auto a_levels = a.type().format.get_all_levels();
    auto b_levels = b.type().format.get_all_levels();

    if (a_levels != b_levels) {
        std::vector<Level> a_missing;
        std::vector<Level> b_missing;

        std::set_difference(b_levels.begin(), b_levels.end(), a_levels.begin(),
                            a_levels.end(), std::back_inserter(a_missing));

        std::set_difference(a_levels.begin(), a_levels.end(), b_levels.begin(),
                            b_levels.end(), std::back_inserter(b_missing));

        // For each level in a - b, do:
        for (const auto &lvl : a_missing) {
            a = Bc::make(lvl.index, a);
        }

        // For each level in b - a, do:
        for (const auto &lvl : b_missing) {
            b = Bc::make(lvl.index, b);
        }
    }

    return true;
}

Expr Add::make(Expr a, Expr b) {
    internal_assert(compatible_and_broadcast(a, b))
        << "Incompatible types being added"; // << a << " + " << b;

    Add *node = new Add;
    node->type = a.type(); // TODO: might not always be a type?
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

Expr Bc::make(std::string index, Expr a) {
    // Broadcast index dimension.
    Format f = a.type().format; // copy

    auto all = f.get_all_levels();
    auto it = std::find_if(all.begin(), all.end(), [&](const Level &lvl) {
        return lvl.index == index; // matching index means level already exists
    });

    internal_assert(it == all.end()) << "Broadcast error: level '" << index
                                     << "' already exists in tensor format";

    f.unordered_levels.insert({index, LevelFormat::Dense});

    Bc *node = new Bc;
    node->type = TensorType{f, a.type().dtype};
    node->index = std::move(index);
    node->a = std::move(a);
    return node;
}

Expr Mul::make(Expr a, Expr b) {
    internal_assert(compatible_and_broadcast(a, b))
        << "Incompatible types being multiplied"; // << a << " + " << b;

    Mul *node = new Mul;
    node->type = a.type(); // TODO: might not always be a type?
    node->a = std::move(a);
    node->b = std::move(b);
    return node;
}

Expr Sum::make(std::string index, Expr a) {
    // Remove index dimension.
    Format f = a.type().format; // copy

    auto all = f.get_all_levels();
    auto it = std::find_if(all.begin(), all.end(), [&](const Level &lvl) {
        return lvl.index == index; // matching index means level already exists
    });

    if (it == all.end()) {
        throw std::runtime_error("Sum error: level '" + index +
                                 "' does not exist in tensor format");
    }
    // TODO: only one of these should do any erasing
    std::size_t n_erased =
        std::erase_if(f.unordered_levels,
                      [&](const Level &lvl) { return lvl.index == index; });

    n_erased += std::erase_if(
        f.levels, [&](const Level &lvl) { return lvl.index == index; });

    internal_assert(n_erased == 1) << "Sum error: erased '" << index
                                   << "' from " << n_erased << " levels.";

    Sum *node = new Sum;
    node->type = TensorType{f, a.type().dtype};
    node->index = std::move(index);
    node->a = std::move(a);
    return node;
}

Expr Tensor::make(TensorType type, std::string name) {
    internal_assert(type.format.unordered_levels.empty())
        << "Tensor cannot be made with a type with unordered levels: " << name;
    Tensor *node = new Tensor;
    node->type = std::move(type);
    node->name = std::move(name);
    return node;
}

Expr operator+(Expr a, Expr b) { return Add::make(std::move(a), std::move(b)); }

Expr operator*(Expr a, Expr b) { return Mul::make(std::move(a), std::move(b)); }

Expr bc(std::string idx, Expr a) {
    return Bc::make(std::move(idx), std::move(a));
}

Expr sum(std::string idx, Expr a) {
    return Sum::make(std::move(idx), std::move(a));
}

} // namespace nacho
