#pragma once

#include "llir/LLIR.h"

#include <vector>

namespace nacho {
namespace llir {

// Represents a callable (host | device) function.
// Should be able to represent basically any C++ function.
struct Function {
    std::vector<std::string> generics;

    enum Attribute {
        device,
        inline_,
        host,
    };

    std::vector<Attribute> attributes;

    lType ret_type;

    std::string name;

    struct Argument {
        bool mutating;
        lType type;
        std::string name;
    };

    std::vector<Argument> args;

    lStmt body;

    void print(std::ostream &os) const;
};

} // namespace llir
} // namespace nacho
