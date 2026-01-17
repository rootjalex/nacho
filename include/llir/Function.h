#pragma once

#include "llir/LLIR.h"

#include <vector>

namespace nacho {
namespace llir {

// Represents a callable (host | device) function.
// Should be able to represent basically any C++ function.
struct Function : lStmtNode<Function> {
    std::vector<std::string> generics;

    enum Attribute {
        global,
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

    static lStmt make(std::vector<std::string> generics, std::vector<Attribute> attributes, std::vector<Argument> args, lType ret_type, std::string name, lStmt body);

    static const lStmtEnum node_type = lStmtEnum::Function;
};

} // namespace llir
} // namespace nacho
