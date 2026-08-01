#include "llir/Function.h"

#include "Printer.h"
#include "Error.h"
namespace nacho {
namespace llir {

lStmt Function::make(std::vector<std::string> generics,
    std::vector<Attribute> attributes,
    std::vector<Argument> args,
    lType ret_type,
    std::string name,
    lStmt body) {

    internal_assert(!name.empty())
        << "Cannot make Function with empty name";
    internal_assert(body.defined())
        << "Cannot make Function with undefined body";
    Function *node = new Function;
    node->generics = std::move(generics);
    node->attributes = std::move(attributes);
    node->args = std::move(args);
    node->ret_type = std::move(ret_type);
    node->name = std::move(name);
    node->body = std::move(body);
    return node;
}
} // namespace llir

void Printer::visit(const llir::Function *node) {
     if (!node->generics.empty()) {
        os << "template<";
        bool first = true;
        for (const auto &g : node->generics) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << "typename " << g;
        }
        os << ">\n";
    }
    for (const auto &a : node->attributes) {
        switch (a) {
            case llir::Function::global: {
                os << "__global__ ";
                break;
            }
            case llir::Function::device: {
                os << "__device__ ";
                break;
            }
            case llir::Function::inline_: {
                os << "__inline__ ";
                break;
            }
            case llir::Function::host: {
                os << "__host__ ";
                break;
            }
            case llir::Function::runnable: {
                os << "__runnable__ ";
                break;
            }
        }
    }

    if (!node->attributes.empty()) {
        os << "\n";
    }

    node->ret_type.accept(this);

    os << " " << node->name << "(";

    for (size_t i = 0, e = node->args.size(); i < e; i++) {
        if (i != 0) {
            os << ", ";
        }
        const auto &arg = node->args[i];
        if (!arg.mutating) {
            os << "const ";
        }
        arg.type.accept(this);
        // if (!arg.mutating && arg.type.is<llir::Ptr_t>()) {
        //     os << " __restrict__";
        // }
        os << " " << arg.name;
    }

    os << ") {\n";
    indent();
    node->body.accept(this);
    dedent();
    os << "}\n";
}

} // namespace nacho
