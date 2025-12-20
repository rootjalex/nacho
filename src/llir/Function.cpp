#include "llir/Function.h"

#include "Printer.h"

namespace nacho {
namespace llir {

void Function::print(std::ostream &os) const {
    Printer printer(os);
    if (!generics.empty()) {
        os << "template<";
        bool first = true;
        for (const auto &g : generics) {
            if (!first) {
                os << ", ";
            }
            first = false;
            os << "typename " << g;
        }
        os << ">\n";
    }

    for (const auto &a : attributes) {
        switch (a) {
            case Function::device: {
                os << "__device__ ";
                break;
            }
            case Function::inline_: {
                os << "__inline__ ";
                break;
            }
            case Function::host: {
                os << "__host__ ";
                break;
            }
        }
    }

    if (!attributes.empty()) {
        os << "\n";
    }

    printer.print(ret_type);

    os << " " << name << "(";

    for (size_t i = 0, e = args.size(); i < e; i++) {
        if (i != 0) {
            os << ", ";
        }
        const auto &arg = args[i];
        if (!arg.mutating) {
            os << "const ";
        }
        printer.print(arg.type);
        if (!arg.mutating && arg.type.is<Ptr_t>()) {
            os << " __restrict__";
        }
        os << " " << arg.name;
    }

    os << ") {\n";
    printer.indent();
    printer.print(body);
    printer.dedent();
    os << "}\n";
}

} // namespace llir
} // namespace nacho
