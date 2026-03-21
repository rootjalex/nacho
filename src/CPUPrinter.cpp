#include "CPUPrinter.h"

#include "llir/LLIR.h"
#include "llir/Function.h"
#include "Error.h"

namespace nacho {

void CPUPrinter::visit(const llir::Float_t *node) {
    switch (node->bits) {
    case 16:
        os << "uint16_t";
        break;
    case 32:
        os << "float";
        break;
    case 64:
        os << "double";
        break;
    default:
        internal_error << "Unhandled bits for float_t: " << node->bits;
    }
}

void CPUPrinter::visit(const llir::Function *node) {
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
        case llir::Function::global:
            // omit __global__
            break;
        case llir::Function::device:
            // omit __device__
            break;
        case llir::Function::inline_:
            os << "inline ";
            break;
        case llir::Function::host:
            // omit __host__
            break;
        }
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
        if (arg.by_reference) {
            os << "&";
        }
        // No __restrict__ on CPU
        os << " " << arg.name;
    }

    os << ") {\n";
    indent();
    node->body.accept(this);
    dedent();
    os << "}\n";
}

void CPUPrinter::visit(const llir::DeviceAlloc *node) {
    print_indent();
    print_no_parens(node->target);
    os << " = (decltype(";
    print_no_parens(node->target);
    os << "))malloc(";
    print_no_parens(node->size_bytes);
    os << ");\n";
}

void CPUPrinter::visit(const llir::DeviceFree *node) {
    print_indent();
    os << "free(";
    print_no_parens(node->ptr);
    os << ");\n";
}

void CPUPrinter::visit(const llir::DeviceTransfer *node) {
    print_indent();
    if (node->kind == llir::DeviceTransfer::Memcpy) {
        if (node->synchronize) {
            // synchronize-only transfer is a no-op on CPU
            os << "// sync (no-op on CPU)\n";
            return;
        }
        os << "memcpy(";
        print_no_parens(node->dst);
        os << ", ";
        print_no_parens(node->src);
        os << ", ";
        print_no_parens(node->size_bytes);
        os << ");\n";
    } else {
        os << "memset(";
        print_no_parens(node->dst);
        os << ", ";
        print_no_parens(node->src);
        os << ", ";
        print_no_parens(node->size_bytes);
        os << ");\n";
    }
}

void CPUPrinter::visit(const llir::PrefixSum *node) {
    // Sequential inclusive scan:
    // output[0] = 0;
    // for (int _ps_i = 0; _ps_i < count; _ps_i++) {
    //   output[_ps_i + 1] = output[_ps_i] + input[_ps_i];
    // }
    print_indent();
    print_no_parens(node->output);
    os << "[0] = 0;\n";
    print_indent();
    os << "for (int _ps_i = 0; _ps_i < ";
    print_no_parens(node->count);
    os << "; _ps_i++) {\n";
    indent();
    print_indent();
    print_no_parens(node->output);
    os << "[_ps_i + 1] = ";
    print_no_parens(node->output);
    os << "[_ps_i] + ";
    print_no_parens(node->input);
    os << "[_ps_i];\n";
    dedent();
    print_indent();
    os << "}\n";
}

void CPUPrinter::visit(const llir::KernelLaunch *node) {
    print_indent();
    os << "gridDim.x = ";
    print_no_parens(node->grid_dim);
    os << ";\n";
    print_indent();
    os << "blockDim.x = ";
    print_no_parens(node->block_dim);
    os << ";\n";
    print_indent();
    os << "for (int _blk = 0; _blk < ";
    print_no_parens(node->grid_dim);
    os << "; _blk++) {\n";
    indent();
    print_indent();
    os << "blockIdx.x = _blk;\n";
    print_indent();
    os << "for (int _thr = 0; _thr < ";
    print_no_parens(node->block_dim);
    os << "; _thr++) {\n";
    indent();
    print_indent();
    os << "threadIdx.x = _thr;\n";
    print_indent();
    os << node->kernel_name;
    if (!node->template_args.empty()) {
        os << "<";
        for (size_t i = 0; i < node->template_args.size(); i++) {
            if (i > 0)
                os << ", ";
            os << node->template_args[i];
        }
        os << ">";
    }
    os << "(";
    for (size_t i = 0; i < node->args.size(); i++) {
        if (i > 0)
            os << ", ";
        print_no_parens(node->args[i]);
    }
    os << ");\n";
    dedent();
    print_indent();
    os << "}\n";
    dedent();
    print_indent();
    os << "}\n";
}

} // namespace nacho
