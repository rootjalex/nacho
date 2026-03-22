#pragma once

#include "Printer.h"

namespace nacho {

struct CPUPrinter : public Printer {
    explicit CPUPrinter(std::ostream &os) : Printer(os) {}

    void visit(const llir::Float_t *) override;
    void visit(const llir::Function *) override;
    void visit(const llir::DeviceAlloc *) override;
    void visit(const llir::DeviceFree *) override;
    void visit(const llir::DeviceTransfer *) override;
    void visit(const llir::PrefixSum *) override;
    void visit(const llir::KernelLaunch *) override;
    void visit(const llir::CubScratchQuery *) override;
    void visit(const llir::SlabAlloc *) override;
    void visit(const llir::InPlacePrefixSum *) override;
};

} // namespace nacho
