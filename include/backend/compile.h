#pragma once

#include "IRFwdDecl.h"
#include "CIN.h"
#include "llir/LLIR.h"
#include "Printer.h"
#include "Type.h"
#include <string>


namespace nacho {
namespace backend {

        struct CINLowerer {

            CIN cin;
            Printer printer;
            CINLowerer(CIN cin, std::ostream &os) : cin(std::move(cin)), printer(os) {}

            void lower_cin();
            void lower_struct_definitions();
            void lower_partition_function();
        };



} // namespace backend

} // namespace nacho
