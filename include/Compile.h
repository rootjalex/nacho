#pragma once

#include "IRFwdDecl.h"

#include <string>

namespace nacho {

CIN compile_to_cin(const Expr &expr, std::string out = "Z");

} // namespace nacho
