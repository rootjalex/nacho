#pragma once

#include "Format.h"
#include "IRFwdDecl.h"

#include <optional>
#include <string>

namespace nacho {

// `out_format`, when given, replaces the result format inferred from `expr`. It must
// cover the same indices; only the level formats (e.g. storing a result dense that
// inference made sparse) may differ.
CIN compile_to_cin(const Expr &expr, std::string out = "Z",
                   std::optional<Format> out_format = std::nullopt);

} // namespace nacho
