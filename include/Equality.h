#pragma once

#include "Seq.h"

#include <map>

namespace nacho {

bool equals(const Seq &a, const Seq &b);

// Structural equality checker. e.g., to use Seq in a std::map
struct SeqLessThan {
    bool operator()(const Seq &a, const Seq &b) const;
};

} // namespace nacho
