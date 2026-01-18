#pragma once

#include "CIN.h"
#include "Equality.h"
#include "Seq.h"

#include <set>

namespace nacho {

// Remove sequence expressions and tensor accesses with undefined parents.
CIN simplify(const std::set<Seq, SeqLessThan> &defined, const CIN &cin);

// Remove `remove` from `orig` and perform simplification.
Seq remove_and_simplify(const Seq &orig, const Seq &remove);

} // namespace nacho
