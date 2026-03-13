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

// Remove locators from a Seq expression.
std::tuple<std::vector<Seq>, std::vector<Seq>, bool>
partition_iterators_locators(const Seq &seq);

// Get all dense locators from a Seq expression.
// This returns the dense locators due to intersection with sparse nodes.
// does not include dense coiteration locators.
std::vector<Seq> get_dense_locators(const Seq &seq);

// Just a helper function
// Gathers all `Index`s
std::vector<Seq> indexes(const Seq &seq);

} // namespace nacho
