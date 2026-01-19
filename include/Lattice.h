#pragma once

#include "Equality.h"
#include "Seq.h"

#include <iostream>
#include <map>

namespace nacho {

struct Lattice {
    static Lattice build(Seq top);

    std::vector<Seq> topological_order() const;

    // Also in topologically sorted order
    std::vector<Seq> sub_points(const Seq &seq) const;

    void dump(std::ostream &os) const;

  private:
    Lattice(Seq top) : top(std::move(top)) {}

    // First node
    Seq top;
    // The graph is a map from a seq to a list of simplified versions of
    // that sequence (the sub points of the lattice point).
    using Graph = std::map<Seq, std::vector<Seq>, SeqLessThan>;
    Graph graph;

    // Returns immediately if point is in graph,
    // otherwise adds the point to the lattice
    // recursively.
    void insert(Seq point);
};

} // namespace nacho