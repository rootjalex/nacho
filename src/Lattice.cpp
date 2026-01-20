#include "Lattice.h"

#include "Mutator.h"
#include "Simplify.h"
#include "Visitor.h"

namespace nacho {

Lattice Lattice::build(Seq top) {
    Lattice lattice(top);
    lattice.insert(std::move(top));
    return lattice;
}

std::vector<Seq> Lattice::topological_order() const { return sub_points(top); }

std::vector<Seq> Lattice::sub_points(const Seq &start) const {
    std::vector<Seq> order;
    std::set<Seq, SeqLessThan> visited;

    std::function<void(const Seq &)> helper = [&](const Seq &seq) -> void {
        if (!seq.defined()) {
            return;
        } else if (visited.count(seq)) {
            return;
        }
        visited.insert(seq);

        const auto &iter = graph.find(seq);
        internal_assert(iter != graph.end())
            << "Failure in topological sorting of lattice, node not found: "
            << seq;

        for (const auto &c : iter->second) {
            helper(c);
        }

        if (!seq.is<Empty>()) {
            order.push_back(seq);
        }
    };

    helper(start);

    std::reverse(order.begin(), order.end());
    return order;
}

void Lattice::dump(std::ostream &os) const {
    std::set<Seq, SeqLessThan> visited;

    std::function<void(int, const Seq &)> helper = [&](int level,
                                                       const Seq &seq) -> void {
        std::string indent(level * 4, ' ');
        if (!seq.defined()) {
            os << indent << "EMPTY\n";
            return;
        }

        os << indent << seq << "\n";

        const auto &iter = graph.find(seq);
        internal_assert(iter != graph.end())
            << "Failure in dumping of lattice, node not found: " << seq;

        if (visited.count(seq)) {
            // Don't duplicate children printing
            if (!iter->second.empty()) {
                os << indent << " (children printed above)\n";
            }
            return;
        }

        visited.insert(seq);

        for (const auto &c : iter->second) {
            helper(level + 1, c);
        }
    };

    os << "{\n";
    helper(0, top);
    os << "}\n";
}

namespace {

// TODO: could speed up with memoization and recursion.
std::set<Seq, SeqLessThan> edges(const Seq &point) {
    struct GetEdges : public Visitor {
        std::set<Seq, SeqLessThan> edges;
        void visit(const Index *node) override { edges.insert(node); }
    };
    GetEdges getter;
    point.accept(&getter);
    return getter.edges;
}

} // namespace

void Lattice::insert(Seq point) {
    if (graph.count(point)) {
        return;
    }

    graph[point] = {};

    std::vector<Seq> &children = graph[point];

    auto es = edges(point);

    // Don't insert children multiple times (e.g., Empty)
    std::set<Seq, SeqLessThan> dedup;

    for (const auto &e : es) {
        Seq child = remove_and_simplify(point, e);
        internal_assert(child.defined())
            << "Removing " << e << " from " << point << " failed.";
        // Useful for debugging
        /*
        std::cout << "Removing " << e << " from " << point << " = " << child <<
        " and child is "; if (dedup.count(child)) { std::cout << "not ";
        }
        std::cout << "unique!\n";
        */
        if (dedup.count(child) == 0) {
            dedup.insert(child);
            children.push_back(child);
            insert(child);
        }
    }
    return;
}

} // namespace nacho
