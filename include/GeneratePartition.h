// #pragma once

// #include "llir/Function.h"

// #include <vector>

// namespace nacho {

// /*
// Generates the partition function for a list of one-dimensional coordinate
// lists. All inputs are assumed to be names of compressed lists. The generated
// function accepts a coordinate list and size for each input name, and a mutable
// partitions array (of length n_threads * names.size()) that it fills. e.g. for
// input {"a", "b"}, the following function is generated:

// Goal is to match partition function here:
// https://github.com/rootjalex/mergepath/blob/5984a8fe2c09958968674eafb1d7f34685db63c3/partition.h#L10
// */
// std::vector<llir::Function>
// generate_nary_1d_partition(const std::vector<std::string> &names);

// } // namespace nacho
