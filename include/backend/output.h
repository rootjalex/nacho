#pragma once

#include <string>

namespace nacho {
namespace backend {

// Directory that every generated source/header is written to. Relative to the
// process working directory unless set_output_directory() is given an absolute path.
const std::string &output_directory();

void set_output_directory(std::string dir);

// Removes output_directory() and recreates it empty.
//
// Call this exactly once, before any code generation. The build globs this
// directory, so wiping it is what makes "compile only the kernels I asked for"
// work: whatever survives the run is exactly what gets compiled.
void reset_output_directory();

} // namespace backend
} // namespace nacho
