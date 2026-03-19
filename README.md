# nacho

Sparse Tensor Algebra compiler for GPUs

### Setup.

1. This project uses the [CMake](https://cmake.org/) build system. On macOS,
```bash
brew install cmake
```

2. Build `nacho`:

```bash
# Option 1: normal
cmake -S . -B build
cmake --build build -j<N PARALLELISM>

# Option 2: debug
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg --config Debug -j<N PARALLELISM>
```

3. Run
```bash
./build/compiler
```

# Testing

TODO

### Acknowledgements

A significant portion of the code in this repository is modeled after, or
directly taken from, the [Halide] compiler. That is because they both have done
incredible work, and because it is the compiler that I (AJR) am most familiar
with navigating and understanding. As a result, this repository benefits heavily
from over a decade of hard work from the Halide developers.

[Halide]: https://github.com/halide/Halide
