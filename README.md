# nacho

Sparse Tensor Algebra compiler for GPUs

## Quick Start

```bash
./setup.sh --compiler        # creates conda env, builds, and runs tests
conda activate nacho
```

## Setup

Requires CMake 3.30+ and a C++20 compiler. The setup script handles everything (including installing CMake via conda if needed):

```bash
./setup.sh --compiler   # compiler only, no CUDA needed
./setup.sh --runtime    # runtime only, needs CUDA + GPU
./setup.sh              # both
```

Or build manually:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

For a debug build (includes UBSan):

```bash
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg --config Debug -j$(nproc)
```

## Testing

```bash
# Run all tests
ctest --test-dir build

# Run a single test by name
./build/compiler --test sparse_vec_mul

# List available tests
./build/compiler --list

# Run all tests with verbose output
./build/compiler
```

Available tests:

| Test | Description |
|------|-------------|
| `sparse_vec_mul` | Sparse vector element-wise multiply (intersection) |
| `sparse_vec_add` | Sparse vector element-wise add (union) |
| `sparse_vec_apb_c` | Sparse vector `(a+b)*c` — 3-operand fused |
| `sparse_vec_ab_pc` | Sparse vector `(a*b)+c` — 3-operand fused |
| `dcsr_mul` | DCSR matrix element-wise multiply (multi-phase) |
| `dcsr_add` | DCSR matrix element-wise add (multi-phase) |
| `format_inference` | Format propagation through add/mul/broadcast |
| `lattice` | Merge lattice construction |
| `locator` | Iterator/locator partitioning optimization |

## Runtime

The `runtime/` directory contains the GPU runtime and benchmark suite (CUDA kernels, Python bindings via nanobind, benchmarks against cuSPARSE). See [`runtime/README.md`](runtime/README.md) for details.

During runtime builds, nacho-generated kernels are produced automatically by
`build/compiler` into the build directory. If the compiler is missing, the
runtime build bootstraps it first.

### Acknowledgements

A significant portion of the code in this repository is modeled after, or
directly taken from, the [Halide] compiler. That is because they both have done
incredible work, and because it is the compiler that I (AJR) am most familiar
with navigating and understanding. As a result, this repository benefits heavily
from over a decade of hard work from the Halide developers.

[Halide]: https://github.com/halide/Halide
