# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nacho** is a sparse tensor algebra compiler targeting GPUs. The architecture is inspired by (and partially derived from) the [Halide](https://github.com/halide/Halide) compiler. Written in C++20.

## Build Commands

Always use the `nacho` conda environment for all build and run commands: `conda run -n nacho <cmd>` or `conda activate nacho` first.

```bash
# Normal build
conda run -n nacho cmake -S . -B build
conda run -n nacho cmake --build build -j$(nproc)

# Debug build (includes UBSan)
conda run -n nacho cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
conda run -n nacho cmake --build build-dbg --config Debug -j$(nproc)

# Run all tests
conda run -n nacho ctest --test-dir build

# Run a single test
conda run -n nacho ./build/compiler --test sparse_vec_mul

# List available tests
conda run -n nacho ./build/compiler --list

# Run everything (all tests, verbose output)
conda run -n nacho ./build/compiler

# Install runtime (requires CUDA)
conda run -n nacho pip install runtime/

# Run runtime tests
conda run -n nacho pytest runtime/tests/

# Run CPU runtime pytest suite (no CUDA needed, included in ctest)
conda run -n nacho pytest tests/test_cpu_generated.py -v --forked
```

Requires CMake 3.30+. Tests are defined in `compiler.cpp` and registered via CTest. CPU runtime tests are in `tests/test_cpu_generated.py` (pytest via ctypes, also registered as `cpu_runtime_pytest` in CTest).

## Code Style

- `.clang-format`: LLVM style, 4-space indentation
- `.clang-tidy` naming conventions:
  - Classes/Enums: `CamelCase`
  - Functions/variables/members/parameters: `lower_case`
  - Constexpr variables: `UPPER_CASE`

## Compiler Architecture

Three-tier IR with progressive lowering:

```
Expr (Frontend DSL)  →  CIN (Concrete Index Notation)  →  LLIR (Low-Level IR)
    compile_to_cin()         backend::CINLowerer::lower_cin()
```

### Frontend (`include/Frontend.h`)
Expression DSL with overloaded `+`, `*` operators. Nodes: `Add`, `Mul`, `Sum`, `Tensor`, `Bc` (broadcast). Each expression carries a `TensorType` (Format + dType). `sum("idx", expr)` contracts a dimension.

### CIN — Concrete Index Notation (`include/CIN.h`)
Mid-level IR representing iteration structure. Key nodes: `Forall` (loop over index), `Where` (precompute workspace), `Accumulate`, `Assign`, `CalculateWork`, `Sequence`. Scalar math nodes: `cAdd`, `cMul`, `cTensor`.

### LLIR — Low-Level IR (`include/llir/LLIR.h`)
Near-machine IR. Types (`lType`): `Generic_t`, `Float_t`, `Int_t`, `Ptr_t`, `Tuple_t`, `Struct_t`. Expressions (`lExpr`) and statements (`lStmt`) for conditionals, loops, array access, function calls, etc.

### Format System (`include/Format.h`)
Tensor storage formats composed of per-dimension level formats: `Dense` or `Compressed` (sparse). `Format::ordered(...)` builds formats. Format inference rules in `src/Format.cpp` propagate sparsity through `add_formats` (union → denser) and `mul_formats` (intersection → sparser).

### Iteration Sequences (`include/Seq.h`)
`Seq` represents iteration space algebra: `Index`, `Intersect` (∩), `Union` (∪), `Universe`, `Empty`. Used by `Lattice` (in `src/Lattice.cpp`) to build merge lattices for co-iteration over sparse structures.

### Backend (`src/backend/`)
GPU kernel generation from CIN → LLIR. Five components:
- **`compile.cpp`** — `CINLowerer`: top-level orchestrator
- **`partition.cpp`** — `PartitionLowerer`: generates partition kernels for work distribution across GPU threads (merge-path strategies)
- **`compute.cpp`** — `ComputeLowerer`: generates compute/pre-compute kernels
- **`tensor.cpp`** — `TensorLowerer`: tensor struct definitions and field access helpers
- **`base_lowerer.cpp`** — `BaseKernelLowerer`: shared infrastructure (kernel naming, offset tracking)

### IR Infrastructure
- **Intrusive pointers** (`IntrusivePtr.h`, `RefCount.h`): reference-counted smart pointers for all IR nodes
- **IRNode/IRHandle** (`IRNode.h`, `IRHandle.h`): template-based node hierarchy with enum dispatch
- **Visitor/Mutator** (`Visitor.h`, `Mutator.h`): visitor pattern for traversal and transformation across all three IR levels
- **Printer** (`src/Printer.cpp`): pretty-printing for Format, Expr, CIN, and LLIR

## Namespace

Everything lives under `nacho::`. Backend code is in `nacho::backend::`. LLIR types are in `nacho::llir::`.

## Setup

```bash
# Full setup (creates 'nacho' conda env, builds compiler + runtime)
./setup.sh

# Or just one component
./setup.sh --compiler   # compiler only, no CUDA needed
./setup.sh --runtime    # runtime only, needs CUDA + GPU
```

After setup, activate the environment: `conda activate nacho`

## Runtime (`runtime/`)

The `runtime/` subdirectory (formerly the `sparse_gpu` repo) is the GPU runtime and benchmark suite. nacho generates CUDA kernel code via `CINLowerer`, which is placed into `runtime/src/` `.cu` files. The runtime compiles kernels with CUDA, exposes them via Python/nanobind, and benchmarks against cuSPARSE.

Build systems are independent by default — nacho's CMakeLists.txt only builds the runtime when `-Dnacho_BUILD_RUNTIME=ON` is passed (requires CUDA).

### Benchmarks

All benchmark commands can be run from the repo root (no `cd` needed). CUDA is auto-loaded.

```bash
conda activate nacho

# Quick smoke test (~5 matrix pairs)
python runtime/tests/run_benchmarks.py --quick

# Specific benchmark(s)
python runtime/tests/run_benchmarks.py csr_add spgemm

# Custom matrix range
python runtime/tests/run_benchmarks.py spgemm -s 500 -e 600

# Full SuiteSparse sweep (1712 matrices), no plots
python runtime/tests/run_benchmarks.py --no-plot

# All benchmarks, full sweep (slow)
python runtime/tests/run_benchmarks.py
```

Available benchmarks: `csr_add`, `coo_add`, `spgemm`, `sparse_vectors`, `broadcast`.

See `runtime/CLAUDE.md` for runtime-specific details.
