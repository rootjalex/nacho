# Repository Guidelines

## Project Structure & Module Organization
The repository has two primary parts:
- `include/` and `src/`: C++20 compiler core (`nacho_library`) and lowering pipeline.
- `compiler.cpp`: compiler CLI entry point (`build/compiler`).
- `runtime/`: CUDA + nanobind Python runtime package.
- `runtime/src/`: operation kernels and bindings (`csr_add/`, `coo_add/`, `spgemm/`, `nacho_generated/` wrappers).
- Runtime-generated CUDA kernels are emitted at build time into the runtime build directory (`build/.../generated/nacho_generated/`).
- `runtime/tests/`: pytest-based correctness tests and benchmark scripts.
- `cmake/`, `CMakeLists.txt`, `setup.sh`: build/configuration entry points.

## Build, Test, and Development Commands
- `./setup.sh --compiler`: create/update `nacho` conda env, build compiler, run CTest.
- `cmake -S . -B build && cmake --build build -j$(nproc)`: manual compiler build.
- `ctest --test-dir build --output-on-failure`: run compiler test suite.
- `./build/compiler --list` or `./build/compiler --test sparse_vec_mul`: inspect/run a single compiler test.
- `./setup.sh --runtime`: install runtime dependencies and `pip install runtime/` (CUDA required).
- `pytest runtime/tests/`: run runtime tests.

## Coding Style & Naming Conventions
- C++ formatting follows `.clang-format` (LLVM base, 4-space indentation).
- Naming is enforced by `.clang-tidy`:
  - `CamelCase`: classes, enums.
  - `lower_case`: functions, variables, parameters, members.
  - `UPPER_CASE`: `constexpr` variables.
- Generated kernels are build artifacts; do not hand-edit generated `.cu` files.
- Add concise comments for non-obvious logic (especially build/codegen wiring) so control flow is human-readable.

## Testing Guidelines
- Compiler tests are registered through CTest and executed via `build/compiler`.
- Runtime tests use `pytest` with files named `test_*.py` under `runtime/tests/`.
- Before opening a PR:
  - Run `ctest --test-dir build`.
  - If runtime or CUDA code changed, run `pytest runtime/tests/` (or targeted tests such as `pytest runtime/tests/test_nacho_generated.py`).

## Commit & Pull Request Guidelines
- Match existing commit style: imperative, concise subject lines (e.g., `Fix DCSR add row completion in generated code`).
- Keep commits focused by concern (compiler IR/lowering vs runtime kernels/tests).
- PRs should include:
  - Clear problem statement and approach.
  - Linked issue (if applicable).
  - Exact validation commands run and notable output.
  - Performance/benchmark notes when kernel behavior changes.
