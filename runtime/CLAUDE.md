# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this subdirectory.

## Project Overview

GPU-accelerated sparse matrix computation library with Python bindings. Built with CUDA, nanobind, and scikit-build-core. Targets Redwood GPU (sm_89).

## Build & Run

Preferred setup is via `./setup.sh --runtime` from the repo root (creates conda env, loads CUDA, installs everything). Otherwise:

```bash
module load cuda
pip install runtime/                          # from repo root
pip install .                                 # or from runtime/ directory

# Development build (faster iteration, from runtime/ directory)
pip install nanobind scikit-build-core[pyproject] torch
pip install --no-build-isolation -ve .
```

### Benchmarks

Run from anywhere (CUDA is auto-loaded):

```bash
conda activate nacho
python runtime/tests/run_benchmarks.py --quick              # smoke test
python runtime/tests/run_benchmarks.py csr_add spgemm       # specific benchmarks
python runtime/tests/run_benchmarks.py spgemm -s 500 -e 600 # custom range
python runtime/tests/run_benchmarks.py                      # full sweep
```

Available: `csr_add`, `coo_add`, `spgemm`, `sparse_vectors`, `broadcast`.

## Architecture

### Core Data Structures (`src/nb_utils.hpp`)

- **CSR<index_t, value_t>** — Compressed Sparse Row: `indptr`, `indices`, `data` tensors + `shape` + timing fields
- **COO<index_t, value_t>** — Coordinate format: `row`, `col`, `data` tensors + `shape`
- **CVector<index_t, coord_t, value_t>** — Sparse vector: `indices`, `data` tensors + `size`
- **COO2D** (namespace `io_coo`) — File I/O for MatrixMarket format, handles symmetric expansion

All GPU tensors are managed via nanobind capsules with `cudaFreeWrapper` for automatic cleanup.

### Python Bindings (`src/nanobind_cuda_example_ext.cpp`)

Single nanobind extension module exposing: `parse2D`, `gpu_csr_add_f32`, `gpu_coo_add_f32`, `gpu_sss_mergepath_test`, `gpu_broadcast_xA`, `spgemm`.

### Operation Modules

Each operation lives in its own `src/<name>/` directory with `.cu` (kernels), `.h` (declarations), `.hpp` (nanobind wrappers), and `.cpp` (binding registrations):

- **spgemm/** — Sparse matrix multiply. Manual implementation uses per-(i,j) work computation, ijk partitioning, and cuCollections hash map for aggregation. Also has cuSPARSE fallback.
- **csr_add/, coo_add/** — Sparse matrix addition in CSR/COO formats. Custom kernels + cuSPARSE implementations.
- **sparse_vector/, sp_ab_c/** — Sparse vector A*B with fused and non-fused strategies.
- **mergepath_utils/** — Balanced work partitioning across threads using diagonal-based mergepath algorithm. Used by multiple operations for load balancing.
- **cuda_utils/** — `CHECK_CUDA()`/`CHECK_CUSPARSE()` error macros, debug printing, array comparison.

### Build System

CMake via scikit-build-core. Uses CPM to fetch cuCollections (cuco). Links against CUDA::cusparse, cuco, CUB, Thrust. CUDA flags include `-arch=sm_89`, `--expt-extended-lambda`, `-G -g -lineinfo`.

### Benchmarking

Tests in `tests/` benchmark against SuiteSparse matrix collection. `tests/parser.py` reads MatrixMarket files. `tests/plotter.py` generates performance plots saved to `benchmark_results/`.

## Relationship with nacho

This directory is part of the [nacho](https://github.com/rootjalex/nacho) monorepo. nacho is a sparse tensor algebra compiler that generates CUDA kernel code from high-level expressions (e.g., `sum("j", a_ij * b_jk)` for SpGEMM). The workflow:

1. nacho compiles a sparse tensor expression and prints CUDA kernels to stdout
2. Generated kernels are placed into the corresponding `.cu` files here
3. Code is adapted to integrate with nanobind bindings, memory management (`cudaMalloc`/`cudaFree`), and error checking macros
4. Benchmarks compare nacho-generated kernels against cuSPARSE baselines

Files with nacho-generated code include `csr_add/csr_add.cu` and `csr_dcsr_mul/dcsr_mul.cu`. The generated code follows a three-phase pattern — partition, precompute, compute — with templated tensor format structs. Some kernels (e.g., `spgemm/`, `coo_add/`) are hand-written reference implementations.
