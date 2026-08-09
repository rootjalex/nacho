# nacho

Sparse Tensor Algebra compiler for GPUs

### Setup

Install first: the CUDA toolkit (`nvcc` on `PATH` or in `CUDACXX`), oneTBB, and Python
3.12 or newer. Then:

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Replace `cu126` with the CUDA version you installed.

Tested with CUDA toolkit 12.6, driver 560.35, gcc 11.4, Python 3.12 and torch 2.8.0+cu128
on an RTX 4090. `python/CMakeLists.txt` builds for `sm_89`; override with
`-DCMAKE_CUDA_ARCHITECTURES`.

1. Build `nacho`:

```bash
# Option 1: normal
cmake -S . -B build
cmake --build build -j<N PARALLELISM>

# Option 2: debug
cmake -S . -B build-dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dbg --config Debug -j<N PARALLELISM>
```

2. Generate kernels into `generated/`:

```bash
./build/compiler                          # every kernel declared in compiler.cpp
./build/compiler --kernels csr_mul,csr_add  # just these
```

Each run clears `generated/` first, so what is left behind is exactly what was asked
for — and exactly what the extension build compiles.

# Python bindings

Kernels are exposed to Python through nanobind. `compiler.cpp` declares a kernel with

```cpp
Kernel("csr_mul").expr(a_csr_ij * b_csr_ij).targets({Target::CPU, Target::GPU}).emit();
```

and the compiler emits, alongside the kernel itself, a wrapper and the tensor classes it
needs. A layout's Python class name comes from `Format::named("CSR")`; formats derived by
the compiler (an expression's result, or the flattened form of a COO operand) recover the
same name from their level structure.

Operand order in both the C++ signature and the Python wrapper is lexicographic by tensor
name, so name operands `a`, `b`, `c` to match the order they appear in the expression.

Build the extension against whatever is currently in `generated/`:

```bash
pip install --no-build-isolation -ve .
```

```python
import nacho
c = nacho.cpu_csr_mul_f32(a, b)          # a, b are nacho.CSR_cpu
c = nacho.gpu_csr_mul_f32(a, b)          # nacho.CSR_gpu, launch geometry optional
```

# Testing

`tests/smoke.py` checks each generated kernel on both devices against scipy on small
random inputs; run it after rebuilding and before any timing work. Kernels missing from
the current build are skipped rather than failing.

```bash
python tests/smoke.py
```

# Benchmarks

The benchmarks read their inputs from two datasets, neither of which lives in this repo.
Set both before running anything:

```bash
export NACHO_SUITESPARSE_DIR=/path/to/suitesparse   # .mtx files, every 2-D benchmark
export NACHO_FROSTT_DIR=/path/to/frostt             # .tns files, frostt_tensors_add.py
```

FROSTT tensors must have lexicographically sorted coordinates and no duplicate
coordinates.

Each script directly in `benchmarks/` is then a standalone entry point:

```bash
python benchmarks/csr_add.py     --device cpu --start 0 --end 1600  # vs cuSPARSE/Taco/MKL
python benchmarks/csr_mul.py     --device cpu --start 0 --end 1600  # vs PyTorch
python benchmarks/coo_add.py     --device cpu --start 0 --end 1600  # vs PyTorch
python benchmarks/coo_mul.py     --device cpu --start 0 --end 1600  # vs PyTorch
python benchmarks/coo_csr_add.py --device cpu --start 0 --end 1600  # vs PyTorch
python benchmarks/csr_add_3.py   --device cpu --start 0 --end 1600  # fused vs unfused
python benchmarks/frostt_tensors_add.py --device cpu                # CSF3 vs COO3D vs torch
python benchmarks/dcsr_lb_comparison.py --device cpu                # partitioning schemes
```

Shared helpers — matrix loading, plotting, timing, result comparison — live in
`benchmarks/common/`. Every setting in `benchmarks/config.py` — dataset paths, which FROSTT
tensors to run, iteration counts, output directory — can be overridden by an environment
variable of the same name prefixed with `NACHO_`.

Building a nacho tensor from a torch one is part of the package rather than the benchmark
harness, so it works anywhere:

```python
import nacho
A = nacho.to_csr(torch_csr, device="cuda")   # also to_dcsr, to_coo, to_coo3, to_csf3
```

### Acknowledgements

A significant portion of the code in this repository is modeled after, or
directly taken from, the [Halide] compiler. That is because they both have done
incredible work, and because it is the compiler that we are most familiar
with navigating and understanding. As a result, this repository benefits heavily
from over a decade of hard work from the Halide developers.

[Halide]: https://github.com/halide/Halide
