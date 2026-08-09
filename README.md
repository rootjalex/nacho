# nacho

Sparse Tensor Algebra compiler for GPUs and multicore CPUs

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

Generated kernels, and the tensor classes they take and return, are exposed to Python
through nanobind. A kernel is declared in [compiler.cpp](compiler.cpp) — consisting of
the tensor formats, the operands over them, and the expression:

```cpp
Format csr = Format::ordered({
    {"i", LevelFormat::Dense},
    {"j", LevelFormat::Compressed_unique},
}).named("CSR");

TensorType csr_f32 = TensorType(csr, dType::Float32);
Expr a_csr_ij = Tensor::make(csr_f32, "a");
Expr b_csr_ij = Tensor::make(csr_f32, "b");

Kernel("csr_mul").expr(a_csr_ij * b_csr_ij).targets({Target::CPU, Target::GPU}).emit();
```

`.named("CSR")` is the Python class name of the layout, giving `nacho.CSR_cpu` and
`nacho.CSR_gpu`; the kernel is exposed as `<cpu|gpu>_<name>_f32`. Operands are passed in
lexicographic order by name unless `.operand_ordering({"b", "a"})` says otherwise.

Before the generated code can be used from Python it has to be compiled into the
extension module:

```bash
pip install --no-build-isolation -ve .
```

```python
import nacho
A, B = nacho.to_csr(A_torch, "cpu"), nacho.to_csr(B_torch, "cpu")
C = nacho.cpu_csr_mul_f32(A, B)          # A, B are nacho.CSR_cpu
C = nacho.gpu_csr_mul_f32(A, B)          # nacho.CSR_gpu, grid size can be optionally provided
```

# Testing

`tests/smoke.py` checks each generated kernel on both devices against scipy on small
random inputs; run it after rebuilding and before any timing work.

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
python benchmarks/csr_add.py     --device both --start 0 --end 1600  # vs cuSPARSE/Taco/MKL
python benchmarks/csr_mul.py     --device both --start 0 --end 1600  # vs PyTorch
python benchmarks/coo_add.py     --device both --start 0 --end 1600  # vs PyTorch
python benchmarks/coo_mul.py     --device both --start 0 --end 1600  # vs PyTorch
python benchmarks/coo_csr_add.py --device cpu --start 0 --end 1600  # vs PyTorch
python benchmarks/csr_add_3.py   --device cpu --start 0 --end 1600  # fused vs unfused
python benchmarks/frostt_tensors_add.py --device both               # CSF3 vs COO3D vs torch
python benchmarks/inner_prod.py  --device both                       # FROSTT, CSF3 vs COO3 vs torch
python benchmarks/dcsr_lb_comparison.py --device both                # partitioning schemes

python benchmarks/spgemm.py --start 0 --end 1300                    # vs cuSPARSE
python benchmarks/sssmm.py  --start 0 --end 1300                    # fused vs unfused vs cuSPARSE
```

Settings in `benchmarks/config.py` — dataset paths, iteration counts, output directory — 
can be overridden by an environment variable of the same name prefixed with `NACHO_`.


### Acknowledgements

A significant portion of the code in this repository is modeled after, or
directly taken from, the [Halide] compiler. That is because they both have done
incredible work, and because it is the compiler that we are most familiar
with navigating and understanding. As a result, this repository benefits heavily
from over a decade of hard work from the Halide developers.

[Halide]: https://github.com/halide/Halide
