#pragma once

// Support header included by every nacho-generated kernel header.
//
// The code generator assumes a small set of primitives exist: fixed-width integer
// types, an unqualified min()/max(), and — on the GPU path — the CUDA runtime and
// CUB. This header is the single place those assumptions are satisfied, so the
// generated headers stay free of target-specific includes.
//
// One header serves both targets: a generated *_cpu.h is compiled by the host
// compiler, a generated *_gpu.h by nvcc.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef __CUDACC__

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>

// min/max come from the CUDA headers as device builtins.

namespace nacho_std = cuda::std;

#else

#include <algorithm>
#include <tuple>

// Generated kernels call min()/max() unqualified. nvcc supplies these; the host
// compiler does not, so pull the standard ones into scope.
using std::max;
using std::min;

namespace nacho_std = std;

#endif // __CUDACC__
