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
#include <cstring>

#ifdef __CUDACC__

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cuda_runtime.h>

// Contracting a scatter reduction sorts the products and sums runs of equal keys.
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

// min/max come from the CUDA headers as device builtins.

namespace nacho_std = cuda::std;

// Reductions accumulate concurrently into one location.
template <typename T>
__device__ inline void nacho_atomic_add(T *address, T value) {
    atomicAdd(address, value);
}

#else

#include <algorithm>
#include <atomic>
#include <tuple>

// Generated kernels call min()/max() unqualified. nvcc supplies these; the host
// compiler does not, so pull the standard ones into scope.
using std::max;
using std::min;

namespace nacho_std = std;

// Host counterpart of the device atomicAdd. Floating point has no lock-free fetch-add, so
// this is the usual compare-exchange loop.
template <typename T>
inline void nacho_atomic_add(T *address, T value) {
    std::atomic_ref<T> target(*address);
    T expected = target.load(std::memory_order_relaxed);
    while (!target.compare_exchange_weak(expected, expected + value,
                                         std::memory_order_relaxed)) {
    }
}

#endif // __CUDACC__
