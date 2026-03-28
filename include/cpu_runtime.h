#pragma once
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

#include <tbb/parallel_for.h>

// CUDA grid/block simulation — thread_local for safe parallel execution
struct dim3 { int x = 0, y = 0, z = 0; };
static thread_local dim3 blockIdx, threadIdx, blockDim, gridDim;

// atomicAdd via std::atomic_ref (C++20) — safe under TBB parallelism
template<typename T>
T atomicAdd(T* addr, T val) {
    std::atomic_ref<T> ref(*addr);
    return ref.fetch_add(val, std::memory_order_relaxed);
}

// Inclusive scan: output[0] = 0, output[i+1] = output[i] + input[i]
template<typename T>
void cpu_inclusive_scan(const T* input, T* output, int count) {
    output[0] = 0;
    for (int i = 0; i < count; i++) {
        output[i + 1] = output[i] + input[i];
    }
}

// In-place exclusive scan: data[i] becomes sum of data[0..i-1]
template<typename T>
void cpu_exclusive_scan_inplace(T* data, int count) {
    T running = 0;
    for (int i = 0; i < count; i++) {
        T cur = data[i];
        data[i] = running;
        running += cur;
    }
}

// CUDA-compatible min/max (bare names, like CUDA's device builtins)
template<typename T>
T min(T a, T b) { return (a < b) ? a : b; }
template<typename T>
T max(T a, T b) { return (a > b) ? a : b; }

// Stream stubs
using cudaStream_t = int;
static const cudaStream_t cudaStreamPerThread = 0;
inline void cudaStreamSynchronize(cudaStream_t) {}
