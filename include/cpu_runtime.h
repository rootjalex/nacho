#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>

// CUDA grid/block simulation
struct dim3 { int x = 0, y = 0, z = 0; };
static dim3 blockIdx, threadIdx, blockDim, gridDim;

// atomicAdd (sequential — no races on CPU)
template<typename T>
T atomicAdd(T* addr, T val) { T old = *addr; *addr += val; return old; }

// CUDA-compatible min/max (bare names, like CUDA's device builtins)
template<typename T>
T min(T a, T b) { return (a < b) ? a : b; }
template<typename T>
T max(T a, T b) { return (a > b) ? a : b; }

// Stream stubs
using cudaStream_t = int;
static const cudaStream_t cudaStreamPerThread = 0;
inline void cudaStreamSynchronize(cudaStream_t) {}
