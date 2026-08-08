#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>

void cudaFreeWrapper(void* ptr) noexcept;
void print_cuda(int * & ptr, int size);
void print_cuda(float * & ptr, int size);
bool compare_cuda(int* & ptr, int* & ptr2, int size);

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("cuda error\n");                                                \
        throw std::runtime_error(cudaGetErrorString(status));                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cusparse error\n");                                            \
        throw std::runtime_error(cusparseGetErrorString(status));              \
    }                                                                          \
}


template<typename index_t,typename coord_t, typename value_t>
struct SparseVector{
    index_t length;
    index_t nnz;
    coord_t * indices;
    value_t * values;

    SparseVector(coord_t * _indices, value_t * _values, const index_t _length, const index_t _nnz)
        : indices(_indices), values(_values), length(_length), nnz(_nnz) {}
};

template<typename index_t, typename value_t>
struct CSRMatrix {
    index_t rows;
    index_t cols;
    index_t nnz;
    index_t * row_offsets;
    index_t * col_indices;
    value_t * values;

    CSRMatrix(index_t * _row_offsets, index_t * _col_indices, value_t * _values, const index_t _rows, const index_t _cols, const index_t _nnz)
        : row_offsets(_row_offsets), col_indices(_col_indices), values(_values), rows(_rows), cols(_cols), nnz(_nnz) {}
};

// Async allocator: routes thrust::device_vector through cudaMallocAsync/cudaFreeAsync
// on stream 0 so that all allocations join the same async dependency chain as the
// raw cudaMallocAsync calls in the same kernels.
//
// pointer = thrust::device_ptr<T> so that the iterator type of async_vec<T> is
// identical to that of thrust::device_vector<T> — this ensures CUB and other code
// that expects thrust::device_ptr-based iterators (e.g. DeviceSegmentedSort offsets)
// does not try to dereference a raw device pointer on the host.
template <typename T>
struct async_allocator {
    using value_type      = T;
    using pointer         = thrust::device_ptr<T>;
    using const_pointer   = thrust::device_ptr<const T>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using is_always_equal = std::true_type;

    template <typename U>
    struct rebind { using other = async_allocator<U>; };

    thrust::device_ptr<T> allocate(std::size_t n) {
        T* raw = nullptr;
        CHECK_CUDA(cudaMallocAsync(&raw, n * sizeof(T), 0));
        return thrust::device_ptr<T>(raw);
    }
    void deallocate(thrust::device_ptr<T> ptr, std::size_t) noexcept {
        cudaFreeAsync(thrust::raw_pointer_cast(ptr), 0);
    }
    bool operator==(const async_allocator&) const noexcept { return true; }
    bool operator!=(const async_allocator&) const noexcept { return false; }
};

template <typename T>
using async_vec = thrust::device_vector<T, async_allocator<T>>;

template<typename index_t, typename value_t>
struct COOMatrix{
    index_t rows;
    index_t cols;
    index_t nnz;
    index_t * row_indices;
    index_t * col_indices;
    value_t * values;
};