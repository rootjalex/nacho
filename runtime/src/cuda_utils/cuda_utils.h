#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <sstream>

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
struct CSRMatrix{
    index_t rows;
    index_t cols;
    index_t nnz;
    index_t * row_offsets;
    index_t * col_indices;
    value_t * values;

    CSRMatrix(index_t * _row_offsets, index_t * _col_indices, value_t * _values, const index_t _rows, const index_t _cols, const index_t _nnz)
        : row_offsets(_row_offsets), col_indices(_col_indices), values(_values), rows(_rows), cols(_cols), nnz(_nnz) {}
};

template<typename index_t, typename value_t>
struct COOMatrix{
    index_t rows;
    index_t cols;
    index_t nnz;
    index_t * row_indices;
    index_t * col_indices;
    value_t * values;
};