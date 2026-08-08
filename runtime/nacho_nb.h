#pragma once

// nanobind vocabulary shared by every generated binding: the ndarray aliases the
// generated tensor classes are built from, and the helpers that move buffers across
// the C++/Python boundary.

#include <cstdint>
#include <initializer_list>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


void cudaFreeWrapper(void *ptr) noexcept;

namespace nacho {

// Tensors are exchanged as torch tensors: 1-D contiguous buffers per level, plus a
// small host-side shape vector.
template <typename T>
using ArrayCPU = nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using ArrayGPU = nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::cuda>;

// Dimension sizes always live on the host, whatever the tensor's device, because the
// kernels read them from the struct on the host side.
template <size_t Rank>
using ShapeTuple = nb::ndarray<nb::pytorch, int32_t, nb::shape<Rank>, nb::c_contig, nb::device::cpu>;

// Kernels hand back buffers they own: malloc'd on the CPU path, cudaMallocAsync'd on
// the GPU path. These wrap such a buffer in an ndarray whose capsule frees it with the
// matching deallocator, transferring ownership to Python.
template <typename T>
ArrayCPU<T> adopt_cpu(T *data, size_t count) {
    size_t shape[1] = {count};
    nb::capsule owner(data, [](void *p) noexcept { std::free(p); });
    return ArrayCPU<T>(data, 1, shape, owner, nullptr, nb::dtype<T>(), nb::device::cpu::value);
}

template <typename T>
ArrayGPU<T> adopt_gpu(T *data, size_t count) {
    size_t shape[1] = {count};
    nb::capsule owner(data, cudaFreeWrapper);
    return ArrayGPU<T>(data, 1, shape, owner, nullptr, nb::dtype<T>(), nb::device::cuda::value);
}

// Shape vectors are built here rather than adopted, so they get a plain delete[].
template <size_t Rank>
ShapeTuple<Rank> make_shape(std::initializer_list<int32_t> sizes) {
    int32_t *data = new int32_t[Rank];
    size_t i = 0;
    for (int32_t size : sizes) {
        data[i++] = size;
    }
    size_t shape[1] = {Rank};
    nb::capsule owner(data, [](void *p) noexcept { delete[] (int32_t *)p; });
    return ShapeTuple<Rank>(data, 1, shape, owner, nullptr, nb::dtype<int32_t>(), nb::device::cpu::value);
}

// Operand buffers stay owned by Python. The generated tensor structs hold non-const
// pointers, so borrowing one costs a const_cast; the kernels only read operands.
template <typename T, typename Array>
T *borrow(const Array &array) {
    return const_cast<T *>(array.data());
}

} // namespace nacho
