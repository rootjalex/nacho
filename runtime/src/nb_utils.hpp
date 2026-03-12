#pragma once
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>


#include "cuda_utils/cuda_utils.h"

namespace nb = nanobind;


template<typename value_t>
using nVector = nb::ndarray<nb::pytorch, value_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using Int32Vector = nb::ndarray<nb::pytorch, int32_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using Int64Vector = nb::ndarray<nb::pytorch, int64_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using UInt32Vector = nb::ndarray<nb::pytorch, uint32_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using UInt64Vector = nb::ndarray<nb::pytorch, uint64_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using Float32Vector = nb::ndarray<nb::pytorch, float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;

using Int32TupleCPU = nb::ndarray<nb::pytorch, int32_t, nb::shape<2>, nb::c_contig, nb::device::cpu>;
using Int64TupleCPU = nb::ndarray<nb::pytorch,int64_t, nb::shape<2>, nb::c_contig, nb::device::cpu>;
using UInt32TupleCPU = nb::ndarray<nb::pytorch, uint32_t, nb::shape<2>, nb::c_contig, nb::device::cpu>;
using UInt64TupleCPU = nb::ndarray<nb::pytorch, uint64_t, nb::shape<2>, nb::c_contig, nb::device::cpu>;

template<typename index_t, typename value_t>
struct CSR {
    using IntVector = nb::ndarray<nb::pytorch, index_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    using FloatVector = nb::ndarray<nb::pytorch, value_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    using ShapeTuple2D = nb::ndarray<nb::pytorch, index_t, nb::shape<2>, nb::c_contig, nb::device::cpu>;
    IntVector indptr;
    IntVector indices;
    FloatVector data;
    ShapeTuple2D shape;

    value_t time_1 = 0;
    value_t time_2 = 0;
    value_t time_3 = 0;
    value_t time_4 = 0;

    CSR(const IntVector &_indptr, const IntVector &_indices,
        const FloatVector &_data, const ShapeTuple2D &_shape)
        : indptr(_indptr), indices(_indices), data(_data), shape(_shape) {}

    // CSR(const index_t N, const index_t M, const uint64_t nnz) {
    //     index_t *_indptr = new index_t[N + 1];
    //     index_t *_indices = new index_t[nnz];
    //     value_t *_data = new value_t[nnz];
    //     index_t *_shape = new index_t[2];
    //     _shape[0] = N;
    //     _shape[1] = M;

    //     // These *have* to be size_t
    //     size_t shape_dense[1] = { (size_t)N + 1 };
    //     size_t shape_nnz[1] = { (size_t)nnz };
    //     size_t shape_scalar[1] = { (size_t)2 };

    //     indptr = IntVector(_indptr, /* ndim = */ 1, shape_dense);
    //     indices = IntVector(_indices, /* ndim = */ 1, shape_nnz);
    //     data = FloatVector(_data, /* ndim = */ 1, shape_nnz);
    //     shape = ShapeTuple2D(_shape, /* ndim = */ 1, shape_scalar);
    // }

    CSR(index_t *_indptr, index_t *_indices, value_t *_data,
        const index_t N, const index_t M, const uint64_t nnz, float time_1 = 0, float time_2 = 0, float time_3 = 0, float time_4 = 0) {
        index_t *_shape = new index_t[2];
        _shape[0] = N;
        _shape[1] = M;

        this->time_1 = time_1;
        this->time_2 = time_2;
        this->time_3 = time_3;
        this->time_4 = time_4;
        // These *have* to be size_t
        size_t shape_dense[1] = { (size_t)N + 1 };
        size_t shape_nnz[1] = { (size_t)nnz };
        size_t shape_scalar[1] = { (size_t)2 };

        nb::capsule owner_indptr(_indptr, cudaFreeWrapper);
        nb::capsule owner_indices(_indices, cudaFreeWrapper);
        nb::capsule owner_data(_data, cudaFreeWrapper);
        nb::capsule owner_shape(_shape, [](void *p) noexcept {
            delete[] (index_t *)p;
        });

        indptr = IntVector(_indptr, /* ndim = */ 1, shape_dense, owner_indptr,  nullptr, nb::dtype<index_t>(), /* explicitly set device type */ nb::device::cuda::value);
        indices = IntVector(_indices, /* ndim = */ 1, shape_nnz, owner_indices,  nullptr, nb::dtype<index_t>(), /* explicitly set device type */ nb::device::cuda::value);
        data = FloatVector(_data, /* ndim = */ 1, shape_nnz, owner_data,  nullptr, nb::dtype<value_t>(), /* explicitly set device type */ nb::device::cuda::value);
        shape = ShapeTuple2D(_shape, /* ndim = */ 1, shape_scalar, owner_shape,  nullptr, nb::dtype<index_t>(), /* explicitly set device type */ nb::device::cpu::value);
    }
};

template<typename index_t, typename value_t>
struct COO {
    using IntVector = nb::ndarray<nb::pytorch, index_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    using FloatVector = nb::ndarray<nb::pytorch, value_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    using ShapeTuple2D = nb::ndarray<nb::pytorch, index_t, nb::shape<2>, nb::c_contig, nb::device::cpu>;
    IntVector row;
    IntVector col;
    FloatVector data;
    ShapeTuple2D shape;

    COO(const IntVector &_row, const IntVector &_col,
        const FloatVector &_data, const ShapeTuple2D &_shape)
        : row(_row), col(_col), data(_data), shape(_shape) {}


    COO(index_t *_row, index_t *_col, value_t *_data,
        const index_t N, const index_t M, const uint64_t nnz) {
        index_t *_shape = new index_t[2];
        _shape[0] = N;
        _shape[1] = M;

        // These *have* to be size_t
        size_t shape_nnz[1] = { (size_t)nnz };
        size_t shape_scalar[1] = { (size_t)2 };

        nb::capsule owner_row(_row, cudaFreeWrapper);
        nb::capsule owner_col(_col, cudaFreeWrapper);
        nb::capsule owner_data(_data, cudaFreeWrapper);
        nb::capsule owner_shape(_shape, [](void *p) noexcept {
            delete[] (size_t *)p;
        });

        row = IntVector(_row, /* ndim = */ 1, shape_nnz, owner_row, nullptr, nb::dtype<index_t>(), /* explicitly set device type */ nb::device::cuda::value);
        col = IntVector(_col, /* ndim = */ 1, shape_nnz, owner_col, nullptr, nb::dtype<index_t>(), /* explicitly set device type */ nb::device::cuda::value);
        data = FloatVector(_data, /* ndim = */ 1, shape_nnz, owner_data, nullptr, nb::dtype<value_t>(), /* explicitly set device type */ nb::device::cuda::value);
        shape = ShapeTuple2D(_shape, /* ndim = */ 1, shape_scalar, owner_shape, nullptr, nb::dtype<index_t>(), /* explicitly set device type */ nb::device::cpu::value);
    }
};

template<typename index_t, typename coord_t, typename value_t>
struct CVector {
    using IntVector = nb::ndarray<nb::pytorch, coord_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    using FloatVector = nb::ndarray<nb::pytorch, value_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    IntVector indices;
    FloatVector data;
    struct _Shape {
        _Shape(const coord_t _size) : size(_size) {}
        const coord_t size;
        const coord_t operator()(const coord_t c) const {
            return size;
        }
    };
    coord_t size = 0;
    const _Shape shape;

    value_t time_1 = 0;
    value_t time_2 = 0;
    value_t time_3 = 0;

    CVector(const IntVector &_indices, const FloatVector &_data, const coord_t &_size)
        : indices(_indices), data(_data), size(_size), shape(_size) {}

    CVector(coord_t *_indices, value_t *_data, const coord_t N, const index_t nnz, float time_1 = 0, float time_2 = 0, float time_3 = 0) : shape(N) {
        // These *have* to be size_t
        size_t shape_nnz[1] = { (size_t)nnz };

        nb::capsule owner_indices(_indices, cudaFreeWrapper);
        nb::capsule owner_data(_data, cudaFreeWrapper);

        indices = IntVector(_indices, /* ndim = */ 1, shape_nnz, owner_indices, nullptr, nb::dtype<coord_t>(), /* explicitly set device type */ nb::device::cuda::value);
        data = FloatVector(_data, /* ndim = */ 1, shape_nnz, owner_data, nullptr, nb::dtype<value_t>(), /* explicitly set device type */ nb::device::cuda::value);
        size = N;
        this->time_1 = time_1;
        this->time_2 = time_2;
        this->time_3 = time_3;
    }
};

template<typename index_t, typename value_t>
struct DCSR {
    using IntVector = nb::ndarray<nb::pytorch, index_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    using FloatVector = nb::ndarray<nb::pytorch, value_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
    IntVector row_indices;   // dim_i_indices
    IntVector row_offsets;   // dim_j_offsets
    IntVector col_indices;   // dim_j_indices
    FloatVector data;        // values
    index_t nrows = 0;       // dim_i_size
    index_t ncols = 0;       // dim_j_size

    DCSR(const IntVector &_row_indices, const IntVector &_row_offsets,
         const IntVector &_col_indices, const FloatVector &_data,
         const index_t &_nrows, const index_t &_ncols)
        : row_indices(_row_indices), row_offsets(_row_offsets),
          col_indices(_col_indices), data(_data),
          nrows(_nrows), ncols(_ncols) {}

    DCSR(index_t *_row_indices, index_t *_row_offsets,
         index_t *_col_indices, value_t *_data,
         const index_t _nrows, const index_t _ncols,
         const index_t nrows_compressed, const index_t nnz) {
        nrows = _nrows;
        ncols = _ncols;

        size_t shape_nrows_comp[1] = { (size_t)nrows_compressed };
        size_t shape_offsets[1] = { (size_t)(nrows_compressed + 1) };
        size_t shape_nnz[1] = { (size_t)nnz };

        nb::capsule owner_ri(_row_indices, cudaFreeWrapper);
        nb::capsule owner_ro(_row_offsets, cudaFreeWrapper);
        nb::capsule owner_ci(_col_indices, cudaFreeWrapper);
        nb::capsule owner_data(_data, cudaFreeWrapper);

        row_indices = IntVector(_row_indices, 1, shape_nrows_comp, owner_ri, nullptr, nb::dtype<index_t>(), nb::device::cuda::value);
        row_offsets = IntVector(_row_offsets, 1, shape_offsets, owner_ro, nullptr, nb::dtype<index_t>(), nb::device::cuda::value);
        col_indices = IntVector(_col_indices, 1, shape_nnz, owner_ci, nullptr, nb::dtype<index_t>(), nb::device::cuda::value);
        data = FloatVector(_data, 1, shape_nnz, owner_data, nullptr, nb::dtype<value_t>(), nb::device::cuda::value);
    }
};

// std::min does weird stuff sometimes, use this instead.
// template<typename T, typename S>
// uint64_t min(const T &t, const S &s) {
//     return std::min((uint64_t)t, (uint64_t)s);
// }

// for locate.
// template<typename index_t, typename value_t>
// index_t binary_search(const value_t &value, index_t low, index_t high, const value_t *array) {
//     while (low < high) {
//         const index_t mid = (uint64_t)low + ((uint64_t)high - (uint64_t)low) / 2;
//         if (array[mid] == value) {
//             return mid;
//         } else if (array[mid] < value) {
//             low = mid + 1;
//         } else {
//             // array[mid] > value
//             high = mid;
//         }
//     }
//     return low;
// }



namespace io_coo {

template<typename T>
using Array = nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

template<typename index_t, typename value_t>
struct COO2D {
    Array<index_t> row, col;
    Array<value_t> data;
    index_t N, M;
};


template<typename index_t, typename value_t>
COO2D<index_t, value_t> parse2D(const std::string filename) {
    COO2D<index_t, value_t> ret;

    std::ifstream file(filename);
    // file.open(filename, std::ios::out);   

    if (file.is_open()) {
        // std::string fline;
        // if (!std::getline(file, fline)) {
        //     std::cerr << "mtx file is empty" << std::endl;
        //     return ret;
        // }

        // std::stringstream flineStream(fline);

        std::string head, type, formats, field, symmetry;
        // formats = [coordinate array]
        // type = [matrix tensor]
        // field = [real integer complex pattern]
        // symmetry = [general symmetric skew-symmetric Hermitian]
        file >> head >> type >> formats >> field >> symmetry;
        if (head != "%%MatrixMarket") {
            std::cerr << "Unknown header of MatrixMarket" << std::endl;
            std::cerr << head << std::endl;
            exit(-1);
            return ret;
        } else if (type != "matrix" && type != "tensor") {
            std::cerr << "Unknown type of MatrixMarket" << std::endl;
            std::cerr << type << std::endl;
            exit(-1);
            return ret;
        } else if (field != "real") {
            std::cerr << "MatrixMarket field not available" << std::endl;
            std::cerr << field << std::endl;
            exit(-1);
            return ret;
        } else if (symmetry != "general" && symmetry != "symmetric" && symmetry != "skew-symmetric") {
            // TODO: handle symmetric.
            std::cerr << "MatrixMarket symmetry not available" << std::endl;
            std::cerr << symmetry << std::endl;
            exit(-1);
            return ret;
        }
        file.ignore(2048, '\n');

        while (file.peek() == '%') {
            // std::string line;
            // std::getline(file, line);
            file.ignore(2048, '\n');
            // file.ignore(1);
            // std::cerr << "ignoring line: " << line << "\n";
        }

        // std::cout << "peeking: " << file.peek() << "\n";
        uint64_t N = 0, M = 0, nnz = 0;
        file >> N >> M >> nnz;

        const bool is_skew = (symmetry == "skew-symmetric");
        const bool is_symm = (symmetry == "symmetric") || is_skew;
        // const uint64_t nnz_alloc = symm ? nnz * 2 : nnz;
        index_t *c0 = new index_t[nnz];
        index_t *c1 = new index_t[nnz];
        value_t *v = new value_t[nnz];

        if (is_symm) {
            uint64_t num_symm = 0;
            for (uint64_t c = 0; c < nnz; c++) {
                uint64_t i, j;
                double val;
                // Only "pattern" has this issue.
                // file >> i >> j;
                // if (file.peek() == '\n') {
                //     val = 1.0f;
                // } else {
                //     file >> val;
                // }
                file >> i >> j >> val;
                c0[c] = (index_t)(i - 1);
                c1[c] = (index_t)(j - 1);
                v[c] = (value_t)val;
                num_symm += (i != j);
            }
            // std::cerr << "Number nnz: " << nnz << "\n";
            // std::cerr << "Number symmetric: " << num_symm << "\n";
            if (num_symm != 0) {
                index_t *_c0 = new index_t[nnz + num_symm];
                index_t *_c1 = new index_t[nnz + num_symm];
                value_t *_v = new value_t[nnz + num_symm];
                // TODO: either we do this n^2 or we don't sort.
                // I'm going with the latter for now.
                std::memcpy(_c0, c0, nnz * sizeof(index_t));
                std::memcpy(_c1, c1, nnz * sizeof(index_t));
                std::memcpy(_v, v, nnz * sizeof(value_t));
                delete[] c0;
                delete[] c1;
                delete[] v;
                c0 = _c0;
                c1 = _c1;
                v = _v;
                uint64_t counter = nnz;

                for (uint64_t c = 0; c < nnz; c++) {
                    if (c0[c] != c1[c]) {
                        c0[counter] = c1[c];
                        c1[counter] = c0[c];
                        if (is_skew) {
                            v[counter] = -v[c];
                        } else {
                            v[counter] = v[c];
                        }
                        counter++;
                    }
                }
                assert(counter == nnz + num_symm);
                nnz += num_symm;
            }
        } else {
            for (uint64_t c = 0; c < nnz; c++) {
                uint64_t i, j;
                double val;
                // Only "pattern" has this issue.
                // file >> i >> j;
                // if (file.peek() == '\n') {
                //     val = 1.0f;
                // } else {
                //     file >> val;
                // }
                file >> i >> j >> val;
                c0[c] = (index_t)(i - 1);
                c1[c] = (index_t)(j - 1);
                v[c] = (value_t)val;
            }
            //assert(counter == nnz);
        }

        file.close();

        // These *have* to be size_t
        size_t shape_nnz[1] = { (size_t)nnz };
        nb::capsule owner_c0(c0, [](void *p) noexcept {
            delete[] (index_t *)p;
        });
        nb::capsule owner_c1(c1, [](void *p) noexcept {
            delete[] (index_t *)p;
        });
        nb::capsule owner_v(v, [](void *p) noexcept {
            delete[] (value_t *)p;
        });

        ret.row = Array<index_t>(c0, /* ndim = */ 1, shape_nnz, owner_c0);
        ret.col = Array<index_t>(c1, /* ndim = */ 1, shape_nnz, owner_c1);
        ret.data = Array<float>(v, /* ndim = */ 1, shape_nnz, owner_v);
        ret.N = N;
        ret.M = M;
    } else {
        std::cerr << "Failed to open (and parse from) file: " << filename << std::endl;
        assert(false);
    }
    return ret;
}


}  // namespace io_coo
