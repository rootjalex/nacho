#include <nanobind/nanobind.h>

#include "csr_add/csr_add.hpp"
#include "coo_add/coo_add.hpp"
#include "sparse_vector/sparse_vector.hpp"
#include "broadcasts/broadcasts.hpp"
#include "spgemm/spgemm.hpp"
#include "mergepath_utils/mergepath_utils.h"
#include "nacho_generated/nacho_ops.hpp"
#include "nb_utils.hpp"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(nacho_runtime_ext, m) {
    nb::class_<io_coo::COO2D<int32_t, float>>(m, "COO2D")
        .def_ro("row", &io_coo::COO2D<int32_t, float>::row, nb::rv_policy::reference)
        .def_ro("col", &io_coo::COO2D<int32_t, float>::col, nb::rv_policy::reference)
        .def_ro("data", &io_coo::COO2D<int32_t, float>::data, nb::rv_policy::reference)
        .def_ro("N", &io_coo::COO2D<int32_t, float>::N, nb::rv_policy::reference)
        .def_ro("M", &io_coo::COO2D<int32_t, float>::M, nb::rv_policy::reference);

    m.def("parse2D", &io_coo::parse2D<int32_t, float>);
    
    
     nb::class_<CSR<int32_t, float>>(m, "CSR")
        .def(nb::init<const Int32Vector &, const Int32Vector &, const Float32Vector &, const Int32TupleCPU &>())
        .def_ro("indptr", &CSR<int32_t, float>::indptr, nb::rv_policy::reference)
        .def_ro("indices", &CSR<int32_t, float>::indices, nb::rv_policy::reference)
        .def_ro("data", &CSR<int32_t, float>::data, nb::rv_policy::reference)
        .def_ro("shape", &CSR<int32_t, float>::shape, nb::rv_policy::reference)
        .def_ro("time_1", &CSR<int32_t, float>::time_1, nb::rv_policy::reference)
        .def_ro("time_2", &CSR<int32_t, float>::time_2, nb::rv_policy::reference)
        .def_ro("time_3", &CSR<int32_t, float>::time_3, nb::rv_policy::reference)
        .def_ro("time_4", &CSR<int32_t, float>::time_4, nb::rv_policy::reference);

    nb::class_<COO<int32_t, float>>(m, "COO")
        .def(nb::init<const Int32Vector &, const Int32Vector &, const Float32Vector &, const Int32TupleCPU &>())
        .def_ro("row", &COO<int32_t, float>::row, nb::rv_policy::reference)
        .def_ro("col", &COO<int32_t, float>::col, nb::rv_policy::reference)
        .def_ro("data", &COO<int32_t, float>::data, nb::rv_policy::reference)
        .def_ro("shape", &COO<int32_t, float>::shape, nb::rv_policy::reference);

    nb::class_<CVector<int32_t, int32_t, float>>(m, "CVector")
        .def(nb::init<const Int32Vector &, const Float32Vector &, const int32_t &>())
        .def_ro("indices", &CVector<int32_t, int32_t, float>::indices, nb::rv_policy::reference)
        .def_ro("data", &CVector<int32_t, int32_t, float>::data, nb::rv_policy::reference)
        .def_ro("size", &CVector<int32_t, int32_t, float>::size, nb::rv_policy::reference)
        .def_ro("time_1", &CVector<int32_t, int32_t, float>::time_1, nb::rv_policy::reference)
        .def_ro("time_2", &CVector<int32_t, int32_t, float>::time_2, nb::rv_policy::reference)
        .def_ro("time_3", &CVector<int32_t, int32_t, float>::time_3, nb::rv_policy::reference);

    nb::class_<COO<int64_t, float>>(m, "COO64")
        .def(nb::init<const Int64Vector &, const Int64Vector &, const Float32Vector &, const Int64TupleCPU &>())
        .def_ro("row", &COO<int64_t, float>::row, nb::rv_policy::reference)
        .def_ro("col", &COO<int64_t, float>::col, nb::rv_policy::reference)
        .def_ro("data", &COO<int64_t, float>::data, nb::rv_policy::reference)
        .def_ro("shape", &COO<int64_t, float>::shape, nb::rv_policy::reference);

     nb::class_<CSR<int64_t, float>>(m, "CSR64")
        .def(nb::init<const Int64Vector &, const Int64Vector &, const Float32Vector &, const Int64TupleCPU &>())
        .def_ro("indptr", &CSR<int64_t, float>::indptr, nb::rv_policy::reference)
        .def_ro("indices", &CSR<int64_t, float>::indices, nb::rv_policy::reference)
        .def_ro("data", &CSR<int64_t, float>::data, nb::rv_policy::reference)
        .def_ro("shape", &CSR<int64_t, float>::shape, nb::rv_policy::reference);

    nb::class_<CVector<int32_t, int64_t, float>>(m, "CVector64")
        .def(nb::init<const Int64Vector &, const Float32Vector &, const int64_t &>())
        .def_ro("indices", &CVector<int32_t, int64_t, float>::indices, nb::rv_policy::reference)
        .def_ro("data", &CVector<int32_t, int64_t, float>::data, nb::rv_policy::reference)
        .def_ro("size", &CVector<int32_t, int64_t, float>::size, nb::rv_policy::reference)
        .def_ro("time_1", &CVector<int32_t, int64_t, float>::time_1, nb::rv_policy::reference)
        .def_ro("time_2", &CVector<int32_t, int64_t, float>::time_2, nb::rv_policy::reference)
        .def_ro("time_3", &CVector<int32_t, int64_t, float>::time_3, nb::rv_policy::reference);

    nb::class_<DCSR<int32_t, float>>(m, "DCSR")
        .def(nb::init<const Int32Vector &, const Int32Vector &, const Int32Vector &, const Float32Vector &, const int32_t &, const int32_t &>())
        .def_ro("row_indices", &DCSR<int32_t, float>::row_indices, nb::rv_policy::reference)
        .def_ro("row_offsets", &DCSR<int32_t, float>::row_offsets, nb::rv_policy::reference)
        .def_ro("col_indices", &DCSR<int32_t, float>::col_indices, nb::rv_policy::reference)
        .def_ro("data", &DCSR<int32_t, float>::data, nb::rv_policy::reference)
        .def_ro("nrows", &DCSR<int32_t, float>::nrows, nb::rv_policy::reference)
        .def_ro("ncols", &DCSR<int32_t, float>::ncols, nb::rv_policy::reference);

    // Add CSR matrix addition function
    m.def("gpu_csr_add_f32", &nb_gpu_csr_add_f32);
    m.def("gpu_coo_add_f32", &nb_gpu_coo_add_f32);
    m.def("gpu_sss_mergepath_test", (&nb_3dmergepath_test<int, int, float>));
    m.def("gpu_sss_mergepath_test64", (&nb_3dmergepath_test<int, int64_t, float>));
    m.def("gpu_broadcast_xA", &broadcast_xA<int32_t, int32_t, float>);
    m.def("spgemm", &spgemm<int32_t, float>);

    // nacho-generated operations
    m.def("nacho_sparse_vec_mul", &nacho_sparse_vec_mul_nb);
    m.def("nacho_sparse_vec_add", &nacho_sparse_vec_add_nb);
    m.def("nacho_sparse_vec_apb_c", &nacho_sparse_vec_apb_c_nb);
    m.def("nacho_sparse_vec_ab_pc", &nacho_sparse_vec_ab_pc_nb);
    m.def("nacho_dcsr_mul", &nacho_dcsr_mul_nb);
    m.def("nacho_dcsr_add", &nacho_dcsr_add_nb);
    m.def("nacho_csr_add", &nacho_csr_add_nb);

    // handwritten
    // m.def("cv_collapse_coo", &cv_collapse_coo<uint32_t, uint32_t, float>);
    // m.def("cv_collapse_coo32to64", &cv_collapse_coo<uint32_t, uint64_t, float>);
    // m.def("coo_hstack_coo_coo", &coo_hstack_coo_coo<uint32_t, float>);
    // m.def("coo_vstack_coo_coo", &coo_vstack_coo_coo<uint32_t, float>);
    // m.def("csr_hstack_csr_csr", &csr_hstack_csr_csr<uint32_t, float>);
    // m.def("csr_vstack_csr_csr", &csr_vstack_csr_csr<uint32_t, float>);
    // m.def("csr_slice_1d_csr", &csr_slice_1d_csr<uint32_t, float>);
    // // portable
    // m.def("coo_split_cv", &coo_split_cv<uint32_t, uint32_t, float>);
    // m.def("coo_split_cv64to32", &coo_split_cv<uint32_t, uint64_t, float>);
    // m.def("csr_split_cv", &csr_split_cv<uint32_t, uint32_t, float>);
    // m.def("csr_split_cv64to32", &csr_split_cv<uint32_t, uint64_t, float>);
    // m.def("csr_hstack_coo_coo", &csr_hstack_coo_coo<uint32_t, float>);
    // m.def("coo_vstack_csr_csr", &coo_vstack_csr_csr<uint32_t, float>);
    // m.def("coo_hstack_csr_csr", &coo_hstack_csr_csr<uint32_t, float>);
    // m.def("coo_slice_1d_csr", &coo_slice_1d_csr<uint32_t, float>);
    // fusion
   }
