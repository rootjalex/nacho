#include "register_runtime.h"

#include "io/parse.h"

namespace nacho {
namespace runtime {

namespace {

void register_io(nb::module_ &m) {
    nb::class_<io::COO2D<int32_t, float>>(m, "COO2D")
        .def_ro("row", &io::COO2D<int32_t, float>::row, nb::rv_policy::reference)
        .def_ro("col", &io::COO2D<int32_t, float>::col, nb::rv_policy::reference)
        .def_ro("data", &io::COO2D<int32_t, float>::data, nb::rv_policy::reference)
        .def_ro("N", &io::COO2D<int32_t, float>::N, nb::rv_policy::reference)
        .def_ro("M", &io::COO2D<int32_t, float>::M, nb::rv_policy::reference);

    nb::class_<io::COO3D<int32_t, float>>(m, "COO3D_parsed")
        .def_ro("row", &io::COO3D<int32_t, float>::row, nb::rv_policy::reference)
        .def_ro("col", &io::COO3D<int32_t, float>::col, nb::rv_policy::reference)
        .def_ro("dep", &io::COO3D<int32_t, float>::dep, nb::rv_policy::reference)
        .def_ro("data", &io::COO3D<int32_t, float>::data, nb::rv_policy::reference)
        .def_ro("N", &io::COO3D<int32_t, float>::N, nb::rv_policy::reference)
        .def_ro("M", &io::COO3D<int32_t, float>::M, nb::rv_policy::reference)
        .def_ro("K", &io::COO3D<int32_t, float>::K, nb::rv_policy::reference)
        .def_ro("nnz", &io::COO3D<int32_t, float>::nnz, nb::rv_policy::reference);

    m.def("parse2D", &io::parse2D<int32_t, float>);
    m.def("parse3D_i32_f32", &io::parse3D<int32_t, float>);
}

} // namespace

void register_runtime(nb::module_ &m) {
    register_io(m);
}

} // namespace runtime
} // namespace nacho
