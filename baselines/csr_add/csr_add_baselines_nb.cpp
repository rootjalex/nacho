#include "baseline_types.h"
#include "csr_add/csr_add_baselines.h"
#include "csr_add/csr_add_taco_cpu.h"

namespace nacho {
namespace baselines {

namespace {

using RawGpuKernel = void (*)(int*, int*, int*, float*, uint64_t,
                              int*, int*, float*, uint64_t,
                              int*&, int*&, float*&, int*);

CSRGpu run_gpu(RawGpuKernel kernel, const CSRGpu &a, const CSRGpu &b) {
    int32_t *row_offsets = nullptr, *col_indices = nullptr;
    float *values = nullptr;
    int32_t nnz = 0;

    kernel(const_cast<int32_t *>(a.shape.data()),
           const_cast<int32_t *>(a.indptr.data()), const_cast<int32_t *>(a.indices.data()),
           const_cast<float *>(a.values.data()), a.values.shape(0),
           const_cast<int32_t *>(b.indptr.data()), const_cast<int32_t *>(b.indices.data()),
           const_cast<float *>(b.values.data()), b.values.shape(0),
           row_offsets, col_indices, values, &nnz);

    const int32_t rows = a.shape.data()[0];
    return CSRGpu{
        adopt_gpu<int32_t>(row_offsets, (size_t)rows + 1),
        adopt_gpu<int32_t>(col_indices, (size_t)nnz),
        adopt_gpu<float>(values, (size_t)nnz),
        make_shape<2>({rows, a.shape.data()[1]}),
    };
}

CSRGpu gpu_csr_add_taco(const CSRGpu &a, const CSRGpu &b) {
    return run_gpu(&gpu_csr_add_taco_f32, a, b);
}

CSRGpu gpu_csr_add_cusparse(const CSRGpu &a, const CSRGpu &b) {
    return run_gpu(&gpu_csr_add_cusparse_f32, a, b);
}

CSRCpu cpu_csr_add_taco(const CSRCpu &a, const CSRCpu &b) {
    int32_t *row_offsets = nullptr, *col_indices = nullptr;
    float *values = nullptr;
    int32_t nnz = 0;

    ::cpu_csr_add_taco<int32_t, float>(
        a.shape.data(),
        a.indptr.data(), a.indices.data(), a.values.data(), a.values.shape(0),
        b.indptr.data(), b.indices.data(), b.values.data(), b.values.shape(0),
        row_offsets, col_indices, values, nnz);

    const int32_t rows = a.shape.data()[0];
    return CSRCpu{
        adopt_cpu<int32_t>(row_offsets, (size_t)rows + 1),
        adopt_cpu<int32_t>(col_indices, (size_t)nnz),
        adopt_cpu<float>(values, (size_t)nnz),
        make_shape<2>({rows, a.shape.data()[1]}),
    };
}

} // namespace

void register_baseline_types(nb::module_ &m) {
    nb::class_<CSRCpu>(m, "BaselineCSR_cpu")
        .def(nb::init<const ArrayCPU<int32_t> &, const ArrayCPU<int32_t> &,
                      const ArrayCPU<float> &, const ShapeTuple<2> &>())
        .def_ro("indptr", &CSRCpu::indptr, nb::rv_policy::reference)
        .def_ro("indices", &CSRCpu::indices, nb::rv_policy::reference)
        .def_ro("values", &CSRCpu::values, nb::rv_policy::reference)
        .def_ro("shape", &CSRCpu::shape, nb::rv_policy::reference);

    nb::class_<CSRGpu>(m, "BaselineCSR_gpu")
        .def(nb::init<const ArrayGPU<int32_t> &, const ArrayGPU<int32_t> &,
                      const ArrayGPU<float> &, const ShapeTuple<2> &>())
        .def_ro("indptr", &CSRGpu::indptr, nb::rv_policy::reference)
        .def_ro("indices", &CSRGpu::indices, nb::rv_policy::reference)
        .def_ro("values", &CSRGpu::values, nb::rv_policy::reference)
        .def_ro("shape", &CSRGpu::shape, nb::rv_policy::reference);
}

void register_csr_add_baselines(nb::module_ &m) {
    m.def("cpu_csr_add_taco_f32", &cpu_csr_add_taco);
    m.def("gpu_csr_add_taco_f32", &gpu_csr_add_taco);
    m.def("gpu_csr_add_cusparse_f32", &gpu_csr_add_cusparse);
}

} // namespace baselines
} // namespace nacho
