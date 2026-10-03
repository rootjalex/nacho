#include "baseline_types.h"
#include "spmv/spmv_baselines.h"

namespace nacho {
namespace baselines {

namespace {

ArrayGPU<float> gpu_spmv_cusparse(const CSRGpu &a, const ArrayGPU<float> &x) {

    float *y = nullptr;

    // a stores CSR(A^T), which has the same physical arrays as CSC(A).
    // If A is m x k, then A^T is k x m.
    const int32_t k = a.shape.data()[0];
    const int32_t m = a.shape.data()[1];

    gpu_spmv_cusparse_f32(
        m, k,
        const_cast<int32_t *>(a.indptr.data()),  // CSC column offsets
        const_cast<int32_t *>(a.indices.data()), // CSC row indices
        const_cast<float *>(a.values.data()),    // CSC values
        static_cast<int64_t>(a.values.shape(0)), // nnz(A)
        const_cast<float *>(x.data()),           // dense input vector x
        y                                        // output vector y
    );

    // y was allocated by gpu_spmv_cusparse_f32.
    // Adopt that GPU allocation and return it as an ArrayGPU<float>.
    return adopt_gpu<float>(y, static_cast<size_t>(m));
}

} // namespace

void register_spmv_baselines(nb::module_ &m) {
    m.def("gpu_spmv_cusparse_f32", &gpu_spmv_cusparse);
}

} // namespace baselines
} // namespace nacho
