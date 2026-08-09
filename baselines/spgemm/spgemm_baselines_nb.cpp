#include "baseline_types.h"
#include "spgemm/spgemm_baselines.h"

namespace nacho {
namespace baselines {

namespace {

CSRGpu gpu_spgemm_cusparse(const CSRGpu &a, const CSRGpu &b) {
    int32_t *row_offsets = nullptr, *col_indices = nullptr;
    float *values = nullptr;
    int32_t nnz = 0;

    const int32_t m = a.shape.data()[0];
    const int32_t k = a.shape.data()[1];
    const int32_t n = b.shape.data()[1];

    gpu_spgemm_cusparse_f32(m, k, n,
        const_cast<int32_t *>(a.indptr.data()), const_cast<int32_t *>(a.indices.data()),
        const_cast<float *>(a.values.data()), (int64_t)a.values.shape(0),
        const_cast<int32_t *>(b.indptr.data()), const_cast<int32_t *>(b.indices.data()),
        const_cast<float *>(b.values.data()), (int64_t)b.values.shape(0),
        row_offsets, col_indices, values, &nnz);

    return CSRGpu{
        adopt_gpu<int32_t>(row_offsets, (size_t)m + 1),
        adopt_gpu<int32_t>(col_indices, (size_t)nnz),
        adopt_gpu<float>(values, (size_t)nnz),
        make_shape<2>({m, n}),
    };
}

} // namespace

void register_spgemm_baselines(nb::module_ &m) {
    m.def("gpu_spgemm_cusparse_f32", &gpu_spgemm_cusparse);
}

} // namespace baselines
} // namespace nacho
