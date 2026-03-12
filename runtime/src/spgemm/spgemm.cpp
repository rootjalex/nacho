#include "spgemm.hpp"

template<typename index_t, typename value_t>
CSR<index_t, value_t> spgemm(CSR<index_t, value_t> A, CSR<index_t, value_t> B, bool use_cusparse){

    index_t * C_nnz = new index_t;

    index_t * C_row_offsets;
    index_t* C_indices;
    value_t* C_values;

    value_t * D_times = new value_t[4];

    if (use_cusparse) {
        spgemm_cusparse_impl(
            CSRMatrix<index_t, value_t>(A.indptr.data(), A.indices.data(), A.data.data(), A.shape(0), A.shape(1), A.indices.shape(0)),
            CSRMatrix<index_t, value_t>(B.indptr.data(), B.indices.data(), B.data.data(), B.shape(0), B.shape(1), B.indices.shape(0)),
            C_row_offsets, C_indices, C_values, C_nnz
        );
    } else {
    spgemm_impl(
        CSRMatrix<index_t, value_t>(A.indptr.data(), A.indices.data(), A.data.data(), A.shape(0), A.shape(1), A.indices.shape(0)),
        CSRMatrix<index_t, value_t>(B.indptr.data(), B.indices.data(), B.data.data(), B.shape(0), B.shape(1), B.indices.shape(0)),
        C_row_offsets, C_indices, C_values, C_nnz, D_times
    );
    }

    CSR<index_t, value_t> C = CSR<index_t, value_t>(C_row_offsets, C_indices, C_values, A.shape(0), B.shape(1), *C_nnz,  D_times[0], D_times[1], D_times[2], D_times[3]);

    return C;
};

template CSR<int, float> spgemm(CSR<int, float> A, CSR<int, float> B, bool use_cusparse);
