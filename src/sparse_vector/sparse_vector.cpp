#include "sparse_vector.hpp"

template<typename index_t, typename coord_t, typename value_t>
CVector<index_t, coord_t, value_t> nb_3dmergepath_test(CVector<index_t, coord_t, value_t> A, CVector<index_t, coord_t, value_t> B, CVector<index_t, coord_t, value_t> C, int num_fused, bool expr) {

    //printf("Start mergetest\n");
    index_t * D_nnz = new index_t;

    coord_t* D_indices;
    value_t* D_values;

    value_t * D_times = new value_t[3];

    if (expr) {
    sparse_vector_fusion_test(
        SparseVector<index_t, coord_t, value_t>(A.indices.data(), A.data.data(), A.size, A.indices.shape(0)),
        SparseVector<index_t, coord_t, value_t>(B.indices.data(), B.data.data(), B.size, B.indices.shape(0)),
        SparseVector<index_t, coord_t, value_t>(C.indices.data(), C.data.data(), C.size, C.indices.shape(0)),
        D_indices, D_values, D_nnz, num_fused, D_times);
    } else {
        sparse_vector_ab_c(
        SparseVector<index_t, coord_t, value_t>(A.indices.data(), A.data.data(), A.size, A.indices.shape(0)),
        SparseVector<index_t, coord_t, value_t>(B.indices.data(), B.data.data(), B.size, B.indices.shape(0)),
        SparseVector<index_t, coord_t, value_t>(C.indices.data(), C.data.data(), C.size, C.indices.shape(0)),
        D_indices, D_values, D_nnz, num_fused, D_times);
    }
    //printf("%f %f %f\n", D_times[0], D_times[1], D_times[2]);
    CVector<index_t, coord_t, value_t> D = CVector<index_t, coord_t, value_t>(D_indices, D_values, A.size, *D_nnz, D_times[0], D_times[1], D_times[2]);
    //print_cuda(D_values,5);
    return D;
        
};

template CVector<int, int, float> nb_3dmergepath_test<int, int, float>(CVector<int, int, float> A, CVector<int, int, float> B, CVector<int, int, float> C, int num_fused, bool expr);
template CVector<int, int64_t, float> nb_3dmergepath_test<int, int64_t, float>(CVector<int, int64_t, float> A, CVector<int, int64_t, float> B, CVector<int, int64_t, float> C, int num_fused, bool expr);
