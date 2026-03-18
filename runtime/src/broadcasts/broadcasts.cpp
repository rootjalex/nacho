#include "broadcasts.hpp"

template<typename index_t, typename coord_t, typename value_t>
CSR<index_t, value_t> broadcast_xA(CVector<index_t, coord_t, value_t> x, CSR<index_t, value_t> A) {

    //printf("Start mergetest\n");
    index_t * D_nnz = new index_t;

    index_t * D_row_offsets;
    coord_t* D_indices;
    value_t* D_values;

    broadcast_xA_impl(
        SparseVector<index_t, coord_t, value_t>(x.indices.data(), x.data.data(), x.size, x.indices.shape(0)),
        CSRMatrix<index_t, value_t>(A.indptr.data(), A.indices.data(), A.data.data(), A.shape(0), A.shape(1), A.indices.shape(0)),
        D_row_offsets, D_indices, D_values, D_nnz
    );


    //printf("%f %f %f\n", D_times[0], D_times[1], D_times[2]);
    CSR<index_t, value_t> D = CSR<index_t, value_t>(D_row_offsets, D_indices, D_values, A.shape(0), A.shape(1), *D_nnz);
    
    return D;
        
};

template CSR<int, float> broadcast_xA(CVector<int, int, float> x, CSR<int, float> A);
