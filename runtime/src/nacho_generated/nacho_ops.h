#pragma once

// Forward declarations for nacho-generated flat-API wrappers.
// These are defined in the corresponding .cu files and explicitly
// instantiated for <int, float>.

template <typename index_t, typename value_t>
void sparse_vec_mul(index_t a_dim_i_size, index_t a_dim_i_length,
                    index_t *a_dim_i_indices, value_t *a_values, index_t a_nnz,
                    index_t b_dim_i_size, index_t b_dim_i_length,
                    index_t *b_dim_i_indices, value_t *b_values, index_t b_nnz,
                    index_t result_dim_i_size, index_t &out_nnz,
                    index_t *&out_dim_i_indices, value_t *&out_values);
