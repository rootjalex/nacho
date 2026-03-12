#pragma once

// Forward declarations for nacho-generated flat-API wrappers.
// These are defined in the corresponding .cu files and explicitly
// instantiated for <int, float>.

// 2-operand sparse vector: a * b
template <typename index_t, typename value_t>
void sparse_vec_mul(index_t a_dim_i_size, index_t a_dim_i_length,
                    index_t *a_dim_i_indices, value_t *a_values, index_t a_nnz,
                    index_t b_dim_i_size, index_t b_dim_i_length,
                    index_t *b_dim_i_indices, value_t *b_values, index_t b_nnz,
                    index_t result_dim_i_size, index_t &out_nnz,
                    index_t *&out_dim_i_indices, value_t *&out_values);

// 2-operand sparse vector: a + b
template <typename index_t, typename value_t>
void sparse_vec_add(index_t a_dim_i_size, index_t a_dim_i_length,
                    index_t *a_dim_i_indices, value_t *a_values, index_t a_nnz,
                    index_t b_dim_i_size, index_t b_dim_i_length,
                    index_t *b_dim_i_indices, value_t *b_values, index_t b_nnz,
                    index_t result_dim_i_size, index_t &out_nnz,
                    index_t *&out_dim_i_indices, value_t *&out_values);

// 3-operand sparse vector: (a + b) * c
template <typename index_t, typename value_t>
void sparse_vec_apb_c(index_t a_dim_i_size, index_t a_dim_i_length,
                      index_t *a_dim_i_indices, value_t *a_values, index_t a_nnz,
                      index_t b_dim_i_size, index_t b_dim_i_length,
                      index_t *b_dim_i_indices, value_t *b_values, index_t b_nnz,
                      index_t c_dim_i_size, index_t c_dim_i_length,
                      index_t *c_dim_i_indices, value_t *c_values, index_t c_nnz,
                      index_t result_dim_i_size, index_t &out_nnz,
                      index_t *&out_dim_i_indices, value_t *&out_values);

// 3-operand sparse vector: (a * b) + c
template <typename index_t, typename value_t>
void sparse_vec_ab_pc(index_t a_dim_i_size, index_t a_dim_i_length,
                      index_t *a_dim_i_indices, value_t *a_values, index_t a_nnz,
                      index_t b_dim_i_size, index_t b_dim_i_length,
                      index_t *b_dim_i_indices, value_t *b_values, index_t b_nnz,
                      index_t c_dim_i_size, index_t c_dim_i_length,
                      index_t *c_dim_i_indices, value_t *c_values, index_t c_nnz,
                      index_t result_dim_i_size, index_t &out_nnz,
                      index_t *&out_dim_i_indices, value_t *&out_values);

// 2-operand DCSR: A * B
template <typename index_t, typename value_t>
void dcsr_mul(index_t A_dim_i_size, index_t A_dim_j_size,
              index_t A_dim_i_length, index_t *A_dim_i_indices,
              index_t *A_dim_j_offsets, index_t A_dim_j_length,
              index_t *A_dim_j_indices, value_t *A_values, index_t A_nnz,
              index_t B_dim_i_size, index_t B_dim_j_size,
              index_t B_dim_i_length, index_t *B_dim_i_indices,
              index_t *B_dim_j_offsets, index_t B_dim_j_length,
              index_t *B_dim_j_indices, value_t *B_values, index_t B_nnz,
              index_t result_dim_i_size, index_t result_dim_j_size,
              index_t &out_nnz, index_t &out_dim_i_length,
              index_t *&out_dim_j_indices, index_t *&out_dim_j_offsets,
              index_t *&out_dim_i_indices, value_t *&out_values);

// 2-operand DCSR: A + B
template <typename index_t, typename value_t>
void dcsr_add(index_t A_dim_i_size, index_t A_dim_j_size,
              index_t A_dim_i_length, index_t *A_dim_i_indices,
              index_t *A_dim_j_offsets, index_t A_dim_j_length,
              index_t *A_dim_j_indices, value_t *A_values, index_t A_nnz,
              index_t B_dim_i_size, index_t B_dim_j_size,
              index_t B_dim_i_length, index_t *B_dim_i_indices,
              index_t *B_dim_j_offsets, index_t B_dim_j_length,
              index_t *B_dim_j_indices, value_t *B_values, index_t B_nnz,
              index_t result_dim_i_size, index_t result_dim_j_size,
              index_t &out_nnz, index_t &out_dim_i_length,
              index_t *&out_dim_j_indices, index_t *&out_dim_j_offsets,
              index_t *&out_dim_i_indices, value_t *&out_values);

// 2-operand CSR: A + B
template <typename index_t, typename value_t>
void csr_add(index_t A_dim_i_size, index_t A_dim_j_size,
             index_t *A_dim_j_offsets, index_t A_dim_j_length,
             index_t *A_dim_j_indices, value_t *A_values, index_t A_nnz,
             index_t B_dim_i_size, index_t B_dim_j_size,
             index_t *B_dim_j_offsets, index_t B_dim_j_length,
             index_t *B_dim_j_indices, value_t *B_values, index_t B_nnz,
             index_t result_dim_i_size, index_t result_dim_j_size,
             index_t &out_nnz, index_t *&out_dim_j_indices,
             index_t *&out_dim_j_offsets, value_t *&out_values);
