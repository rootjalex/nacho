#include <cstdlib>

// Forward declarations of the generated flat-API functions
template <typename index_t, typename value_t>
void sparse_vec_mul(index_t, index_t, index_t*, value_t*, index_t,
                    index_t, index_t, index_t*, value_t*, index_t,
                    index_t, index_t&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void sparse_vec_add(index_t, index_t, index_t*, value_t*, index_t,
                    index_t, index_t, index_t*, value_t*, index_t,
                    index_t, index_t&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void sparse_vec_apb_c(index_t, index_t, index_t*, value_t*, index_t,
                      index_t, index_t, index_t*, value_t*, index_t,
                      index_t, index_t, index_t*, value_t*, index_t,
                      index_t, index_t&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void sparse_vec_ab_pc(index_t, index_t, index_t*, value_t*, index_t,
                      index_t, index_t, index_t*, value_t*, index_t,
                      index_t, index_t, index_t*, value_t*, index_t,
                      index_t, index_t&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void csr_add(index_t, index_t, index_t*, index_t, index_t*, value_t*, index_t,
             index_t, index_t, index_t*, index_t, index_t*, value_t*, index_t,
             index_t, index_t, index_t&, index_t*&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void dcsr_mul(index_t, index_t, index_t, index_t*, index_t*, index_t, index_t*,
              value_t*, index_t, index_t, index_t, index_t, index_t*, index_t*,
              index_t, index_t*, value_t*, index_t, index_t, index_t, index_t&,
              index_t&, index_t*&, index_t*&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void dcsr_add(index_t, index_t, index_t, index_t*, index_t*, index_t, index_t*,
              value_t*, index_t, index_t, index_t, index_t, index_t*, index_t*,
              index_t, index_t*, value_t*, index_t, index_t, index_t, index_t&,
              index_t&, index_t*&, index_t*&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void tcsf_add(index_t, index_t, index_t, index_t, index_t*, index_t*, index_t,
              index_t*, index_t*, index_t, index_t*, value_t*, index_t, index_t,
              index_t, index_t, index_t, index_t*, index_t*, index_t, index_t*,
              index_t*, index_t, index_t*, value_t*, index_t, index_t, index_t,
              index_t, index_t&, index_t&, index_t&, index_t*&, index_t*&,
              index_t*&, index_t*&, index_t*&, value_t*&);

template <typename index_t, typename value_t>
void coo_add(index_t, index_t, index_t, index_t*, index_t, index_t*, value_t*,
             index_t, index_t, index_t, index_t, index_t*, index_t, index_t*,
             value_t*, index_t, index_t, index_t, index_t&, index_t&, index_t*&,
             index_t*&, value_t*&);

template <typename index_t, typename value_t>
void coo_mul(index_t, index_t, index_t, index_t*, index_t, index_t*, value_t*,
             index_t, index_t, index_t, index_t, index_t*, index_t, index_t*,
             value_t*, index_t, index_t, index_t, index_t&, index_t&, index_t*&,
             index_t*&, value_t*&);

extern "C" {

void cpu_free(void *ptr) { free(ptr); }

// ---- sparse_vec_mul ----
void cpu_sparse_vec_mul(
    int a_dim_i_size, int a_dim_i_length, int *a_dim_i_indices, float *a_values,
    int a_nnz, int b_dim_i_size, int b_dim_i_length, int *b_dim_i_indices,
    float *b_values, int b_nnz, int result_dim_i_size, int *out_nnz,
    int **out_dim_i_indices, float **out_values) {
    sparse_vec_mul<int, float>(
        a_dim_i_size, a_dim_i_length, a_dim_i_indices, a_values, a_nnz,
        b_dim_i_size, b_dim_i_length, b_dim_i_indices, b_values, b_nnz,
        result_dim_i_size, *out_nnz, *out_dim_i_indices, *out_values);
}

// ---- sparse_vec_add ----
void cpu_sparse_vec_add(
    int a_dim_i_size, int a_dim_i_length, int *a_dim_i_indices, float *a_values,
    int a_nnz, int b_dim_i_size, int b_dim_i_length, int *b_dim_i_indices,
    float *b_values, int b_nnz, int result_dim_i_size, int *out_nnz,
    int **out_dim_i_indices, float **out_values) {
    sparse_vec_add<int, float>(
        a_dim_i_size, a_dim_i_length, a_dim_i_indices, a_values, a_nnz,
        b_dim_i_size, b_dim_i_length, b_dim_i_indices, b_values, b_nnz,
        result_dim_i_size, *out_nnz, *out_dim_i_indices, *out_values);
}

// ---- sparse_vec_apb_c: (a+b)*c ----
void cpu_sparse_vec_apb_c(
    int a_dim_i_size, int a_dim_i_length, int *a_dim_i_indices, float *a_values,
    int a_nnz, int b_dim_i_size, int b_dim_i_length, int *b_dim_i_indices,
    float *b_values, int b_nnz, int c_dim_i_size, int c_dim_i_length,
    int *c_dim_i_indices, float *c_values, int c_nnz, int result_dim_i_size,
    int *out_nnz, int **out_dim_i_indices, float **out_values) {
    sparse_vec_apb_c<int, float>(
        a_dim_i_size, a_dim_i_length, a_dim_i_indices, a_values, a_nnz,
        b_dim_i_size, b_dim_i_length, b_dim_i_indices, b_values, b_nnz,
        c_dim_i_size, c_dim_i_length, c_dim_i_indices, c_values, c_nnz,
        result_dim_i_size, *out_nnz, *out_dim_i_indices, *out_values);
}

// ---- sparse_vec_ab_pc: (a*b)+c ----
void cpu_sparse_vec_ab_pc(
    int a_dim_i_size, int a_dim_i_length, int *a_dim_i_indices, float *a_values,
    int a_nnz, int b_dim_i_size, int b_dim_i_length, int *b_dim_i_indices,
    float *b_values, int b_nnz, int c_dim_i_size, int c_dim_i_length,
    int *c_dim_i_indices, float *c_values, int c_nnz, int result_dim_i_size,
    int *out_nnz, int **out_dim_i_indices, float **out_values) {
    sparse_vec_ab_pc<int, float>(
        a_dim_i_size, a_dim_i_length, a_dim_i_indices, a_values, a_nnz,
        b_dim_i_size, b_dim_i_length, b_dim_i_indices, b_values, b_nnz,
        c_dim_i_size, c_dim_i_length, c_dim_i_indices, c_values, c_nnz,
        result_dim_i_size, *out_nnz, *out_dim_i_indices, *out_values);
}

// ---- csr_add ----
void cpu_csr_add(int A_dim_i_size, int A_dim_j_size, int *A_dim_j_offsets,
                 int A_dim_j_length, int *A_dim_j_indices, float *A_values,
                 int A_nnz, int B_dim_i_size, int B_dim_j_size,
                 int *B_dim_j_offsets, int B_dim_j_length, int *B_dim_j_indices,
                 float *B_values, int B_nnz, int result_dim_i_size,
                 int result_dim_j_size, int *out_nnz, int **out_dim_j_indices,
                 int **out_dim_j_offsets, float **out_values) {
    csr_add<int, float>(
        A_dim_i_size, A_dim_j_size, A_dim_j_offsets, A_dim_j_length,
        A_dim_j_indices, A_values, A_nnz, B_dim_i_size, B_dim_j_size,
        B_dim_j_offsets, B_dim_j_length, B_dim_j_indices, B_values, B_nnz,
        result_dim_i_size, result_dim_j_size, *out_nnz, *out_dim_j_indices,
        *out_dim_j_offsets, *out_values);
}

// ---- dcsr_mul ----
void cpu_dcsr_mul(
    int A_dim_i_size, int A_dim_j_size, int A_dim_i_length,
    int *A_dim_i_indices, int *A_dim_j_offsets, int A_dim_j_length,
    int *A_dim_j_indices, float *A_values, int A_nnz, int B_dim_i_size,
    int B_dim_j_size, int B_dim_i_length, int *B_dim_i_indices,
    int *B_dim_j_offsets, int B_dim_j_length, int *B_dim_j_indices,
    float *B_values, int B_nnz, int result_dim_i_size, int result_dim_j_size,
    int *out_nnz, int *out_dim_i_length, int **out_dim_j_indices,
    int **out_dim_j_offsets, int **out_dim_i_indices, float **out_values) {
    dcsr_mul<int, float>(
        A_dim_i_size, A_dim_j_size, A_dim_i_length, A_dim_i_indices,
        A_dim_j_offsets, A_dim_j_length, A_dim_j_indices, A_values, A_nnz,
        B_dim_i_size, B_dim_j_size, B_dim_i_length, B_dim_i_indices,
        B_dim_j_offsets, B_dim_j_length, B_dim_j_indices, B_values, B_nnz,
        result_dim_i_size, result_dim_j_size, *out_nnz, *out_dim_i_length,
        *out_dim_j_indices, *out_dim_j_offsets, *out_dim_i_indices,
        *out_values);
}

// ---- dcsr_add ----
void cpu_dcsr_add(
    int A_dim_i_size, int A_dim_j_size, int A_dim_i_length,
    int *A_dim_i_indices, int *A_dim_j_offsets, int A_dim_j_length,
    int *A_dim_j_indices, float *A_values, int A_nnz, int B_dim_i_size,
    int B_dim_j_size, int B_dim_i_length, int *B_dim_i_indices,
    int *B_dim_j_offsets, int B_dim_j_length, int *B_dim_j_indices,
    float *B_values, int B_nnz, int result_dim_i_size, int result_dim_j_size,
    int *out_nnz, int *out_dim_i_length, int **out_dim_j_indices,
    int **out_dim_j_offsets, int **out_dim_i_indices, float **out_values) {
    dcsr_add<int, float>(
        A_dim_i_size, A_dim_j_size, A_dim_i_length, A_dim_i_indices,
        A_dim_j_offsets, A_dim_j_length, A_dim_j_indices, A_values, A_nnz,
        B_dim_i_size, B_dim_j_size, B_dim_i_length, B_dim_i_indices,
        B_dim_j_offsets, B_dim_j_length, B_dim_j_indices, B_values, B_nnz,
        result_dim_i_size, result_dim_j_size, *out_nnz, *out_dim_i_length,
        *out_dim_j_indices, *out_dim_j_offsets, *out_dim_i_indices,
        *out_values);
}

// ---- tcsf_add ----
void cpu_tcsf_add(
    int A_dim_i_size, int A_dim_j_size, int A_dim_k_size, int A_dim_i_length,
    int *A_dim_i_indices, int *A_dim_j_offsets, int A_dim_j_length,
    int *A_dim_j_indices, int *A_dim_k_offsets, int A_dim_k_length,
    int *A_dim_k_indices, float *A_values, int A_nnz, int B_dim_i_size,
    int B_dim_j_size, int B_dim_k_size, int B_dim_i_length,
    int *B_dim_i_indices, int *B_dim_j_offsets, int B_dim_j_length,
    int *B_dim_j_indices, int *B_dim_k_offsets, int B_dim_k_length,
    int *B_dim_k_indices, float *B_values, int B_nnz, int result_dim_i_size,
    int result_dim_j_size, int result_dim_k_size, int *out_nnz,
    int *out_dim_i_length, int *out_dim_j_length, int **out_dim_k_indices,
    int **out_dim_k_offsets, int **out_dim_j_indices, int **out_dim_j_offsets,
    int **out_dim_i_indices, float **out_values) {
    tcsf_add<int, float>(
        A_dim_i_size, A_dim_j_size, A_dim_k_size, A_dim_i_length,
        A_dim_i_indices, A_dim_j_offsets, A_dim_j_length, A_dim_j_indices,
        A_dim_k_offsets, A_dim_k_length, A_dim_k_indices, A_values, A_nnz,
        B_dim_i_size, B_dim_j_size, B_dim_k_size, B_dim_i_length,
        B_dim_i_indices, B_dim_j_offsets, B_dim_j_length, B_dim_j_indices,
        B_dim_k_offsets, B_dim_k_length, B_dim_k_indices, B_values, B_nnz,
        result_dim_i_size, result_dim_j_size, result_dim_k_size, *out_nnz,
        *out_dim_i_length, *out_dim_j_length, *out_dim_k_indices,
        *out_dim_k_offsets, *out_dim_j_indices, *out_dim_j_offsets,
        *out_dim_i_indices, *out_values);
}

// ---- coo_add ----
void cpu_coo_add(
    int A_dim_i_size, int A_dim_j_size, int A_dim_i_length,
    int *A_dim_i_indices, int A_dim_j_length, int *A_dim_j_indices,
    float *A_values, int A_nnz, int B_dim_i_size, int B_dim_j_size,
    int B_dim_i_length, int *B_dim_i_indices, int B_dim_j_length,
    int *B_dim_j_indices, float *B_values, int B_nnz,
    int result_dim_i_size, int result_dim_j_size, int *out_nnz,
    int *out_dim_i_length, int **out_dim_j_indices,
    int **out_dim_i_indices, float **out_values) {
    coo_add<int, float>(
        A_dim_i_size, A_dim_j_size, A_dim_i_length, A_dim_i_indices,
        A_dim_j_length, A_dim_j_indices, A_values, A_nnz,
        B_dim_i_size, B_dim_j_size, B_dim_i_length, B_dim_i_indices,
        B_dim_j_length, B_dim_j_indices, B_values, B_nnz,
        result_dim_i_size, result_dim_j_size, *out_nnz, *out_dim_i_length,
        *out_dim_j_indices, *out_dim_i_indices, *out_values);
}

// ---- coo_mul ----
void cpu_coo_mul(
    int A_dim_i_size, int A_dim_j_size, int A_dim_i_length,
    int *A_dim_i_indices, int A_dim_j_length, int *A_dim_j_indices,
    float *A_values, int A_nnz, int B_dim_i_size, int B_dim_j_size,
    int B_dim_i_length, int *B_dim_i_indices, int B_dim_j_length,
    int *B_dim_j_indices, float *B_values, int B_nnz,
    int result_dim_i_size, int result_dim_j_size, int *out_nnz,
    int *out_dim_i_length, int **out_dim_j_indices,
    int **out_dim_i_indices, float **out_values) {
    coo_mul<int, float>(
        A_dim_i_size, A_dim_j_size, A_dim_i_length, A_dim_i_indices,
        A_dim_j_length, A_dim_j_indices, A_values, A_nnz,
        B_dim_i_size, B_dim_j_size, B_dim_i_length, B_dim_i_indices,
        B_dim_j_length, B_dim_j_indices, B_values, B_nnz,
        result_dim_i_size, result_dim_j_size, *out_nnz, *out_dim_i_length,
        *out_dim_j_indices, *out_dim_i_indices, *out_values);
}

} // extern "C"
