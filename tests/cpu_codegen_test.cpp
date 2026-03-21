#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static bool approx(float a, float b) { return std::fabs(a - b) < 1e-5f; }

// Forward declarations of generated flat-API functions
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

// -----------------------------------------------------------------------
// sparse_vec_mul: a * b  (intersection)
// a = [_, 2.0, _, 4.0, 5.0]  indices=[1,3,4]
// b = [1.0, _, 3.0, 4.0, _]  indices=[0,2,3]
// result = [_, _, _, 16.0, _] indices=[3]
// -----------------------------------------------------------------------
void test_sparse_vec_mul() {
    int a_idx[] = {1, 3, 4};
    float a_val[] = {2.0f, 4.0f, 5.0f};
    int b_idx[] = {0, 2, 3};
    float b_val[] = {1.0f, 3.0f, 4.0f};

    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_mul<int, float>(
        /*a_dim_i_size=*/5, /*a_dim_i_length=*/3, a_idx, a_val, /*a_nnz=*/3,
        /*b_dim_i_size=*/5, /*b_dim_i_length=*/3, b_idx, b_val, /*b_nnz=*/3,
        /*result_dim_i_size=*/5, out_nnz, out_indices, out_values);

    assert(out_nnz == 1);
    assert(out_indices[0] == 3);
    assert(approx(out_values[0], 16.0f));
    free(out_indices);
    free(out_values);
    printf("  sparse_vec_mul: PASS\n");
}

// -----------------------------------------------------------------------
// sparse_vec_add: a + b  (union)
// a = [_, 2.0, _, 4.0, _]  indices=[1,3]
// b = [1.0, _, 3.0, 4.0, _]  indices=[0,2,3]
// result = [1.0, 2.0, 3.0, 8.0] indices=[0,1,2,3]
// -----------------------------------------------------------------------
void test_sparse_vec_add() {
    int a_idx[] = {1, 3};
    float a_val[] = {2.0f, 4.0f};
    int b_idx[] = {0, 2, 3};
    float b_val[] = {1.0f, 3.0f, 4.0f};

    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_add<int, float>(
        /*a_dim_i_size=*/5, /*a_dim_i_length=*/2, a_idx, a_val, /*a_nnz=*/2,
        /*b_dim_i_size=*/5, /*b_dim_i_length=*/3, b_idx, b_val, /*b_nnz=*/3,
        /*result_dim_i_size=*/5, out_nnz, out_indices, out_values);

    assert(out_nnz == 4);
    assert(out_indices[0] == 0);
    assert(out_indices[1] == 1);
    assert(out_indices[2] == 2);
    assert(out_indices[3] == 3);
    assert(approx(out_values[0], 1.0f));
    assert(approx(out_values[1], 2.0f));
    assert(approx(out_values[2], 3.0f));
    assert(approx(out_values[3], 8.0f));
    free(out_indices);
    free(out_values);
    printf("  sparse_vec_add: PASS\n");
}

// -----------------------------------------------------------------------
// sparse_vec_apb_c: (a+b)*c
// a = [_, 2.0, _, 4.0, _]  indices=[1,3]
// b = [1.0, _, 3.0, 4.0, _]  indices=[0,2,3]
// c = [_, _, 5.0, 6.0, _]  indices=[2,3]
// (a+b) = [1, 2, 3, 8] at indices [0,1,2,3]
// (a+b)*c = intersection: at 2: 3*5=15, at 3: 8*6=48
// result = [15.0, 48.0] indices=[2,3]
// -----------------------------------------------------------------------
void test_sparse_vec_apb_c() {
    int a_idx[] = {1, 3};
    float a_val[] = {2.0f, 4.0f};
    int b_idx[] = {0, 2, 3};
    float b_val[] = {1.0f, 3.0f, 4.0f};
    int c_idx[] = {2, 3};
    float c_val[] = {5.0f, 6.0f};

    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_apb_c<int, float>(
        /*a_dim_i_size=*/5, /*a_dim_i_length=*/2, a_idx, a_val, /*a_nnz=*/2,
        /*b_dim_i_size=*/5, /*b_dim_i_length=*/3, b_idx, b_val, /*b_nnz=*/3,
        /*c_dim_i_size=*/5, /*c_dim_i_length=*/2, c_idx, c_val, /*c_nnz=*/2,
        /*result_dim_i_size=*/5, out_nnz, out_indices, out_values);

    assert(out_nnz == 2);
    assert(out_indices[0] == 2);
    assert(out_indices[1] == 3);
    assert(approx(out_values[0], 15.0f));
    assert(approx(out_values[1], 48.0f));
    free(out_indices);
    free(out_values);
    printf("  sparse_vec_apb_c: PASS\n");
}

// -----------------------------------------------------------------------
// sparse_vec_ab_pc: (a*b)+c
// a = [_, 2.0, _, 4.0, _]  indices=[1,3]
// b = [1.0, _, 3.0, 4.0, _]  indices=[0,2,3]
// c = [_, 10.0, _, _, 7.0]  indices=[1,4]
// a*b = intersection: at 3: 4*4=16
// (a*b)+c = union: at 1: 10, at 3: 16, at 4: 7
// result = [10.0, 16.0, 7.0] indices=[1,3,4]
// -----------------------------------------------------------------------
void test_sparse_vec_ab_pc() {
    int a_idx[] = {1, 3};
    float a_val[] = {2.0f, 4.0f};
    int b_idx[] = {0, 2, 3};
    float b_val[] = {1.0f, 3.0f, 4.0f};
    int c_idx[] = {1, 4};
    float c_val[] = {10.0f, 7.0f};

    int out_nnz;
    int *out_indices;
    float *out_values;

    sparse_vec_ab_pc<int, float>(
        /*a_dim_i_size=*/5, /*a_dim_i_length=*/2, a_idx, a_val, /*a_nnz=*/2,
        /*b_dim_i_size=*/5, /*b_dim_i_length=*/3, b_idx, b_val, /*b_nnz=*/3,
        /*c_dim_i_size=*/5, /*c_dim_i_length=*/2, c_idx, c_val, /*c_nnz=*/2,
        /*result_dim_i_size=*/5, out_nnz, out_indices, out_values);

    assert(out_nnz == 3);
    assert(out_indices[0] == 1);
    assert(out_indices[1] == 3);
    assert(out_indices[2] == 4);
    assert(approx(out_values[0], 10.0f));
    assert(approx(out_values[1], 16.0f));
    assert(approx(out_values[2], 7.0f));
    free(out_indices);
    free(out_values);
    printf("  sparse_vec_ab_pc: PASS\n");
}

// -----------------------------------------------------------------------
// csr_add: A + B  (CSR format: dense i, sparse j)
// A = 3x4 matrix:
//   row 0: [(1, 1.0), (3, 2.0)]
//   row 1: [(0, 3.0), (2, 4.0)]
//   row 2: [(1, 5.0)]
// B = 3x4 matrix:
//   row 0: [(1, 10.0)]
//   row 1: [(0, 20.0), (1, 30.0)]
//   row 2: [(1, 40.0), (3, 50.0)]
// A+B:
//   row 0: [(1, 11.0), (3, 2.0)]
//   row 1: [(0, 23.0), (1, 30.0), (2, 4.0)]
//   row 2: [(1, 45.0), (3, 50.0)]
// -----------------------------------------------------------------------
void test_csr_add() {
    // A: CSR
    int A_offsets[] = {0, 2, 4, 5};
    int A_indices[] = {1, 3, 0, 2, 1};
    float A_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // B: CSR
    int B_offsets[] = {0, 1, 3, 5};
    int B_indices[] = {1, 0, 1, 1, 3};
    float B_values[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

    int out_nnz;
    int *out_indices;
    int *out_offsets;
    float *out_values;

    csr_add<int, float>(
        /*A_dim_i_size=*/3, /*A_dim_j_size=*/4,
        A_offsets, /*A_dim_j_length=*/5, A_indices, A_values, /*A_nnz=*/5,
        /*B_dim_i_size=*/3, /*B_dim_j_size=*/4,
        B_offsets, /*B_dim_j_length=*/5, B_indices, B_values, /*B_nnz=*/5,
        /*result_dim_i_size=*/3, /*result_dim_j_size=*/4,
        out_nnz, out_indices, out_offsets, out_values);

    assert(out_nnz == 7);

    // Check offsets
    assert(out_offsets[0] == 0);
    assert(out_offsets[1] == 2);
    assert(out_offsets[2] == 5);
    assert(out_offsets[3] == 7);

    // Row 0: (1, 11.0), (3, 2.0)
    assert(out_indices[0] == 1);
    assert(approx(out_values[0], 11.0f));
    assert(out_indices[1] == 3);
    assert(approx(out_values[1], 2.0f));

    // Row 1: (0, 23.0), (1, 30.0), (2, 4.0)
    assert(out_indices[2] == 0);
    assert(approx(out_values[2], 23.0f));
    assert(out_indices[3] == 1);
    assert(approx(out_values[3], 30.0f));
    assert(out_indices[4] == 2);
    assert(approx(out_values[4], 4.0f));

    // Row 2: (1, 45.0), (3, 50.0)
    assert(out_indices[5] == 1);
    assert(approx(out_values[5], 45.0f));
    assert(out_indices[6] == 3);
    assert(approx(out_values[6], 50.0f));

    free(out_indices);
    free(out_offsets);
    free(out_values);
    printf("  csr_add: PASS\n");
}

int main() {
    printf("CPU codegen tests:\n");
    test_sparse_vec_mul();
    test_sparse_vec_add();
    test_sparse_vec_apb_c();
    test_sparse_vec_ab_pc();
    test_csr_add();
    printf("All CPU codegen tests passed.\n");
    return 0;
}
