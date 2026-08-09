#pragma once

// TACO-equivalent CSR addition on the CPU: the row loop is parallelised directly, one
// task per row, with no merge-path partitioning of work between threads. This is the
// comparison point for what nacho's load balancing buys on skewed inputs.
//
// Result buffers are malloc'd, matching the generated kernels, so Python can adopt them
// with the same deallocator.

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace {

template<typename index_t, typename value_t>
struct a_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  const index_t * __restrict__ dim_j_offsets;
  index_t dim_j_length;
  const index_t * __restrict__ dim_j_indices;
  const value_t* __restrict__ values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct b_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  const index_t* __restrict__ dim_j_offsets;
  index_t dim_j_length;
  const index_t* __restrict__ dim_j_indices;
  const value_t* __restrict__ values;
  index_t nnz;
};
template<typename index_t, typename value_t>
struct Z_tensor_format {
  index_t dim_i_size;
  index_t dim_j_size;
  index_t* __restrict__ dim_j_offsets;
  index_t dim_j_length;
  index_t* __restrict__ dim_j_indices;
  value_t* __restrict__ values;
  index_t nnz;
};
template<typename index_t>
struct result_per_thread_count {
  index_t* __restrict__ dim_j_count;
};

template<typename index_t, typename value_t>
void assembly(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, result_per_thread_count<index_t> count_offsets, const int32_t row_id) {
  index_t jpA = a.dim_j_offsets[row_id];
  const index_t jpA_end = a.dim_j_offsets[row_id + 1];
  index_t jpB = b.dim_j_offsets[row_id];
  const index_t jpB_end = b.dim_j_offsets[row_id + 1];

  index_t jpZ = 0;

  while (jpA < jpA_end && jpB < jpB_end) {
    index_t jA = a.dim_j_indices[jpA];
    index_t jB = b.dim_j_indices[jpB];
    index_t j = std::min(jA, jB);
    if (j == jA && j == jB) {
      jpZ++;
    } else if (j == jA) {
      jpZ++;
    } else { // j == jB
      jpZ++;
    }

    jpA += (jA == j);
    jpB += (jB == j);
  }

  while (jpA < jpA_end) {
    jpZ++;
    jpA++;
  }

  while (jpB < jpB_end) {
    jpZ++;
    jpB++;
  }

  count_offsets.dim_j_count[row_id] = jpZ;
}

template<typename index_t, typename value_t>
void compute(const a_tensor_format<index_t, value_t> a, const b_tensor_format<index_t, value_t> b, const result_per_thread_count<index_t> count_offsets, Z_tensor_format<index_t, value_t> Z, const int32_t row_id) {
  index_t jpA = a.dim_j_offsets[row_id];
  const index_t jpA_end = a.dim_j_offsets[row_id + 1];
  index_t jpB = b.dim_j_offsets[row_id];
  const index_t jpB_end = b.dim_j_offsets[row_id + 1];

  index_t jpZ = count_offsets.dim_j_count[row_id];

  if (row_id == 0) {
    Z.dim_j_offsets[0] = 0;
  }

  while (jpA < jpA_end && jpB < jpB_end) {
    index_t jA = a.dim_j_indices[jpA];
    index_t jB = b.dim_j_indices[jpB];
    index_t j = std::min(jA, jB);
    if (j == jA && j == jB) {
      Z.dim_j_indices[jpZ] = j;
      Z.values[jpZ] = a.values[jpA] + b.values[jpB];
      jpZ++;
    } else if (j == jA) {
      Z.dim_j_indices[jpZ] = j;
      Z.values[jpZ] = a.values[jpA];
      jpZ++;
    } else { // j == jB
      Z.dim_j_indices[jpZ] = j;
      Z.values[jpZ] = b.values[jpB];
      jpZ++;
    }

    jpA += (jA == j);
    jpB += (jB == j);
  }

  while (jpA < jpA_end) {
    Z.dim_j_indices[jpZ] = a.dim_j_indices[jpA];
    Z.values[jpZ] = a.values[jpA];
    jpZ++;
    jpA++;
  }

  while (jpB < jpB_end) {
    Z.dim_j_indices[jpZ] = b.dim_j_indices[jpB];
    Z.values[jpZ] = b.values[jpB];
    jpZ++;
    jpB++;
  }

  Z.dim_j_offsets[row_id + 1] = jpZ;
}

} // namespace

template<typename index_t, typename value_t>
void cpu_csr_add_taco(
    const index_t* __restrict__ shape, const index_t* __restrict__ rowOffsA,
    const index_t* __restrict__ colIndsA, const value_t* __restrict__ ValsA, const uint64_t nnzA,
    const index_t* __restrict__ rowOffsB, const index_t* __restrict__ colIndsB,
    const value_t* __restrict__ ValsB, const uint64_t nnzB,
    index_t* __restrict__ &rowOffsC, index_t* __restrict__ &colIndsC, value_t* __restrict__ &ValsC,
    index_t &nnzC) {

    // setup identical to GPU version
    a_tensor_format<index_t,value_t> A; b_tensor_format<index_t,value_t> B; Z_tensor_format<index_t,value_t> C;
    A.dim_i_size=shape[0]; A.dim_j_size=shape[1]; A.dim_j_offsets=rowOffsA;
    A.dim_j_indices=colIndsA; A.dim_j_length=nnzA; A.nnz=nnzA; A.values=ValsA;
    B.dim_i_size=shape[0]; B.dim_j_size=shape[1]; B.dim_j_offsets=rowOffsB;
    B.dim_j_indices=colIndsB; B.dim_j_length=nnzB; B.nnz=nnzB; B.values=ValsB;

    const int n_threads = shape[0];
    result_per_thread_count<index_t> count_offsets;
    count_offsets.dim_j_count = (index_t*)malloc(sizeof(index_t) * n_threads);

    tbb::parallel_for(0, n_threads, [&](int thread_id) {
        assembly(A, B, count_offsets, thread_id);
    });

    // TODO: this doesn't work in-place
    /*
    index_t last_count = count_offsets.dim_j_count[n_threads - 1];  // save before overwrite

    std::exclusive_scan(std::execution::par,
        count_offsets.dim_j_count,
        count_offsets.dim_j_count + n_threads,
        count_offsets.dim_j_count,
        0);

    nnzC = count_offsets.dim_j_count[n_threads - 1] + last_count;
    */
    nnzC = tbb::parallel_scan(
        tbb::blocked_range<int>(0, n_threads),
        0,                          // identity / initial sum
        [&](const tbb::blocked_range<int>& r, int sum, bool is_final) -> int {
            for (int i = r.begin(); i != r.end(); ++i) {
                int v = count_offsets.dim_j_count[i];
                if (is_final)
                    count_offsets.dim_j_count[i] = sum;  // write prefix, not original
                sum += v;
            }
            return sum;
        },
        std::plus<int>{}            // combine partial sums across chunks
    );

    rowOffsC = (index_t*)calloc(shape[0] + 1, sizeof(index_t));
    colIndsC = (index_t*)malloc(sizeof(index_t) * nnzC);
    ValsC    = (value_t*)malloc(sizeof(value_t) * nnzC);
    C.dim_i_size=shape[0]; C.dim_j_size=shape[1];
    C.dim_j_offsets=rowOffsC; C.dim_j_indices=colIndsC; C.values=ValsC;

    tbb::parallel_for(0, n_threads, [&](int thread_id) {
        compute(A, B, count_offsets, C, thread_id);
    });

    free(count_offsets.dim_j_count);
}