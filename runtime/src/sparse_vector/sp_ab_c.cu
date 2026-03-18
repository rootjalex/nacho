#include "sp_ab_c.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>




// kernels to compute (a*b)+c through full fusion (l.b on a,b,c)
template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_full_lb_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    );

    template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_full_lb_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    );


// kernels to compute (a+b)*c through partial fusion (l.b on a,b)
template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_partial_lb_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    );

    template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_partial_lb_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    );


// kernels to compute (a*b)+c through no fusion (l.b on c)
template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_no_fusion_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    );

template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_no_fusion_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    );


// sparse_vector_fusion_test computes (A+B)*C where A,B,C are sparse vectors
// num_fused = 3 means full fusion (l.b on A,B,C), 
// num_fused = 2 means partial fusion (l.b on A,B), 
// num_fused = 1 means no fusion (l.b on C)
template<typename index_t, typename coord_t, typename value_t>
void sparse_vector_ab_c(const SparseVector<index_t, coord_t, value_t> A, const SparseVector<index_t, coord_t, value_t> B, const SparseVector<index_t, coord_t, value_t> C, 
    coord_t * & D_indices, value_t * & D_values, index_t * & D_nnz, int num_fused, value_t * & D_times) {

    int num_blocks = 256;
    int threads_per_block = 256;
    index_t * mergepath_boundaries;

    SparseVector<index_t, coord_t, value_t>* g_vectors;

    CHECK_CUDA(cudaMalloc(&g_vectors, 3 * sizeof(SparseVector<index_t, coord_t, value_t>)));


    CHECK_CUDA(cudaMemcpy(g_vectors, &A, sizeof(SparseVector<index_t, coord_t, value_t>), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g_vectors + 1, &B, sizeof(SparseVector<index_t, coord_t, value_t>), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g_vectors + 2, &C, sizeof(SparseVector<index_t, coord_t, value_t>), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&mergepath_boundaries, num_fused * num_blocks* threads_per_block * sizeof(index_t)));

    index_t total_size, per_thread_work;

    //printf("Finding mergepath boundaries\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bool old = false;
    if (num_fused == 3) {
        total_size = A.nnz + B.nnz + C.nnz;
        per_thread_work = (total_size)/(num_blocks*threads_per_block) + 2; // +(num_vectors-1) to account for boundary shifts due to balancing 
        
        kern_mergepath_partition_sparse_vectors<<<num_blocks, threads_per_block>>>(
        3,
        total_size,
        g_vectors,
        mergepath_boundaries,
        per_thread_work,
        old
        );

        CHECK_CUDA(cudaGetLastError());
        
    } else if (num_fused == 2) {
        total_size = A.nnz + B.nnz;
        per_thread_work = (total_size)/(num_blocks*threads_per_block) + 1; // +(num_vectors-1) to account for boundary shifts due to balancing
        
        
        kern_mergepath_partition_sparse_vectors<<<num_blocks, threads_per_block>>>(
        2,
        total_size,
        g_vectors,
        mergepath_boundaries,
        per_thread_work,
        old
        );
        CHECK_CUDA(cudaGetLastError());
        
    } else {
        total_size = C.nnz;
        per_thread_work = (total_size)/(num_blocks*threads_per_block) + 1; 
        
        kern_mergepath_partition_sparse_vectors<<<num_blocks, threads_per_block>>>(
        1,
        total_size,
        g_vectors+2,
        mergepath_boundaries,
        per_thread_work,
        old
        );

        CHECK_CUDA(cudaGetLastError());
        //print_cuda(mergepath_boundaries, 10);

    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    D_times[0]=milliseconds;


    //CHECK_CUDA(cudaDeviceSynchronize());
    // print_cuda(mergepath_boundaries, 20);
    //  int * offset;

    // if(num_fused >=2){
    //     offset = mergepath_boundaries + num_blocks*threads_per_block;
    //     print_cuda(offset,20);
    // }

    // if(num_fused >=3){
    //     offset = mergepath_boundaries + 2 * num_blocks*threads_per_block;
    //     print_cuda(offset, 20);
    // }
    // printf("Starting pre-compute pass\n");

    index_t * nnz_count;


    CHECK_CUDA(cudaMalloc(&nnz_count, num_blocks * threads_per_block * sizeof(index_t)));
    CHECK_CUDA(cudaMemset(nnz_count, 0, num_blocks * threads_per_block * sizeof(index_t)));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (num_fused == 3) {
       sp_ab_c_full_lb_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    } else if (num_fused == 2) {
        sp_ab_c_partial_lb_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    } else {
        sp_ab_c_no_lb_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    D_times[1]=milliseconds;

    //print_cuda(nnz_count, 20);
  
    //CHECK_CUDA(cudaGetLastError());
    //printf("Computing nnz prefix sum\n");
    index_t * nnz_prefix;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cudaMalloc((void**)&nnz_prefix, sizeof(index_t)*((num_blocks * threads_per_block) + 1)));
    CHECK_CUDA(cudaMemset(nnz_prefix, 0, sizeof(index_t)*((num_blocks * threads_per_block) + 1)));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnz_prefix+1, num_blocks * threads_per_block));

    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnz_prefix+1, num_blocks * threads_per_block));
    CHECK_CUDA( cudaGetLastError() );
    
    cudaFree(d_temp_storage);

    CHECK_CUDA(cudaMemcpy(D_nnz, nnz_prefix + num_blocks * threads_per_block, sizeof(index_t), cudaMemcpyDeviceToHost));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if(*D_nnz == 0) {
        CHECK_CUDA(cudaMalloc(&D_indices, 1 * sizeof(coord_t)));
        CHECK_CUDA(cudaMalloc(&D_values, 1 * sizeof(value_t)));
        CHECK_CUDA(cudaMemset(D_indices, 0, sizeof(coord_t)));
        CHECK_CUDA(cudaMemset(D_values, 0, sizeof(value_t)));

    } else{

        CHECK_CUDA(cudaMalloc(&D_indices, (*D_nnz) * sizeof(coord_t)));
        CHECK_CUDA(cudaMalloc(&D_values, (*D_nnz) * sizeof(value_t)));

       // printf("Starting final paas\n");
        if( num_fused == 3) {
            sp_ab_c_full_lb_compute<<<num_blocks, threads_per_block>>>(
                    g_vectors,
                    mergepath_boundaries,
                    per_thread_work,
                    D_indices, D_values,
                    nnz_prefix
                );
                CHECK_CUDA(cudaGetLastError());
        } else if (num_fused == 2) {
            sp_ab_c_partial_lb_compute<<<num_blocks, threads_per_block>>>(
                    g_vectors,
                    mergepath_boundaries,
                    per_thread_work,
                    D_indices, D_values,
                    nnz_prefix
                );
                CHECK_CUDA(cudaGetLastError());
        } else {
            sp_ab_c_no_lb_compute<<<num_blocks, threads_per_block>>>(
                    g_vectors,
                    mergepath_boundaries,
                    per_thread_work,
                    D_indices, D_values,
                    nnz_prefix
                );
                CHECK_CUDA(cudaGetLastError());
        }
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    D_times[2]=milliseconds;
    //printf("Finished Computation\n");
    cudaFree(mergepath_boundaries);

    cudaFree(nnz_count);
    cudaFree(nnz_prefix);

    cudaFree(g_vectors);
    return;
};

template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_full_lb_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnzs
    ) {

    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;

    index_t nnz_count = 0;
  const int n_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  index_t idx_a = 0, end_a = A->nnz - 1;
  index_t idx_b = 0, end_b = B->nnz - 1;
  index_t idx_c = 0, end_c = C->nnz - 1;

  idx_a = mergepath_partitions[0 * n_threads + tid];
  
  if(tid<n_threads-1) {
      end_a = mergepath_partitions[0 * n_threads + tid + 1];
  }
  
  idx_b = mergepath_partitions[1 * n_threads + tid];
    
  if(tid<n_threads-1){
  end_b = mergepath_partitions[1 * n_threads + tid + 1];
  }
  idx_c = mergepath_partitions[2 * n_threads + tid];
    
  if(tid<n_threads-1){
  end_c = mergepath_partitions[2 * n_threads + tid + 1];
  }

  idx_a++;
    idx_b++;
    idx_c++;

  while((idx_a <= end_a) && (idx_b <= end_b) && (idx_c <= end_c)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = min(min(crd_a, crd_b), crd_c);

    if ((crd_a == crd) && (crd_b == crd) && (crd_c == crd)) {
      nnz_count++;
    } else if ((crd_a == crd) && (crd_b == crd)) {
      nnz_count++;
    } else if ((crd_c == crd)) {
      nnz_count++;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_a <= end_a) && (idx_b <= end_b)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd = min(crd_a, crd_b);

    if ((crd_a == crd) && (crd_b == crd)) {
      nnz_count++;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
  }

  while((idx_c <= end_c)) {
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = crd_c;

    if ((crd_c == crd)) {
      nnz_count++;
    }
    idx_c += (index_t)(crd_c == crd);
  }

  // if(nnz_count>5) {
  //     printf("Thread %d: nnz_count %d\n", tid, nnz_count);
  // }
  nnzs[tid] = nnz_count;
  return;

};

template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_full_lb_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    ) {
    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;

  const int n_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  index_t idx_a = 0, end_a = A->nnz - 1;
  index_t idx_b = 0, end_b = B->nnz - 1;
  index_t idx_c = 0, end_c = C->nnz - 1;
  idx_a = mergepath_partitions[0 * n_threads + tid];
  if(tid<n_threads-1) {
      end_a = mergepath_partitions[0 * n_threads + tid + 1];
  }
  idx_b = mergepath_partitions[1 * n_threads + tid];
    if(tid<n_threads-1)
  end_b = mergepath_partitions[1 * n_threads + tid + 1];
  idx_c = mergepath_partitions[2 * n_threads + tid];
    if(tid<n_threads-1)
  end_c = mergepath_partitions[2 * n_threads + tid + 1];
  index_t idx_output = nnz_prefix[tid];

  idx_a++;
    idx_b++;
    idx_c++;
  while((idx_a <= end_a) && (idx_b <= end_b) && (idx_c <= end_c)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = min(min(crd_a, crd_b), crd_c);

    if ((crd_a == crd) && (crd_b == crd) && (crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      value_t c = C->values[idx_c];
      D_values[idx_output++] = ((a * b) + c);
    } else if ((crd_a == crd) && (crd_b == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      D_values[idx_output++] = (a * b);
    } else if ((crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t c = C->values[idx_c];
      D_values[idx_output++] = c;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_a <= end_a) && (idx_b <= end_b)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd = min(crd_a, crd_b);

    if ((crd_a == crd) && (crd_b == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      D_values[idx_output++] = (a * b);
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
  }

  while((idx_c <= end_c)) {
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = crd_c;

    if ((crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t c = C->values[idx_c];
      D_values[idx_output++] = c;
    }
    idx_c += (index_t)(crd_c == crd);
  }

    return;

};

template<typename index_t, typename coord_t, typename value_t>
__device__  __inline__ index_t locate( const SparseVector<index_t,coord_t,value_t> * vect, coord_t index, index_t lo, index_t hi) {

    lo = max((index_t)0, lo);
    hi = min(vect->nnz - 1,hi);

    if(lo == hi)
        return lo;

    index_t mid;
    while (lo < hi) {
        //printf("tid %d, lo %d, hi %d\n",threadIdx.x, lo, hi);
        mid = (lo + hi) / 2;
        if (vect->indices[mid] < index) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo; 
};

template<typename index_t, typename coord_t, typename value_t>
__device__  __inline__ index_t lower_bound(const SparseVector<index_t, coord_t, value_t>* a, coord_t crd) {
    // First index > crd
    index_t low = 0, high = a->nnz;
    while (low < high) {
        index_t mid = low + (high - low) / 2;
        if (a->indices[mid] <= crd) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

template<typename index_t, typename coord_t, typename value_t>
__device__  __inline__ index_t upper_bound(const SparseVector<index_t,coord_t, value_t>* a, coord_t crd) {
    // Last index <= crd
    index_t low = 0, high = a->nnz;
    while (low < high) {
        index_t mid = low + (high - low) / 2;
        if (a->indices[mid] <= crd) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low - 1;
}


template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_partial_lb_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnzs
    ) {

 const SparseVector<index_t, coord_t, value_t> * A = vectors;
 const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
 const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;
  
 index_t nnz_count = 0;
  const int n_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  index_t idx_a = 0, end_a = A->nnz - 1;
  index_t idx_b = 0, end_b = B->nnz - 1;
  index_t idx_c = 0, end_c = C->nnz - 1;
  idx_a = mergepath_partitions[0 * n_threads + tid];
  if(tid<n_threads-1)
  end_a = mergepath_partitions[0 * n_threads + tid + 1];
  idx_b = mergepath_partitions[1 * n_threads + tid];
  if(tid<n_threads-1)
  end_b = mergepath_partitions[1 * n_threads + tid + 1];

     if(idx_a>=A->nnz-1 && idx_b>=B->nnz-1){
      nnzs[tid]=0;
      return;
  }


  coord_t crd_lb_c = max(idx_a>=0 ? A->indices[idx_a] : -1, idx_b>=0 ? B->indices[idx_b] : -1);
  coord_t crd_ub_c = max(end_a>=0 ? A->indices[end_a]: -1, end_b>=0 ? B->indices[end_b] : -1);

  if(tid!=0)
  idx_c = lower_bound(C, crd_lb_c);
  end_c = upper_bound(C, crd_ub_c);

  if(end_a == A->nnz -1 && end_b == B->nnz -1){
      end_c = C->nnz -1;
  }

  // if(tid==0){
  //   printf("Thread %d: idx_a %d, end_a %d, idx_b %d, end_b %d, idx_c %d, end_c %d\n", tid, idx_a, end_a, idx_b, end_b, idx_c, end_c);
  // }

    idx_a++;
  idx_b++;


  while((idx_a <= end_a) && (idx_b <= end_b) && (idx_c <= end_c)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = min(min(crd_a, crd_b), crd_c);

    if ((crd_a == crd) && (crd_b == crd) && (crd_c == crd)) {
      nnz_count++;
    } else if ((crd_a == crd) && (crd_b == crd)) {
      nnz_count++;
    } else if ((crd_c == crd)) {
      nnz_count++;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_a <= end_a) && (idx_b <= end_b)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd = min(crd_a, crd_b);

    if ((crd_a == crd) && (crd_b == crd)) {
      nnz_count++;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
  }

  while((idx_c <= end_c)) {
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = crd_c;

    if ((crd_c == crd)) {
      nnz_count++;
    }
    idx_c += (index_t)(crd_c == crd);
  }
  nnzs[tid] = nnz_count;

return;
    
};

template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_partial_lb_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    ) {
     const SparseVector<index_t, coord_t, value_t> * A = vectors;
 const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
 const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;
  
 index_t nnz_count = 0;
  const int n_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  index_t idx_a = 0, end_a = A->nnz - 1;
  index_t idx_b = 0, end_b = B->nnz - 1;
  index_t idx_c = 0, end_c = C->nnz - 1;
  idx_a = mergepath_partitions[0 * n_threads + tid];
  if(tid<n_threads-1)
  end_a = mergepath_partitions[0 * n_threads + tid + 1];
  idx_b = mergepath_partitions[1 * n_threads + tid];
  if(tid<n_threads-1)
  end_b = mergepath_partitions[1 * n_threads + tid + 1];

    if(idx_a>=A->nnz-1 && idx_b>=B->nnz-1){
      return;
  }

  coord_t crd_lb_c = max(idx_a>=0 ? A->indices[idx_a] : -1, idx_b>=0 ? B->indices[idx_b] : -1);
  coord_t crd_ub_c = max(end_a>=0 ? A->indices[end_a]: -1, end_b>=0 ? B->indices[end_b] : -1);

  idx_a++;
  idx_b++;

if(tid!=0)
  idx_c = lower_bound(C, crd_lb_c);
  end_c = upper_bound(C, crd_ub_c);
  
   if(end_a == A->nnz -1 && end_b == B->nnz -1){
      end_c = C->nnz -1;
  }


  index_t idx_output = nnz_prefix[tid];
    while((idx_a <= end_a) && (idx_b <= end_b) && (idx_c <= end_c)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = min(min(crd_a, crd_b), crd_c);

    if ((crd_a == crd) && (crd_b == crd) && (crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      value_t c = C->values[idx_c];
      D_values[idx_output++] = ((a * b) + c);
    } else if ((crd_a == crd) && (crd_b == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      D_values[idx_output++] = (a * b);
    } else if ((crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t c = C->values[idx_c];
      D_values[idx_output++] = c;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_a <= end_a) && (idx_b <= end_b)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd = min(crd_a, crd_b);

    if ((crd_a == crd) && (crd_b == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      D_values[idx_output++] = (a * b);
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
  }

  while((idx_c <= end_c)) {
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = crd_c;

    if ((crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t c = C->values[idx_c];
      D_values[idx_output++] = c;
    }
    idx_c += (index_t)(crd_c == crd);
  }
  return;
};


template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_no_lb_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnzs
    ) {
    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;


      index_t nnz_count = 0;
  const int n_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  index_t idx_a = 0, end_a = A->nnz - 1;
  index_t idx_b = 0, end_b = B->nnz - 1;
  index_t idx_c = 0, end_c = C->nnz - 1;

  
  idx_c = mergepath_partitions[0 * n_threads + tid];

  if(tid<n_threads-1)
  end_c = mergepath_partitions[0 * n_threads + tid + 1];

  if(idx_c>=C->nnz-1)
    {return;}

  bool found = false;

  if(end_c == C->nnz -1){
    found = true;
  }
  
    // if(idx_c>C->nnz -100){
    // printf("Thread %d: %d idx_c %d, end_c %d\n", tid, C->nnz, idx_c, end_c);
    // }
  

  coord_t crd_lb_a = idx_c>=0 ? C->indices[idx_c]: -1;
  coord_t crd_ub_a = end_c>=0 ? C->indices[end_c]: -1;

  if(tid !=0) idx_a = lower_bound(A, crd_lb_a);

  end_a = upper_bound(A, crd_ub_a);
  coord_t crd_lb_b = idx_c>=0 ? C->indices[idx_c]: -1;
  coord_t crd_ub_b = end_c>=0 ? C->indices[end_c]: -1;

  if(tid !=0) idx_b = lower_bound(B, crd_lb_b);

  end_b = upper_bound(B, crd_ub_b);

  if(end_c==C->nnz -1 && idx_c<C->nnz -1){
    //printf("tid %d, idx_c %d, end_c %d\n", tid, idx_c, end_c);
      end_a = A->nnz -1;
      end_b = B->nnz -1;
  }

  idx_c++;
  while((idx_a <= end_a) && (idx_b <= end_b) && (idx_c <= end_c)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = min(min(crd_a, crd_b), crd_c);

    if ((crd_a == crd) && (crd_b == crd) && (crd_c == crd)) {
      nnz_count++;
    } else if ((crd_c == crd)) {
      nnz_count++;
    } else if ((crd_a == crd) && (crd_b == crd)) {
      nnz_count++;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_c <= end_c)) {
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = crd_c;

    if ((crd_c == crd)) {
      nnz_count++;
    }
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_a <= end_a) && (idx_b <= end_b)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd = min(crd_a, crd_b);

    if ((crd_a == crd) && (crd_b == crd)) {
      nnz_count++;
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
  }
  nnzs[tid] = nnz_count;
return;
};

template<typename index_t, typename coord_t, typename value_t>
__global__ void sp_ab_c_no_lb_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    ) {
   const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;


      index_t nnz_count = 0;
  const int n_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  index_t idx_a = 0, end_a = A->nnz - 1;
  index_t idx_b = 0, end_b = B->nnz - 1;
  index_t idx_c = 0, end_c = C->nnz - 1;

    

  idx_c = mergepath_partitions[0 * n_threads + tid];
  if(tid<n_threads-1)
  end_c = mergepath_partitions[0 * n_threads + tid + 1];


    if(idx_c>=C->nnz-1)
    {return;}

  coord_t crd_lb_a = idx_c>=0 ? C->indices[idx_c]:-1;
  coord_t crd_ub_a = end_c>=0 ? C->indices[end_c]:-1;

   if(tid !=0) idx_a = lower_bound(A, crd_lb_a);

  end_a = upper_bound(A, crd_ub_a);
  coord_t crd_lb_b = idx_c>=0 ? C->indices[idx_c]:-1;
  coord_t crd_ub_b = end_c>=0 ? C->indices[end_c]:-1;

   if(tid !=0) idx_b = lower_bound(B, crd_lb_b);
  end_b = upper_bound(B, crd_ub_b);

    if(end_c==C->nnz -1 && idx_c<C->nnz -1){
      end_a = A->nnz -1;
      end_b = B->nnz -1;
  }

  coord_t idx_output = nnz_prefix[tid];
        idx_c++;
  while((idx_a <= end_a) && (idx_b <= end_b) && (idx_c <= end_c)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = min(min(crd_a, crd_b), crd_c);

    if ((crd_a == crd) && (crd_b == crd) && (crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      value_t c = C->values[idx_c];
      D_values[idx_output++] = ((a * b) + c);
    } else if ((crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t c = C->values[idx_c];
      D_values[idx_output++] = c;
    } else if ((crd_a == crd) && (crd_b == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      D_values[idx_output++] = (a * b);
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_c <= end_c)) {
    coord_t crd_c = C->indices[idx_c];
    coord_t crd = crd_c;

    if ((crd_c == crd)) {
      D_indices[idx_output] = crd;
      value_t c = C->values[idx_c];
      D_values[idx_output++] = c;
    }
    idx_c += (index_t)(crd_c == crd);
  }

  while((idx_a <= end_a) && (idx_b <= end_b)) {
    coord_t crd_a = A->indices[idx_a];
    coord_t crd_b = B->indices[idx_b];
    coord_t crd = min(crd_a, crd_b);

    if ((crd_a == crd) && (crd_b == crd)) {
      D_indices[idx_output] = crd;
      value_t a = A->values[idx_a];
      value_t b = B->values[idx_b];
      D_values[idx_output++] = (a * b);
    }
    idx_a += (index_t)(crd_a == crd);
    idx_b += (index_t)(crd_b == crd);
  }
  
    return;
};

template void sparse_vector_ab_c(const SparseVector<int, int, float> A, const SparseVector<int, int, float> B, const SparseVector<int, int, float> C, 
    int * & D_indices, float * & D_values, int * & D_nnz, int num_fused, float * & D_times);

template void sparse_vector_ab_c(const SparseVector<int, int64_t, float> A, const SparseVector<int, int64_t, float> B, const SparseVector<int, int64_t, float> C, 
    int64_t * & D_indices, float * & D_values, int * & D_nnz, int num_fused, float * & D_times);