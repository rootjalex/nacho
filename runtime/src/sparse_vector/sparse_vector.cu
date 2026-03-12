#include "sparse_vector.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>




// kernels to compute (a+b)*c through full fusion (l.b on a,b,c)
template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_full_fusion_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    );

    template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_full_fusion_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    );


// kernels to compute (a+b)*c through partial fusion (l.b on a,b)
template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_partial_fusion_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    );

    template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_partial_fusion_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    );


// kernels to compute (a+b)*c through no fusion (l.b on c)
template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_no_fusion_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    );

template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_no_fusion_compute(
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
void sparse_vector_fusion_test(const SparseVector<index_t, coord_t, value_t> A, const SparseVector<index_t, coord_t, value_t> B, const SparseVector<index_t, coord_t, value_t> C, 
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
    // print_cuda(mergepath_boundaries, 10);
    // int * offset;

    // if(num_fused >=2){
    //     offset = mergepath_boundaries + num_blocks*threads_per_block;
    //     print_cuda(offset, 10);
    // }

    // if(num_fused >=3){
    //     offset = mergepath_boundaries + 2 * num_blocks*threads_per_block;
    //     print_cuda(offset, 10);
    // }
    //printf("Starting pre-compute pass\n");

    index_t * nnz_count;


    CHECK_CUDA(cudaMalloc(&nnz_count, num_blocks * threads_per_block * sizeof(index_t)));
    CHECK_CUDA(cudaMemset(nnz_count, 0, num_blocks * threads_per_block * sizeof(index_t)));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (num_fused == 3) {
        sparse_vectors_full_fusion_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    } else if (num_fused == 2) {
        sparse_vectors_partial_fusion_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    } else {
        sparse_vectors_no_fusion_precompute<<<num_blocks, threads_per_block>>>(
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

    //CHECK_CUDA(cudaDeviceSynchronize());
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
        CHECK_CUDA(cudaMalloc(&D_indices, 1 * sizeof(index_t)));
        CHECK_CUDA(cudaMalloc(&D_values, 1 * sizeof(value_t)));
        CHECK_CUDA(cudaMemset(D_indices, 0, sizeof(index_t)));
        CHECK_CUDA(cudaMemset(D_values, 0, sizeof(value_t)));

    } else{
        //printf("Total nnz in output: %d\n", *D_nnz);

        CHECK_CUDA(cudaMalloc(&D_indices, (*D_nnz) * sizeof(coord_t)));
        CHECK_CUDA(cudaMalloc(&D_values, (*D_nnz) * sizeof(value_t)));

        //printf("Starting final paas\n");
        if( num_fused == 3) {
            sparse_vectors_full_fusion_compute<<<num_blocks, threads_per_block>>>(
                    g_vectors,
                    mergepath_boundaries,
                    per_thread_work,
                    D_indices, D_values,
                    nnz_prefix
                );
                CHECK_CUDA(cudaGetLastError());
        } else if (num_fused == 2) {
            sparse_vectors_partial_fusion_compute<<<num_blocks, threads_per_block>>>(
                    g_vectors,
                    mergepath_boundaries,
                    per_thread_work,
                    D_indices, D_values,
                    nnz_prefix
                );
                CHECK_CUDA(cudaGetLastError());
        } else {
            sparse_vectors_no_fusion_compute<<<num_blocks, threads_per_block>>>(
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
__global__ void sparse_vectors_full_fusion_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    ) {

    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    index_t start_A, start_B, start_C, end_A, end_B, end_C;
    
    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    start_C = mergepath_partitions[2*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
        end_C = mergepath_partitions[2*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
        end_C = C->nnz - 1;
    }

    index_t count = 0;

    index_t idx_A = start_A + 1;
    index_t idx_B = start_B + 1;
    index_t idx_C = start_C + 1;

    while( (idx_A <= end_A) && (idx_B <= end_B) && (idx_C <= end_C) ) {
       coord_t ia0 = A->indices[idx_A];
        coord_t ib0 = B->indices[idx_B];
        coord_t ic0 = C->indices[idx_C];

        coord_t i = min(ia0, min(ib0, ic0));
        if ((ia0 == i && ib0 == i) && ic0 == i) {
            count++;
        }
        else if (ia0 == i && ic0 == i) {
            count++;
        }
        else if (ib0 == i && ic0 == i) {
            count++;
        }
        
        idx_A += (index_t)(ia0 == i);
        idx_B += (index_t)(ib0 == i);
        idx_C += (index_t)(ic0 == i);
    }

    while( (idx_B <= end_B) && (idx_C <= end_C) ) {
        coord_t ib0 = B->indices[idx_B];
        coord_t ic0 = C->indices[idx_C];

        coord_t i = min(ib0, ic0);

        if (ib0 == i && ic0 == i) {
            count++;
        }

        idx_B += (index_t)(ib0 == i);
        idx_C += (index_t)(ic0 == i);

    }

    while( (idx_A <= end_A) && (idx_C <= end_C) ) {
        coord_t ia0 = A->indices[idx_A];
        coord_t ic0 = C->indices[idx_C];

        coord_t i = min(ia0, ic0);

        if (ia0 == i && ic0 == i) {
            count++;
        }

        idx_A += (index_t)(ia0 == i);
        idx_C += (index_t)(ic0 == i);

    }

    nnz_count[tid] = count;
    return;

};

template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_full_fusion_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    ) {
    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_A, start_B, start_C, end_A, end_B, end_C;

    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    start_C = mergepath_partitions[2*num_threads + tid];

    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
        end_C = mergepath_partitions[2*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
        end_C = C->nnz - 1;
    }

    index_t idx_A = start_A + 1;
    index_t idx_B = start_B + 1;
    index_t idx_C = start_C + 1;

    index_t idx_D = nnz_prefix[tid];

    while( (idx_A <= end_A) && (idx_B <= end_B) && (idx_C <= end_C) ) {
        coord_t ia0 = A->indices[idx_A];
        coord_t ib0 = B->indices[idx_B];
        coord_t ic0 = C->indices[idx_C];

        coord_t i = min(ia0, min(ib0, ic0));
        if ((ia0 == i && ib0 == i) && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C];
            idx_D++;

            //printf("\n new D_value : %f %f", D_values[idx_D-1], (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C]);
        }
        else if (ia0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
            //printf("\n new D_value : %f %f", D_values[idx_D-1], A->values[idx_A]  * C->values[idx_C]);
        }
        else if (ib0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
            //printf("\n new D_value : %f %f", D_values[idx_D-1], B->values[idx_B]  * C->values[idx_C]);
        }
        
        idx_A += (index_t)(ia0 == i);
        idx_B += (index_t)(ib0 == i);
        idx_C += (index_t)(ic0 == i);
    }

    while( (idx_B <= end_B) && (idx_C <= end_C) ) {
        coord_t ib0 = B->indices[idx_B];
        coord_t ic0 = C->indices[idx_C];

        coord_t i = min(ib0, ic0);

        if (ib0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
        }

        idx_B += (index_t)(ib0 == i);
        idx_C += (index_t)(ic0 == i);

    }

    while( (idx_A <= end_A) && (idx_C <= end_C) ) {
        coord_t ia0 = A->indices[idx_A];
        coord_t ic0 = C->indices[idx_C];

        coord_t i = min(ia0, ic0);

        if (ia0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
        }

        idx_A += (index_t)(ia0 == i);
        idx_C += (index_t)(ic0 == i);

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
__global__ void sparse_vectors_partial_fusion_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    ) {
    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_A, start_B, end_A, end_B;

    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
    }

    index_t count = 0;

    index_t idx_A = start_A + 1;
    index_t idx_B = start_B + 1;

    index_t idx_C = 0;
    index_t end_C = locate(C, max(A->indices[end_A>=0 ? end_A: 0], B->indices[end_B>=0 ? end_B: 0]), (index_t)0, C->nnz - 1);

    while( (idx_A <= end_A) && (idx_B <= end_B)) {
        coord_t ia0 = A->indices[idx_A];
        coord_t ib0 = B->indices[idx_B];

        coord_t i = min(ia0, ib0);

        idx_C = locate(C, i, idx_C, end_C);
        if (C->indices[idx_C] == i) {
            count++;
        }

        idx_A += (index_t)(ia0 == i);
        idx_B += (index_t)(ib0 == i);
    }

    while( (idx_B <= end_B)) {
        coord_t ib0 = B->indices[idx_B];

        coord_t i = ib0;

        idx_C = locate(C, i, idx_C, end_C);
        if (C->indices[idx_C] == i) {
            count++;
        }

        idx_B += (index_t)(ib0 == i);
    }

    while( (idx_A <= end_A)) {
        coord_t ia0 = A->indices[idx_A];
        coord_t i = ia0;

        idx_C = locate(C, i, idx_C, end_C);
        if (C->indices[idx_C] == i) {
            count++;
        }

        idx_A += (index_t)(ia0 == i);

    }

    nnz_count[tid] = count;

    return;
    
};

template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_partial_fusion_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    ) {
    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_A, start_B, end_A, end_B;

    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
    }


    index_t idx_A = start_A + 1;
    index_t idx_B = start_B + 1;

    index_t idx_D = nnz_prefix[tid];

    index_t idx_C = 0;
    index_t end_C = locate(C, max(A->indices[end_A>=0 ? end_A: 0], B->indices[end_B>=0 ? end_B: 0]), (index_t)0, C->nnz - 1);


    while( (idx_A <= end_A) && (idx_B <= end_B)) {
        coord_t ia0 = A->indices[idx_A];
        coord_t ib0 = B->indices[idx_B];

        coord_t i = min(ia0, ib0);
        idx_C = locate(C, i, idx_C, end_C);
        if (C->indices[idx_C] == i) {
            D_indices[idx_D] = i;
            if(ia0 ==i && ib0==i) {
                D_values[idx_D] = (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C];
            } else if(ia0==i) {
                D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            } else if(ib0==i) {
                 D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            }
            idx_D++;
        }

        idx_A += (index_t)(ia0 == i);
        idx_B += (index_t)(ib0 == i);
    }


    while( (idx_B <= end_B)) {
        coord_t ib0 = B->indices[idx_B];

        coord_t i = ib0;

        idx_C = locate(C, i, idx_C, end_C);
        if (C->indices[idx_C] == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
        }

        idx_B += (index_t)(ib0 == i);
    }

    while( (idx_A <= end_A)) {
        coord_t ia0 = A->indices[idx_A];

        coord_t i = ia0;

        idx_C = locate(C, i, idx_C, end_C);
        if (C->indices[idx_C] == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
        }

        idx_A += (index_t)(ia0 == i);

    }


    return;
    
};


template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_no_fusion_precompute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    index_t * nnz_count
    ) {
    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;


     int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_C, end_C;

    start_C = mergepath_partitions[0*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_C = mergepath_partitions[0*num_threads + tid + 1];
    } else {
        end_C = C->nnz - 1;
    }

    index_t idx_C = start_C + 1;

    index_t count = 0;

    index_t idx_A = 0;
    index_t idx_B = 0;
    index_t end_A = locate(A, C->indices[end_C>=0 ? end_C: 0], (index_t)0, A->nnz - 1);
    index_t end_B = locate(B, C->indices[end_C>=0 ? end_C: 0], (index_t)0, B->nnz - 1);
    while( (idx_C <= end_C)) {
        coord_t ic0 = C->indices[idx_C];

        coord_t i = ic0;

        idx_A = locate(A, i, idx_A, end_A);
        idx_B = locate(B, i, idx_B, end_B);
        if (i == A->indices[idx_A] || i == B->indices[idx_B]) {
            count++;
        }

        idx_C += (index_t)(ic0 == i);
    }

    nnz_count[tid] = count;
    return;
};

template<typename index_t, typename coord_t, typename value_t>
__global__ void sparse_vectors_no_fusion_compute(
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work,
    coord_t * D_indices, value_t * D_values,
    index_t *nnz_prefix
    ) {
    const SparseVector<index_t, coord_t, value_t> * A = vectors;
    const SparseVector<index_t, coord_t, value_t> * B = vectors + 1;
    const SparseVector<index_t, coord_t, value_t> * C = vectors + 2;


    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_C, end_C;

    start_C = mergepath_partitions[0*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_C = mergepath_partitions[0*num_threads + tid + 1];
    } else {
        end_C = C->nnz - 1;
    }

    index_t idx_C = start_C + 1;

    index_t idx_D = nnz_prefix[tid];

    index_t idx_A = 0;
    index_t idx_B = 0;
    index_t end_A = locate(A, C->indices[end_C>=0 ? end_C: 0], (index_t)0, A->nnz - 1);
    index_t end_B = locate(B, C->indices[end_C>=0 ? end_C: 0], (index_t)0, B->nnz - 1);
    while( (idx_C <= end_C)) {
        coord_t ic0 = C->indices[idx_C];

        coord_t i = ic0;

        idx_A = locate(A, i, idx_A, end_A);
        idx_B = locate(B, i, idx_B, end_B);

        if (i == A->indices[idx_A] && i == B->indices[idx_B]) {
            D_indices[idx_D] = i;
            D_values[idx_D] = (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C];
            idx_D++;
        } else if (i == A->indices[idx_A]) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
        } else if (i == B->indices[idx_B]) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
        }

        idx_C += (index_t)(ic0 == i);
    }
    return;
};

template void sparse_vector_fusion_test(const SparseVector<int, int, float> A, const SparseVector<int, int, float> B, const SparseVector<int, int, float> C, 
    int * & D_indices, float * & D_values, int * & D_nnz, int num_fused, float * & D_times);

template void sparse_vector_fusion_test(const SparseVector<int, int64_t, float> A, const SparseVector<int, int64_t, float> B, const SparseVector<int, int64_t, float> C, 
    int64_t * & D_indices, float * & D_values, int * & D_nnz, int num_fused, float * & D_times);