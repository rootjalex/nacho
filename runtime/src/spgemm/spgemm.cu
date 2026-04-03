#include "spgemm.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/distance.h>
#include <iostream>
#include <sstream>


template<typename index_t> 
struct output_index{
    index_t i;
    index_t k;
    __host__ __device__ output_index() {}
    __host__ __device__ output_index(index_t x) : i{x}, k{x} {}
    __host__ __device__ output_index(index_t i, index_t k) : i{i}, k{k} {}
};

template<typename index_t> 
struct custom_hash {
  __device__ uint32_t operator()(output_index<index_t>  const& k) const noexcept { 
        return  cuco::default_hash_function<index_t>()(k.i) ^ cuco::default_hash_function<index_t>()(k.k);
   };
};

template<typename index_t> 
struct custom_compare {
    __host__ __device__
    bool operator()(const output_index<index_t>& a, const output_index<index_t>& b) const {
        if(a.i == b.i) {
            return a.k < b.k; // If 'i' is the same, sort by 'k' 
        }
        return a.i < b.i; // Sort by 'i'
    }
};


// User-defined device key equal callable
template<typename index_t> 
struct custom_key_equal {
  __device__ bool operator()(output_index<index_t> const& lhs, output_index<index_t> const& rhs) const noexcept
  {
    return lhs.i == rhs.i && lhs.k == rhs.k;
  }
};

template<typename index_t>
__global__ void spgemm_ij_compute_kernel(
        index_t A_rows, index_t * A_row_offsets, index_t * A_col_indices, index_t A_nnz,
        index_t B_rows, index_t * B_row_offsets, index_t * B_col_indices, index_t B_nnz,
        index_t * T_work_offsets, index_t per_thread_work
){
        // Compute the work for each thread
        index_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

        index_t start = thread_id * per_thread_work;
        index_t end = min(start + per_thread_work - 1, A_nnz - 1);
        for(index_t idx = start; idx <= end; idx++) {
            T_work_offsets[idx] = B_row_offsets[A_col_indices[idx]+1] - B_row_offsets[A_col_indices[idx]];
        }
        return;
};


template<typename index_t>
__device__ index_t row_search(index_t * row_offsets, index_t rows, index_t search) {
    // Binary search to find the row index for given offset
    index_t left = 0;
    index_t right = rows - 1;

    while (left < right) {
        index_t mid = left + (right - left + 1) / 2;
        if (row_offsets[mid] <= search && row_offsets[mid + 1] > search) {
            return mid;
        } else if (row_offsets[mid] <= search) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }
    //printf("Thread %d: row_search(%d) = %d\n", blockIdx.x * blockDim.x + threadIdx.x, search, left);
    return left; // Not found
};

template<typename index_t>
__global__ void spgemm_ijk_partitions_kernel(
        index_t A_rows, index_t * A_row_offsets, index_t * A_col_indices, index_t A_nnz,
        index_t B_rows, index_t * B_row_offsets, index_t * B_col_indices, index_t B_nnz,
        index_t * T_work_offsets,
        index_t * partitions_ijk_A_i, index_t * partitions_ijk_A_j, index_t * partitions_ijk_B_k,
        index_t per_thread_work
){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    index_t count = thread_id * per_thread_work;

    if(count ==0 ) {
        partitions_ijk_A_i[thread_id] = 0;
        partitions_ijk_A_j[thread_id] = 0;
        partitions_ijk_B_k[thread_id] = B_row_offsets[A_col_indices[0]]-1;
        return;
    }

    if(count >= T_work_offsets[A_nnz -1]) {
        partitions_ijk_A_i[thread_id] = A_rows -1;
        partitions_ijk_A_j[thread_id] = A_nnz -1;
        partitions_ijk_B_k[thread_id] = B_row_offsets[A_col_indices[A_nnz -1]+1]-1;
        return;
    }

    index_t start = 0;
    index_t end = A_nnz - 1;

    index_t mid = (start + end) / 2;

    while(start < end) {
        mid = (start + end + 1) / 2;
        index_t ub_work = T_work_offsets[mid];
        index_t lb_work = mid ==0 ? 0: T_work_offsets[mid - 1];


        if(ub_work  < count) {
            start = mid;
        } else if (lb_work >= count) {
            end = mid - 1;
        } else{
            // lb_work < count <= ub_work
            start = mid;
            end = mid;
            break;
        }
    }

    index_t j = start;

    index_t rem = count - (j==0? 0 : T_work_offsets[j-1]);

    index_t k = B_row_offsets[A_col_indices[j]] + rem-1;

    

    partitions_ijk_A_i[thread_id] = row_search(A_row_offsets, A_rows, j);
    partitions_ijk_A_j[thread_id] = j;
    partitions_ijk_B_k[thread_id] = k;
};


template<typename index_t, typename value_t, typename map_ref_t>
__global__ void spgemm_ijk_compute_scatter_reduce_kernel(
        index_t A_rows, index_t * A_row_offsets, index_t * A_col_indices, value_t * A_values, index_t A_nnz,
        index_t B_rows, index_t * B_row_offsets, index_t * B_col_indices, value_t * B_values, index_t B_nnz,
        index_t * partitions_ijk_A_i, index_t * partitions_ijk_A_j, index_t * partitions_ijk_B_k,
        index_t * C_row_offsets, index_t * C_indices, value_t * C_values, map_ref_t map_ref
){
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    index_t start_i, start_j, start_k;

    // this threads iterates from (start_i, start_j, start_k) to (end_i, end_j, end_k). 
    // start is exclusive, end is inclusive

    start_i = partitions_ijk_A_i[thread_id];
    start_j = partitions_ijk_A_j[thread_id];
    start_k = partitions_ijk_B_k[thread_id];



    index_t end_i, end_j, end_k;
    if(thread_id != gridDim.x * blockDim.x -1) {
        end_i = partitions_ijk_A_i[thread_id + 1];
        end_j = partitions_ijk_A_j[thread_id + 1];
        end_k = partitions_ijk_B_k[thread_id + 1];
    } else {
        end_i = A_rows-1;
        end_j = A_nnz-1;
        end_k = B_row_offsets[A_col_indices[A_nnz -1]+1]-1;
    }


    start_k ++; // increment k before beginning co-iteration as (start_i, start_j, start_k) is exclusive

    for(index_t i = start_i; i <= end_i; i++) {
        index_t count_i = 0;

        index_t curr_j_start = (i == start_i ? start_j : A_row_offsets[i]);
        index_t curr_j_end = (i == end_i ? end_j : A_row_offsets[i+1]-1);
      
        for(index_t j = curr_j_start; j <= curr_j_end; j++) {

            index_t curr_k_start = ( (i==start_i && j == start_j) ? start_k : B_row_offsets[A_col_indices[j]]);
            index_t curr_k_end = ( (i==end_i && j == end_j) ? end_k : B_row_offsets[A_col_indices[j]+1]-1);

            for(index_t k = curr_k_start; k <= curr_k_end; k++) {
                // perform computation for (i,j,k)

                
                value_t a_val = A_values[j];
                value_t b_val = B_values[k];
                value_t prod = a_val * b_val;
            
                index_t k_ind = B_col_indices[k];
                output_index<index_t> key{i, k_ind};
               
                auto found = map_ref.find(key);
                index_t offset = found->second;
                index_t old = atomicCAS(&C_indices[offset], -1, k_ind);
                if(old == -1) {
                    count_i++;
                }

                // if(i==2 && B_col_indices[k] == 2){
                //     printf("Thread %d processing (i,j,k) = (%d, %d, %d).  a_val: %f, b_val: %f, prod: %f, offset: %d\n", thread_id, i, A_col_indices[j], B_col_indices[k], a_val, b_val, prod, offset);
                // }
                // if(offset  <8 ){
                //     printf("Thread %d processing (i,j,k) = (%d, %d, %d).  a_val: %f, b_val: %f, prod: %f, offset: %d\n", thread_id, i, A_col_indices[j], B_col_indices[k], a_val, b_val, prod, offset);
                // }

                atomicAdd(&C_values[offset], prod);
                
            }
        }
        atomicAdd(&C_row_offsets[i+1], count_i);
    }

};



template<typename index_t, typename map_ref_t>
__global__ void spgemm_ijk_precompute_scatter_reduce_kernel(
        index_t A_rows, index_t * A_row_offsets, index_t * A_col_indices, index_t A_nnz,
        index_t B_rows, index_t * B_row_offsets, index_t * B_col_indices, index_t B_nnz,
        index_t * partitions_ijk_A_i, index_t * partitions_ijk_A_j, index_t * partitions_ijk_B_k,
        map_ref_t map_ref
){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // This kernel pre-computes the required hashmap and also computes the total nnz in C.
    

    index_t start_i, start_j, start_k;

    // this threads iterates from (start_i, start_j, start_k) to (end_i, end_j, end_k). 
    // start is exclusive, end is inclusive

    start_i = partitions_ijk_A_i[thread_id];
    start_j = partitions_ijk_A_j[thread_id];
    start_k = partitions_ijk_B_k[thread_id];



    index_t end_i, end_j, end_k;
    if(thread_id != gridDim.x * blockDim.x -1) {
        end_i = partitions_ijk_A_i[thread_id + 1];
        end_j = partitions_ijk_A_j[thread_id + 1];
        end_k = partitions_ijk_B_k[thread_id + 1];
    } else {
        end_i = A_rows-1;
        end_j = A_nnz-1;
        end_k = B_row_offsets[A_col_indices[A_nnz -1]+1]-1;
    }

    // if(thread_id==0){
    //     printf("Thread %d processing from (i,j,k) = (%d, %d, %d) to (i,j,k) = (%d, %d, %d)\n", thread_id, start_i, start_j, start_k, end_i, end_j, end_k);
    // }


    start_k ++; // increment k before beginning co-iteration as (start_i, start_j, start_k) is exclusive

    for(index_t i = start_i; i <= end_i; i++) {

        index_t curr_j_start = (i == start_i ? start_j : A_row_offsets[i]);
        index_t curr_j_end = (i == end_i ? end_j : A_row_offsets[i+1]-1);
      
        for(index_t j = curr_j_start; j <= curr_j_end; j++) {

            index_t curr_k_start = ( (i==start_i && j == start_j) ? start_k : B_row_offsets[A_col_indices[j]]);
            index_t curr_k_end = ( (i==end_i && j == end_j) ? end_k : B_row_offsets[A_col_indices[j]+1]-1);

            for(index_t k = curr_k_start; k <= curr_k_end; k++) {
            
                // if(i==0 && B_col_indices[k] == 1){
                //     printf("Thread %d processing (i,j,k) = (%d, %d, %d)\n", thread_id, i, A_col_indices[j], B_col_indices[k]);
                // }

                 // insert (i,k) into the hashmap
                map_ref.insert(cuco::pair{output_index<index_t>(i, B_col_indices[k]), 0});

            }
        }
        
    }

};

template<typename index_t, typename map_ref_t>
__global__ void assign_offsets_kernel(
        output_index<index_t> * keys,
        map_ref_t map_ref,
        size_t num_keys
){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    for(index_t i = thread_id; i < num_keys; i += blockDim.x * gridDim.x) {

        //auto key_ptr = thrust::device_pointer_cast(&keys[i]);
        auto found = map_ref.find(keys[i]);
        auto ref = cuda::atomic_ref<typename map_ref_t::mapped_type, cuda::thread_scope_device>{found->second};

        ref.store(i);
    }

    return;
};

void spgemm_impl(CSRMatrix<int, float> A, CSRMatrix<int, float> B,
    int * & C_row_offsets, int * & C_indices, float * & C_values, int * & C_nnz, float * & D_times) {

    int num_blocks = 256;
    int threads_per_block = 256;
    int num_threads = num_blocks * threads_per_block;

    // CIN
    // forall $i \in i_A$:
    //   forall $j \in j_A[i]$ with $j_B = j$:
    //     forall $k \in k_B[j]$:
    //       C[i,k] += A[i, j] * B[j, k] 

    
    // Step 1: Consider first sparse intersection j 
    //
    // Consider first 2 loops of CIN
    // forall $i \in i_A$:
    //   forall $j \in j_A[i]$ with $j_B = j$:
    // 
    //    Substep a: Create partitions for i,j in A and B.
    //              1. W_a = count_nnz_lt(A ,i ,j) 
    //              2. W_b = 0 as j is dense in B so iteration work for first 2 loops for B is 0 (just a lookup).


    //    Substep b: Pre-Compute Thread_offsets 

    //    Substep c: Compute paas
    //              1: Compute intersection j (same as A for this case) and write i_C , j_C to T.
    //              2: Compute work_offsets T' for next partition, where [i,j] = W_k(i,j) = nnzB_k(i,j) i.e amount of work we have to do
    //                      till next sparse intersection (which is k) for every, i,j value.
    //              3: T' <- prefix_sum(T')

    // Step 2: Now, frist sparse intersection is resolved. no more outer intersections left. consider the whole CIN.
    // forall $i \in i_C$:
    //   forall $j \in j_C$:
    //     forall $k \in k_B[j]$:
    //       C[i,k] += A[i, j] * B[j, k]
    
    //    Substep a: Create partitions i,j,k in A and B.
    //               1. W_a = count_nnz_lt(A, i, j)
    //               2. W_b = T'[W_a-1]

    //    Substep b: Scatter Reduction - Use Hashmap based pre-compute

    //    Substep c: Allocate C and Compute using Hashmap

    


    // Step 1 - Substep a b and c are merged into a single kernel for this case as we have just one sparse tensor (A) 
    //   in the first two dimensions i,j. Hence the partitions and thread offsets computation is trivial.


    int * T_work_offsets;

    CHECK_CUDA(cudaMalloc(&T_work_offsets, sizeof(int)*A.nnz));


    int per_thread_work = (A.nnz ) / num_threads + 1;

    cudaEvent_t start, stop;
    
    spgemm_ij_compute_kernel<<<num_blocks, threads_per_block>>>(
        A.rows, A.row_offsets, A.col_indices, A.nnz,
        B.rows, B.row_offsets, B.col_indices, B.nnz,
        T_work_offsets,
        per_thread_work
    );


    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int * T_work_offsets_prefix;

    CHECK_CUDA(cudaMalloc((void**)&T_work_offsets_prefix, sizeof(int)*A.nnz));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, T_work_offsets, T_work_offsets_prefix, A.nnz));

    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, T_work_offsets, T_work_offsets_prefix, A.nnz));
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaFree(T_work_offsets));

    


    // Step 2 - Substep a

    int * total_work = new int[1];
    CHECK_CUDA(cudaMemcpy(total_work, &T_work_offsets_prefix[A.nnz - 1], sizeof(int), cudaMemcpyDeviceToHost));

    per_thread_work = total_work[0] / num_threads + 1;

    int * partitions_ijk_A_i, * partitions_ijk_A_j, * partitions_ijk_B_k;
    CHECK_CUDA(cudaMalloc(&partitions_ijk_A_i, sizeof(int) * num_threads));
    CHECK_CUDA(cudaMalloc(&partitions_ijk_A_j, sizeof(int) * num_threads));
    CHECK_CUDA(cudaMalloc(&partitions_ijk_B_k, sizeof(int) * num_threads));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    spgemm_ijk_partitions_kernel<<<num_blocks, threads_per_block>>>(
        A.rows, A.row_offsets, A.col_indices, A.nnz,
        B.rows, B.row_offsets, B.col_indices, B.nnz,
        T_work_offsets_prefix,
        partitions_ijk_A_i, partitions_ijk_A_j, partitions_ijk_B_k,
        per_thread_work
    );
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    D_times[0]=milliseconds;

    CHECK_CUDA(cudaFree(T_work_offsets_prefix));



    // Step 2 - Substep b

   auto map = cuco::static_map{std::size_t(1.2* float(total_work[0])),
                              cuco::empty_key{output_index<int>{-1}},
                              cuco::empty_value{-1},
                              custom_key_equal<int>{},
                              cuco::linear_probing<1, custom_hash<int>>{}};

    auto insert_ref = map.ref(cuco::insert);
    

     cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    spgemm_ijk_precompute_scatter_reduce_kernel<<<num_blocks, threads_per_block>>>(
        A.rows, A.row_offsets, A.col_indices, A.nnz,
        B.rows, B.row_offsets, B.col_indices, B.nnz,
        partitions_ijk_A_i, partitions_ijk_A_j, partitions_ijk_B_k,
        insert_ref
    );
     cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    D_times[1]=milliseconds;


    auto find_ref = map.ref(cuco::find);
    

    { // explicit scoping for d_keys and d_values to free memory after use - probably not needed
        thrust::device_vector<output_index<int>> d_keys(map.size());
        thrust::device_vector<int> d_values(map.size());
        auto keys_begin = d_keys.begin();
        auto values_begin = d_values.begin();

        auto [keys_end, values_end] = map.retrieve_all(keys_begin, values_begin);
        
        thrust::sort(d_keys.begin(), keys_end, custom_compare<int>{});
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        assign_offsets_kernel<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(d_keys.data()),
            find_ref,
            thrust::distance(d_keys.begin(), keys_end)
        );
         cudaEventRecord(stop);

        cudaEventSynchronize(stop);
         milliseconds = 0;
         cudaEventElapsedTime(&milliseconds, start, stop);
            D_times[2]=milliseconds;
    }

    *C_nnz = map.size();

    //std::cout << "C nnz: " << *C_nnz << std::endl;
    // Step 2 - Substep c

    int * C_row_offsets_temp;

    CHECK_CUDA(cudaMalloc(&C_row_offsets_temp, sizeof(int)*(A.rows+1)));
    CHECK_CUDA(cudaMalloc(&C_indices, sizeof(int)*(*C_nnz)));
    CHECK_CUDA(cudaMalloc(&C_values, sizeof(float)*(*C_nnz)));

    CHECK_CUDA(cudaMemset(C_indices, -1, sizeof(int)*(*C_nnz)));
    CHECK_CUDA(cudaMemset(C_values, 0, sizeof(float)*(*C_nnz)));
    CHECK_CUDA(cudaMemset(C_row_offsets_temp, 0, sizeof(int)*(A.rows+1)));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    spgemm_ijk_compute_scatter_reduce_kernel<<<num_blocks, threads_per_block>>>(
        A.rows, A.row_offsets, A.col_indices, A.values, A.nnz,
        B.rows, B.row_offsets, B.col_indices, B.values, B.nnz,
        partitions_ijk_A_i, partitions_ijk_A_j, partitions_ijk_B_k,
        C_row_offsets_temp, C_indices, C_values, find_ref
    );
     cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    D_times[3]=milliseconds;



    // prefix sum C_row_offsets
    void *d_temp_storage_2 = nullptr;
    size_t temp_storage_bytes_2 = 0;
    CHECK_CUDA(cudaMalloc((void**)&C_row_offsets, sizeof(int)*(A.rows+1)));
    CHECK_CUDA(cudaMemset(C_row_offsets, 0, sizeof(int)*(A.rows+1)));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage_2, temp_storage_bytes_2, C_row_offsets_temp+1, C_row_offsets+1, A.rows));

    CHECK_CUDA(cudaMalloc(&d_temp_storage_2, temp_storage_bytes_2));

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage_2, temp_storage_bytes_2, C_row_offsets_temp+1, C_row_offsets+1, A.rows));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_temp_storage_2));
    CHECK_CUDA(cudaFree(C_row_offsets_temp));

    CHECK_CUDA(cudaFree(partitions_ijk_A_i));
    CHECK_CUDA(cudaFree(partitions_ijk_A_j));
    CHECK_CUDA(cudaFree(partitions_ijk_B_k));
};



void spgemm_cusparse_impl(
            CSRMatrix<int, float> A, CSRMatrix<int, float> B,
            int * & C_row_offsets, int * & C_indices, float * & C_values, int * & C_nnz
    ) {
    
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                      (void*)A.row_offsets, (void*)A.col_indices, (void*)A.values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateCsr(&matB, B.rows, B.cols, B.nnz,
                      (void*)B.row_offsets, (void*)B.col_indices, (void*)B.values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

     CHECK_CUDA(cudaMalloc(&C_row_offsets, sizeof(int)*(A.rows + 1)));

    cusparseCreateCsr(&matC, A.rows, B.cols, 0,
                      C_row_offsets, nullptr, nullptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);


     void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    float alpha = 1.0f;
    float beta = 0.0f;

   cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )


    *C_nnz = static_cast<int>(C_nnz1);
   
    CHECK_CUDA(cudaMalloc(&C_indices, sizeof(int)*(*C_nnz)));
    CHECK_CUDA(cudaMalloc(&C_values, sizeof(float)*(*C_nnz)));
    cusparseCsrSetPointers(matC, C_row_offsets, C_indices, C_values);
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    cusparseDestroy(handle);
}