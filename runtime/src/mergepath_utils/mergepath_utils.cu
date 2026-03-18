#include "mergepath_utils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

# define MAX_NUM_VECTORS 6

template<typename index_t,typename coord_t, typename value_t>
__global__ void kern_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work, bool old);


template<typename index_t, typename coord_t, typename value_t>
__device__  void find_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const index_t total_size,
    const index_t count,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions);


template<typename index_t , typename coord_t, typename value_t>
__device__  void balanced_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t *partitions);

template<typename index_t , typename coord_t, typename value_t, int NUM_VECTORS>
__device__  void find_balanced_mergepath_partition_sparse_vectors(
    const index_t total_size,
    const index_t count,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions);

template<typename index_t, typename coord_t, typename value_t>
void sparse_vectors_balanced_mergepath(
    const uint num_vectors,
    index_t * mergepath_partitions,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    const uint num_blocks,
    const uint threads_per_block,
    const index_t per_thread_work,
    bool old
) {
    if(num_vectors>MAX_NUM_VECTORS){
        throw std::runtime_error("More than MAX_NUM_VECTORS");          
    }
    int num_threads = num_blocks * threads_per_block;


    kern_mergepath_partition_sparse_vectors<<<num_blocks, threads_per_block>>>(
        num_vectors,
        total_size,
        vectors,
        mergepath_partitions,
        per_thread_work,
        old
    );
    
    //CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    return ;

};


// kern_mergepath_partition_sparse_vectors finds the merge path partitions for each thread
// 
// params:
//   num_vectors : number of input vectors
//   total_size : total number of non-zeros across all sparse vectors
//   vectors : array of input sparse vectors
//   mergepath_partitions : (Must be preallocated) mergepath_partitions[i*num_threads+t] is the start of the partition for i^th tensor for the j^th thread
//                        boundary is exclusive, i.e. the first index that should be processed by that t^th thread is the index after mergepath_partitions[i*num_threads+t]
//   
//   per_thread_work :  number of non-zeros each thread should process (this could be +-1 due to balancing)
template<typename index_t, typename coord_t, typename value_t>
__global__ void kern_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions,
    const index_t per_thread_work, bool old) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    index_t count = (  (tid) * per_thread_work );

    if(old || num_vectors<=3){


    find_mergepath_partition_sparse_vectors(
        num_vectors,
        total_size,
        count,
        vectors,
        mergepath_partitions
    );
    
    //__syncthreads();
    //if(tid<30 && num_vectors==3)printf("tid %d : %d %d %d\n", tid, mergepath_partitions[tid],mergepath_partitions[gridDim.x * blockDim.x+tid], mergepath_partitions[2*gridDim.x * blockDim.x+tid]);

    balanced_mergepath_partition_sparse_vectors(
        num_vectors,
        vectors,
        mergepath_partitions
    );
    // for(int i=0; i<num_vectors; i++) {
    //     mergepath_partitions[i*gridDim.x * blockDim.x+tid] = partitions[i];
    // }
}  else {
    switch (num_vectors) {
        case 1:
            find_balanced_mergepath_partition_sparse_vectors<index_t, coord_t, value_t, 1>(
                total_size,
                count,
                vectors,
                mergepath_partitions
            );
            break;
        case 2:
            find_balanced_mergepath_partition_sparse_vectors<index_t, coord_t, value_t, 2>(
                total_size,
                count,
                vectors,
                mergepath_partitions
            );
            break;
        case 3:
            find_balanced_mergepath_partition_sparse_vectors<index_t, coord_t, value_t, 3>(
                total_size,
                count,
                vectors,
                mergepath_partitions
            );
            break;
        case 4:
            find_balanced_mergepath_partition_sparse_vectors<index_t, coord_t, value_t, 4>(
                total_size,
                count,
                vectors,
                mergepath_partitions
            );
            break;
        case 5:
            find_balanced_mergepath_partition_sparse_vectors<index_t, coord_t, value_t, 5>(
                total_size,
                count,
                vectors,
                mergepath_partitions
            );
            break;
        case 6:
            find_balanced_mergepath_partition_sparse_vectors<index_t, coord_t, value_t, 6>(
                total_size,
                count,
                vectors,
                mergepath_partitions
            );
            break;
    }
      
} 
    return;
}

// passing NUM_VECTORS as template as don't expect this to be very large,
// and can allocate arrays in stack of size NUM_VECTORS
template<typename index_t , typename coord_t, typename value_t, int NUM_VECTORS>
__device__  void find_balanced_mergepath_partition_sparse_vectors(
    const index_t total_size,
    const index_t count,
    const SparseVector<index_t, coord_t, value_t> * vectors,
    index_t * mergepath_partitions
) {
    int num_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(count == 0) {
        for(int i=0; i<NUM_VECTORS; i++) {
            mergepath_partitions[i*num_threads+tid] = -1;
        }
        return;
    }
   
    if(count >= total_size) {
        for(int i=0; i<NUM_VECTORS; i++) {
            mergepath_partitions[i*num_threads+tid] = vectors[i].nnz - 1;
        }
        return;
    }

    if (NUM_VECTORS ==1) {
        index_t index = min(count-1, vectors[0].nnz-1);
        mergepath_partitions[0*num_threads+tid] = index;
        return;
    }

    coord_t lo = vectors[0].indices[0] , hi = vectors[0].indices[vectors[0].nnz - 1];

    for(int i=1; i<NUM_VECTORS; i++) {
        lo = min(lo, vectors[i].indices[0]);
        hi = max(hi, vectors[i].indices[vectors[i].nnz - 1]);
    }

    index_t boundaries[NUM_VECTORS];
    coord_t curr;
    int sum = 0;

    // Using break condition separately as once lo=hi, need to run the loop once more
    // so that boundaries are recalculated for the correct value
    while(true) {
        
        curr = (lo+hi)/2;
        
        sum = 0;
        for(int i=0;i<NUM_VECTORS;i++) {

            index_t vec_lo = -1;
            index_t vec_hi = vectors[i].nnz-1;
            index_t vec_mid;
            
            while(vec_lo < vec_hi) {
                vec_mid = (vec_lo + vec_hi + 1)/2;

                if(vectors[i].indices[vec_mid] > curr) {
                    vec_hi = vec_mid-1;
                } else if(vectors[i].indices[vec_mid] <= curr) {
                    vec_lo = vec_mid;
                }
            }
            vec_mid = vec_lo;

            sum+=vec_mid + 1;
            
            boundaries[i] = vec_mid;
        }

        if(lo==hi) {
            break;
        }
        
        if(sum >= count) {
            hi = curr;
        } else {
            lo = curr+1;
        }
    }
    curr = lo;

    for(int i=0;i<NUM_VECTORS;i++) {
        mergepath_partitions[i*num_threads+tid] = boundaries[i];
    }

    return;
};


template<typename index_t, typename coord_t, typename value_t>
__device__  void find_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const index_t total_size,
    const index_t count,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions) {

    int num_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(num_vectors > 3 ) {
        //throw std::runtime_error("Error: find_mergepath_partition_sparse_vectors only supports num_vectors <= 3");
        return;
    }


    if(count == 0) {
        for(int i=0; i<num_vectors; i++) {
            mergepath_partitions[i*num_threads+tid] = -1;
        }
        return;
    }
   
    if(count >= total_size) {
        for(int i=0; i<num_vectors; i++) {
           mergepath_partitions[i*num_threads+tid] = vectors[i].nnz - 1;
        }
        return;
    }

    // num_vectors = 1 is just dividing a single vector equally, added just for generality in implementation
    if (num_vectors ==1) {
        index_t index = min(count-1, vectors[0].nnz-1);
        mergepath_partitions[0*num_threads+tid] = index;
        return;
    }
    else if (num_vectors ==2) {
        index_t lo = -1;
        index_t hi = min(count-1, vectors[0].nnz-1);
        index_t a_index, b_index;

        // a_index + b_index == count-2,  as (indices are 0 indexed) (if count = 1, we want a_index = 0, b_index = -1 to be the output)
        while(lo < hi ) {
            a_index = (lo + hi+1)/2;
            b_index = count - 2 - a_index;
            
            if( b_index + 1> vectors[1].nnz-1) {
                lo = a_index;
                continue;
            }

            if(vectors[0].indices[a_index] > vectors[1].indices[b_index + 1]) {
                hi = a_index-1;
            } else {
                lo = a_index;
            }
        }
       
    
        a_index = lo;

        b_index = min(count - 2 - a_index, (index_t)(vectors[1].nnz-1)); // min behaving wierd when one value is -ve. porbably because it is expecting uints?
        
        mergepath_partitions[0*num_threads+tid] = a_index; mergepath_partitions[1*num_threads+tid] = b_index;
        return;
    }  else if (num_vectors == 3) {
        // General case for num_tensors > 2
      
        index_t lo = -1;
        index_t hi = min(count-1, vectors[0].nnz-1);
        index_t a_index;
        index_t start_c=-1, end_c=vectors[2].nnz-1, start_b=-1, end_b=vectors[1].nnz-1;

        
        while(lo < hi ) {
            a_index = (lo + hi+1)/2;       
           
            index_t rem_count = count - (a_index + 1);

            if(rem_count <0) {
                hi = a_index-1;
                continue;
            }

            index_t lo_b = start_b;
            index_t hi_b = min(rem_count-1, end_b);

            if(hi_b < lo_b) {
                hi = a_index-1;
                continue;
            }
            // __syncthreads();
            // if(tid==992){
            //     printf("tid %d : %d %d,  lo_a %d lo_b %d lo_a %d hi_b %d\n",threadIdx.x, blockIdx.x,  tid, lo, lo_b, hi, hi_b);
            // }
            // __syncthreads();

            index_t b_index, c_index;
            bool a_too_big = false;
            bool cut = false;
            // a_index + b_index == count-2,  as (indices are 0 indexed) (if count = 1, we want a_index = 0, b_index = -1 to be the output)

            
            while(lo_b < hi_b ) {

                b_index = (lo_b + hi_b+1)/2;
                c_index = rem_count - 2 - b_index;
                if( c_index + 1 > end_c) {
                    lo_b = b_index;
                    continue;
                }

                if(c_index < start_c) {
                    hi_b = b_index-1;
                    continue;
                }
                    
                // if(c_index + 1 <= vectors[2].nnz -1 &&  vectors[1].indices[b_index] > vectors[2].indices[c_index+1]) {
                //     if(c_index + 1<= vectors[2].nnz -1 &&  vectors[0].indices[a_index] > vectors[2].indices[c_index+1]) {
                //         start_c = c_index;
                //     }
                // }

                // if(a_index +1 <= vectors[0].nnz -1 &&  vectors[2].indices[c_index] > vectors[0].indices[a_index+1]) {
                //     if(b_index +1 <= vectors[1].nnz -1 &&  vectors[2].indices[c_index] > vectors[1].indices[b_index+1]) {
                //         end_c = c_index;
                //     }
                // }
                
                if(b_index +1 <= vectors[1].nnz -1 &&  vectors[0].indices[a_index] > vectors[1].indices[b_index+1]) {
                    if(c_index +1 <= vectors[2].nnz -1 &&  vectors[0].indices[a_index] > vectors[2].indices[c_index+1]) {
                        a_too_big = true;
                        cut = true;
                        break;
                    }
                    if(b_index +1 <= vectors[1].nnz -1 &&  c_index>=0 && vectors[2].indices[c_index] > vectors[1].indices[b_index+1]) {
                        start_b = b_index;
                    }
                    lo_b = b_index;
                    continue;
                }


                if(a_index +1 <= vectors[0].nnz -1 &&  vectors[1].indices[b_index] > vectors[0].indices[a_index+1]) {
                  if(a_index +1 <= vectors[0].nnz -1 &&  c_index>=0 && vectors[2].indices[c_index] > vectors[0].indices[a_index+1]) {
                    a_too_big = false;
                    cut = true;
                    break;
                  }
                  if(c_index +1 <= vectors[2].nnz -1 &&  vectors[1].indices[b_index] > vectors[2].indices[c_index+1]) {
                        end_b = b_index;
                  }
                    hi_b = b_index-1;
                    continue;
                }
                
                //lo_b = b_index; hi_b = b_index;
                if(c_index+1<=vectors[2].nnz-1 && vectors[1].indices[b_index] > vectors[2].indices[c_index + 1]) {
                    hi_b = b_index-1;
                } else {
                    lo_b = b_index;
                }
            }
            //if(cut>5)printf("cuts %d\n", cut);
        
            b_index = lo_b;

            c_index = min(rem_count - 2 - b_index, (index_t)(vectors[2].nnz-1)); // min behaving wierd when one value is -ve. porbably because it is expecting uints?
            // if(c_index < 0) {
                
            // }
            // __syncthreads();
            // if(tid==992){
            //     printf("tid %d : %d %d,  a_index %d b_index %d c_index %d for count %d\n",threadIdx.x, blockIdx.x,  tid, a_index, b_index, c_index, count);
            // }
            // __syncthreads();
            //for(int i=1; i<num_vectors; i++) {
                
                //index_t b_index = partitions[i];
            if(!cut) {
                // if the next element after b_index is less than a_index that means , a_index is too big as we can instead include more elements from b
                // and thus we need to decrease a_index
                if(b_index +1 <= vectors[1].nnz -1 &&  vectors[0].indices[a_index] > vectors[1].indices[b_index+1]) {
                    a_too_big = true;
                    //break;
                }

                if(c_index +1 <= vectors[2].nnz -1 &&  vectors[0].indices[a_index] > vectors[2].indices[c_index+1]) {
                    a_too_big = true;
                    //break;
                }
            }
            //}

            if(a_too_big) {
                hi = a_index-1;
            } else {
                lo = a_index;
            }
            
        }

        a_index = lo;

           
        index_t rem_count = count - (a_index + 1);

        index_t lo_b = -1;
        index_t hi_b = min(rem_count-1, vectors[1].nnz-1);
        index_t b_index, c_index;

    // a_index + b_index == count-2,  as (indices are 0 indexed) (if count = 1, we want a_index = 0, b_index = -1 to be the output)
        while(lo_b < hi_b ) {
            b_index = (lo_b + hi_b+1)/2;
            c_index = rem_count - 2 - b_index;
        
            if( c_index + 1 > vectors[2].nnz-1) {
                lo_b = b_index;
                continue;
            }

            if(c_index+1<=vectors[2].nnz-1 && vectors[1].indices[b_index] > vectors[2].indices[c_index + 1]) {
                hi_b = b_index-1;
            } else {
                lo_b = b_index;
            }
        }
    

        b_index = lo_b;

        c_index = min(rem_count - 2 - b_index, (index_t)(vectors[2].nnz-1)); 
        // min behaving wierd when one value is -ve. porbably because it is expecting uints?

        mergepath_partitions[0*num_threads + tid] = a_index;
        mergepath_partitions[1*num_threads + tid] = b_index;
        mergepath_partitions[2*num_threads + tid] = c_index;
        
    }
};


template <typename index_t, typename coord_t, typename value_t>
__device__  void balanced_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const SparseVector<index_t, coord_t, value_t> * vectors, 
    index_t * mergepath_partitions) {
    
    if (num_vectors ==1) {
        return;
    }
   

    int num_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
     //__syncthreads();
     //if(tid<30)printf("starting balancing %d\n", vectors[0].nnz);
    int permutation[MAX_NUM_VECTORS];

   
    
    for(int i=0; i<num_vectors; i++) {
        permutation[i] = i;
    }
     //__syncthreads();
     //if(tid<30)printf("starting sort %d\n", vectors[0].nnz);
    // sort indices and get permutation
     // Bubble sort - not efficient but n is small so should be fine (can implement better sort later if needed)
    for(int i=0; i<num_vectors; i++) {
        for(int j=0; j<num_vectors-i-1; j++) {
            index_t index_1 = mergepath_partitions[permutation[j]*num_threads+tid];
            index_t index_2 = mergepath_partitions[permutation[j+1]*num_threads+tid];

            if(index_2< 0) {
                int temp = permutation[j];
                permutation[j] = permutation[j+1];
                permutation[j+1] = temp;
            } else if (index_1>=0 && index_2>=0 && 
                vectors[permutation[j]].indices[index_1] >  vectors[permutation[j+1]].indices[index_2]) {

                int temp = permutation[j];
                permutation[j] = permutation[j+1];
                permutation[j+1] = temp;
            }
        }
    }

    //if(tid<30)printf("finsihed sort\n");
    // Now arr is sorted, we can balance the partitions
    for(int i=1; i<num_vectors; i++) {
        for(int j=0; j<i; j++) {
            int vector_i = permutation[i];
            int vector_j = permutation[j];
            index_t index_i = mergepath_partitions[vector_i*num_threads+tid];
            index_t index_j = mergepath_partitions[vector_j*num_threads+tid];

            if(index_i>=0 && index_j + 1 <= vectors[vector_j].nnz - 1 && vectors[vector_i].indices[index_i] == vectors[vector_j].indices[index_j + 1]) {
                // We can shift the boundary to the right for vector_j
                mergepath_partitions[vector_j*num_threads+tid] = index_j + 1;

            }
            
        }
    }
    
    return;
    };

template<typename index_t, typename coord_t, typename value_t>
void vector_matrix_mergepath_partition(
    index_t * mergepath_partitions_matrix_row,
    index_t * mergepath_partitions_matrix_col,
    index_t * mergepath_partitions_vector,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vector,
    const CSRMatrix<index_t, value_t> * matrix,
    const uint num_blocks,
    const uint threads_per_block,
    const index_t per_thread_work
) {

    kern_mergepath_partition_vector_matrix<<<num_blocks, threads_per_block>>>(
        mergepath_partitions_matrix_row,
        mergepath_partitions_matrix_col,
        mergepath_partitions_vector,
        total_size,
        vector,
        matrix,
        per_thread_work
    );

    CHECK_CUDA(cudaGetLastError());

    return;

};

template<typename index_t, typename coord_t, typename value_t>
__global__ void kern_mergepath_partition_vector_matrix(
    index_t * mergepath_partitions_matrix_row,
    index_t * mergepath_partitions_matrix_col,
    index_t * mergepath_partitions_vector,
    const index_t total_size,
    const SparseVector<index_t, coord_t, value_t> * vector,
    const CSRMatrix<index_t, value_t> * matrix,
    const index_t per_thread_work
) {
    int num_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    index_t count = (  (tid) * per_thread_work );

    if(count == 0) {
        mergepath_partitions_vector[tid] = -1;
        mergepath_partitions_matrix_row[tid] = -1;
        mergepath_partitions_matrix_col[tid] = -1;
    }

    if(count >= total_size) {
        mergepath_partitions_vector[tid] = vector->nnz - 1;
        mergepath_partitions_matrix_row[tid] = matrix->rows - 1;
        mergepath_partitions_matrix_col[tid] = matrix->nnz - 1;
    }
    
    index_t lo_row = 0;
    index_t hi_row = matrix->rows - 1;

    // if(tid == 30) {
    //     printf("lo_row %d hi_row %d count %d\n", lo_row, hi_row, count);
    // }

    while(lo_row < hi_row ) {
        index_t mid_row = (lo_row + hi_row)/2;
        
        index_t matrix_nnz = matrix->row_offsets[mid_row+1];

        index_t total_nnz = matrix_nnz + vector->nnz* (mid_row + 1); 
        // if(tid == 30) {
        // printf("lo_row %d hi_row %d count %d total_nnz %d\n", lo_row, hi_row, count, total_nnz);
        // }
        if(total_nnz > count) {
            hi_row = mid_row;
        } else {
            lo_row = mid_row + 1;
        }
    }

    // if(tid == 30) {
    //     printf("lo_row %d hi_row %d count %d\n", lo_row, hi_row, count);
    // }

    mergepath_partitions_matrix_row[tid] = lo_row;

    
    index_t remaining = count - (matrix->row_offsets[lo_row] + vector->nnz * lo_row);

    // if(tid<5) {
    //     printf("tid %d : row partition %d count %d remaining %d\n", tid, lo_row, count, remaining);
    // }
    
    // Now find partition in vector and matrix col
    index_t lo = -1;
    index_t hi = min(remaining-1, vector->nnz-1);
    index_t a_index, b_index;

    index_t num_cols = matrix->row_offsets[lo_row+1] - matrix->row_offsets[lo_row];
    index_t col_offset = matrix->row_offsets[lo_row];
    // a_index + b_index == remaining-2,  as (indices are 0 indexed) (if count = 1, we want a_index = 0, b_index = -1 to be the output)
    while(lo < hi ) {
        a_index = (lo + hi+1)/2;
        b_index = remaining - 2 - a_index;
        
        if( b_index + 1> num_cols-1) {
            lo = a_index;
            continue;
        }

        if(vector->indices[a_index] > matrix->col_indices[col_offset + b_index + 1]) {
            hi = a_index-1;
        } else {
            lo = a_index;
        }
    }
    

    a_index = lo;

    b_index = min(remaining - 2 - a_index, (index_t)(num_cols - 1)); // min behaving wierd when one value is -ve. porbably because it is expecting uints?

    //if(tid==4)printf("tid %d : row partition %d count %d remaining %d a_index %d b_index %d col_ind %d.  %d %d\n", tid, lo_row, count, remaining, a_index, b_index, col_offset + b_index + 1,  vector->indices[a_index], matrix->col_indices[col_offset + b_index + 1]);

    if(a_index>=0 && b_index+1 <= num_cols-1 && vector->indices[a_index]== matrix->col_indices[col_offset + b_index + 1]) {
        // Can shift boundary to right for matrix
        b_index = b_index + 1;
    } 

    if(b_index>=0 && a_index + 1 <= vector->nnz -1 && matrix->col_indices[col_offset + b_index] == vector->indices[a_index + 1]) {
        // Can shift boundary to right for vector
        a_index = a_index + 1;
    }
    //if(tid==4)printf("tid %d : row partition %d count %d remaining %d a_index %d b_index %d\n", tid, lo_row, count, remaining, a_index, b_index);

    mergepath_partitions_vector[tid] = a_index; mergepath_partitions_matrix_col[tid] = col_offset + b_index;

    

    return;
};


template __global__ void kern_mergepath_partition_sparse_vectors<int, int, float>(
    const uint num_vectors,
    const int total_size,
    const SparseVector<int, int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    bool old);

template __global__ void kern_mergepath_partition_sparse_vectors<int, int64_t, float>(
    const uint num_vectors,
    const int total_size,
    const SparseVector<int, int64_t, float> * vectors,
    int * mergepath_partitions,
    const int per_thread_work,
    bool old);

template __global__ void kern_mergepath_partition_vector_matrix(
    int * mergepath_partitions_matrix_row,
    int * mergepath_partitions_matrix_col,
    int * mergepath_partitions_vector,
    const int total_size,
    const SparseVector<int, int, float> * vector,
    const CSRMatrix<int, float> * matrix,
    const int per_thread_work
);