#include "csr_add.h"
#include "../cuda_utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>


// returns r in [0, m-1] such that row_off[r] <= x < row_off[r+1]
// returns -1 when x<0
__device__ inline int row_search(const int *row_off, int m, int x) {
    if(x<0)return -1;
    int lo = 0, hi = m; // hi = m as row_off[m] is total_work
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (row_off[mid] <= x) lo = mid;
        else hi = mid;
    }
    return lo;
}


// find the mergepath boundaries for each thread 
__global__ void mergepath_boundary_find(
    const int m, const int n,
    const int *A_row_ptr, const int *A_col,
    const int *B_row_ptr, const int *B_col, 
    int * mergepath_boundary_A_row, int* mergepath_boundary_B_row,
    int * mergepath_boundary_A_col, int* mergepath_boundary_B_col,
    const int per_thread_work) {

    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Thread %d In mergepath_boundary_find kernel\n",tid);
    int diag_boundary = (  (tid) * per_thread_work );

    if(diag_boundary <= 0) {
        mergepath_boundary_A_row[tid] = -1;mergepath_boundary_A_col[tid] = -1;
        mergepath_boundary_B_row[tid] = -1; mergepath_boundary_B_col[tid] = -1;
        return;
    }
    if(diag_boundary >= A_row_ptr[m] + B_row_ptr[m]) {
        mergepath_boundary_A_row[tid] = m-1; mergepath_boundary_A_col[tid] = A_row_ptr[m]-1;
        mergepath_boundary_B_row[tid] = m-1; mergepath_boundary_B_col[tid] = B_row_ptr[m]-1;
        return;
    }

    //if(tid==5014858)printf("Thread %d diag_boundary %d\n", tid, diag_boundary);
   
    int lo = -1;
    int hi = min(diag_boundary-1, A_row_ptr[m]-1);
    int mid, a_col, b_col, rA, rB;
    while(lo < hi ) {
        mid = (lo + hi+1)/2;
        if(diag_boundary - mid -1 > B_row_ptr[m]-1) {
            lo = mid;
            continue;
        }
        rA = row_search(A_row_ptr, m, mid);
        rB = row_search(B_row_ptr, m, diag_boundary - mid - 1);
        
        if(rA > rB) {
            hi = mid-1;
            continue;
        } else if (rA < rB)
        {
            lo = mid;
            continue;
        }
        
        // rA == rB

        a_col =  A_col[mid];
        b_col =  B_col[diag_boundary - mid-1]; 
    
        //  if(diag_boundary==50148580)
        //  printf("lo %d hi %d mid %d diag_boundary %d rA %d rB %d a_col %d b_col %d\n", lo, hi, mid, diag_boundary, rA, rB, a_col, b_col);
        if(a_col > b_col) {
            hi = mid-1;
        } else {
            lo = mid;
        }
    }
    //printf("Thread %d finished binary search lo %d hi %d diag_boundary %d\n", tid, lo, hi, diag_boundary);

    mid = lo;
    rA = row_search(A_row_ptr, m, mid);
    rB = row_search(B_row_ptr, m, diag_boundary - mid-2);

    

    int a_col_index = mid;
    int b_col_index = min(diag_boundary - mid-2, B_row_ptr[m]-1);

    //printf("Thread %d: initial boundary A [%d, %d] B [%d, %d]\n", tid, rA, a_col_index, rB, b_col_index);
    mergepath_boundary_A_row[tid] = rA; mergepath_boundary_A_col[tid] = a_col_index;
    
    mergepath_boundary_B_row[tid] = rB; mergepath_boundary_B_col[tid] = b_col_index;
    // if(tid == 504) {
    //     printf("rB %d, row_search(A_row_ptr, m, mid+1) %d, mid %d, diag_boundary %d, A_col[mid+1] %d, B_col[diag_boundary - mid] %d\n", rB, row_search(A_row_ptr, m, mid+1), mid,diag_boundary, A_col[mid+1], B_col[diag_boundary - mid]);
    // }
    if(row_search(A_row_ptr, m, a_col_index+1) == rB && a_col_index+1 < A_row_ptr[m] && b_col_index>=0 && A_col[a_col_index+1] == B_col[b_col_index]) {
        // printf("Adjusting A boundary from %d to %d\n", mid, mid+1);
        // //printf("A_col[%d] = %d B_col[%d] = %d\n", mid+1, A_col[mid+1], diag_boundary - mid, B_col[diag_boundary - mid]);
        mergepath_boundary_A_row[tid] = rB;
        mergepath_boundary_A_col[tid] = a_col_index + 1;
    } else if(rA == row_search(B_row_ptr,m, b_col_index + 1) && b_col_index + 1 < B_row_ptr[m] && a_col_index>=0 && B_col[b_col_index+1] == A_col[a_col_index] ) {
        // printf("Adjusting B boundary from %d to %d\n", diag_boundary - mid, diag_boundary - mid + 1);
        mergepath_boundary_B_row[tid] = rA;
        mergepath_boundary_B_col[tid] = b_col_index + 1;
    }

    //printf("Thread %d: final boundary A [%d, %d] B [%d, %d]\n", tid, mergepath_boundary_A_row[tid], mergepath_boundary_A_col[tid], mergepath_boundary_B_row[tid], mergepath_boundary_B_col[tid]);
    return;
}

__global__ void count_nnz_mergepath_kernel(
    const int m, const int n,
    const int *A_row_ptr, const int *A_col,     // CSR A
    const int *B_row_ptr, const int *B_col,   // CSR B                   
    int * mergepath_boundary_A_row, int* mergepath_boundary_B_row,
    int * mergepath_boundary_A_col, int* mergepath_boundary_B_col,                 
    int *row_counts, int * nnz_count)                            // output per-row counts (zeroed before launch)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //printf("Thread %d: start %lld end %lld total_work %d\n", tid, start, end, total_work);
    
    
    int rA = mergepath_boundary_A_row[tid], rB = mergepath_boundary_B_row[tid];
    int curr_rA_count =0, curr_rB_count =0;
    int A_index = mergepath_boundary_A_col[tid];
    int B_index = mergepath_boundary_B_col[tid];

    
    int end_A_col, end_B_col;
    if(tid != blockDim.x*gridDim.x -1 ){ // check as tid+1 is illegal access for last thread
        end_A_col = mergepath_boundary_A_col[tid+1];
        end_B_col = mergepath_boundary_B_col[tid+1];
    } else {
        end_A_col = A_row_ptr[m]-1;
        end_B_col = B_row_ptr[m]-1;
    }

    //if(rA<=196610 && rA>= 196607)printf("Thread %d: start_A [%d, %d] start_B [%d, %d], endA_col %d, endB_col %d\n", tid, rA, A_index, rB, B_index, end_A_col, end_B_col);
    //if(blockIdx.x ==0 && threadIdx.x <10)printf("Thread %d: start_A   [%d, %d] start_B   [%d, %d]\n", tid, rA, rB, A_index, B_index);

    int total_nnz_processed = 0;
    
    A_index++;
    while(A_index == A_row_ptr[rA+1]) {
        rA++;
    }

    B_index++;
    while(B_index == B_row_ptr[rB+1]) {
        rB++;
    }

    while(A_index <= end_A_col && B_index <= end_B_col) {
        if(rA < rB) {
            //  if(rA==999){
            //      printf("Thread %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
            A_index++;
            curr_rA_count++;
            total_nnz_processed++;
        } else if (rB < rA) {
            //   if(rB==999){
            //      printf("Thread %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
            B_index++;
            curr_rB_count++;
            total_nnz_processed++;
        } else {
            // rA == rB
            if(A_col[A_index] < B_col[B_index]) {
            //      if(rA==999){
            //      printf("Thread lg %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
                A_index++;
                curr_rA_count++;
                total_nnz_processed++;

            } else if (B_col[B_index] < A_col[A_index]) {
            //      if(rB==999){
            //      printf("Thread ds %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
                B_index++;
                curr_rB_count++;
                total_nnz_processed++;
            } else {
                //  if(rB==999){
                //  printf("Thread avs %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
                //  }
                A_index++; B_index++; // increment both but Add to only one of the two counter
                curr_rA_count++; 
                total_nnz_processed++;
            }
        }

        if(A_index == A_row_ptr[rA+1]) {
           
            atomicAdd(&row_counts[rA], curr_rA_count);
            //int after = atomicAdd(&row_counts[rA], 0);
            //if(tid==98438 && rA==196608)printf("Thread %d A: p %d after %d atomicAdd row %d count %d, address %p\n", tid,p,after, rA, curr_rA_count, (void*)&row_counts[rA]);
            while(A_index == A_row_ptr[rA+1]){
                rA++; 
                //if(tid==98438 && rA==196609)printf("Thread %d A: atomicAdd row %d count %d\n", tid, rA, curr_rA_count);
                curr_rA_count = 0;
            }
            
        }
        if(B_index == B_row_ptr[rB+1]) {
            
            atomicAdd(&row_counts[rB], curr_rB_count);
            //int after = atomicAdd(&row_counts[rB], 0);
            //if(tid==98438 && rB==196608)printf("Thread %d B: p %d after %d atomicAdd row %d count %d. address %p\n", tid,p,after, rB, curr_rB_count, (void*)&row_counts[rB]);
            while(B_index == B_row_ptr[rB+1]){
                rB++; 
                //if(tid==98438 && rB==196609)printf("Thread %d B : atomicAdd row %d count %d\n", tid, rB, curr_rB_count);
                curr_rB_count = 0;
            }
        }
        
    }
   // __threadfence(); 
    //if(tid==98438){int v = atomicAdd(&row_counts[196608], 0); printf("calue of row_count %d %p\n",v, (void*)&row_counts[196608]);}
    // if(tid==0)
    // printf("Thread %d: finished main loop A_index %d end_A %d B_index %d end_B %d\n", tid, A_index, end_A[1], B_index, end_B[1]);
    // // finish off remaining entries in A or B

    while(A_index <= end_A_col) {
        //  if(rA==999){
        //          printf("Thread ml %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
        //  }
        A_index++;
        curr_rA_count++;
        total_nnz_processed++;
        if(A_index == A_row_ptr[rA+1]) {
            //if(tid==98438)printf("Thread %d A: atomicAdd row %d count %d\n", tid, rA, curr_rA_count);
            atomicAdd(&row_counts[rA], curr_rA_count);
            while(A_index == A_row_ptr[rA+1]){
                rA++; 
                curr_rA_count = 0;
            }
        }
    }
    //if(tid==98438)printf("Thread ss %d A: atomicAdd row %d count %d\n", tid, rA, curr_rA_count);
    atomicAdd(&row_counts[rA], curr_rA_count);
    while(B_index <= end_B_col) {
        //  if(rB==999){
        //          printf("Thread lk %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
        //  }
        B_index++;
        curr_rB_count++;
        total_nnz_processed++;
        if(B_index == B_row_ptr[rB+1]) {
            //if(tid==98438)printf("Thread %d B: atomicAdd row %d count %d\n", tid, rB, curr_rB_count);
            atomicAdd(&row_counts[rB], curr_rB_count);
            while(B_index == B_row_ptr[rB+1]){
                rB++; 
                curr_rB_count = 0;
            }
        }
    }
    //if(tid==98438)printf("Thread la %d B: atomicAdd row %d count %d\n", tid, rB, curr_rB_count);
    atomicAdd(&row_counts[rB], curr_rB_count);
    nnz_count[tid] = total_nnz_processed;
    return;
}


__global__ void csr_add_mergepath(
    const int m, const int n,
    const int *A_row_ptr, const int *A_col, const float *A_val,    
    const int *B_row_ptr, const int *B_col, const float *B_val,  // CSR B  
    const int *C_row_ptr, int *C_col, float *C_val,            
    int * mergepath_boundary_A_row, int* mergepath_boundary_B_row,
    int * mergepath_boundary_A_col, int* mergepath_boundary_B_col,      
    int *nnzC_prefix
) {
        
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int rA = mergepath_boundary_A_row[tid], rB = mergepath_boundary_B_row[tid];

    int A_index = mergepath_boundary_A_col[tid];
    int B_index = mergepath_boundary_B_col[tid];

    // if(tid<10) {
    //     printf("tid %d: nnzc_prefix %d\n", tid, nnzC_prefix[tid]);
    // } 

    int C_index = nnzC_prefix[tid];
    // if(tid<10 && C_index == 0) {
    //     printf("tid %d: nnzc_prefix %d\n", tid, C_index);
    // }

    int end_A_col, end_B_col;
    if(tid != blockDim.x*gridDim.x -1 ){
        end_A_col = mergepath_boundary_A_col[tid+1];
        end_B_col = mergepath_boundary_B_col[tid+1];
    } else {
        end_A_col = A_row_ptr[m]-1;
        end_B_col = B_row_ptr[m]-1;
    }

    //if(blockIdx.x==0)printf("Thread %d:  [rA %d, A_index %d, A_col = %d] , [rB %d, B_index %d, B_col = %d] , end_A_col %d end_B_col %d\n", tid, rA, A_index, A_col[A_index], rB, B_index, B_col[B_index], end_A_col, end_B_col);
    A_index++;
    while(A_index == A_row_ptr[rA+1]) {
        rA++;
    }

    B_index++;
    while(B_index == B_row_ptr[rB+1]) {
        rB++;
    }

    while(A_index <= end_A_col && B_index <= end_B_col) {
        if(rA < rB) {
            //  if(C_index==0){
            //      printf("Thread %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
            C_col[C_index]=A_col[A_index];
            C_val[C_index]=A_val[A_index];
            A_index++;
            C_index++;
            
        } else if (rB < rA) {
            //   if(C_index==49518669){
            //      printf("Thread %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
            C_col[C_index]=B_col[B_index];
            C_val[C_index]=B_val[B_index];
            B_index++;
            C_index++;
        } else {
            // rA == rB
            if(A_col[A_index] < B_col[B_index]) {
            //      if(C_index==49518669){
            //      printf("Thread lg %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
                C_col[C_index]=A_col[A_index];
                C_val[C_index]=A_val[A_index];
                A_index++;
                C_index++;
                

            } else if (B_col[B_index] < A_col[A_index]) {
            //     if(C_index==49518669) {
            //      printf("Thread ds %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
                C_col[C_index]=B_col[B_index];
                C_val[C_index]=B_val[B_index];
                B_index++;
                C_index++;
            } else {
                //  if(C_index==49518669){
                //  printf("Thread avs %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
                //  }
                C_col[C_index]=A_col[A_index];
                C_val[C_index]=A_val[A_index]+B_val[B_index];
                A_index++; B_index++; 
                C_index++;
            }
        }

        if(A_index == A_row_ptr[rA+1]) {
            //printf("Thread %d A: atomicAdd row %d count %d\n", tid, rA, curr_rA_count);
            while(A_index == A_row_ptr[rA+1]){
                rA++; 
            }
            
        }
        if(B_index == B_row_ptr[rB+1]) {
            //printf("Thread %d B: atomicAdd row %d count %d\n", tid, rB, curr_rB_count);
            while(B_index == B_row_ptr[rB+1]){
                rB++; 
            }
        }
        
    }

    while(A_index <= end_A_col) {
        //  if(C_index==49518669){
        //          printf("Thread ml %d: rA %d A_index %d A_col %d rB %d B_index %d B_col %d \n", tid, rA, A_index, A_col[A_index], rB, B_index, B_col[B_index]);
        //  }
        C_col[C_index]=A_col[A_index];
        C_val[C_index]=A_val[A_index];
        A_index++;
        C_index++;
        if(A_index == A_row_ptr[rA+1]) {
            
            while(A_index == A_row_ptr[rA+1]){
                rA++; 

            }
        }
    }
    
    while(B_index <= end_B_col) {
        //  if(C_index==49518669){
        //          printf("Thread lk %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
        //  }
        C_col[C_index]=B_col[B_index];
        C_val[C_index]=B_val[B_index];
        B_index++;
        C_index++;
        if(B_index == B_row_ptr[rB+1]) {
            
            while(B_index == B_row_ptr[rB+1]){
                rB++; 
            }
        }
    }

    return ;
}

void gpu_manual_csr_add_f32(int* shape, 
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA, 
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB, 
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* &nnzC) {

        
        int * rowCountC, *nnz_count;
        
        CHECK_CUDA(cudaMalloc((void**)&rowCountC, sizeof(int)*(shape[0]+1)));
        CHECK_CUDA(cudaMalloc((void**)&rowOffsC, sizeof(int)*(shape[0]+1)));
        CHECK_CUDA(cudaMemset(rowCountC, 0, sizeof(int)*(shape[0]+1)));
        CHECK_CUDA(cudaMemset(rowOffsC, 0, sizeof(int)*(shape[0]+1)));
        
        int blockSize = 256;
        int numBlocks = 256; // tune this parameter
        int per_thread_work = (nnzA+nnzB)/(numBlocks*blockSize) + 1; 
        //int numBlocks = (nnzA + nnzB)/ (blockSize * per_thread_work) + 1;
        //printf("nnzA %d nnzB %d\n", nnzA, nnzB);
        
        //printf("Using %d blocks of size %d\n per_thread_work %d\n", numBlocks, blockSize, per_thread_work);
        int * mergepath_boundary_A_row;
        int * mergepath_boundary_B_row;
        int * mergepath_boundary_A_col;
        int * mergepath_boundary_B_col;
        CHECK_CUDA(cudaMalloc((void**)&nnz_count, sizeof(int)*((numBlocks*blockSize))));
        CHECK_CUDA(cudaMalloc((void**)&mergepath_boundary_A_row, sizeof(int)*((numBlocks*blockSize))));
        CHECK_CUDA(cudaMalloc((void**)&mergepath_boundary_A_col, sizeof(int)*((numBlocks*blockSize))));
        CHECK_CUDA(cudaMalloc((void**)&mergepath_boundary_B_row, sizeof(int)*((numBlocks*blockSize))));
        CHECK_CUDA(cudaMalloc((void**)&mergepath_boundary_B_col, sizeof(int)*((numBlocks*blockSize))));



        //printf("Starting mergepath boundary find with %d blocks of size %d\n", numBlocks, blockSize);
        mergepath_boundary_find<<<numBlocks, blockSize>>>(
            shape[0], shape[1],
            rowOffsA, colIndsA,
            rowOffsB, colIndsB,
            mergepath_boundary_A_row, mergepath_boundary_B_row,
            mergepath_boundary_A_col, mergepath_boundary_B_col,
            per_thread_work
        );
        CHECK_CUDA(cudaGetLastError());
        
        

        //printf("Starting csradd kernel with %d blocks of size %d\n", numBlocks, blockSize);
        count_nnz_mergepath_kernel<<<numBlocks, blockSize>>>(
            shape[0], shape[1],
            rowOffsA, colIndsA,
            rowOffsB, colIndsB,
            mergepath_boundary_A_row, mergepath_boundary_B_row,
            mergepath_boundary_A_col, mergepath_boundary_B_col,
            rowCountC, nnz_count
        );      
        CHECK_CUDA( cudaGetLastError() );
        //printf("Finished csradd kernel\n", numBlocks, blockSize);

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // First call: just to get temp storage size
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, rowCountC, rowOffsC+1, shape[0]);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, rowCountC, rowOffsC+1, shape[0]);
        CHECK_CUDA( cudaMemcpy(nnzC, rowOffsC+shape[0], sizeof(int), cudaMemcpyDeviceToHost) );
        
        cudaFree(d_temp_storage);
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;

        int * nnzC_prefix;
        CHECK_CUDA(cudaMalloc((void**)&nnzC_prefix, sizeof(int)*((numBlocks*blockSize))));
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnzC_prefix, numBlocks*blockSize);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnzC_prefix, numBlocks*blockSize);
        cudaFree(d_temp_storage);
        
        
        CHECK_CUDA(cudaMalloc((void**)&colIndsC, sizeof(int)*(*nnzC)));
        CHECK_CUDA(cudaMalloc((void**)&ValsC, sizeof(float)*(*nnzC)));

        
        //print_cuda(rowOffsC, 100);
        //print_cuda(nnzC_prefix, 10);

        //printf("Starting add mergepath kernel with %d blocks of size %d\n", numBlocks, blockSize);
        csr_add_mergepath<<<numBlocks, blockSize>>>(
            shape[0], shape[1],
            rowOffsA, colIndsA, ValsA,
            rowOffsB, colIndsB, ValsB,
            rowOffsC, colIndsC, ValsC,
            mergepath_boundary_A_row, mergepath_boundary_B_row,
            mergepath_boundary_A_col, mergepath_boundary_B_col,
            nnzC_prefix
        );
        CHECK_CUDA( cudaGetLastError() );

        cudaFree(mergepath_boundary_A_row);
        cudaFree(mergepath_boundary_B_row);
        cudaFree(mergepath_boundary_A_col);
        cudaFree(mergepath_boundary_B_col);
        cudaFree(rowCountC);
        cudaFree(nnz_count);
        cudaFree(nnzC_prefix);

        

        return;
}


void gpu_cusparse_csr_add_f32(int* shape, 
        int* rowOffsA, int* colIndsA, float* ValsA, uint64_t nnzA, 
        int* rowOffsB, int* colIndsB, float* ValsB, uint64_t nnzB, 
        int* &rowOffsC, int* &colIndsC, float* &ValsC, int* &nnzC) {
    
    int m = shape[0];
    int n = shape[1];
    
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create matrix descriptors
    cusparseMatDescr_t descrA, descrB, descrC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));

    // Set matrix types (general sparse matrix with 0-based indexing)
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));

    // Get buffer size for the operation
    size_t bufferSize;
      // C = alpha*A + beta*B

    float alpha_h = 1.0f;     // host value
    

    float beta_h = 1.0f;     // host value
    
    CHECK_CUDA(cudaMalloc((void**)&rowOffsC, sizeof(int)*(m+1)));
    
    cusparseScsrgeam2_bufferSizeExt(handle, m, n,
        &alpha_h,
        descrA, nnzA, ValsA, rowOffsA, colIndsA,
        &beta_h,
        descrB, nnzB, ValsB, rowOffsB, colIndsB,
        descrC,
        ValsC, rowOffsC, colIndsC,
        &bufferSize);

    
    // Allocate workspace buffer
    void* buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer, sizeof(char)*bufferSize));
    // Get number of non-zero elements in result matrix

    //printf("nnzA %ld nnZB %ld buffer size = %ld\n", nnzA, nnzB, sizeof(char)*bufferSize);
    
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, m, n,
        descrA, nnzA, rowOffsA, colIndsA,
        descrB, nnzB, rowOffsB, colIndsB,
        descrC, rowOffsC, nnzC,
        buffer));

    
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA(cudaMalloc((void**)&colIndsC, sizeof(int)*(*nnzC)));
    CHECK_CUDA(cudaMalloc((void**)&ValsC, sizeof(float)*(*nnzC)));
    // Perform the actual matrix addition C = alpha*A + beta*B
    CHECK_CUSPARSE(cusparseScsrgeam2(handle, m, n,
        &alpha_h,
        descrA, nnzA, ValsA, rowOffsA, colIndsA,
        &beta_h,
        descrB, nnzB, ValsB, rowOffsB, colIndsB,
        descrC,
        ValsC, rowOffsC, colIndsC,
        buffer));

    // Clean up
    CHECK_CUDA(cudaFree(buffer));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrB));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA( cudaGetLastError() );
}

