#include "coo_add.h"
#include "../cuda_utils/cuda_utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>


__global__ void mergepath_boundary_find(
    const uint nnzA, const uint nnzB,
    const int *A_row, const int *A_col,
    const int *B_row, const int *B_col, 
    int * mergepath_boundary_A, int* mergepath_boundary_B,
    const int per_thread_work) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //if(tid<100)printf("Thread %d In mergepath_boundary_find kernel\n",tid);
    int diag_boundary = (  (tid) * per_thread_work );

    if(diag_boundary <= 0) {
        mergepath_boundary_A[tid] = -1;mergepath_boundary_B[tid] = -1;
        return;
    }
    if(diag_boundary >= nnzA + nnzB) {
        mergepath_boundary_A[tid] = nnzA-1; 
        mergepath_boundary_B[tid] = nnzB-1; 
        return;
    }

    //if(tid<100)printf("Thread %d diag_boundary %d\n", tid, diag_boundary);
   
    int lo = -1;
    int hi = min(diag_boundary-1, nnzA-1);
    int mid, a_col, b_col, rA, rB;
    while(lo < hi ) {
        mid = (lo + hi+1)/2;
        if(diag_boundary - mid -1 > nnzB-1) {
            lo = mid;
            continue;
        }
        rA = A_row[mid];
        rB = B_row[diag_boundary - mid - 1];
        
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
    //if(tid<10)printf("Thread %d finished binary search lo %d hi %d diag_boundary %d\n", tid, lo, hi, diag_boundary);

    mid = lo;

    int a_index = mid;
    //if(tid<10) printf("Thread %d: b_index %d %d \n", tid, diag_boundary - mid-2, nnzB-1);

    int b_index = min(diag_boundary-mid-2, (int)(nnzB-1)); // min behaving wierd when one value is -ve. porbably because it is expecting uints?
    // int b_index = diag_boundary - mid-2;
    // if(b_index> (int)(nnzB-1)) {
    //     b_index = nnzB -1;
    // }

   

    //if(tid<5)printf("Thread %d: initial boundary A [%d] B [%d]\n", tid, a_index, b_index);
    mergepath_boundary_A[tid] = a_index; mergepath_boundary_B[tid] = b_index;
    
    // if(tid == 504) {
    //     printf("rB %d, row_search(A_row_ptr, m, mid+1) %d, mid %d, diag_boundary %d, A_col[mid+1] %d, B_col[diag_boundary - mid] %d\n", rB, row_search(A_row_ptr, m, mid+1), mid,diag_boundary, A_col[mid+1], B_col[diag_boundary - mid]);
    // }
    if(a_index+1 < nnzA && b_index>=0 && A_row[a_index+1] == B_row[b_index] && A_col[a_index+1] == B_col[b_index]) {
        // printf("Adjusting A boundary from %d to %d\n", mid, mid+1);
        // //printf("A_col[%d] = %d B_col[%d] = %d\n", mid+1, A_col[mid+1], diag_boundary - mid, B_col[diag_boundary - mid]);
        mergepath_boundary_A[tid] = a_index + 1;
    } else if(b_index + 1 < nnzB && a_index>=0 && A_row[a_index] == B_row[b_index+1]  && B_col[b_index+1] == A_col[a_index] ) {
        // printf("Adjusting B boundary from %d to %d\n", diag_boundary - mid, diag_boundary - mid + 1);
        mergepath_boundary_B[tid] = b_index + 1;
    }

    //printf("Thread %d: final boundary A [%d, %d] B [%d, %d]\n", tid, mergepath_boundary_A_row[tid], mergepath_boundary_A_col[tid], mergepath_boundary_B_row[tid], mergepath_boundary_B_col[tid]);
    return;
    }


__global__ void count_nnz_mergepath_kernel(
    const uint nnzA, const uint nnzB,
    const int *A_row, const int *A_col,     // CSR A
    const int *B_row, const int *B_col,   // CSR B                   
    int * mergepath_boundary_A, int* mergepath_boundary_B,              
    int * nnz_count)                            // output per-row counts (zeroed before launch)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
   
    
    int A_index = mergepath_boundary_A[tid];
    int B_index = mergepath_boundary_B[tid];

    
    int end_A_index, end_B_index;
    if(tid != blockDim.x*gridDim.x -1 ){ // check as tid+1 is illegal access for last thread
        end_A_index = mergepath_boundary_A[tid+1];
        end_B_index = mergepath_boundary_B[tid+1];
    } else {
        end_A_index = nnzA-1;
        end_B_index = nnzB-1;
    }


    //if(tid<10)printf("Thread %d: start_A [%d] start_B [%d], endA_col %d, endB_col %d\n", tid, A_index, B_index, end_A_index, end_B_index);
    //if(blockIdx.x ==0 && threadIdx.x <10)printf("Thread %d: start_A   [%d, %d] start_B   [%d, %d]\n", tid, rA, rB, A_index, B_index);

    int total_nnz_processed = 0;
    
    A_index++;
    B_index++;


    while(A_index <= end_A_index && B_index <= end_B_index) {
        if(A_row[A_index] < B_row[B_index]) {
            //  if(rA==999){
            //      printf("Thread %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
            A_index++;
            total_nnz_processed++;
        } else if (B_row[B_index] < A_row[A_index]) {
            //   if(rB==999){
            //      printf("Thread %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
            B_index++;
            total_nnz_processed++;
        } else {
            // rA == rB
            if(A_col[A_index] < B_col[B_index]) {
            //      if(rA==999){
            //      printf("Thread lg %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
                A_index++;
                total_nnz_processed++;

            } else if (B_col[B_index] < A_col[A_index]) {
            //      if(rB==999){
            //      printf("Thread ds %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
                B_index++;
                total_nnz_processed++;
            } else {
                //  if(rB==999){
                //  printf("Thread avs %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
                //  }
                A_index++; B_index++; // increment both but Add to only one of the two counter
                total_nnz_processed++;
            }
        }
    }
   // __threadfence(); 
    //if(tid==98438){int v = atomicAdd(&row_counts[196608], 0); printf("calue of row_count %d %p\n",v, (void*)&row_counts[196608]);}
    // if(tid==0)
    // printf("Thread %d: finished main loop A_index %d end_A %d B_index %d end_B %d\n", tid, A_index, end_A[1], B_index, end_B[1]);
    // // finish off remaining entries in A or B

    while(A_index <= end_A_index) {
        //  if(rA==999){
        //          printf("Thread ml %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
        //  }
        A_index++;
        total_nnz_processed++;
    }

    while(B_index <= end_B_index) {
        //  if(rB==999){
        //          printf("Thread lk %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
        //  }
        B_index++;
        total_nnz_processed++;
    }
    //if(tid<10)printf("Tid %d : nnz_count %d\n",tid, total_nnz_processed);
    nnz_count[tid] = total_nnz_processed;
    return;
}



__global__ void coo_add_mergepath(
    const uint nnzA, const uint nnzB,
    const int *A_row, const int *A_col, const float *A_val,       // CSR A
    const int *B_row, const int *B_col, const float *B_val,      // CSR B   
    int *C_row, int *C_col, float *C_val,                
    int * mergepath_boundary_A, int* mergepath_boundary_B,              
    int *nnzC_prefix)                            // output per-row counts (zeroed before launch)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //printf("Thread %d: start %lld end %lld total_work %d\n", tid, start, end, total_work);
    
    int A_index = mergepath_boundary_A[tid];
    int B_index = mergepath_boundary_B[tid];

    int C_index = nnzC_prefix[tid];
    //if(tid<10)printf("Tid %d: C_index %d\n",tid,  nnzC_prefix[tid]);

    int end_A_index, end_B_index;
    if(tid != blockDim.x*gridDim.x -1 ){ // check as tid+1 is illegal access for last thread
        end_A_index = mergepath_boundary_A[tid+1];
        end_B_index = mergepath_boundary_B[tid+1];
    } else {
        end_A_index = nnzA-1;
        end_B_index = nnzB-1;
    }

    //if(rA<=196610 && rA>= 196607)printf("Thread %d: start_A [%d, %d] start_B [%d, %d], endA_col %d, endB_col %d\n", tid, rA, A_index, rB, B_index, end_A_col, end_B_col);
    //if(blockIdx.x ==0 && threadIdx.x <10)printf("Thread %d: start_A   [%d, %d] start_B   [%d, %d]\n", tid, rA, rB, A_index, B_index);
    
    A_index++;
    B_index++;


    while(A_index <= end_A_index && B_index <= end_B_index) {
        if(A_row[A_index] < B_row[B_index]) {
            //  if(rA==999){
            //      printf("Thread %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
            C_row[C_index]=A_row[A_index];
            C_col[C_index]=A_col[A_index];
            C_val[C_index]=A_val[A_index];

            A_index++;
            C_index++;
        } else if (B_row[B_index] < A_row[A_index]) {
            //   if(rB==999){
            //      printf("Thread %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
            C_row[C_index]=B_row[B_index];
            C_col[C_index]=B_col[B_index];
            C_val[C_index]=B_val[B_index];

            B_index++;
            C_index++;
        } else {
            // rA == rB
            if(A_col[A_index] < B_col[B_index]) {
            //      if(rA==999){
            //      printf("Thread lg %d: A_index %d A_col %d \n", tid, A_index, A_col[A_index]);
            //  }
                C_row[C_index]=A_row[A_index];
                C_col[C_index]=A_col[A_index];
                C_val[C_index]=A_val[A_index];

                A_index++;
                C_index++;

            } else if (B_col[B_index] < A_col[A_index]) {
            //      if(rB==999){
            //      printf("Thread ds %d: B_index %d B_col %d \n", tid, B_index, B_col[B_index]);
            //  }
                C_row[C_index]=B_row[B_index];
                C_col[C_index]=B_col[B_index];
                C_val[C_index]=B_val[B_index];

                B_index++;
                C_index++;
            } else {
                //  if(rB==999){
                //  printf("Thread avs %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
                //  }
                C_row[C_index]=A_row[A_index];
                C_col[C_index]=A_col[A_index];
                C_val[C_index]=A_val[A_index]+B_val[B_index];
                A_index++; B_index++; // increment both but Add to only one of the two counter
                C_index++;
            }
        }
    }
   // __threadfence(); 
    //if(tid==98438){int v = atomicAdd(&row_counts[196608], 0); printf("calue of row_count %d %p\n",v, (void*)&row_counts[196608]);}
    // if(tid==0)
    // printf("Thread %d: finished main loop A_index %d end_A %d B_index %d end_B %d\n", tid, A_index, end_A[1], B_index, end_B[1]);
    // // finish off remaining entries in A or B

    while(A_index <= end_A_index) {
        //  if(rA==999){
        //          printf("Thread ml %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
        //  }
        C_row[C_index]=A_row[A_index];
        C_col[C_index]=A_col[A_index];
        C_val[C_index]=A_val[A_index];

        A_index++;
        C_index++;
    }

    while(B_index <= end_B_index) {
        //  if(rB==999){
        //          printf("Thread lk %d: A_index %d A_col %d B_index %d B_col %d \n", tid, A_index, A_col[A_index], B_index, B_col[B_index]);
        //  }
        C_row[C_index]=B_row[B_index];
        C_col[C_index]=B_col[B_index];
        C_val[C_index]=B_val[B_index];

        B_index++;
        C_index++;
    }

    return;
}


void gpu_manual_coo_add_f32(int* shape, 
        int* rowsA, int* colsA, float* ValsA, uint64_t nnzA, 
        int* rowsB, int* colsB, float* ValsB, uint64_t nnzB, 
        int* &rowsC, int* &colsC, float* &ValsC, int* &nnzC) {


        int blockSize = 256;
        int numBlocks = 256; // tune this parameter
        int per_thread_work = (nnzA+nnzB)/(numBlocks*blockSize) + 1; 

        int * nnz_count;
        int * mergepath_boundary_A;
        int * mergepath_boundary_B;
        CHECK_CUDA(cudaMalloc((void**)&nnz_count, sizeof(int)*((numBlocks*blockSize))));
        CHECK_CUDA(cudaMalloc((void**)&mergepath_boundary_A, sizeof(int)*((numBlocks*blockSize))));
        CHECK_CUDA(cudaMalloc((void**)&mergepath_boundary_B, sizeof(int)*((numBlocks*blockSize))));

        //printf("Starting mergepath boundary find with %d blocks of size %d\n", numBlocks, blockSize);
        mergepath_boundary_find<<<numBlocks, blockSize>>>(
            nnzA, nnzB,
            rowsA, colsA,
            rowsB, colsB,
            mergepath_boundary_A, mergepath_boundary_B,
            per_thread_work
        );
        CHECK_CUDA(cudaGetLastError());
        //CHECK_CUDA(cudaDeviceSynchronize());
        //printf("Starting cooadd kernel with %d blocks of size %d\n", numBlocks, blockSize);
        count_nnz_mergepath_kernel<<<numBlocks, blockSize>>>(
            nnzA, nnzB,
            rowsA, colsA,
            rowsB, colsB,
            mergepath_boundary_A, mergepath_boundary_B,
            nnz_count
        );      
        CHECK_CUDA( cudaGetLastError() );
        //CHECK_CUDA(cudaDeviceSynchronize());
        //print_cuda(nnz_count, 100);
        //printf("Finished count nnz with %d blocks of size %d\n", numBlocks, blockSize);
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        
        int * nnzC_prefix;
        CHECK_CUDA(cudaMalloc((void**)&nnzC_prefix, sizeof(int)*((numBlocks*blockSize) + 1)));
        CHECK_CUDA(cudaMemset(nnzC_prefix, 0, sizeof(int)*((numBlocks*blockSize) + 1)));
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnzC_prefix+1, numBlocks*blockSize));

        CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnzC_prefix+1, numBlocks*blockSize));
        CHECK_CUDA( cudaGetLastError() );
       
        CHECK_CUDA(cudaFree(d_temp_storage));

        //printf("Finished inclusive sum with %d blocks of size %d\n", numBlocks, blockSize);
        //*nnzC = 10;
        CHECK_CUDA( cudaMemcpy(nnzC, nnzC_prefix+(numBlocks*blockSize), sizeof(int), cudaMemcpyDeviceToHost) );

        //printf("Total num zeros in C %d\n", *nnzC);
        CHECK_CUDA(cudaMalloc((void**)&rowsC, sizeof(int)*(*nnzC)));
        CHECK_CUDA(cudaMalloc((void**)&colsC, sizeof(int)*(*nnzC)));
        CHECK_CUDA(cudaMalloc((void**)&ValsC, sizeof(float)*(*nnzC)));

        //printf("Starting add mergepath kernel with %d blocks of size %d\n", numBlocks, blockSize);
        coo_add_mergepath<<<numBlocks, blockSize>>>(
            nnzA, nnzB,
            rowsA, colsA, ValsA,
            rowsB, colsB, ValsB,
            rowsC, colsC, ValsC,
            mergepath_boundary_A, mergepath_boundary_B,
            nnzC_prefix
        );
        CHECK_CUDA( cudaGetLastError() );

        cudaFree(mergepath_boundary_A);
        cudaFree(mergepath_boundary_B);
        cudaFree(nnz_count);
        cudaFree(nnzC_prefix);

        return;
}


