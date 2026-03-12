#include "cuda_utils.h"
#include <iostream>
#include <cuda_runtime.h>

// Wrapper function for cudaFree
void cudaFreeWrapper(void* ptr) noexcept {
    // Can't do this due to `noexcept`
    // cudaError_t error = cudaFree(ptr);
    // if (error != cudaSuccess) {
    //     throw std::runtime_error(cudaGetErrorString(error));
    // }
    cudaFree(ptr);
}

void print_cuda(int * & ptr, int size) {
    int* h_ptr = new int[size];
    CHECK_CUDA(cudaMemcpy(h_ptr, ptr, size*sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nDevice array: ");
    for(int i=0; i<size; i++) {
        printf(" %d", h_ptr[i]);
    }
    
    delete[] h_ptr;
}

void print_cuda(float * & ptr, int size) {
    float* h_ptr = new float[size];
    CHECK_CUDA(cudaMemcpy(h_ptr, ptr, size*sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nDevice array: ");
    for(int i=0; i<size; i++) {
        printf(" %f", h_ptr[i]);
    }
    
    delete[] h_ptr;
}


bool compare_cuda(int* & ptr, int* & ptr2, int size) {
    int* h_ptr = new int[3*size];
    CHECK_CUDA(cudaMemcpy(h_ptr, ptr, 3*size*sizeof(int), cudaMemcpyDeviceToHost));
    int* h_ptr2 = new int[3*size];
    CHECK_CUDA(cudaMemcpy(h_ptr2, ptr2, 3*size*sizeof(int), cudaMemcpyDeviceToHost));
    for(int i=0; i<size; i++) {
        printf("[%d %d %d], [%d %d %d]\n", h_ptr[i],h_ptr[size+i],h_ptr[2*size+i], h_ptr2[i],h_ptr2[size+i],h_ptr2[2*size+i]);
        if(h_ptr[i]!=h_ptr2[i] || h_ptr[size+i]!= h_ptr2[size+i] || h_ptr[2*size+i] !=h_ptr2[2*size+i]) {
            printf("Mismatch at index %d\n",i);
            return false;
        }
    }

    return true;
}