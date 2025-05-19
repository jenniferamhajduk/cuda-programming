#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

void sumArraysOnDevice(float *a, float *b, float *c, const int n) {
    for (int idx = 0; idx < n; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char **argv) {
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_B, nBytes);
    cudaMalloc((void **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

   
    sumArraysOnDevice(h_A, h_B, h_C, nElem);


    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (nElem + blockSize - 1) / blockSize;

    // Launch kernel to perform summation on the device
    sumArraysOnDevice<<<gridSize, blockSize>>>(d_A, d_B, d_C, nElem);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    
    return(0);
    }


