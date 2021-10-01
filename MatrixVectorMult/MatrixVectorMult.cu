// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void MatrixMulCUDA(double* C, double* A, double* B) {
    int index = threadIdx.x;

    __shared__ double shared_b[1024];
    shared_b[index] = B[index];
    __syncthreads();
    for (int i = 0; i < 1024; i++)
        C[index] += (A[1024 * index + i] * shared_b[i]);
    __syncthreads();
}

void ConstantInit(double* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(const dim3& dimsA, const dim3& dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(double) * size_A;
    double* h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(double) * size_B;
    double* h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    cudaStream_t stream;

    ConstantInit(h_A, size_A);
    ConstantInit(h_B, size_B);

    // Allocate device memory
    double* d_A, * d_B, * d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, 1, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(double);
    double* h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(1024);
    dim3 grid(1);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, stream));

    MatrixMulCUDA <<< grid, threads, 0, stream >>> (d_C, d_A, d_B);

    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("Total time in msec: %f\n", msecTotal);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    return 0;
}


int main(int argc, char** argv) {

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char**)argv);

    int oneK = 1024;

    dim3 dimsA(oneK, oneK, 1);
    dim3 dimsB(oneK, 1, 1);

    int matrix_result = MatrixMultiply(dimsA, dimsB);

    exit(matrix_result);
}

