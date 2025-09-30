#include <iostream>
#include <vector>
#include <cmath> 
#include <cuda_runtime.h>

// Check CUDA API calls for errors
#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}

// 1. Unfused Kernels 
// Kernel 1: Vector addition B = A + V
__global__ void AddKernel(const float *A, const float *V, float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        B[idx] = A[idx] + V[idx];
    }
}

// Kernel 2: Vector scalar multiplication C = B * scale
__global__ void ScaleKernel(const float *B, float *C, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = B[idx] * scale;
    }
}

// 2. Fused Kernel 
// Kernel 3: Fused operation C = (A + V) * scale
__global__ void FusedKernel(const float *A, const float *V, float *C, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) { 
        float temp_B = A[idx] + V[idx]; 
        C[idx] = temp_B * scale;
    } 
}

void setupAndRun(int N, float scale) {

    // 1. Host Data Initialization
    std::vector<float> h_A(N), h_V(N), h_C_unfused(N), h_C_fused(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;        
        h_V[i] = (float)i * 0.1f; 
    }

    // 2. Device Memory Allocation and Data Transfer
    float *d_A, *d_V, *d_B, *d_C_unfused, *d_C_fused;
    size_t bytes = N * sizeof(float);

    // Allocations 
    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_V, bytes));
    CHECK(cudaMalloc(&d_B, bytes)); 
    CHECK(cudaMalloc(&d_C_unfused, bytes));
    CHECK(cudaMalloc(&d_C_fused, bytes));

    CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    const int BLOCK_SIZE = 256;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;


    // Run Unfused Kernels (AddKernel + ScaleKernel)
    std::cout << "Running Unfused Kernels...\n";
    CHECK(cudaEventRecord(start));
    
    // Kernel 1: B = A + V
    AddKernel<<<numBlocks, BLOCK_SIZE>>>(d_A, d_V, d_B, N);
    CHECK(cudaDeviceSynchronize()); 
    
    // Kernel 2: C = B * scale
    ScaleKernel<<<numBlocks, BLOCK_SIZE>>>(d_B, d_C_unfused, scale, N);
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "  [Unfused] Time: " << milliseconds << " ms\n";
    
    CHECK(cudaMemcpy(h_C_unfused.data(), d_C_unfused, bytes, cudaMemcpyDeviceToHost));

    // Run Fused Kernel (FusedKernel)
    std::cout << "Running Fused Kernel...\n";
    CHECK(cudaEventRecord(start));
    
    // Single Kernel: C = (A + V) * scale
    FusedKernel<<<numBlocks, BLOCK_SIZE>>>(d_A, d_V, d_C_fused, scale, N);
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "  [Fused] Time: " << milliseconds << " ms\n";

    CHECK(cudaMemcpy(h_C_fused.data(), d_C_fused, bytes, cudaMemcpyDeviceToHost));

    // Validation
    bool success = true;
    for (int i = 0; i < 5; ++i) {
        float expected = (h_A[i] + h_V[i]) * scale;
        if (std::abs(h_C_unfused[i] - expected) > 1e-5 || std::abs(h_C_fused[i] - expected) > 1e-5) {
            success = false;
            break;
        }
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_B); 
    cudaFree(d_C_unfused);
    cudaFree(d_C_fused);
}

int main() {
    const int N = 5 * 1024 * 1024; 
    setupAndRun(N, 2.0f);
    return 0;
}