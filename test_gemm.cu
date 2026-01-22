/**
 * test_gemm.cu - Test and Benchmark for GEMM Kernel
 * Compares custom kernel against cuBLAS for correctness and performance
 */

#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#include "gemm_kernel.cuh"

// ==================== Utility Functions ====================

// Initialize matrix with random values
void init_matrix(half* h_mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        h_mat[i] = __float2half((float)(rand() % 10) / 10.0f - 0.5f);
    }
}

// Compare two matrices and return max error
float compare_matrices(half* h_C, half* h_C_ref, int rows, int cols) {
    float max_error = 0.0f;
    int error_count = 0;
    float total_error = 0.0f;
    
    for (int i = 0; i < rows * cols; i++) {
        float val = __half2float(h_C[i]);
        float ref = __half2float(h_C_ref[i]);
        float error = fabs(val - ref);
        float rel_error = (fabs(ref) > 1e-6f) ? error / fabs(ref) : error;
        
        total_error += error;
        if (error > max_error) {
            max_error = error;
        }
        
        // Count significant errors (relative error > 1% or absolute error > 0.1)
        if (rel_error > 0.01f && error > 0.1f) {
            error_count++;
            if (error_count <= 10) {
                printf("Error at index %d (row=%d, col=%d): got %f, expected %f, diff=%f\n",
                       i, i / cols, i % cols, val, ref, error);
            }
        }
    }
    
    printf("Total elements: %d\n", rows * cols);
    printf("Elements with significant error: %d (%.4f%%)\n", error_count, 100.0f * error_count / (rows * cols));
    printf("Max absolute error: %f\n", max_error);
    printf("Average absolute error: %f\n", total_error / (rows * cols));
    
    return max_error;
}

// ==================== Main ====================

int main() {
    // Allocate host memory
    half *h_A, *h_B, *h_C, *h_C_ref;
    h_A = (half*)malloc(SIZE_M * SIZE_K * sizeof(half));
    h_B = (half*)malloc(SIZE_K * SIZE_N * sizeof(half));
    h_C = (half*)malloc(SIZE_M * SIZE_N * sizeof(half));
    h_C_ref = (half*)malloc(SIZE_M * SIZE_N * sizeof(half));
    
    // Initialize matrices with random values
    srand(42);
    init_matrix(h_A, SIZE_M, SIZE_K);
    init_matrix(h_B, SIZE_K, SIZE_N);
    
    // Allocate device memory
    half *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, SIZE_M * SIZE_K * sizeof(half));
    cudaMalloc(&d_B, SIZE_K * SIZE_N * sizeof(half));
    cudaMalloc(&d_C, SIZE_M * SIZE_N * sizeof(half));
    cudaMalloc(&d_C_ref, SIZE_M * SIZE_N * sizeof(half));
    
    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, SIZE_M * SIZE_K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE_K * SIZE_N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE_M * SIZE_N * sizeof(half));
    cudaMemset(d_C_ref, 0, SIZE_M * SIZE_N * sizeof(half));
    
    // ==================== cuBLAS Reference ====================
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    
    // Warmup cuBLAS
    cublasHgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                SIZE_N, SIZE_M, SIZE_K,
                &alpha,
                d_B, SIZE_N,
                d_A, SIZE_K,
                &beta,
                d_C_ref, SIZE_N);
    cudaDeviceSynchronize();
    
    // Reset and time cuBLAS
    cudaMemset(d_C_ref, 0, SIZE_M * SIZE_N * sizeof(half));
    
    cudaEvent_t cublas_start, cublas_stop;
    cudaEventCreate(&cublas_start);
    cudaEventCreate(&cublas_stop);
    
    cudaEventRecord(cublas_start);
    cublasHgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                SIZE_N, SIZE_M, SIZE_K,
                &alpha,
                d_B, SIZE_N,
                d_A, SIZE_K,
                &beta,
                d_C_ref, SIZE_N);
    cudaEventRecord(cublas_stop);
    cudaEventSynchronize(cublas_stop);
    
    float cublas_ms = 0;
    cudaEventElapsedTime(&cublas_ms, cublas_start, cublas_stop);
    
    double flops = 2.0 * SIZE_M * SIZE_N * SIZE_K;
    double cublas_tflops = flops / (cublas_ms / 1000.0) / 1e12;
    printf("cuBLAS time: %.3f ms\n", cublas_ms);
    printf("cuBLAS performance: %.2f TFLOPS\n", cublas_tflops);
    
    // ==================== Custom Kernel ====================
    dim3 block(number_of_warps * 32);
    dim3 grid((SIZE_N + shared_mem_n - 1) / shared_mem_n, (SIZE_M + shared_mem_m - 1) / shared_mem_m);
    
    printf("Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);
    
    // Warmup
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // Reset C and run again for timing
    cudaMemset(d_C, 0, SIZE_M * SIZE_N * sizeof(half));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Calculate TFLOPS
    double tflops = flops / (ms / 1000.0) / 1e12;
    printf("Custom kernel time: %.3f ms\n", ms);
    printf("Custom kernel performance: %.2f TFLOPS\n", tflops);
    printf("Efficiency vs cuBLAS: %.1f%%\n", (cublas_ms / ms) * 100.0);
    
    // ==================== Validation ====================
    // Copy results back to host
    cudaMemcpy(h_C, d_C, SIZE_M * SIZE_N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, SIZE_M * SIZE_N * sizeof(half), cudaMemcpyDeviceToHost);
    
    printf("\n==================== Validation ====================\n");
    float max_error = compare_matrices(h_C, h_C_ref, SIZE_M, SIZE_N);
    
    // FP16 tolerance check
    float tolerance = 1.0f;
    if (max_error < tolerance) {
        printf("\n✓ PASSED: Results match within tolerance (%.2f)\n", tolerance);
    } else {
        printf("\n✗ FAILED: Results differ beyond tolerance (%.2f)\n", tolerance);
    }
    
    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(cublas_start);
    cudaEventDestroy(cublas_stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return 0;
}
