#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
const int SIZE_M = 8192;
const int SIZE_N = 8192;
const int SIZE_K = 8192;
const int register_file_m = 64;
const int register_file_n = 64;
const int register_file_k = 32;
const int shared_mem_m = 256;
const int shared_mem_n = 128;
const int shared_mem_k = 64;
const int number_of_warps = 8;
const int mma_m = 16;
const int mma_n = 8;
const int mma_k = 16;
const int iter_within_global = SIZE_K / shared_mem_k;
const int iter_within_shared = shared_mem_k / register_file_k;

const int mma_tiles_per_warp_m = register_file_m / mma_m;
const int mma_tiles_per_warp_k = register_file_k / mma_k;
const int mma_tiles_per_warp_n = register_file_n / mma_n;

__device__ inline int swizzle_A(int logical_flattened){
            int old_row = logical_flattened / 32;
            int small_shift = logical_flattened % 4;
            int row = (logical_flattened / 32) % 8;
            int col = (logical_flattened % 32) / 4;
            int new_col = row ^ col;
            return old_row * 32 + new_col * 4 + small_shift;
}           
__device__ inline int swizzle_B(int logical_flattened){
            int old_row = logical_flattened / 64;
            int small_shift = logical_flattened % 4;
            int shift_col = (logical_flattened % 64) / 32;
            int row = (logical_flattened / 64) % 8;
            int col = (logical_flattened % 32) / 4;
            int new_col = row ^ col;
            return shared_mem_m * shared_mem_k / 2 + old_row * 64 + new_col * 4 + shift_col * 32 + small_shift;
}

__global__ void gemm_kernel(half* A, half* B, half* C) {

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_x = warp_id % 2;
    int warp_y = warp_id / 2;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    half* start_block_x = B + block_x * shared_mem_n;
    half* start_block_y = A + block_y * shared_mem_m * SIZE_K;

    int start_warp_x = warp_x * 64 / 2; // shared mem perspective
    int start_warp_y = warp_y * 64 * 64 / 2; // shared mem perspective
    float* start_block_x_f32 = reinterpret_cast<float*>(start_block_x);
    float* start_block_y_f32 = reinterpret_cast<float*>(start_block_y);

    __shared__ half shmem[shared_mem_m * shared_mem_k + shared_mem_k * shared_mem_n];
    float* shmem_f32 = reinterpret_cast<float*>(shmem);

    uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4];
    uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2];
    uint32_t CD_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];

    // Initialize CD_register to zero
    #pragma unroll
    for(int i = 0; i < mma_tiles_per_warp_m; i++){

        #pragma unroll
        for(int j = 0; j < mma_tiles_per_warp_n; j++){
            CD_register[i][j][0] = 0;
            CD_register[i][j][1] = 0;
        }
    }

    #pragma unroll
    for(int iter_global = 0; iter_global < iter_within_global; iter_global++){

        // load A swizzed from global memory to shared memory
        #pragma unroll
        for(int i = threadIdx.x; i < shared_mem_m * shared_mem_k / 2; i += blockDim.x){
            int old_row = i / 32;
            int old_col = i % 32;
            uint32_t smem_ptr = __cvta_generic_to_shared(shmem_f32 + swizzle_A(i));
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                         :
                         : "r"(smem_ptr), "l"(start_block_y_f32 + old_row * SIZE_K / 2 + old_col + iter_global * shared_mem_k / 2), "n"(4));
            }
        // load B swizzeld from global memory to shared memory
        #pragma unroll
        for(int i = threadIdx.x; i < shared_mem_k * shared_mem_n / 2; i += blockDim.x){
            int old_row = i / 64;
            int old_col = i % 64;
            uint32_t smem_ptr = __cvta_generic_to_shared(shmem_f32 + swizzle_B(i));
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                        :
                        : "r"(smem_ptr), "l"(start_block_x_f32 + old_row * SIZE_N / 2 + old_col + iter_global * shared_mem_k * SIZE_N / 2), "n"(4));
        } 

        asm volatile("cp.async.wait_all;");
        __syncthreads();
        #pragma unroll
        for(int iter_shared = 0; iter_shared < iter_within_shared; iter_shared++){

            // Load A from shared memory to register
            #pragma unroll
            for(int i = 0; i < mma_tiles_per_warp_m; i++){

                #pragma unroll
                for(int j = 0; j < mma_tiles_per_warp_k; j++){
                    int row = lane_id % 16;
                    int col = lane_id / 16;
                    int flattened_row = i * 16 * 32 + row * 32 + j * 8 + start_warp_y + iter_shared * 16 + col * 4;
                    uint32_t smem_ptr = __cvta_generic_to_shared(shmem_f32 + swizzle_A(flattened_row));
                    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                                "{%0, %1, %2, %3}, [%4];"
                                : "=r"(A_register[i][j][0]), "=r"(A_register[i][j][1]), "=r"(A_register[i][j][2]), "=r"(A_register[i][j][3])
                                : "r"(smem_ptr));
                }
            }

            // Load B from shared memory to register
            #pragma unroll
            for(int i = 0; i < mma_tiles_per_warp_n; i++){

                #pragma unroll
                for(int j = 0; j < mma_tiles_per_warp_k; j++){
                    int row = lane_id % 16;
                    int flattened_row = j * 16 * 64 + row * 64 + i * 4 + start_warp_x + iter_shared * 32 * 64;
                    int new_col = swizzle_B(flattened_row);
                    uint32_t smem_ptr = __cvta_generic_to_shared(shmem_f32 + new_col);
                    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16"
                                "{%0, %1}, [%2];"
                                : "=r"(B_register[j][i][0]), "=r"(B_register[j][i][1])
                                : "r"(smem_ptr));
                }
            }

            // MMA compute
            #pragma unroll
            for(int i = 0; i < mma_tiles_per_warp_m; i++){

                #pragma unroll
                for(int j = 0; j < mma_tiles_per_warp_n; j++){

                    #pragma unroll
                    for(int k = 0; k < mma_tiles_per_warp_k; k++){
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                                    " { %0, %1}, "
                                    " { %2, %3, %4, %5 }, "
                                    " { %6, %7 }, "
                                    " { %8, %9}; "
                                    : "=r"(CD_register[i][j][0]), "=r"(CD_register[i][j][1])
                                    : "r"(A_register[i][k][0]), "r"(A_register[i][k][1]), "r"(A_register[i][k][2]), "r"(A_register[i][k][3]), "r"(B_register[k][j][0]), "r"(B_register[k][j][1]),
                                        "r"(CD_register[i][j][0]), "r"(CD_register[i][j][1]));

                    }
            }
            }

        }
        __syncthreads();
    }
    // writing to global memory the result
    int offset_block = block_y * shared_mem_m * SIZE_N + shared_mem_n * block_x;
    int offset_warp = warp_y * register_file_m * SIZE_N + warp_x * 64;
    int final_offset = offset_block + offset_warp;
    uint32_t* output = reinterpret_cast<uint32_t*>(C + final_offset);
    #pragma unroll
    for(int i = 0; i < mma_tiles_per_warp_m; i++){

        #pragma unroll
        for(int j = 0; j < mma_tiles_per_warp_n; j++){
        output[i * mma_m * SIZE_N / 2 + (lane_id / 4) * SIZE_N / 2 + j * mma_n / 2 + (lane_id % 4)] = CD_register[i][j][0];
        output[i * mma_m * SIZE_N / 2 + 8 * SIZE_N / 2 + (lane_id / 4) * SIZE_N / 2  + j * mma_n / 2 + (lane_id % 4)] = CD_register[i][j][1];
        }
    }
    
}

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

int main(){
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