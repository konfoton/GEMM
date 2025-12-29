#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>
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
const int iter_within_shared = shared_mem_k / mma_k;

const int mma_tiles_per_warp_m = register_file_m / mma_m;
const int mma_tiles_per_warp_k = register_file_k / mma_k;
const int mma_tiles_per_warp_n = register_file_n / mma_n;

__device__ inline int swizzle_A(int logical_flattened){
            int old_row = logical_flattened / 32;
            int old_col = logical_flattened % 32;
            int small_shift = logical_flattened % 4;
            int row = (logical_flattened / 32) % 8;
            int col = (logical_flattened % 32) / 4;
            int new_col = row ^ col;
            return old_row * 32 + new_col + small_shift;
}           
__device__ inline int swizzle_B(int logical_flattened){
            int old_row = logical_flattened / 64;
            int old_col = logical_flattened % 32;
            int small_shift = logical_flattened % 4;
            int shift_col = (logical_flattened % 64) / 32;
            int row = (logical_flattened / 64) % 8;
            int col = (logical_flattened % 32) / 4;
            int new_col = row ^ col;
            return shared_mem_m * shared_mem_k / 2 + old_row * 64 + new_col + shift_col * 32 + small_shift;
}

__global__ void gemm_kernel(half* A, half* B, half* C) {

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_x = warp_id % 2;
    int warp_y = warp_id / 2;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    half* start_block_x = B + block_x * shared_mem_n;
    half* start_block_y = A + block_y * shared_mem_m * SIZE_M;

    int start_warp_x = warp_x * 64 / 2;
    int start_warp_y = warp_y * 64 * 64 / 2;
    float* start_block_x_f32 = reinterpret_cast<float*>(start_block_x);
    float* start_block_y_f32 = reinterpret_cast<float*>(start_block_y);

    __shared__ half shmem[shared_mem_m * shared_mem_k + shared_mem_k * shared_mem_n];
    float* shmem_f32 = reinterpret_cast<float*>(shmem);

    uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4];
    uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2];
    uint32_t CD_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];


    float* A_float = reinterpret_cast<float*>(start_block_x);
    float* B_float = reinterpret_cast<float*>(start_block_y);
    for(int iter_global = 0; iter_global < iter_within_global; iter_global++){

        // load A swizzed from global memory to shared memory
        for(int i = 0; i < shared_mem_m * shared_mem_k / 2; i += blockDim.x){
            int old_row = i / 32;
            int old_col = i % 32;
            shmem_f32[swizzle_A(i)] = start_block_x_f32[old_row * SIZE_M / 2 + old_col + iter_global * SIZE_K];
        } 
        // load B swizzeld from global memory to shared memory
        for(int i = 0; i < shared_mem_k * shared_mem_n / 2; i += blockDim.x){
            int old_row = i / 64;
            int old_col = i % 32;
            shmem_f32[swizzle_B(i)] = start_block_y_f32[old_row * SIZE_N / 2 + old_col + iter_global * SIZE_K * SIZE_N / 2];
        } 
        __syncthreads();

        for(int iter_shared = 0; iter_shared < iter_within_shared; iter_shared++){

            // Load A from shared memory to register
            for(int i = 0; i < mma_tiles_per_warp_m; i++){
                for(int j = 0; j < mma_tiles_per_warp_k; j++){
                    int row = threadIdx.x % 16;
                    int col = threadIdx.x / 16;
                    int flattened_row = i * 16 * 32 + row * 32 + j * 8 + start_warp_y + iter_shared * 16 + col * 4;
                    uint32_t smem_ptr = __cvta_generic_to_shared(shmem_f32 + swizzle_A(flattened_row));
                    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                                "{%0, %1, %2, %3}, [%4];"
                                : "=r"(A_register[i][j][0]), "=r"(A_register[i][j][1]), "=r"(A_register[i][j][2]), "=r"(A_register[i][j][3])
                                : "r"(smem_ptr));
                }
            }

            // Load B from shared memory to register
            for(int i = 0; i < mma_tiles_per_warp_n; i++){
                for(int j = 0; j < mma_tiles_per_warp_k; j++){
                    int row = threadIdx.x;
                    int flattened_row = i * 16 * 64 + row * 64 + j * 4 + start_warp_x + iter_shared * 32 * 64;
                    uint32_t smem_ptr = __cvta_generic_to_shared(shmem_f32 + swizzle_B(flattened_row));
                    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16"
                                "{%0, %1}, [%2];"
                                : "=r"(B_register[i][j][0]), "=r"(B_register[i][j][1])
                                : "r"(smem_ptr));
                }
            }

            // MMA compute
            for(int i = 0; i < mma_tiles_per_warp_m; i++){
                for(int j = 0; j < mma_tiles_per_warp_n; j++){
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
    for(int i = 0; i < mma_tiles_per_warp_m; i++){
         for(int j = 0; j < mma_tiles_per_warp_n; j++){
            output[i * mma_m * SIZE_N / 2 + (lane_id / 4) * SIZE_N / 2 + j * mma_k / 2 + (lane_id % 4)] = CD_register[i][j][0];
            output[i * mma_m * SIZE_N / 2 + (lane_id / 4) * SIZE_N / 2 + 8 + j * mma_k / 2 + (lane_id % 4)] = CD_register[i][j][1];
         }
    }
    
}

int main(){
    half* A, * B, * C;
    cudaMalloc(&A, SIZE_M * SIZE_K * sizeof(half));
    cudaMalloc(&B, SIZE_K * SIZE_N * sizeof(half));
    cudaMalloc(&C, SIZE_M * SIZE_N * sizeof(half));

    dim3 block(number_of_warps * 32);
    dim3 grid((SIZE_N + shared_mem_n - 1) / shared_mem_n, (SIZE_M + shared_mem_m - 1) / shared_mem_m);

    gemm_kernel<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();
    return 0;
}