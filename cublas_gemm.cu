// cuBLAS GEMM (FP16) with timing and FLOPS
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#ifndef SIZE_M
#define SIZE_M 8192
#endif
#ifndef SIZE_N
#define SIZE_N 8192
#endif
#ifndef SIZE_K
#define SIZE_K 8192
#endif

#define CUDA_CHECK(expr)                                                   \
	do {                                                                  \
		cudaError_t _err = (expr);                                        \
		if (_err != cudaSuccess) {                                        \
			fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
					cudaGetErrorString(_err), __FILE__, __LINE__);        \
			exit(EXIT_FAILURE);                                           \
		}                                                                 \
	} while (0)

#define CUBLAS_CHECK(expr)                                                 \
	do {                                                                  \
		cublasStatus_t _stat = (expr);                                    \
		if (_stat != CUBLAS_STATUS_SUCCESS) {                             \
			fprintf(stderr, "cuBLAS error %d at %s:%d\n",               \
					static_cast<int>(_stat), __FILE__, __LINE__);         \
			exit(EXIT_FAILURE);                                           \
		}                                                                 \
	} while (0)

static void init_half_matrix(std::vector<__half>& h, int rows, int cols) {
	// Deterministic pattern without extra libs; fill in column-major layout
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			float val = static_cast<float>((r % 17) - (c % 13)) * 0.01f;
			h[c * rows + r] = __float2half(val); // column-major index
		}
	}
}

int main() {
	const int M = SIZE_M;
	const int N = SIZE_N;
	const int K = SIZE_K;
	const size_t sizeA = static_cast<size_t>(M) * K;
	const size_t sizeB = static_cast<size_t>(K) * N;
	const size_t sizeC = static_cast<size_t>(M) * N;

	// Show device info
	int device = 0;
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	printf("Device: %s, CC %d.%d, globalMem %.1f GiB\n", prop.name, prop.major, prop.minor,
		   prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

	// Host buffers
	std::vector<__half> hA(sizeA);
	std::vector<__half> hB(sizeB);
	std::vector<__half> hC(sizeC, __float2half(0.0f));
	init_half_matrix(hA, M, K);
	init_half_matrix(hB, K, N);

	// Device buffers
	__half *dA = nullptr, *dB = nullptr, *dC = nullptr;
	CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(__half)));
	CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(__half)));
	CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(__half)));
	CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA * sizeof(__half), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB * sizeof(__half), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(dC, 0, sizeC * sizeof(__half)));

	// cuBLAS setup
	cublasHandle_t handle;
	CUBLAS_CHECK(cublasCreate(&handle));
	CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

	const cublasOperation_t opA = CUBLAS_OP_N;
	const cublasOperation_t opB = CUBLAS_OP_N;
	const int lda = M; // leading dimension of A (rows); opA==N and A is MxK => lda=M (column-major)
	const int ldb = K; // leading dimension of B (rows); B is KxN => ldb=K (column-major)
	const int ldc = M; // leading dimension of C (rows); C is MxN => ldc=M (column-major)


	__half alpha_h = __float2half(1.0f);
	__half beta_h  = __float2half(0.0f);

	// Timing with CUDA events
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	// Perform GEMM: C = alpha * A * B + beta * C
	CUBLAS_CHECK(cublasGemmEx(
		handle,
		opA, opB,
		M, N, K,
		&alpha_h,
		dA, CUDA_R_16F, lda,
		dB, CUDA_R_16F, ldb,
		&beta_h,
		dC, CUDA_R_16F, ldc,
		CUBLAS_COMPUTE_16F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP));

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

	// Compute FLOPS: 2 * M * N * K operations
	const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
	const double seconds = static_cast<double>(ms) / 1000.0;
	const double gflops = ops / seconds / 1e9;
	const double tflops = gflops / 1000.0;

	printf("GEMM FP16: M=%d N=%d K=%d\n", M, N, K);
	printf("Time: %.3f ms\n", ms);
	printf("Performance: %.3f GFLOPS (%.3f TFLOPS)\n", gflops, tflops);

	// Cleanup
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUBLAS_CHECK(cublasDestroy(handle));
	CUDA_CHECK(cudaFree(dA));
	CUDA_CHECK(cudaFree(dB));
	CUDA_CHECK(cudaFree(dC));

	return 0;
}

