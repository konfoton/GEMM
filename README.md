# PTX-GEMM

A high-performance FP16 GEMM (General Matrix Multiplication) implementation using CUDA PTX inline assembly and Tensor Cores.

## Overview

This project implements a GEMM kernel from scratch using:
- **PTX inline assembly** for direct hardware control
- **Tensor Core MMA instructions** (`mma.sync.aligned.m16n8k16`)
- **Shared memory with swizzling** to avoid bank conflicts
- **`ldmatrix` instructions** for efficient shared memory to register loading
- **Hierarchical tiling** (global → shared memory → registers)

## Performance

| Metric | Value |
|--------|-------|
| Matrix Size | 8192 × 8192 × 8192 |
| Data Type | FP16 |
| **Custom Kernel vs cuBLAS** | **51%** |

## Implementation Details

### Tiling Strategy
- **Block tile**: 256×128 (M×N) per thread block
- **Warp tile**: 64×64 per warp
- **MMA tile**: 16×8×16 (m×n×k) per Tensor Core operation

### Memory Hierarchy
1. **Global → Shared Memory**: Cooperative loading with all threads in a block
2. **Shared → Registers**: Using `ldmatrix.sync.aligned` PTX instructions
3. **Swizzling**: XOR-based swizzle pattern to eliminate shared memory bank conflicts

### Key Features
- 8 warps per block (256 threads)
- Register file usage: 64×64×32 per warp
- Shared memory: 256×64 (A) + 64×128 (B) half-precision elements
- Validation against cuBLAS reference implementation

## Building

```bash
nvcc -o gemm gemm.cu -lcublas -arch=sm_86 -O3
```

## Running

```bash
./gemm
```

## Future Optimizations

- Double buffering for global memory loads
- Software pipelining
- Warp specialization
- Tune tile sizes for different GPU architectures
