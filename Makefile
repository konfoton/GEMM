# Makefile for PTX-GEMM project

# Compiler
NVCC := nvcc

# GPU architecture (adjust for your GPU)
# sm_70 = V100, sm_80 = A100, sm_89 = RTX 4090
CUDA_ARCH := -arch=sm_86

# Compiler flags
NVCC_FLAGS := -O3 -std=c++17 $(CUDA_ARCH)
DEBUG_FLAGS := -G -g -lineinfo

# Libraries
LIBS := -lcublas

# Source files
KERNEL_HDR := gemm_kernel.cuh
TEST_SRC := test_gemm.cu

# Output
TEST_BIN := test_gemm

# Default target
all: $(TEST_BIN)

# Build test binary
$(TEST_BIN): $(TEST_SRC) $(KERNEL_HDR)
	$(NVCC) $(NVCC_FLAGS) $(TEST_SRC) -o $(TEST_BIN) $(LIBS)

# Debug build
debug: $(TEST_SRC) $(KERNEL_HDR)
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $(TEST_SRC) -o $(TEST_BIN)_debug $(LIBS)

# Build with profiling support
profile: $(TEST_SRC) $(KERNEL_HDR)
	$(NVCC) $(NVCC_FLAGS) -lineinfo $(TEST_SRC) -o $(TEST_BIN)_profile $(LIBS)

# Run test
run: $(TEST_BIN)
	./$(TEST_BIN)

test: run

# Clean
clean:
	rm -f $(TEST_BIN) $(TEST_BIN)_debug $(TEST_BIN)_profile

# Help
help:
	@echo "PTX-GEMM Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build test_gemm (default)"
	@echo "  debug    - Build with debug symbols"
	@echo "  profile  - Build with profiling support"
	@echo "  run/test - Build and run tests"
	@echo "  clean    - Remove binaries"
	@echo ""
	@echo "Files:"
	@echo "  gemm_kernel.cuh - GEMM kernel header"
	@echo "  test_gemm.cu    - Test and benchmark code"

.PHONY: all debug profile run test clean help
