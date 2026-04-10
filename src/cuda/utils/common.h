#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) cuda::utils::__CudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNAL_CHECK(call) cuda::utils::__KernelCheck(__FILE__, __LINE__)

#define BLOCK_DIM_X 32U
#define BLOCK_DIM_Y 32U
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)
#define GRID_DIM(image_dim, block_dim) \
  (static_cast<uint32_t>(std::ceil(static_cast<float>(image_dim) / block_dim)))

namespace cuda {
namespace utils {

void __CudaCheck(const cudaError_t &err, const char *file, const int32_t &line);

void __KernelCheck(const char *file, const int32_t &line);

cudaError_t ErrorCheck(const cudaError_t &error_code, const char *file_name,
                       const int32_t &line_number);

int32_t SetGPU();

}  // namespace utils
}  // namespace cuda
