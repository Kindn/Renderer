#ifndef _CORE_H_
#define _CORE_H_

#if USE_CUDA
#include <cuda_runtime.h>
#define DEVICE_FUNC __device__
#define HOST_DEVICE_FUNC __host__ __device__
#define INLINE_DEVICE_FUNC __device__ __forceinline__
#define INLINE_HOST_DEVICE_FUNC __host__ __device__ __forceinline__
#else
#define HOST_DEVICE_FUNC
#endif

#endif // _CORE_H_