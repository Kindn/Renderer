#include "cuda/utils/common.h"

namespace cuda {
namespace utils {

void __CudaCheck(const cudaError_t &err, const char *file,
                 const int32_t &line) {
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE: %s, DETAIL: %s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
    exit(1);
  }
}

void __KernelCheck(const char *file, const int32_t &line) {
  const cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    printf("ERROR: %s:%d, ", file, line);
    printf("CODE: %s, DETAIL: %s\n", cudaGetErrorName(err),
           cudaGetErrorString(err));
    exit(1);
  }
}

cudaError_t ErrorCheck(const cudaError_t &error_code, const char *file_name,
                       const int32_t &line_number) {
  if (error_code != cudaSuccess) {
    printf("CUDA error: \r\ncode=%d, name=%s, desc=%s\r\nfile=%s, line=%d\r\n",
           error_code, cudaGetErrorName(error_code),
           cudaGetErrorString(error_code), file_name, line_number);
  }
  return error_code;
}

int32_t SetGPU() {
  int32_t num_devices = 0;
  cudaError_t error =
      ErrorCheck(cudaGetDeviceCount(&num_devices), __FILE__, __LINE__);
  if (error != cudaSuccess || num_devices == 0) {
    printf("No CUDA compatable GPU found! \n");
    // exit(-1);
    return -1;
  } else {
    printf("The number of GPUs is %d. \n", num_devices);
  }

  int32_t device = 0;
  error = ErrorCheck(cudaSetDevice(device), __FILE__, __LINE__);
  if (error != cudaSuccess) {
    printf("Failed to set GPU %d for computing. ", device);
    // exit(-1);
    return -1;
  } else {
    printf("Set GPU %d for computing. \n", device);
  }

  return 0;
}

} // namespace utils
} // namespace cuda