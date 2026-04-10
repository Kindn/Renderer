#include <cuda_runtime.h>
#include <numeric>
#include <limits>


namespace cuda {
namespace utils {
namespace constants {

extern __constant__ float eps_f;
extern __constant__ float inf_f;

} // namespace constants
} // namespace utils
} // namespace cuda
