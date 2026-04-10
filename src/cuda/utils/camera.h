#pragma once

#include "utils.h"

namespace cuda {
namespace utils {

template <typename DType>
DEVICE_FUNC __forceinline__ void
back_project(DType const u, DType const v, DType const fx, DType const fy,
             DType const alpha, DType const cx, DType const cy,
             DType const depth, DType p[3]) {
  p[1] = (v - cy) / fy;
  p[0] = (u - alpha * p[1] - cx) / fx;
  p[0] *= depth;
  p[1] *= depth;
  p[2] = depth;
}

} // namespace utils
} // namespace cuda
