#pragma once

#include <algorithm>
#include <cmath>

#include "math/matrix.h"
#include "math/quaterion.h"

namespace math {

template <typename IntType>
INLINE_HOST_DEVICE_FUNC IntType DivUp(IntType const num, IntType const den) {
  return (num + den - 1) / den;
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC DType NormalizeAngle(DType const original) {
  float normalized{static_cast<float>(original)};
  float const kDoublePi{2.0f * M_PIf32};
  while (static_cast<float>(normalized) >= M_PIf32) {
    normalized -= kDoublePi;
  }
  while (static_cast<float>(normalized) < -M_PIf32) {
    normalized += kDoublePi;
  }

  return static_cast<DType>(normalized);
}

}  // namespace math
