#pragma once

#include <cmath>
#include <algorithm>

#include "math/matrix.h"
#include "math/quaterion.h"

namespace math {

template <typename IntType>
HOST_DEVICE_FUNC IntType DivUp(IntType const num, IntType const den) {
  return (num + den - 1) / den;
}

}
