#pragma once

#include "math/matrix.h"

namespace geometry {

template <typename DType, uint32_t Dim> class AABB {
public:
  typedef math::Matrix<DType, Dim, 1U> Point;

  HOST_DEVICE_FUNC AABB() noexcept {}

  HOST_DEVICE_FUNC AABB(Point const &lb, Point const &ub) noexcept
      : lb_{lb}, ub_{ub} {}

  INLINE_HOST_DEVICE_FUNC bool Contain(Point const &p) const noexcept {
    return p.x() >= lb_.x() && p.x() <= ub_.x() && p.y() >= lb_.y() &&
           p.y() <= ub_.y() && p.z() >= lb_.z() && p.z() <= ub_.z();
  }

private:
  Point lb_{};
  Point ub_{};
};

typedef AABB<float, 2U> AABB2f;
typedef AABB<double, 2U> AABB2d;
typedef AABB<float, 3U> AABB3f;
typedef AABB<double, 3U> AABB3d;

} // namespace geometry