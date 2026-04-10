/*
 * filename: Ray.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _RAY_H_
#define _RAY_H_

#include "utils.h"

class Ray {
public:
  HOST_DEVICE_FUNC Ray(const Point3D &origin = Point3D::Zero(),
                  const math::Vector3f &direction = math::Vector3f::UnitX())
      : origin_{origin}, direction_{direction.Normalized()} {
    assert(direction_.Norm() > 1.0e-32);
  }

  HOST_DEVICE_FUNC const Point3D &getOrigin() const { return origin_; }

  /**
   * @brief Get the unit direction vector
   */
  HOST_DEVICE_FUNC const math::Vector3f &getDirection() const { return direction_; }

  HOST_DEVICE_FUNC void setOrigin(const Point3D &origin) { origin_ = origin; }

  HOST_DEVICE_FUNC void setDirection(const math::Vector3f &direction) {
    assert(direction.Norm() > 1.0e-32);
    direction_ = direction.Normalized();
  }

  HOST_DEVICE_FUNC Point3D at(float t) const { return origin_ + t * direction_; }

private:
  Point3D origin_;
  math::Vector3f direction_;
};

#endif // _RAY_H_
