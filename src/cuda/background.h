#pragma once

#include "Ray.h"

namespace cuda {

class SimpleSkyBackground {
 public:
  HOST_DEVICE_FUNC SimpleSkyBackground() {}
  HOST_DEVICE_FUNC SimpleSkyBackground(const math::Vector3f &color1,
                                       const math::Vector3f &color2)
      : color1_{color1}, color2_{color2} {}

 public:
  HOST_DEVICE_FUNC math::Vector3f GetRayColor(const Ray &ray) const;

 private:
  math::Vector3f color1_{1.0f, 1.0f, 1.0f};
  math::Vector3f color2_{0.5f, 0.7f, 1.0f};
};

}  // namespace cuda