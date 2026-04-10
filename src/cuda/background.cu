#include "cuda/background.h"

namespace cuda {

HOST_DEVICE_FUNC math::Vector3f SimpleSkyBackground::GetRayColor(
    const Ray &ray) const {
  const math::Vector3f &direction = ray.getDirection();
  const float a = 0.5f * (-direction.y() + 1.0f);

  return (1.0f - a) * color1_ + a * color2_;
}

}  // namespace cuda