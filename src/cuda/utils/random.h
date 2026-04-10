#pragma once

#include <thrust/random.h>

#include "utils.h"

namespace cuda {
namespace utils {

class RandomNumberGenerator {
 public:
  // INLINE_HOST_DEVICE_FUNC RandomNumberGenerator() : seed_(time(NULL)),
  // engine_(seed_) {}

  INLINE_HOST_DEVICE_FUNC RandomNumberGenerator(const uint64_t &seed)
      : seed_(seed), engine_(seed_) {}

  INLINE_HOST_DEVICE_FUNC ~RandomNumberGenerator() {}

 public:
  INLINE_HOST_DEVICE_FUNC void setSeed(const uint64_t &seed) {
    seed_ = seed;
    engine_.seed(seed_);
  }

  INLINE_HOST_DEVICE_FUNC float uniform01() { return urd01_(engine_); }

  INLINE_HOST_DEVICE_FUNC float uniformReal(const float &lower,
                                            const float &upper) {
    assert(lower <= upper);
    return lower + (upper - lower) * urd01_(engine_);
  }

  // From: "Uniform Random Rotations", Ken Shoemake, Graphics Gems III,
  //       pg. 124-132
  INLINE_HOST_DEVICE_FUNC void uniformQuaternion(float &x, float &y, float &z,
                                                 float &w) {
    float t = urd01_(engine_);
    float r1 = sqrtf(1.0f - t), r2 = sqrtf(t);
    float th1 = 2.0f * M_PI * urd01_(engine_);
    float th2 = 2.0f * M_PI * urd01_(engine_);
    float c1 = cosf(th1), s1 = sinf(th1);
    float c2 = cosf(th2), s2 = sinf(th2);
    x = s1 * r1;
    y = c1 * r1;
    z = s2 * r2;
    w = c2 * r2;
  }

  template <size_t Rows, size_t Cols>
  INLINE_HOST_DEVICE_FUNC math::Matrix<float, Rows, Cols> uniformRealMatrix(
      const float &lower = 0.0f, const float &upper = 1.0f) {
    math::Matrix<float, Rows, Cols> ret;
    for (size_t r = 0; r < Rows; ++r) {
      for (size_t c = 0; c < Cols; ++c) {
        ret(r, c) = lower + uniform01() * (upper - lower);
      }
    }
    return ret;
  }

  INLINE_HOST_DEVICE_FUNC math::Vector3f uniformPoint3DOnUnitSphere3D() {
    const float u = uniformReal(0.0f, 1.0f);
    const float v = uniformReal(-1.0f, 1.0f);
    const float phi = 2.0f * M_PI * u;
    const float theta = acosf(v);
    return math::Vector3f{sinf(theta) * cosf(phi),
                          sinf(theta) * sinf(phi), cosf(theta)};
  }

  INLINE_HOST_DEVICE_FUNC math::Vector3f uniformPoint3DOnUnitHemiSphere3D(
      const math::Vector3f &normal) {
    const math::Vector3f pr = uniformPoint3DOnUnitSphere3D();
    return (pr.Dot(normal) > 0.0f) ? pr : -pr;
  }

  INLINE_HOST_DEVICE_FUNC Point2D uniformPoint2DInUnitCircle() {
    const float r = sqrtf(uniform01());
    const float theta = 2.0f * M_PI * uniform01();
    return Point2D{r * cosf(theta), r * sinf(theta)};
  }

 private:
  uint32_t seed_;
  thrust::default_random_engine engine_;
  thrust::uniform_real_distribution<float> urd01_{0.0f, 1.0f};
  thrust::normal_distribution<float> nd_{0.0f, 1.0f};
};

}  // namespace utils
}  // namespace cuda