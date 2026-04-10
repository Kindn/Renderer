/*
 * filename: utils.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _UTILS_H_
#define _UTILS_H_

#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "math/math.h"

// namespace Eigen {

typedef math::Matrix<uint8_t, 3, 1> Color;
typedef math::Matrix<size_t, 2, 1> PixCoord;
typedef math::Vector3f Point3D;
typedef math::Vector2f Point2D;

template <typename T>
T inf() {
  return std::numeric_limits<T>::infinity();
}

template <typename T>
class Interval {
 public:
  // static constexpr T inf = std::numeric_limits<T>::infinity();

  HOST_DEVICE_FUNC constexpr Interval(const T &min = inf<T>(),
                                      const T &max = -inf<T>())
      : min_{min}, max_{max} {}

  HOST_DEVICE_FUNC T size() const { return max_ - min_; }

  HOST_DEVICE_FUNC bool isEmpty() const { return min_ > max_; }

  HOST_DEVICE_FUNC bool contains(const T &x) const {
    return x >= min_ && x <= max_;
  }

  HOST_DEVICE_FUNC bool surrounds(const T &x) const {
    return x > min_ && x < max_;
  }

  HOST_DEVICE_FUNC T &min() { return min_; }

  HOST_DEVICE_FUNC const T &min() const { return min_; }

  HOST_DEVICE_FUNC T &max() { return max_; }

  HOST_DEVICE_FUNC const T &max() const { return max_; }

  HOST_DEVICE_FUNC T clamp(const T &x) const {
    return (x > max_) ? max_ : ((x < min_) ? min_ : x);
  }

  HOST_DEVICE_FUNC static Interval<T> positive() {
    return Interval<T>(0, inf<T>());
  }

  HOST_DEVICE_FUNC static Interval<T> negative() {
    return Interval<T>(-inf<T>(), 0);
  }

  HOST_DEVICE_FUNC static Interval<T> all() {
    return Interval<T>(-inf<T>(), inf<T>());
  }

 private:
  T min_;
  T max_;
};

typedef Interval<float> Intervalf;
typedef Interval<int> Intervali;

extern const float inff;
extern const float epsf;

class RandomNumberGenerator {
 public:
  RandomNumberGenerator() : seed_(time(NULL)), engine_(seed_) {}

  RandomNumberGenerator(const std::uint_fast32_t &seed)
      : seed_(seed), engine_(seed_) {}

  ~RandomNumberGenerator() {}

 public:
  void setSeed(const std::uint_fast32_t &seed) {
    seed_ = seed;
    engine_.seed(seed_);
  }

  float uniform01() { return urd01_(engine_); }

  float uniformReal(const float &lower, const float &upper) {
    assert(lower <= upper);
    return lower + (upper - lower) * urd01_(engine_);
  }

  int uniformInteger(const int &lower, const int &upper) {
    assert(lower <= upper);
    std::uniform_int_distribution<int> uid(lower, upper);
    return uid(engine_);
  }

  size_t uniformIndex(const size_t &lower, const size_t &upper) {
    assert(lower <= upper);
    std::uniform_int_distribution<size_t> uid(lower, upper);
    return uid(engine_);
  }

  // From: "Uniform Random Rotations", Ken Shoemake, Graphics Gems III,
  //       pg. 124-132
  void uniformQuaternion(float &x, float &y, float &z, float &w) {
    float t = urd01_(engine_);
    float r1 = sqrt(1.0f - t), r2 = sqrt(t);
    float th1 = 2.0f * M_PI * urd01_(engine_);
    float th2 = 2.0f * M_PI * urd01_(engine_);
    float c1 = cos(th1), s1 = sin(th1);
    float c2 = cos(th2), s2 = sin(th2);
    x = s1 * r1;
    y = c1 * r1;
    z = s2 * r2;
    w = c2 * r2;
  }

  template <size_t Rows, size_t Cols>
  math::Matrix<float, Rows, Cols> uniformRealMatrix(
      const float &lower = 0.0f, const float &upper = 1.0f) {
    math::Matrix<float, Rows, Cols> ret;
    for (size_t r = 0; r < Rows; ++r) {
      for (size_t c = 0; c < Cols; ++c) {
        ret(r, c) = lower + uniform01() * (upper - lower);
      }
    }
    return ret;
  }

  Point3D uniformPoint3DOnUnitSphere3D() {
    const float u = uniformReal(0.0f, 1.0f);
    const float v = uniformReal(-1.0f, 1.0f);
    const float phi = 2.0f * M_PI * u;
    const float theta = std::acos(v);
    return Point3D{std::sin(theta) * std::cos(phi),
                   std::sin(theta) * std::sin(phi), std::cos(theta)};
  }

  Point3D uniformPoint3DOnUnitHemiSphere3D(const math::Vector3f &normal) {
    const Point3D pr = uniformPoint3DOnUnitSphere3D();
    return (pr.Dot(normal) > 0.0f) ? pr : -pr;
  }

  Point2D uniformPoint2DInUnitCircle() {
    const float r = std::sqrt(uniform01());
    const float theta = 2.0f * M_PI * uniform01();
    return Point2D{r * std::cos(theta), r * std::sin(theta)};
  }

 private:
  std::uint_fast32_t seed_;
  std::mt19937 engine_;
  std::uniform_real_distribution<float> urd01_{0.0f, 1.0f};
  std::normal_distribution<float> nd_{0.0f, 1.0f};
};

class ThreadGuard {
 private:
  std::thread &t;

 public:
  explicit ThreadGuard(std::thread &t_) : t(t_) {}

  ~ThreadGuard() {
    if (t.joinable()) {
      t.join();
    }
  }
};

class TicToc {
 public:
  TicToc() { tic(); }

  void tic() { start = std::chrono::steady_clock::now(); }

  float toc() {
    end = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    return elapsed_seconds.count();
  }

 private:
  std::chrono::steady_clock::time_point start, end;
};

HOST_DEVICE_FUNC inline float linearToGamma(const float &x,
                                             const float &gamma) {
  return (x > 0.0f) ? powf(x, 1.0f / gamma) : 0.0f;
}

HOST_DEVICE_FUNC inline math::Vector3f reflect(const math::Vector3f &v,
                                               const math::Vector3f &n) {
  return v - 2.0f * v.Dot(n) * n;
}

HOST_DEVICE_FUNC inline math::Vector3f refract(const math::Vector3f &uv,
                                               const math::Vector3f &n,
                                               const float &etai_over_etat) {
  const float cos_theta = fmin(-uv.Dot(n), 1.0f);
  const math::Vector3f r_out_perp = etai_over_etat * (uv + cos_theta * n);
  const math::Vector3f r_out_para =
      -sqrtf(1.0f - r_out_perp.SquaredNorm()) * n;
  return r_out_perp + r_out_para;
}

void writeColorToOStream(std::ostream &out, const Color &color);

void writeColorToOStream(std::ostream &out, const math::Vector3f &color);

// }

#endif  // _UITLS_H_
