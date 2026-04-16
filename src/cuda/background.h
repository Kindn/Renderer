#pragma once

#include "Ray.h"
#include "cuda/utils/common.h"

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

class HdriSkyBackground {
 public:
  struct HostDataPack {
    float *h__tex_r{};
    float *h__tex_g{};
    float *h__tex_b{};
    uint32_t rows{};
    uint32_t cols{};
  };

  HdriSkyBackground(HostDataPack const &host_data);

 public:
  DEVICE_FUNC math::Vector3f GetRayColor(Ray const &ray) const noexcept;

  void Release();

 private:
  DEVICE_FUNC math::Vector3f GetRayColor(
      Ray const &ray, cudaTextureObject_t tex_r, cudaTextureObject_t tex_g,
      cudaTextureObject_t tex_b) const noexcept;

 private:
  struct DeviceDataPack {
    cudaArray_t d__tex_arr_r{};
    cudaArray_t d__tex_arr_g{};
    cudaArray_t d__tex_arr_b{};
    cudaTextureObject_t d__tex_r{};
    cudaTextureObject_t d__tex_g{};
    cudaTextureObject_t d__tex_b{};
    uint32_t rows{};
    uint32_t cols{};
  };

  DeviceDataPack device_data_{};
};

}  // namespace cuda