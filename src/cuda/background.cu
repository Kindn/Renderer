#include "cuda/background.h"

namespace cuda {

HOST_DEVICE_FUNC math::Vector3f SimpleSkyBackground::GetRayColor(
    const Ray &ray) const {
  const math::Vector3f &direction = ray.getDirection();
  const float a = 0.5f * (-direction.y() + 1.0f);

  return (1.0f - a) * color1_ + a * color2_;
}

HdriSkyBackground::HdriSkyBackground(HostDataPack const &host_data) {
  {
    cudaChannelFormatDesc const ch_desc{cudaCreateChannelDesc(
        sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat)};
    CUDA_CHECK(cudaMallocArray(&device_data_.d__tex_arr_r, &ch_desc,
                               host_data.cols, host_data.rows));
  }
  {
    cudaChannelFormatDesc const ch_desc{cudaCreateChannelDesc(
        sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat)};
    CUDA_CHECK(cudaMallocArray(&device_data_.d__tex_arr_g, &ch_desc,
                               host_data.cols, host_data.rows));
  }
  {
    cudaChannelFormatDesc const ch_desc{cudaCreateChannelDesc(
        sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat)};
    CUDA_CHECK(cudaMallocArray(&device_data_.d__tex_arr_b, &ch_desc,
                               host_data.cols, host_data.rows));
  }

  CUDA_CHECK(cudaMemcpy2DToArray(
      device_data_.d__tex_arr_r, 0UL, 0UL, host_data.h__tex_r,
      host_data.cols * sizeof(float), host_data.cols * sizeof(float),
      host_data.rows, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy2DToArray(
      device_data_.d__tex_arr_g, 0UL, 0UL, host_data.h__tex_g,
      host_data.cols * sizeof(float), host_data.cols * sizeof(float),
      host_data.rows, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy2DToArray(
      device_data_.d__tex_arr_b, 0UL, 0UL, host_data.h__tex_b,
      host_data.cols * sizeof(float), host_data.cols * sizeof(float),
      host_data.rows, cudaMemcpyHostToDevice));

  {
    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = device_data_.d__tex_arr_r;
    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeMirror;
    tex_desc.addressMode[1] = cudaAddressModeMirror;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    cudaCreateTextureObject(&device_data_.d__tex_r, &res_desc, &tex_desc, NULL);
  }
  {
    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = device_data_.d__tex_arr_g;
    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeMirror;
    tex_desc.addressMode[1] = cudaAddressModeMirror;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    cudaCreateTextureObject(&device_data_.d__tex_g, &res_desc, &tex_desc, NULL);
  }
  {
    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = device_data_.d__tex_arr_b;
    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeMirror;
    tex_desc.addressMode[1] = cudaAddressModeMirror;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    cudaCreateTextureObject(&device_data_.d__tex_b, &res_desc, &tex_desc, NULL);
  }
}

DEVICE_FUNC math::Vector3f HdriSkyBackground::GetRayColor(
    Ray const &ray) const noexcept {
  return GetRayColor(ray, device_data_.d__tex_r, device_data_.d__tex_g,
                     device_data_.d__tex_b);
}

DEVICE_FUNC math::Vector3f HdriSkyBackground::GetRayColor(
    Ray const &ray, cudaTextureObject_t tex_r, cudaTextureObject_t tex_g,
    cudaTextureObject_t tex_b) const noexcept {
  auto const &dir{ray.getDirection()};
  float const u{atan2f(dir.z(), dir.x()) / (2.0f * M_PIf32) + 0.5f};
  float const v{asinf(dir.y()) / M_PIf32 + 0.5f};

  math::Vector3f color{};
  color.x() = tex2D<float>(tex_r, u, v);
  color.y() = tex2D<float>(tex_g, u, v);
  color.z() = tex2D<float>(tex_b, u, v);

  return color;
}

void HdriSkyBackground::Release() {
  CUDA_CHECK(cudaDestroyTextureObject(device_data_.d__tex_r));
  CUDA_CHECK(cudaDestroyTextureObject(device_data_.d__tex_g));
  CUDA_CHECK(cudaDestroyTextureObject(device_data_.d__tex_b));
  CUDA_CHECK(cudaFreeArray(device_data_.d__tex_arr_r));
  CUDA_CHECK(cudaFreeArray(device_data_.d__tex_arr_g));
  CUDA_CHECK(cudaFreeArray(device_data_.d__tex_arr_b));
}

}  // namespace cuda