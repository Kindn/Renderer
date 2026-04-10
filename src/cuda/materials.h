#pragma once

#include "cuda/hit_data.h"
#include "cuda/utils/random.h"

namespace cuda {

struct DielectricProperty {
  float refraction_index{1.0f};
};

struct LambertianProperty {
  math::Vector3f albedo{};
};

struct MetalProperty {
  float fuzz{};
  math::Vector3f albedo{};
};

HOST_DEVICE_FUNC bool scatter_dielectric(
    const Ray &ray_in, const HitRecordCuda &hit_record,
    DielectricProperty const *const property,
    utils::RandomNumberGenerator *const rng, math::Vector3f &attenuation,
    Ray &scattered_ray);

HOST_DEVICE_FUNC bool scatter_lambertian(
    const Ray &ray_in, const HitRecordCuda &hit_record,
    LambertianProperty const *const property,
    utils::RandomNumberGenerator *const rng, math::Vector3f &attenuation,
    Ray &scattered_ray);

HOST_DEVICE_FUNC bool scatter_metal(const Ray &ray_in,
                                    const HitRecordCuda &hit_record,
                                    MetalProperty const *const property,
                                    utils::RandomNumberGenerator *const rng,
                                    math::Vector3f &attenuation,
                                    Ray &scattered_ray);

HOST_DEVICE_FUNC bool scatter(const Ray &ray_in,
                              const HitRecordCuda &hit_record,
                              utils::RandomNumberGenerator *const rng,
                              math::Vector3f &attenuation, Ray &scattered_ray);

}  // namespace cuda