#include <thrust/random.h>

#include "cuda/materials.h"
#include "cuda/utils/random.h"

namespace cuda {

HOST_DEVICE_FUNC static float reflectance(float const &cosine,
                                           float const &refraction_index) {
  // Use Schlick's approximation for reflectance.
  auto r0 = (1 - refraction_index) / (1 + refraction_index);
  r0 = r0 * r0;
  return r0 + (1 - r0) * powf((1 - cosine), 5);
}

HOST_DEVICE_FUNC bool scatter_dielectric(
    const Ray &ray_in, const HitRecordCuda &hit_record,
    DielectricProperty const *const property,
    utils::RandomNumberGenerator *const rng, math::Vector3f &attenuation,
    Ray &scattered_ray) {
  if (nullptr == property) {
    return false;
  }

  attenuation = math::Vector3f{1.0f, 1.0f, 1.0f};
  const float ri = hit_record.front_face ? (1.0f / property->refraction_index)
                                          : property->refraction_index;

  const math::Vector3f &ray_dir = ray_in.getDirection();
  const float cos_theta = fmin(-ray_dir.Dot(hit_record.normal), 1.0f);
  const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
  const bool cannot_refract = (ri * sin_theta > 1.0f);
  math::Vector3f scattered;
  if (cannot_refract || reflectance(cos_theta, ri) > rng->uniform01()) {
    scattered = reflect(ray_dir, hit_record.normal);
  } else {
    scattered = refract(ray_in.getDirection(), hit_record.normal, ri);
  }
  scattered_ray = Ray(hit_record.p, scattered);

  return true;
}

HOST_DEVICE_FUNC bool scatter_lambertian(
    const Ray &ray_in, const HitRecordCuda &hit_record,
    LambertianProperty const *const property,
    utils::RandomNumberGenerator *const rng, math::Vector3f &attenuation,
    Ray &scattered_ray) {
  if (nullptr == property) {
    return false;
  }

  math::Vector3f scatter_direction =
      hit_record.normal + rng->uniformPoint3DOnUnitSphere3D();
  if (scatter_direction.Norm() <= 1.0e-32) {
    scatter_direction = hit_record.normal;
  }

  scattered_ray = Ray(hit_record.p, scatter_direction.Normalized());
  attenuation = property->albedo;

  return true;
}

HOST_DEVICE_FUNC bool scatter_metal(const Ray &ray_in,
                                    const HitRecordCuda &hit_record,
                                    MetalProperty const *const property,
                                    utils::RandomNumberGenerator *const rng,
                                    math::Vector3f &attenuation,
                                    Ray &scattered_ray) {
  if (nullptr == property) {
    return false;
  }

  math::Vector3f reflected = reflect(ray_in.getDirection(), hit_record.normal);
  reflected = reflected.Normalized() +
              property->fuzz * rng->uniformPoint3DOnUnitSphere3D();
  scattered_ray = Ray(hit_record.p, reflected.Normalized());
  attenuation = property->albedo;

  // Absorb scattered rays that are below the surface
  return scattered_ray.getDirection().Dot(hit_record.normal) > 0;
}

HOST_DEVICE_FUNC bool scatter(const Ray &ray_in,
                              const HitRecordCuda &hit_record,
                              utils::RandomNumberGenerator *const rng,
                              math::Vector3f &attenuation, Ray &scattered_ray) {
  if (nullptr == hit_record.material_property) {
    return false;
  }

  if (MaterialType::MATERIAL_DIELECTRIC == hit_record.material) {
    return scatter_dielectric(
        ray_in, hit_record,
        (DielectricProperty const *)hit_record.material_property, rng,
        attenuation, scattered_ray);
  } else if (MaterialType::MATERIAL_LAMBERTIAN == hit_record.material) {
    return scatter_lambertian(
        ray_in, hit_record,
        (LambertianProperty const *)hit_record.material_property, rng,
        attenuation, scattered_ray);
  } else if (MaterialType::MATERIAL_METAL == hit_record.material) {
    return scatter_metal(ray_in, hit_record,
                         (MetalProperty const *)hit_record.material_property,
                         rng, attenuation, scattered_ray);
  }

  return false;
}

}  // namespace cuda