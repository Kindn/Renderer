/*
 * filename: Lambertian.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "Lambertian.h"

bool Lambertian::scatter(const Ray &ray_in, const HitRecord &hit_record,
                         math::Vector3f &attenuation,
                         Ray &scattered_ray) const {
  math::Vector3f scatter_direction =
      hit_record.normal + rng_->uniformPoint3DOnUnitSphere3D();
  if (scatter_direction.Norm() <= std::numeric_limits<float>::epsilon()) {
    scatter_direction = hit_record.normal;
  }

  scattered_ray = Ray(hit_record.p, scatter_direction.Normalized());
  attenuation = albedo_;

  return true;
}
