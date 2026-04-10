/*
 * filename: Metal.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "Metal.h"

bool Metal::scatter(const Ray &ray_in, const HitRecord &hit_record,
                    math::Vector3f &attenuation, Ray &scattered_ray) const {
  math::Vector3f reflected = reflect(ray_in.getDirection(), hit_record.normal);
  reflected =
      reflected.Normalized() + fuzz_ * rng_->uniformPoint3DOnUnitSphere3D();
  scattered_ray = Ray(hit_record.p, reflected.Normalized());
  attenuation = albedo_;

  // Absorb scattered rays that are below the surface
  return scattered_ray.getDirection().Dot(hit_record.normal) > 0;
}
