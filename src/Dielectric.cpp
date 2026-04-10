/*
 * filename: Dielectric.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "Dielectric.h"

bool Dielectric::scatter(const Ray &ray_in, const HitRecord &hit_record,
                         math::Vector3f &attenuation,
                         Ray &scattered_ray) const {
  attenuation = math::Vector3f{1.0f, 1.0f, 1.0f};
  const float ri =
      hit_record.front_face ? (1.0f / refraction_index_) : refraction_index_;

  const math::Vector3f &ray_dir = ray_in.getDirection();
  const float cos_theta = std::min(-ray_dir.Dot(hit_record.normal), 1.0f);
  const float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
  const bool cannot_refract = (ri * sin_theta > 1.0f);
  math::Vector3f scattered;
  if (cannot_refract || reflectance(cos_theta, ri) > rng_->uniform01()) {
    scattered = reflect(ray_dir, hit_record.normal);
  } else {
    scattered = refract(ray_in.getDirection(), hit_record.normal, ri);
  }
  scattered_ray = Ray(hit_record.p, scattered);

  return true;
}
