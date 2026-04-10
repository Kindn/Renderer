/*
 * filename: Lambertian.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _LAMBERTIAN_H_
#define _LAMBERTIAN_H_

#include "MaterialBase.h"

class Lambertian : public MaterialBase {
public:
  typedef std::shared_ptr<Lambertian> Ptr;

  Lambertian(const math::Vector3f &albedo,
             const std::uint_fast32_t &rng_seed = time(NULL))
      : albedo_{albedo} {
    rng_ = std::make_unique<RandomNumberGenerator>(rng_seed);
  }

  math::Vector3f getAlbedo() const { return albedo_; }

  void setAlbedo(const math::Vector3f &albedo) { albedo_ = albedo; }

  

  bool scatter(const Ray &ray_in, const HitRecord &hit_record,
               math::Vector3f &attenuation, Ray &scattered_ray) const override;

private:
  std::unique_ptr<RandomNumberGenerator> rng_;
  math::Vector3f albedo_;
};

#endif // _LAMBERTIAN_H_