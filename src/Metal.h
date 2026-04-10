/*
 * filename: Metal.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _METAL_H_
#define _METAL_H_

#include "MaterialBase.h"

class Metal : public MaterialBase {
public:
  typedef std::shared_ptr<Metal> Ptr;

  Metal(const math::Vector3f &albedo, const float fuzz = 0.0f,
        const std::uint_fast32_t &rng_seed = time(NULL))
      : rng_{std::make_unique<RandomNumberGenerator>(rng_seed)},
        albedo_{albedo}, fuzz_{fuzz} {}

  

  math::Vector3f getAlbedo() const { return albedo_; }

  void setAlbedo(const math::Vector3f &albedo) { albedo_ = albedo; }

  const float &getFuzz() const { return fuzz_; }

  void setFuzz(const float &fuzz) { fuzz_ = fuzz; }

  bool scatter(const Ray &ray_in, const HitRecord &hit_record,
               math::Vector3f &attenuation, Ray &scattered_ray) const override;

private:
  std::unique_ptr<RandomNumberGenerator> rng_;
  math::Vector3f albedo_;
  float fuzz_{0.0f};
};

#endif // _METAL_H_
