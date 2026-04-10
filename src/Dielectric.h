/*
 * filename: Dielectric.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _DIELECTRIC_H_
#define _DIELECTRIC_H_

#include "MaterialBase.h"

class Dielectric : public MaterialBase {
public:
  typedef std::shared_ptr<Dielectric> Ptr;

  Dielectric(const float &refraction_index = 1.0f,
             const std::uint_fast32_t &rng_seed = time(NULL))
      : rng_{std::make_unique<RandomNumberGenerator>(rng_seed)},
        refraction_index_{refraction_index} {
    assert(refraction_index >= 0.0f);
  }

  

  virtual bool scatter(const Ray &ray_in, const HitRecord &hit_record,
                       math::Vector3f &attenuation,
                       Ray &scattered_ray) const override;

private:
  static float reflectance(const float &cosine,
                            const float &refraction_index) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5);
  }

  std::unique_ptr<RandomNumberGenerator> rng_;
  // Refractive index in vacuum or air, or the ratio of the material's
  // refractive index over the refractive index of the enclosing media
  float refraction_index_{1.0f};
};

#endif // _DIELECTRIC_H_
