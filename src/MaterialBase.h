/*
 * filename: MaterialBase.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _MATERIAL_BASE_H_
#define _MATERIAL_BASE_H_

#include "HittableBase.h"
#include "utils.h"

/* Forward declarations. */
class HitRecord;

class MaterialBase {
public:
  typedef std::shared_ptr<MaterialBase> Ptr;

  virtual ~MaterialBase() {}

  virtual bool scatter(const Ray &ray_in, const HitRecord &hit_record,
                       math::Vector3f &attenuation, Ray &scattered_ray) const {
    return false;
  }
};

#endif // _MATERIAL_BASE_H_