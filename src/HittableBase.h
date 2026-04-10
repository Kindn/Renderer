/*
 * filename: Hittable.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Abstract class for objects hittable by rays
 */

#ifndef _HITTABLE_BASE_H_
#define _HITTABLE_BASE_H_

#include "Ray.h"
#include "utils.h"

/* Forward declarations. */
class MaterialBase;

struct HitRecord {
  Point3D p;
  math::Vector3f normal;
  std::shared_ptr<MaterialBase> material;
  float t;
  bool front_face;

  void setFaceNormal(const Ray &ray, const math::Vector3f &outward_normal) {
    front_face = (ray.getDirection().Dot(outward_normal) < 0);
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class HittableBase {
public:
  typedef std::shared_ptr<HittableBase> Ptr;

  HittableBase(const std::shared_ptr<MaterialBase> &material = nullptr) {
    setMaterial(material);
  }

  virtual ~HittableBase() = default;

  virtual bool hit(const Ray &ray, const Intervalf &interval,
                   HitRecord &hit_record) const = 0;

  std::shared_ptr<MaterialBase> getMaterial() const { return material_; }

  virtual void setMaterial(const std::shared_ptr<MaterialBase> &material) {
    // assert(material != nullptr);
    material_ = material;
  }

protected:
  std::shared_ptr<MaterialBase> material_;
};

#endif // _HITTABLE_BASE_H_
