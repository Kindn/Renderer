/*
 * filename: Sphere.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "HittableBase.h"

class Sphere : public HittableBase {
public:
  typedef std::shared_ptr<Sphere> Ptr;

  Sphere(const Point3D &center = Point3D::Zero(), const float &radius = 1.0f,
         const std::shared_ptr<MaterialBase> &material = nullptr);

  const Point3D &getCenter() const { return center_; }

  const float &getRadius() const { return radius_; }

  void setCenter(const Point3D &center) { center_ = center; }

  void setRadius(const float &radius) {
    assert(radius >= 0);
    radius_ = radius;
  }

  virtual bool hit(const Ray &ray, const Intervalf &interval,
                   HitRecord &hit_record) const override;

private:
  Point3D center_{0.0f, 0.0f, 0.0f};
  float radius_{1.0f};
};

#endif // _SPHERE_H_
