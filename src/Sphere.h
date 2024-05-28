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

    Sphere(const Point3D &center = Point3D::Zero(), 
           const double &radius = 1.0, 
           const MaterialBase::Ptr &material = nullptr); 

    const Point3D &getCenter() const {
        return center_; 
    }

    const double &getRadius() const {
        return radius_; 
    }

    void setCenter(const Point3D &center) {
        center_ = center; 
    }

    void setRadius(const double &radius) {
        assert(radius >= 0); 
        radius_ = radius; 
    }

    virtual bool hit(const Ray &ray, 
                     const Intervald &interval, 
                     HitRecord &hit_record) const override; 

private: 
    Point3D center_{0.0, 0.0, 0.0}; 
    double radius_{1.0};
}; 

#endif // _SPHERE_H_ 
