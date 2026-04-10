/*
 * filename: Sphere.cpp 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "Sphere.h" 

Sphere::Sphere(const Point3D &center, 
               const float &radius, 
               const std::shared_ptr<MaterialBase> &material): 
HittableBase(material), 
center_{center}, 
radius_{radius} {
    assert(radius_ >= 0); 
}

bool Sphere::hit(const Ray &ray, 
                 const Intervalf &interval,
                 HitRecord &hit_record) const {
    const Point3D &ray_dir = ray.getDirection(); 
    const math::Vector3f &ray_ori = ray.getOrigin(); 
    const math::Vector3f oc = center_ - ray_ori; 
    const float a = ray_dir.SquaredNorm(); 
    const float c = oc.SquaredNorm() - radius_ * radius_; 
    const float h = oc.Dot(ray_dir);

    const float delta = h * h - a * c; 
    if (delta < 0) {
        return false; 
    }

    // Find the nearest root that lies in the acceptable range 
    const float sqrtd = std::sqrt(delta); 
    float root = (h - sqrtd) / a; 
    if (!interval.surrounds(root)) {
        root = (h + sqrtd) / a; 
        if (!interval.surrounds(root)) {
            return false; 
        }
    }
    hit_record.t = root; 
    hit_record.p = ray.at(root); 
    const math::Vector3f outward_normal = (hit_record.p - center_).Normalized(); 
    hit_record.setFaceNormal(ray, outward_normal); 
    hit_record.material = this->material_; 

    return true; 
}
