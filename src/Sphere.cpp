/*
 * filename: Sphere.cpp 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "Sphere.h" 

Sphere::Sphere(const Point3D &center, 
               const double &radius, 
               const MaterialBase::Ptr &material): 
HittableBase(material), 
center_{center}, 
radius_{radius} {
    assert(radius_ >= 0); 
}

bool Sphere::hit(const Ray &ray, 
                 const Intervald &interval,
                 HitRecord &hit_record) const {
    const Point3D &ray_dir = ray.getDirection(); 
    const Eigen::Vector3d &ray_ori = ray.getOrigin(); 
    const Eigen::Vector3d oc = center_ - ray_ori; 
    const double a = ray_dir.squaredNorm(); 
    const double c = oc.squaredNorm() - radius_ * radius_; 
    const double h = oc.dot(ray_dir);

    const double delta = h * h - a * c; 
    if (delta < 0) {
        return false; 
    }

    // Find the nearest root that lies in the acceptable range 
    const double sqrtd = std::sqrt(delta); 
    double root = (h - sqrtd) / a; 
    if (!interval.surrounds(root)) {
        root = (h + sqrtd) / a; 
        if (!interval.surrounds(root)) {
            return false; 
        }
    }
    hit_record.t = root; 
    hit_record.p = ray.at(root); 
    const Eigen::Vector3d outward_normal = (hit_record.p - center_).normalized(); 
    hit_record.setFaceNormal(ray, outward_normal); 
    hit_record.material = this->material_; 

    return true; 
}
