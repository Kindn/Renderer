/*
 * filename: Lambertian.cpp 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#include "Lambertian.h" 

bool Lambertian::scatter(const Ray &ray_in, 
                         const HitRecord &hit_record, 
                         Eigen::Vector3d &attenuation, 
                         Ray &scattered_ray) const {
    Eigen::Vector3d scatter_direction = 
        hit_record.normal + rng_->uniformPoint3DOnUnitSphere3D(); 
    if (scatter_direction.norm() <= std::numeric_limits<double>::epsilon()) {
        scatter_direction = hit_record.normal; 
    }

    scattered_ray = Ray(hit_record.p, scatter_direction); 
    attenuation = albedo_; 

    return true; 
}
