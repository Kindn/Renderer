/*
 * filename: Dielectric.cpp  
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#include "Dielectric.h" 

bool Dielectric::scatter(const Ray &ray_in, 
                         const HitRecord &hit_record, 
                         Eigen::Vector3d &attenuation, 
                         Ray &scattered_ray) const {
    attenuation = Eigen::Vector3d{1.0, 1.0, 1.0};  
    const double ri = 
        hit_record.front_face ? (1.0 / refraction_index_) : refraction_index_; 

    const Eigen::Vector3d &ray_dir = ray_in.getDirection(); 
    const double cos_theta = std::min(-ray_dir.dot(hit_record.normal), 1.0); 
    const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta); 
    const bool cannot_refract = (ri * sin_theta > 1.0); 
    Eigen::Vector3d scattered; 
    if (cannot_refract || 
        reflectance(cos_theta, ri) > rng_->uniform01()) {
        scattered = reflect(ray_dir, hit_record.normal); 
    } else {
        scattered = 
            refract(ray_in.getDirection(), hit_record.normal, ri); 
    }
    scattered_ray = Ray(hit_record.p, scattered); 

    return true; 
}
