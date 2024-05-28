/*
 * filename: Lambertian.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#ifndef _LAMBERTIAN_H_
#define _LAMBERTIAN_H_

#include "MaterialBase.h" 

class Lambertian : public MaterialBase {
public: 
    typedef std::shared_ptr<Lambertian> Ptr; 

    Lambertian(const Eigen::Vector3d &albedo, 
               const std::uint_fast32_t &rng_seed = time(NULL)): 
    albedo_{albedo} {
        rng_ = std::make_unique<RandomNumberGenerator>(rng_seed); 
    } 

    Eigen::Vector3d getAlbedo() const {
        return albedo_; 
    }

    void setAlbedo(const Eigen::Vector3d &albedo) {
        albedo_ = albedo; 
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool scatter(const Ray &ray_in, 
                 const HitRecord &hit_record, 
                 Eigen::Vector3d &attenuation, 
                 Ray &scattered_ray) const override; 

private: 
    std::unique_ptr<RandomNumberGenerator> rng_; 
    Eigen::Vector3d albedo_; 
}; 

#endif // _LAMBERTIAN_H_