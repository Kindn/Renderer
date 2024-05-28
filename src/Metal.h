/*
 * filename: Metal.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#ifndef _METAL_H_ 
#define _METAL_H_ 

#include "MaterialBase.h" 

class Metal : public MaterialBase {
public: 
    typedef std::shared_ptr<Metal> Ptr; 

    Metal(const Eigen::Vector3d &albedo, 
          const double fuzz = 0.0, 
          const std::uint_fast32_t &rng_seed = time(NULL)): 
    rng_{std::make_unique<RandomNumberGenerator>(rng_seed)}, 
    albedo_{albedo}, fuzz_{fuzz} {

    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d getAlbedo() const {
        return albedo_; 
    }

    void setAlbedo(const Eigen::Vector3d &albedo) {
        albedo_ = albedo; 
    }

    const double &getFuzz() const {
        return fuzz_; 
    }

    void setFuzz(const double &fuzz) {
        fuzz_ = fuzz; 
    }

    bool scatter(const Ray &ray_in, 
                 const HitRecord &hit_record, 
                 Eigen::Vector3d &attenuation, 
                 Ray &scattered_ray) const override; 

private: 
    std::unique_ptr<RandomNumberGenerator> rng_; 
    Eigen::Vector3d albedo_; 
    double fuzz_{0.0}; 
}; 

#endif // _METAL_H_ 
