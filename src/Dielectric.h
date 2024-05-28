/*
 * filename: Dielectric.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#ifndef _DIELECTRIC_H_
#define _DIELECTRIC_H_

#include "MaterialBase.h"

class Dielectric : public MaterialBase {
public: 
    typedef std::shared_ptr<Dielectric> Ptr; 

    Dielectric(const double &refraction_index = 1.0, 
               const std::uint_fast32_t &rng_seed = time(NULL)): 
    rng_{std::make_unique<RandomNumberGenerator>(rng_seed)}, 
    refraction_index_{refraction_index} {
        assert(refraction_index >= 0.0); 
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    virtual bool scatter(const Ray &ray_in, 
                         const HitRecord &hit_record, 
                         Eigen::Vector3d &attenuation, 
                         Ray &scattered_ray) const override; 

private: 
    static double reflectance(const double &cosine, const double &refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1-r0) * std::pow((1 - cosine), 5);
    }

    std::unique_ptr<RandomNumberGenerator> rng_; 
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index_{1.0}; 
}; 

#endif // _DIELECTRIC_H_
