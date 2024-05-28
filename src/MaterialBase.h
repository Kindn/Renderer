/*
 * filename: MaterialBase.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#ifndef _MATERIAL_BASE_H_
#define _MATERIAL_BASE_H_

#include "utils.h" 
#include "HittableBase.h"

/* Forward declarations. */ 
class HitRecord; 

class MaterialBase {
public: 
    typedef std::shared_ptr<MaterialBase> Ptr; 

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual ~MaterialBase() {} 

    virtual bool scatter(const Ray &ray_in, 
                         const HitRecord &hit_record, 
                         Eigen::Vector3d &attenuation, 
                         Ray &scattered_ray) const {
        return false; 
    }
}; 

#endif // _MATERIAL_BASE_H_