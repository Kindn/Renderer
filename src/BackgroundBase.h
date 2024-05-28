/*
 * filename: BackgroundBase.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#ifndef _BACKGROUND_BASE_H_
#define _BACKGROUND_BASE_H_

#include "utils.h"
#include "Ray.h"

class BackgroundBase {
public: 
    typedef std::shared_ptr<BackgroundBase> Ptr; 

    virtual ~BackgroundBase() = default; 

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public: 
    virtual Eigen::Vector3d getRayColor(const Ray &ray) const = 0; 
};

#endif // _BACKGROUND_BASE_H_
