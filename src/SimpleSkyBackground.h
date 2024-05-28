/*
 * filename: SimpleSkyBackground.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#ifndef _SIMPLE_SKY_BACKGOUND_H_
#define _SIMPLE_SKY_BACKGOUND_H_

#include "BackgroundBase.h"

class SimpleSkyBackground : public BackgroundBase {
public: 
    typedef std::shared_ptr<SimpleSkyBackground> Ptr; 

    SimpleSkyBackground() {} 
    SimpleSkyBackground(const Eigen::Vector3d &color1, 
                        const Eigen::Vector3d &color2): 
    color1_{color1}, color2_{color2} {

    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

public: 
    Eigen::Vector3d getRayColor(const Ray &ray) const override;

private: 
    Eigen::Vector3d color1_{1.0, 1.0, 1.0}; 
    Eigen::Vector3d color2_{0.5, 0.7, 1.0}; 
}; 

#endif 