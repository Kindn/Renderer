/*
 * filename: SimpleSkyBackground.cpp 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#include "SimpleSkyBackground.h" 

Eigen::Vector3d SimpleSkyBackground::getRayColor(const Ray &ray) const {
    const Eigen::Vector3d &direction = ray.getDirection(); 
    const double a = 0.5 * (-direction.y() + 1.0); 
    const Eigen::Vector3d color1(1.0, 1.0, 1.0); 
    const Eigen::Vector3d color2(0.5, 0.7, 1.0); 

    return (1.0 - a) * color1 + a * color2; 
}