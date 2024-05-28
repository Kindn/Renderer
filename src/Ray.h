/*
 * filename: Ray.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _RAY_H_
#define _RAY_H_

#include "utils.h"

class Ray {
public: 
    Ray(const Point3D &origin = Point3D::Zero(), 
        const Eigen::Vector3d &direction = Eigen::Vector3d::UnitX()): 
    origin_{origin}, direction_{direction.normalized()} {
        assert(direction_.norm() > std::numeric_limits<double>::epsilon()); 
    }

    const Point3D &getOrigin() const {
        return origin_; 
    }

    /**
     * @brief Get the unit direction vector 
    */
    const Eigen::Vector3d &getDirection() const {
        return direction_; 
    }

    void setOrigin(const Point3D &origin) {
        origin_ = origin; 
    }

    void setDirection(const Eigen::Vector3d &direction) {
        assert(direction.norm() > std::numeric_limits<double>::epsilon()); 
        direction_ = direction.normalized(); 
    }

    Point3D at(double t) const {
        return origin_ + t * direction_; 
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private: 
    Point3D origin_; 
    Eigen::Vector3d direction_; 
}; 

#endif // _RAY_H_
