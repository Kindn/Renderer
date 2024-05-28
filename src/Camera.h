/*
 * filename: Camera.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "Ray.h" 

class Camera {
public: 
    Camera(const size_t &image_width = 640, 
             const size_t &image_height = 480, 
             const double &fov_width = 1.2, 
             const double &fov_height = 0.9, 
             const double &focal_length = 1.0, 
             const double &defocus_angle = 0.0, 
             const Point3D &position = Point3D::Zero(), 
             const Eigen::Quaterniond &rotation = Eigen::Quaterniond::Identity(), 
             const std::uint_fast32_t &rng_seed = time(NULL));  

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

public: 
    const size_t &getImageWidth() const {
        return image_width_; 
    } 

    const size_t &getImageHeight() const {
        return image_height_; 
    } 

    const double &getFOVWidth() const {
        return fov_width_; 
    } 

    const double &getFOVHeight() const {
        return fov_height_; 
    } 

    const double &getImageCenterX() const {
        return image_center_x_; 
    }

    const double &getImageCenterY() const {
        return image_center_y_; 
    }

    const double &getFocalLength() const {
        return focal_length_; 
    } 

    const double &getDefocusAngle() const {
        return defocus_angle_; 
    }

    const Point3D &getPosition() const {
        return position_; 
    }

    const Eigen::Quaterniond &getRotation() const {
        return rotation_; 
    } 

    double getPixelSizeX() const {
        return fov_width_ / image_width_; 
    }

    double getPixelSizeY() const {
        return fov_height_ / image_height_; 
    }

    void setImageWidth(const size_t &image_width) {
        assert(image_width > 0); 
        image_width_ = image_width; 
    }

    void setImageHeight(const size_t &image_height) {
        assert(image_height > 0); 
        image_height_ = image_height; 
    }

    void setFOVWidth(const double &fov_width) {
        assert(fov_width > std::numeric_limits<double>::epsilon()); 
        fov_width_ = fov_width; 
    }

    void setFOVHeight(const double &fov_height) {
        assert(fov_height > std::numeric_limits<double>::epsilon()); 
        fov_height_ = fov_height; 
    }

    void setImageCenterX(const double &image_center_x) {
        image_center_x_ = image_center_x; 
    }

    void setImageCenterY(const double &image_center_y) {
        image_center_y_ = image_center_y; 
    }

    void setFocalLength(const double &focal_length) {
        assert(focal_length > std::numeric_limits<double>::epsilon()); 
        focal_length_ = focal_length; 
        defocus_radius_ = focal_length * std::tan(defocus_angle_ / 2.0);  
    } 

    void setDefocusAngle(const double &defocus_angle) {
        assert(defocus_angle > std::numeric_limits<double>::epsilon() && 
               defocus_angle < M_PI); 
        defocus_angle_ = defocus_angle;
        defocus_radius_ = focal_length_ * std::tan(defocus_angle / 2.0);  
    }

    void setPosition(const Point3D &position) {
        position_ = position; 
    }

    void setRotation(const Eigen::Quaterniond &rotation) {
        rotation_ = rotation; 
    } 

    Point3D getPixelCameraCoordinate(const PixCoord &pixel_coordinate) const; 

    Point3D getPerturbedPixelCameraCoordinate(const PixCoord &pixel_coordinate, 
                                              const double &max_pert) const; 
    
    Point3D defocusDiskSample() const; 

    Ray getRay(const Point3D &camera_coordinate) const;  

    Ray getRay(const PixCoord &pixel_coordinate) const; 

    Ray getPerturbedRay(const PixCoord &pixel_coordinate, 
                        const double &max_perturb) const; 
    
    Ray getDefocusPerturbedRay(const PixCoord &pixel_coordinate, 
                               const double &max_perturb) const; 

private: 
    std::unique_ptr<RandomNumberGenerator> rng_; 

    size_t image_width_; 
    size_t image_height_; 
    double fov_width_;  
    double fov_height_; 

    double image_center_x_; 
    double image_center_y_; 

    double focal_length_; 
    double defocus_angle_{0.0}; 
    double defocus_radius_; 

    Point3D position_; 
    Eigen::Quaterniond rotation_; 
};  

#endif // _CAMERA_H_
