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
  Camera(const size_t &image_width = 640, const size_t &image_height = 480,
         const float &fov_width = 1.2f, const float &fov_height = 0.9,
         const float &focal_length = 1.0f, const float &defocus_angle = 0.0f,
         const Point3D &position = Point3D::Zero(),
         const math::Quaternionf &rotation = math::Quaternionf::Identity(),
         const std::uint_fast32_t &rng_seed = time(NULL));

  

public:
  const size_t &getImageWidth() const { return image_width_; }

  const size_t &getImageHeight() const { return image_height_; }

  const float &getFOVWidth() const { return fov_width_; }

  const float &getFOVHeight() const { return fov_height_; }

  const float &getImageCenterX() const { return image_center_x_; }

  const float &getImageCenterY() const { return image_center_y_; }

  const float &getFocalLength() const { return focal_length_; }

  const float &getDefocusAngle() const { return defocus_angle_; }

  const Point3D &getPosition() const { return position_; }

  const math::Quaternionf &getRotation() const { return rotation_; }

  float getPixelSizeX() const { return fov_width_ / image_width_; }

  float getPixelSizeY() const { return fov_height_ / image_height_; }

  void setImageWidth(const size_t &image_width) {
    assert(image_width > 0);
    image_width_ = image_width;
  }

  void setImageHeight(const size_t &image_height) {
    assert(image_height > 0);
    image_height_ = image_height;
  }

  void setFOVWidth(const float &fov_width) {
    assert(fov_width > std::numeric_limits<float>::epsilon());
    fov_width_ = fov_width;
  }

  void setFOVHeight(const float &fov_height) {
    assert(fov_height > std::numeric_limits<float>::epsilon());
    fov_height_ = fov_height;
  }

  void setImageCenterX(const float &image_center_x) {
    image_center_x_ = image_center_x;
  }

  void setImageCenterY(const float &image_center_y) {
    image_center_y_ = image_center_y;
  }

  void setFocalLength(const float &focal_length) {
    assert(focal_length > std::numeric_limits<float>::epsilon());
    focal_length_ = focal_length;
    defocus_radius_ = focal_length * std::tan(defocus_angle_ / 2.0f);
  }

  void setDefocusAngle(const float &defocus_angle) {
    assert(defocus_angle > std::numeric_limits<float>::epsilon() &&
           defocus_angle < M_PI);
    defocus_angle_ = defocus_angle;
    defocus_radius_ = focal_length_ * std::tan(defocus_angle / 2.0f);
  }

  void setPosition(const Point3D &position) { position_ = position; }

  void setRotation(const math::Quaternionf &rotation) { rotation_ = rotation; }

  Point3D getPixelCameraCoordinate(const PixCoord &pixel_coordinate) const;

  Point3D getPerturbedPixelCameraCoordinate(const PixCoord &pixel_coordinate,
                                            const float &max_pert) const;

  Point3D defocusDiskSample() const;

  Ray getRay(const Point3D &camera_coordinate) const;

  Ray getRay(const PixCoord &pixel_coordinate) const;

  Ray getPerturbedRay(const PixCoord &pixel_coordinate,
                      const float &max_perturb) const;

  Ray getDefocusPerturbedRay(const PixCoord &pixel_coordinate,
                             const float &max_perturb) const;

private:
  std::unique_ptr<RandomNumberGenerator> rng_;

  size_t image_width_;
  size_t image_height_;
  float fov_width_;
  float fov_height_;

  float image_center_x_;
  float image_center_y_;

  float focal_length_;
  float defocus_angle_{0.0f};
  float defocus_radius_;

  Point3D position_;
  math::Quaternionf rotation_;
};

#endif // _CAMERA_H_
