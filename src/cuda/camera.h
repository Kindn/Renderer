#pragma once

#include "Ray.h"
#include "cuda/utils/random.h"

namespace cuda {

class Camera {
 public:
  HOST_DEVICE_FUNC
  Camera(const size_t &image_width = 640, const size_t &image_height = 480,
         const float &fov_width = 1.2f, const float &fov_height = 0.9,
         const float &focal_length = 1.0f, const float &defocus_angle = 0.0f,
         const math::Vector3f &position = math::Vector3f::Zero(),
         const math::Quaternionf &rotation = math::Quaternionf::Identity());

 public:
  HOST_DEVICE_FUNC const size_t &getImageWidth() const { return image_width_; }

  HOST_DEVICE_FUNC const size_t &getImageHeight() const {
    return image_height_;
  }

  HOST_DEVICE_FUNC const float &getFOVWidth() const { return fov_width_; }

  HOST_DEVICE_FUNC const float &getFOVHeight() const { return fov_height_; }

  HOST_DEVICE_FUNC const float &getImageCenterX() const {
    return image_center_x_;
  }

  HOST_DEVICE_FUNC const float &getImageCenterY() const {
    return image_center_y_;
  }

  HOST_DEVICE_FUNC const float &getFocalLength() const {
    return focal_length_;
  }

  HOST_DEVICE_FUNC const float &getDefocusAngle() const {
    return defocus_angle_;
  }

  HOST_DEVICE_FUNC const math::Vector3f &getPosition() const { return position_; }

  HOST_DEVICE_FUNC const math::Quaternionf &getRotation() const {
    return rotation_;
  }

  HOST_DEVICE_FUNC float getPixelSizeX() const {
    return fov_width_ / image_width_;
  }

  HOST_DEVICE_FUNC float getPixelSizeY() const {
    return fov_height_ / image_height_;
  }

  HOST_DEVICE_FUNC void setImageWidth(const size_t &image_width) {
    // assert(image_width > 0);
    image_width_ = image_width;
  }

  HOST_DEVICE_FUNC void setImageHeight(const size_t &image_height) {
    // assert(image_height > 0);
    image_height_ = image_height;
  }

  HOST_DEVICE_FUNC void setFOVWidth(const float &fov_width) {
    // assert(fov_width > 1.0e-32);
    fov_width_ = fov_width;
  }

  HOST_DEVICE_FUNC void setFOVHeight(const float &fov_height) {
    // assert(fov_height > 1.0e-32);
    fov_height_ = fov_height;
  }

  HOST_DEVICE_FUNC void setImageCenterX(const float &image_center_x) {
    image_center_x_ = image_center_x;
  }

  HOST_DEVICE_FUNC void setImageCenterY(const float &image_center_y) {
    image_center_y_ = image_center_y;
  }

  HOST_DEVICE_FUNC void setFocalLength(const float &focal_length) {
    // assert(focal_length > 1.0e-32);
    focal_length_ = focal_length;
    defocus_radius_ = focal_length * std::tan(defocus_angle_ / 2.0f);
  }

  HOST_DEVICE_FUNC void setDefocusAngle(const float &defocus_angle) {
    // assert(defocus_angle > 1.0e-32 &&
          //  defocus_angle < M_PI);
    defocus_angle_ = defocus_angle;
    defocus_radius_ = focal_length_ * std::tan(defocus_angle / 2.0f);
  }

  HOST_DEVICE_FUNC void setPosition(const math::Vector3f &position) {
    position_ = position;
  }

  HOST_DEVICE_FUNC void setRotation(const math::Quaternionf &rotation) {
    rotation_ = rotation;
  }

  HOST_DEVICE_FUNC math::Vector3f
  getPixelCameraCoordinate(const PixCoord &pixel_coordinate) const;

  HOST_DEVICE_FUNC math::Vector3f getPerturbedPixelCameraCoordinate(
      const PixCoord &pixel_coordinate, const float &max_pert,
      utils::RandomNumberGenerator *const rng) const;

  HOST_DEVICE_FUNC math::Vector3f
  defocusDiskSample(utils::RandomNumberGenerator *const rng) const;

  HOST_DEVICE_FUNC Ray getRay(const math::Vector3f &camera_coordinate) const;

  HOST_DEVICE_FUNC Ray getRay(const PixCoord &pixel_coordinate) const;

  HOST_DEVICE_FUNC Ray
  getPerturbedRay(const PixCoord &pixel_coordinate, const float &max_perturb,
                  utils::RandomNumberGenerator *const rng) const;

  HOST_DEVICE_FUNC Ray getDefocusPerturbedRay(
      const PixCoord &pixel_coordinate, const float &max_perturb,
      utils::RandomNumberGenerator *const rng) const;

 private:
  size_t image_width_;
  size_t image_height_;
  float fov_width_;
  float fov_height_;

  float image_center_x_;
  float image_center_y_;

  float focal_length_;
  float defocus_angle_{0.0f};
  float defocus_radius_;

  math::Vector3f position_;
  math::Quaternionf rotation_;
};

}  // namespace cuda
