/*
 * filename: Camera.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "Camera.h"

Camera::Camera(const size_t &image_width, const size_t &image_height,
               const float &fov_width, const float &fov_height,
               const float &focal_length, const float &defocus_angle,
               const Point3D &position, const math::Quaternionf &rotation,
               const std::uint_fast32_t &rng_seed) {
  setImageWidth(image_width);
  setImageHeight(image_height);
  setFOVWidth(fov_width);
  setFOVHeight(fov_height);
  setFocalLength(focal_length);
  setDefocusAngle(defocus_angle);
  setPosition(position);
  setRotation(rotation);
  image_center_x_ = static_cast<float>(image_width_) / 2;
  image_center_y_ = static_cast<float>(image_height) / 2;
  rng_ = std::make_unique<RandomNumberGenerator>(rng_seed);
}

Point3D
Camera::getPixelCameraCoordinate(const PixCoord &pixel_coordinate) const {
  const Point2D top_left = Point2D(-fov_width_ / 2, -fov_height_ / 2);
  return Point3D(top_left.x() + (pixel_coordinate.x() + 0.5f) * getPixelSizeX(),
                 top_left.y() + (pixel_coordinate.y() + 0.5f) * getPixelSizeY(),
                 focal_length_);
}

Point3D
Camera::getPerturbedPixelCameraCoordinate(const PixCoord &pixel_coordinate,
                                          const float &max_perturb) const {
  const float a = std::abs(max_perturb);
  const float pert_x = rng_->uniformReal(-a, a);
  const float pert_y = rng_->uniformReal(-a, a);
  const Point2D top_left = Point2D(-fov_width_ / 2, -fov_height_ / 2);
  return Point3D(
      top_left.x() + (pixel_coordinate.x() + pert_x) * getPixelSizeX(),
      top_left.y() + (pixel_coordinate.y() + pert_y) * getPixelSizeY(),
      focal_length_);
}

Point3D Camera::defocusDiskSample() const {
  const Point2D xy_c = rng_->uniformPoint2DInUnitCircle() * defocus_radius_;
  return position_ + rotation_ * math::Vector3f(xy_c.x(), xy_c.y(), 0.0f);
}

Ray Camera::getRay(const Point3D &camera_coordinate) const {
  const math::Vector3f direction = rotation_ * camera_coordinate.Normalized();
  return Ray(position_, direction);
}

Ray Camera::getRay(const PixCoord &pixel_coordinate) const {
  return getRay(getPixelCameraCoordinate(pixel_coordinate));
}

Ray Camera::getPerturbedRay(const PixCoord &pixel_coordinate,
                            const float &max_perturb) const {
  return getRay(
      getPerturbedPixelCameraCoordinate(pixel_coordinate, max_perturb));
}

Ray Camera::getDefocusPerturbedRay(const PixCoord &pixel_coordinate,
                                   const float &max_perturb) const {
  const Point3D pix_c =
      getPerturbedPixelCameraCoordinate(pixel_coordinate, max_perturb);
  const Point3D origin = defocusDiskSample();
  const math::Vector3f direction = position_ + rotation_ * pix_c - origin;
  return Ray(origin, direction);
}
