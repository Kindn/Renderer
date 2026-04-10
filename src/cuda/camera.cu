#include "cuda/camera.h"

namespace cuda {

HOST_DEVICE_FUNC Camera::Camera(
    const size_t &image_width, const size_t &image_height,
    const float &fov_width, const float &fov_height,
    const float &focal_length, const float &defocus_angle,
    const math::Vector3f &position, const math::Quaternionf &rotation) {
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
}

HOST_DEVICE_FUNC math::Vector3f
Camera::getPixelCameraCoordinate(const PixCoord &pixel_coordinate) const {
  const Point2D top_left = Point2D(-fov_width_ / 2, -fov_height_ / 2);
  return math::Vector3f(top_left.x() + (pixel_coordinate.x() + 0.5f) * getPixelSizeX(),
                 top_left.y() + (pixel_coordinate.y() + 0.5f) * getPixelSizeY(),
                 focal_length_);
}

HOST_DEVICE_FUNC math::Vector3f Camera::getPerturbedPixelCameraCoordinate(
    const PixCoord &pixel_coordinate, const float &max_perturb,
    utils::RandomNumberGenerator *const rng) const {
  const float a = abs(max_perturb);
  const float pert_x = rng->uniformReal(-a, a);
  const float pert_y = rng->uniformReal(-a, a);
  // float const pert_x{0.0f};
  // float const pert_y{0.0f};
  const Point2D top_left = Point2D(-fov_width_ / 2, -fov_height_ / 2);
  return math::Vector3f(
      top_left.x() + (pixel_coordinate.x() + pert_x) * getPixelSizeX(),
      top_left.y() + (pixel_coordinate.y() + pert_y) * getPixelSizeY(),
      focal_length_);
}

HOST_DEVICE_FUNC math::Vector3f
Camera::defocusDiskSample(utils::RandomNumberGenerator *const rng) const {
  const Point2D xy_c = rng->uniformPoint2DInUnitCircle() * defocus_radius_;
  // const Point2D xy_c{0.0f, 0.0f};
  return position_ + rotation_ * math::Vector3f(xy_c.x(), xy_c.y(), 0.0f);
}

HOST_DEVICE_FUNC Ray Camera::getRay(const math::Vector3f &camera_coordinate) const {
  const math::Vector3f direction = rotation_ * camera_coordinate.Normalized();
  return Ray(position_, direction);
}

HOST_DEVICE_FUNC Ray Camera::getRay(const PixCoord &pixel_coordinate) const {
  return getRay(getPixelCameraCoordinate(pixel_coordinate));
}

HOST_DEVICE_FUNC Ray Camera::getPerturbedRay(
    const PixCoord &pixel_coordinate, const float &max_perturb,
    utils::RandomNumberGenerator *const rng) const {
  return getRay(
      getPerturbedPixelCameraCoordinate(pixel_coordinate, max_perturb, rng));
}

HOST_DEVICE_FUNC Ray Camera::getDefocusPerturbedRay(
    const PixCoord &pixel_coordinate, const float &max_perturb,
    utils::RandomNumberGenerator *const rng) const {
  const math::Vector3f pix_c =
      getPerturbedPixelCameraCoordinate(pixel_coordinate, max_perturb, rng);
  const math::Vector3f origin = defocusDiskSample(rng);
  const math::Vector3f direction = position_ + rotation_ * pix_c - origin;
  return Ray(origin, direction);
}

}  // namespace cuda