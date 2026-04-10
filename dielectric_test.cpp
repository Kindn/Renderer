#include "Camera.h"
#include "Dielectric.h"
#include "HittableList.h"
#include "Lambertian.h"
#include "Metal.h"
#include "RayTracer.h"
#include "SimpleSkyBackground.h"
#include "Sphere.h"

int main(int argc, char **argv) {
  HittableList world;

  MaterialBase::Ptr material_ground =
      std::make_shared<Lambertian>(math::Vector3f{0.8, 0.8, 0.0f});
  MaterialBase::Ptr material_center =
      std::make_shared<Lambertian>(math::Vector3f{0.1, 0.2f, 0.5f});
  MaterialBase::Ptr material_left = std::make_shared<Dielectric>(1.50);
  MaterialBase::Ptr material_bubble = std::make_shared<Dielectric>(1.00 / 1.50);
  MaterialBase::Ptr material_right =
      std::make_shared<Metal>(math::Vector3f{0.8, 0.6, 0.2f}, 1.0f);

  BackgroundBase::Ptr background = std::make_shared<SimpleSkyBackground>();

  world.add(
      std::make_shared<Sphere>(Point3D{0.0f, 0.0f, 2.7}, 1.5f, material_center));
  world.add(
      std::make_shared<Sphere>(Point3D{0.0f, 101.5f, 2.0f}, 100, material_ground));
  world.add(
      std::make_shared<Sphere>(Point3D{-3.0f, 0, 2.0f}, 1.5f, material_left));
  world.add(
      std::make_shared<Sphere>(Point3D{-3.0f, 0, 2.0f}, 1.2f, material_bubble));
  world.add(
      std::make_shared<Sphere>(Point3D{3.0f, 0, 2.0f}, 1.5f, material_right));

  const float aspect_ratio = 16.0f / 9.0f;
  const size_t image_width = 400;
  const size_t image_height = static_cast<size_t>(image_width / aspect_ratio);
  const float fov_height = 0.5f;
  const float fov_width =
      fov_height * (static_cast<float>(image_width) / image_height);
  const float focal_length = 1.0f;
  const float defocus_angle = 10.0f * M_PI / 180.0f;
  const Point3D camera_position(-4, -4, -2);
  const math::Vector3f camera_rotation_rpy{-std::atan2(1.0f, std::sqrt(2.0f)),
                                            M_PI / 4.0f, 0.0f};
  const math::Quaternionf camera_rotation =
      math::Quaternionf{std::cos(camera_rotation_rpy.z() / 2.0f), 0, 0,
                         std::sin(camera_rotation_rpy.z() / 2.0f)} *
      math::Quaternionf{std::cos(camera_rotation_rpy.y() / 2.0f), 0,
                         std::sin(camera_rotation_rpy.y() / 2.0f), 0} *
      math::Quaternionf{std::cos(camera_rotation_rpy.x() / 2.0f),
                         std::sin(camera_rotation_rpy.x() / 2.0f), 0, 0};

  const Camera camera(image_width, image_height, fov_width, fov_height,
                      focal_length, defocus_angle, camera_position,
                      camera_rotation);

  RayTracer::Config renderer_config;
  renderer_config.background = background;
  renderer_config.samples_per_pixel = 20;
  RayTracer renderer(renderer_config);
  renderer.render(camera, world, std::cout);

  return 0;
}
