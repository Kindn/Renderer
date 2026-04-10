#include "Camera.h"
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
  MaterialBase::Ptr material_left =
      std::make_shared<Metal>(math::Vector3f{0.8, 0.8, 0.8}, 0.3);
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
      std::make_shared<Sphere>(Point3D{3.0f, 0, 2.0f}, 1.5f, material_right));

  const float aspect_ratio = 16.0f / 9.0f;
  const size_t image_width = 400;
  const size_t image_height = static_cast<size_t>(image_width / aspect_ratio);
  const float fov_height = 2.0f;
  const float fov_width =
      fov_height * (static_cast<float>(image_width) / image_height);
  const float focal_length = 1.0f;
  const Point3D camera_position(0, 0, -focal_length);

  const Camera camera(image_width, image_height, fov_width, fov_height,
                      focal_length, 0.0f, camera_position);

  RayTracer::Config renderer_config;
  renderer_config.background = background;
  renderer_config.samples_per_pixel = 20;
  RayTracer renderer(renderer_config);
  renderer.render(camera, world, std::cout);

  return 0;
}
