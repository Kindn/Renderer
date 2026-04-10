#include "Camera.h"
#include "Dielectric.h"
#include "HittableList.h"
#include "Lambertian.h"
#include "Metal.h"
#include "RayTracer.h"
#include "SimpleSkyBackground.h"
#include "Sphere.h"

#include <fstream>

math::Quaternionf getCameraRotation(const math::Vector3f &lookfrom,
                                    const math::Vector3f &lookat,
                                    const math::Vector3f &vup) {
  math::Matrix3f R;
  auto const R_col_2 = (lookat - lookfrom).Normalized();
  auto const R_col_0 = -(vup.Cross(R_col_2)).Normalized();
  auto const R_col_1 = R_col_2.Cross(R_col_0);
  R(0, 0) = R_col_0.x();
  R(1, 0) = R_col_0.y();
  R(2, 0) = R_col_0.z();
  R(0, 1) = R_col_1.x();
  R(1, 1) = R_col_1.y();
  R(2, 1) = R_col_1.z();
  R(0, 2) = R_col_2.x();
  R(1, 2) = R_col_2.y();
  R(2, 2) = R_col_2.z();
  std::cout << "cam R: \n" << R << std::endl;
  std::cout << "check: \n" << math::Quaternionf{R}.ToRotationMatrix() << std::endl;
  return math::Quaternionf{R};
}

int main(int argc, char **argv) {
  RandomNumberGenerator rng;

  HittableList world;

  const MaterialBase::Ptr ground_material =
      std::make_shared<Lambertian>(math::Vector3f{0.5f, 0.5f, 0.5f});
  world.add(
      std::make_shared<Sphere>(Point3D{0, 0, -1000}, 1000, ground_material));

  for (int a = -11; a < 11; ++a) {
    for (int b = -11; b < 11; ++b) {
      const float choose_material = rng.uniform01();
      const Point3D center{a + 0.9f * rng.uniformReal(-1.0f, 1.0f),
                           b + 0.9f * rng.uniformReal(-1.0f, 1.0f), 0.2f};

      if ((center - Point3D{4.0f, 0.2f, 0.0f}).Norm() > 0.9f) {
        MaterialBase::Ptr material;
        if (choose_material < 0.8f) {
          //* Diffuse
          const math::Vector3f albedo =
              rng.uniformRealMatrix<3, 1>().CwiseProduct(
                  rng.uniformRealMatrix<3, 1>());
          std::cout << "lambertian, albedo: " << albedo << std::endl;
          material = std::make_shared<Lambertian>(albedo);
          world.add(std::make_shared<Sphere>(center, 0.2f, material));
        } else if (choose_material < 0.95f) {
          //* Metal
          const math::Vector3f albedo = rng.uniformRealMatrix<3, 1>(0.5f, 1.0f);
          std::cout << "metal, albedo: " << albedo << std::endl;
          const float fuzz = rng.uniformReal(0.0f, 0.5f);
          material = std::make_shared<Metal>(albedo, fuzz);
          world.add(std::make_shared<Sphere>(center, 0.2f, material));
        } else {
          //* Glass
          material = std::make_shared<Dielectric>(1.5f);
          world.add(std::make_shared<Sphere>(center, 0.2f, material));
        }
      }
    }
  }

  const MaterialBase::Ptr material1 = std::make_shared<Dielectric>(1.5f);
  world.add(std::make_shared<Sphere>(Point3D{0, 0, 1}, 1.0f, material1));

  const MaterialBase::Ptr material2 =
      std::make_shared<Lambertian>(math::Vector3f{0.4, 0.2f, 0.1});
  world.add(std::make_shared<Sphere>(Point3D{-4, 0, 1}, 1.0f, material2));

  const MaterialBase::Ptr material3 =
      std::make_shared<Metal>(math::Vector3f{0.7, 0.6, 0.5f}, 0.0f);
  world.add(std::make_shared<Sphere>(Point3D{4, 0, 1}, 1.0f, material3));

  const float aspect_ratio = 3.0f / 4.0f;
  const size_t image_width = 320UL;
  const size_t image_height = static_cast<size_t>(image_width / aspect_ratio);
  const float vfov = 60.0f * M_PI / 180.0f;
  const float focal_length = 1.0f;
  const float fov_height = 2.0f * std::tan(vfov / 2.0f) * focal_length;
  const float fov_width =
      fov_height * (static_cast<float>(image_width) / image_height);
  const float defocus_angle = 0.6 * M_PI / 180.0f;
  const Point3D camera_position{8, 0, 2};
  const Point3D lookat{0.0f, 0.0f, 0.0f};
  const math::Vector3f vup{0.0f, 0.0f, 1.0f};
  const math::Quaternionf camera_rotation =
      getCameraRotation(camera_position, lookat, vup);

  Camera camera(image_width, image_height, fov_width, fov_height, focal_length,
                defocus_angle, camera_position, camera_rotation);

  RayTracer::Config renderer_config;
  renderer_config.background = std::make_shared<SimpleSkyBackground>();
  renderer_config.samples_per_pixel = 500;
  renderer_config.max_depth = 50;
  renderer_config.max_sample_pert = 0.5f;
  RayTracer renderer(renderer_config);
  const size_t max_num_threads = 8UL;
  std::string const output_path{argv[1]};
  std::ofstream ofs{output_path};
  if (!ofs.is_open()) {
    std::cout << "Error! Cannot open output path " << output_path << std::endl;
  }
  TicToc tictoc;
  tictoc.tic();
  renderer.renderMultiThread(camera, world, max_num_threads, ofs);
  const float rendering_time = tictoc.toc();
  std::clog << "Rendering time: " << int(rendering_time) / 3600 << ":"
            << (int(rendering_time) % 3600) / 60 << ":"
            << (rendering_time - int(rendering_time) / 60 * 60) << "."
            << std::endl;

  ofs.close();

  return 0;
}