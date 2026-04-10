#include "cuda/rendering.h"

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
  return math::Quaternionf{R};
}

int main(int argc, char **argv) {
  cuda::utils::SetGPU();

  RandomNumberGenerator rng;

  cuda::HostHittableList host_world{};

  cuda::MaterialType const ground_material{
      cuda::MaterialType::MATERIAL_LAMBERTIAN};
  auto ground_material_prop{new cuda::LambertianProperty};
  ground_material_prop->albedo = math::Vector3f{0.5f, 0.5f, 0.5f};
  host_world.objects().spheres.emplace_back(
      Point3D{0, 0, -1000}, 1000.0f, ground_material, ground_material_prop);

  for (int a = -11; a < 11; ++a) {
    for (int b = -11; b < 11; ++b) {
      const float choose_material = rng.uniform01();
      const Point3D center{a + 0.9f * rng.uniformReal(-1.0f, 1.0f),
                           b + 0.9f * rng.uniformReal(-1.0f, 1.0f), 0.2f};

      if ((center - Point3D{4.0f, 0.2f, 0.0f}).Norm() > 0.9f) {
        if (choose_material < 0.8f) {
          //* Diffuse
          math::Vector3f const albedo{
              rng.uniformRealMatrix<3, 1>().CwiseProduct(
                  rng.uniformRealMatrix<3, 1>())};
          cuda::MaterialType const material{
              cuda::MaterialType::MATERIAL_LAMBERTIAN};
          auto material_prop{new cuda::LambertianProperty};
          material_prop->albedo = albedo;
          host_world.objects().spheres.emplace_back(center, 0.2f, material,
                                                    material_prop);
        } else if (choose_material < 0.95f) {
          //* Metal
          math::Vector3f const albedo = rng.uniformRealMatrix<3, 1>(0.5f, 1.0f);
          float const fuzz = rng.uniformReal(0.0f, 0.5f);
          cuda::MaterialType const material{cuda::MaterialType::MATERIAL_METAL};
          auto material_prop{new cuda::MetalProperty};
          material_prop->albedo = albedo;
          material_prop->fuzz = fuzz;
          host_world.objects().spheres.emplace_back(center, 0.2f, material,
                                                    material_prop);
        } else {
          //* Glass
          cuda::MaterialType const material{
              cuda::MaterialType::MATERIAL_DIELECTRIC};
          auto material_prop{new cuda::DielectricProperty};
          material_prop->refraction_index = 1.5f;
          host_world.objects().spheres.emplace_back(center, 0.2f, material,
                                                    material_prop);
        }
      }
    }
  }

  cuda::MaterialType const material_1{cuda::MaterialType::MATERIAL_DIELECTRIC};
  auto material_prop_1{new cuda::DielectricProperty};
  material_prop_1->refraction_index = 1.5f;
  host_world.objects().spheres.emplace_back(Point3D{0, 0, 1}, 1.0f, material_1,
                                            material_prop_1);

  cuda::MaterialType const material_2{cuda::MaterialType::MATERIAL_LAMBERTIAN};
  auto material_prop_2{new cuda::LambertianProperty};
  material_prop_2->albedo = math::Vector3f{0.4, 0.2f, 0.1};
  host_world.objects().spheres.emplace_back(Point3D{-4, 0, 1}, 1.0f, material_2,
                                            material_prop_2);

  cuda::MaterialType const material_3{cuda::MaterialType::MATERIAL_METAL};
  auto material_prop_3{new cuda::MetalProperty};
  material_prop_3->albedo = math::Vector3f{0.7f, 0.6f, 0.5f};
  material_prop_3->fuzz = 0.0f;
  host_world.objects().spheres.emplace_back(Point3D{4, 0, 1}, 1.0f, material_3,
                                            material_prop_3);

  cuda::DeviceHittableList device_world{};
  if (!device_world.Upload(host_world)) {
    std::cout << "Error! Failed to upload world to GPU device. " << std::endl;
  }

  const float aspect_ratio = 4.0f / 3.0f;
  const size_t image_width = 640UL;
  const size_t image_height = static_cast<size_t>(image_width / aspect_ratio);
  const float vfov = 60.0f * M_PI / 180.0f;
  const float focal_length = 1.0f;
  const float fov_height = 2.0f * std::tan(vfov / 2.0f) * focal_length;
  const float fov_width =
      fov_height * (static_cast<float>(image_width) / image_height);
  const float defocus_angle = 0.6 * M_PI / 180.0f;
  Point3D camera_position{8, 0, 2};
  Point3D lookat{0.0f, 0.0f, 0.0f};
  const math::Vector3f vup{0.0f, 0.0f, 1.0f};
  math::Quaternionf camera_rotation =
      getCameraRotation(camera_position, lookat, vup);

  cuda::Camera camera(image_width, image_height, fov_width, fov_height,
                      focal_length, defocus_angle, camera_position,
                      camera_rotation);

  cv::Mat image{image_height, image_width, CV_8UC3};
  // cv::cuda::GpuMat device_image{image_height, image_width, CV_8UC3};
  uint8_t *d__image{};
  CUDA_CHECK(cudaMalloc((void **)&d__image, image_height * image_width * 3));
  // device_image.upload(image);
  std::cout << "image_size: " << image.cols << "x" << image.rows << std::endl;

  cuda::RayTracerConfig config{};
  config.max_depth = 10;
  config.max_sample_pert = 0.5f;
  config.samples_per_pixel = 100;

  cuda::SimpleSkyBackground background{};
  // cuda::render(camera, device_world, config, background, image_width,
  //              image_height, device_image.data);

  std::string const output_video_path{argv[1]};
  float const fps{30.0f};
  float const dt{1.0f / fps};
  cv::VideoWriter writer{};
  writer.open(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
              fps, cv::Size(image_width, image_height), true);
  if (!writer.isOpened()) {
    std::cout << "Failed to open " << output_video_path << std::endl;
    return -1;
  }

  float const angv{M_PI / 4.0f};
  float curr_t{0.0};
  float const traj_radius{10.0f};
  float const traj_z{5.0f};
  float const video_duration{std::atof(argv[2])};

  const int bar_length = 100;

  while (curr_t <= video_duration) {
    float const theta{curr_t * angv};
    camera_position.x() = traj_radius * cosf(theta);
    camera_position.y() = traj_radius * sinf(theta);
    camera_position.z() = traj_z;
    camera_rotation = getCameraRotation(camera_position, lookat, vup);
    camera.setPosition(camera_position);
    camera.setRotation(camera_rotation);
    cuda::render(camera, device_world, config, background, image_width,
                 image_height, d__image);
    // cuda::render2(camera, device_world, config, background, image_width,
    //              image_height, d__image);

    CUDA_CHECK(cudaMemcpy(image.data, d__image, image_height * image_width * 3,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    const float progress = curr_t / video_duration;
    std::clog << "\rRendering: [";
    for (int i = 0; i < progress * bar_length; ++i) {
      std::clog << "*";
    }
    for (int i = progress * bar_length; i < bar_length; ++i) {
      std::clog << " ";
    }
    std::clog << "] " << int(std::ceil(100 * progress)) << "%" << std::flush;

    cv::imshow("Rendered Image", image);
    cv::waitKey(1);

    writer.write(image);
    curr_t += dt;
  }

  host_world.Release();
  device_world.Release();

  writer.release();

  CUDA_CHECK(cudaFree(d__image));

  return 0;
}