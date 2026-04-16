#include "cuda/bhlite.h"
#include "cuda/rendering.h"

#include <fstream>
#include <iomanip>
#include <time.h>

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
  // std::cout << "cam R: \n" << R << std::endl;
  // std::cout << "check: \n"
  //           << math::Quaternionf{R}.ToRotationMatrix() << std::endl;
  return math::Quaternionf{R};
}

cv::Mat cv_background_img{};
cv::Mat cv_background_img_f{};
std::vector<cv::Mat> bg_tex{};
cuda::HdriSkyBackground::HostDataPack
LoadBackgroundTexture(std::string const &path, bool const hdr = false) {
  if (hdr) {
    cv_background_img = cv::imread(path, cv::IMREAD_ANYDEPTH);
  } else {
    cv_background_img = cv::imread(path, cv::IMREAD_COLOR);
  }
  cv::cvtColor(cv_background_img, cv_background_img, cv::COLOR_BGR2RGB);
  cv_background_img.convertTo(cv_background_img_f, CV_32FC3, 1.0 / 255.0, 0.0);
  cv::split(cv_background_img_f, bg_tex);

  cuda::HdriSkyBackground::HostDataPack data{};
  data.h__tex_r = (float *)bg_tex[0].data;
  data.h__tex_g = (float *)bg_tex[1].data;
  data.h__tex_b = (float *)bg_tex[2].data;
  data.rows = cv_background_img.rows;
  data.cols = cv_background_img.cols;

  return data;
}

cv::Mat cv_accretion_disk_img{};
cv::Mat cv_accretion_disk_img_f{};
cuda::BlackHole::HostAccretionDiskTexture
LoadAccretionDiskTexture(std::string const &path) {
  cv_accretion_disk_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
  cv_accretion_disk_img.convertTo(cv_accretion_disk_img_f, CV_32F, 1.0 / 255.0,
                                  0.0);

  if (cv_accretion_disk_img.empty() || cv_accretion_disk_img_f.empty()) {
    std::cout << "Error! Cannot load accretion disk texture from " << path
              << std::endl;
  }

  cuda::BlackHole::HostAccretionDiskTexture data{};
  data.h__texture = (float *)cv_accretion_disk_img_f.data;
  data.rows = cv_accretion_disk_img.rows;
  data.cols = cv_accretion_disk_img.cols;

  return data;
}

int main(int argc, char **argv) {
  bool render_video{false};
  for (int i{0}; i < argc; ++i) {
    if ("-v" == std::string(argv[i])) {
      render_video = true;
      break;
    }
  }

  int device_id{};
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
  std::cout << "GPU " << device_id << ", "
            << "num SMs: " << props.multiProcessorCount << std::endl;

  RandomNumberGenerator rng;

  cuda::HostSchwarzschildSpace host_world{};

  cuda::BlackHole::Config blackhole_cfg{};
  blackhole_cfg.position.SetZero();
  blackhole_cfg.rotation.SetIdentity();
  blackhole_cfg.sr = 1.0f;
  blackhole_cfg.inner_disk_radius = 1.8f;
  blackhole_cfg.outer_disk_radius = 12.0f;
  blackhole_cfg.disk_attenuation = {169.0f / 255.0f, 169.0f / 255.0f,
                                    169.0f / 255.0f};
  blackhole_cfg.disk_emission = {255.0f / 255.0f, 165.0f / 255.0f,
                                 153.0f / 255.0f};
  blackhole_cfg.disk_thickness = 0.01f;
  blackhole_cfg.max_eff_range = 20.0f;
  blackhole_cfg.brightness_scale =
      20.0f * std::pow(blackhole_cfg.inner_disk_radius, 3.0f);
  std::string const accretion_disk_texture_path{argv[1]};
  auto accretion_disk_texture{
      LoadAccretionDiskTexture(accretion_disk_texture_path)};
  accretion_disk_texture.type =
      cuda::BlackHole::AccretionDiskTextureType::PARTICLE_DENSITY;
  host_world.objects().black_holes.emplace_back(blackhole_cfg,
                                                accretion_disk_texture);

  cuda::DeviceSchwarzschildSpace::Config device_world_cfg{};
  device_world_cfg.max_marching_time = 30.0f;
  cuda::DeviceSchwarzschildSpace device_world{device_world_cfg};
  if (!device_world.Upload(host_world)) {
    std::cout << "Error! Failed to upload world to GPU device. " << std::endl;
  }

  const float aspect_ratio = 16.0f / 8.0f;
  const size_t image_width = 2160UL;
  const size_t image_height = static_cast<size_t>(image_width / aspect_ratio);
  const float vfov = 45.0f * M_PIf32 / 180.0f;
  const float focal_length = 1.0f;
  const float fov_height = 2.0f * std::tan(vfov / 2.0f) * focal_length;
  const float fov_width =
      fov_height * (static_cast<float>(image_width) / image_height);
  const float defocus_angle = 0.0f;
  Point3D camera_position{8.0f, 0.0f, 1.6f};
  Point3D lookat{0.0f, 0.0f, 0.0f};
  math::Vector3f vup{0.0f, -1.0f, 2.0f};
  // Point3D camera_position{3.0f, 3.0f, 0.2f};
  // Point3D lookat{0.0f, 2.6f, 0.0f};
  // math::Vector3f vup{0.0f, 0.0f, 1.0f};
  // Point3D camera_position{8.0f, 0.0f, 0.1f};
  // Point3D lookat{0.0f, 0.0f, 0.0f};
  // math::Vector3f vup{0.0f, -0.5f, 1.0f};
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
  config.max_depth = 3;
  config.max_sample_pert = 0.5f;
  config.samples_per_pixel = 4;

  // cuda::SimpleSkyBackground background{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f,
  // 0.1f}};
  std::string const background_texture_path{argv[2]};
  auto const background_data{
      LoadBackgroundTexture(background_texture_path, false)};
  cuda::HdriSkyBackground background{background_data};

  cuda::Bloom::Config bloom_cfg{};
  bloom_cfg.image_height = image_height;
  bloom_cfg.image_width = image_width;
  bloom_cfg.kernal_size = 5;
  bloom_cfg.sigma = 1.0f;
  bloom_cfg.bloom_weight = 0.07f;
  bloom_cfg.num_mip_levels = 5U;
  bloom_cfg.brightness_threshold = 1.0f;
  auto bloom{std::make_shared<cuda::Bloom>(bloom_cfg)};

  if (!render_video) {
      device_world.SetCameraPrior(
          host_world.GetCameraPrior(camera.getPosition(),
          camera.getRotation()));
      // cuda::render(camera, device_world, config, background, image_width,
      //              image_height, d__image);
      cuda::render2(camera, host_world, device_world, config, background,
                    image_width, image_height, bloom, d__image);
      // cuda::bhlite::render(camera, background, image_width, image_height,
      // d__image);

      // device_image.download(image);
      CUDA_CHECK(cudaMemcpy(image.data, d__image, image_height * image_width
      * 3,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());

      std::string const output_image_path{argv[3]};
      cv::imwrite(output_image_path, image);
      cv::imshow("Rendered Image", image);
      cv::waitKey();
  } else {
    std::string const output_video_path{argv[3]};
    float const fps{30.0f};
    float const dt{1.0f / fps};
    cv::VideoWriter writer{};
    writer.open(output_video_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                fps, cv::Size(image_width, image_height), true);
    if (!writer.isOpened()) {
      std::cout << "Failed to open " << output_video_path << std::endl;
      return -1;
    }

    //* Circular trajectory
    float const angv{M_PI / 6.0f};
    float curr_t{0.0f};
    float const video_duration{12.0f};
    camera_position = {8.0f, 0.0f, 0.1f};
    math::Vector3f const r_vec{(camera_position - blackhole_cfg.position)};
    float const traj_radius{r_vec.Norm()};
    math::Vector3f const v0{0.0f, angv * traj_radius, 0.0f};
    math::Vector3f const traj_axis{r_vec.Cross(v0).Normalized()};
    math::Vector3f const angv_vec{angv * traj_axis};
    vup = math::Vector3f::UnitZ();

    //* Straight line trajectory
    // float const trans_v{-3.0f};
    // float curr_t{0.0};
    // float const video_duration{10.0f};
    // camera_position = {15.0f, 3.5f, 0.2f};

    //* Newtonian orbit for a single blackhole located at (0, 0, 0)
    // float curr_t{0.0};
    // camera_position = {8.0f, 0.0f, 0.1f};
    // float const r{camera_position.Norm()};
    // lookat = {0.0f, 0.0f, 0.0f};

    // math::Vector3f v{-3.0f, 0.0f, -3.0f};
    // vup = camera_position.Cross(v).Normalized();
    // float const v_norm_inf{0.7f};
    // float const E{0.5 * v_norm_inf * v_norm_inf};
    // float const v_norm{std::sqrt(4.0f * E + 2.0f / camera_position.Norm())};
    // float const video_duration{25.0f};
    // v = v.Normalized() * v_norm;
    // math::Vector6f x{};
    // x(0) = camera_position.x();
    // x(1) = camera_position.y();
    // x(2) = camera_position.z();
    // x(3) = v.x();
    // x(4) = v.y();
    // x(5) = v.z();

    camera_rotation = getCameraRotation(camera_position, lookat, vup);
    camera.setPosition(camera_position);
    camera.setRotation(camera_rotation);

    auto const functor{[](float const t, math::Vector6f const &x) {
      float const r_norm{hypotf(x(0), hypotf(x(1), x(2)))};
      float const inv_r_norm3{1.0f / (r_norm * r_norm * r_norm)};
      float const half_gm{20.0f};
      math::Vector6f dx{};
      dx(0) = x(3);
      dx(1) = x(4);
      dx(2) = x(5);
      dx(3) = -half_gm * inv_r_norm3 * x(0);
      dx(4) = -half_gm * inv_r_norm3 * x(1);
      dx(5) = -half_gm * inv_r_norm3 * x(2);

      return dx;
    }};
    math::Rk4OdeSolver<float, 6U, decltype(functor)> ode{};

    const int bar_length = 100;

    float time_cost{0.0f};

    // float curr_t{0.0};
    while (curr_t <= video_duration) {
      clock_t const start{clock()};
      const float progress = curr_t / video_duration;
      std::clog << "\rRendering: [";
      for (int i = 0; i < progress * bar_length; ++i) {
        std::clog << "*";
      }
      for (int i = progress * bar_length; i < bar_length; ++i) {
        std::clog << " ";
      }
      std::clog << "] " << int(std::ceil(100 * progress)) << "%, ";
      float time_remain{-1.0f};
      if (curr_t > 0.0f) {
        time_remain = time_cost * (video_duration - curr_t) / curr_t;
        int32_t const hours{static_cast<int32_t>(time_remain / 3600.0f)};
        int32_t const minutes{
            static_cast<int32_t>((time_remain - hours * 3600.0f) / 60.0f)};
        int32_t const seconds{static_cast<int32_t>(
            std::round(time_remain - hours * 3600.0f - minutes * 60.0f))};
        std::clog << "time remain: " << hours << ":" << minutes << ":"
                  << seconds << std::flush;
      } else {
        std::clog << "time remain: "
                  << "--:--:--" << std::flush;
      }

      //* Circular trajectory
      float const theta{curr_t * angv};
      float const half_theta{0.5f * theta};
      // camera_position.x() = traj_radius * sinf(theta);
      // camera_position.y() = 0.0f;
      // camera_position.z() = traj_radius * cosf(theta);
      float const sin_half_theta{std::sin(half_theta)};
      float const cos_half_theta{std::cos(half_theta)};
      math::Quaternionf const q{cos_half_theta, sin_half_theta * traj_axis.x(),
                                sin_half_theta * traj_axis.y(),
                                sin_half_theta * traj_axis.z()};
      camera_position = blackhole_cfg.position + q * r_vec;
      // if (camera_position.z() > 0.1f) {
      //   vup = -math::Vector3f::UnitX();
      // } else if (camera_position.z() < -0.1f) {
      //   vup = math::Vector3f::UnitX();
      // } else {
      //   vup = math::Vector3f::UnitZ();
      // }

      //* Straight line trajectory
      // camera_position.x() += trans_v * dt;
      // vup =   vup = math::Vector3f::UnitZ();

      //* Newtonian orbit
      // math::Vector6f dx{};
      // ode.Step(functor, x, curr_t, dt, dx);
      // x += dx;
      // camera_position.x() = x(0);
      // camera_position.y() = x(1);
      // camera_position.z() = x(2);

      camera_rotation = getCameraRotation(camera_position, lookat, vup);
      camera.setPosition(camera_position);
      camera.setRotation(camera_rotation);
      // std::cout << "position: [" << camera_position.Transpose() << "], "
      //           << "rotation: [" << camera_rotation << "]" << std::endl;
      device_world.SetCameraPrior(host_world.GetCameraPrior(
          camera.getPosition(), camera.getRotation()));
      // cuda::render(camera, device_world, config, background, image_width,
      //              image_height, d__image);
      cuda::render2(camera, host_world, device_world, config, background,
                    image_width, image_height, bloom, d__image);
      // cuda::bhlite::render(camera, background, image_width, image_height,
      //                      d__image);

      CUDA_CHECK(cudaMemcpy(image.data, d__image,
                            image_height * image_width * 3,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());

      cv::imshow("Rendered Image", image);
      cv::waitKey(10);

      writer.write(image);
      curr_t += dt;
      clock_t const end{clock()};
      time_cost += double(end - start) / CLOCKS_PER_SEC;
    }

    writer.release();
  }

  host_world.Release();
  device_world.Release();
  background.Release();

  CUDA_CHECK(cudaFree(d__image));

  return 0;
}