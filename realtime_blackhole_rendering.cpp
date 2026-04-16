#include "cuda/bhlite.h"

#include <opencv2/opencv.hpp>

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
cudaTextureObject_t d__disk_tex{};
void LoadAccretionDiskTexture(std::string const &path) {
  cv_accretion_disk_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
  cv_accretion_disk_img.convertTo(cv_accretion_disk_img_f, CV_32F, 1.0 / 255.0,
                                  0.0);

  if (cv_accretion_disk_img.empty() || cv_accretion_disk_img_f.empty()) {
    std::cout << "Error! Cannot load accretion disk texture from " << path
              << std::endl;
  }

  cudaArray_t d__tex_arr{};
  cudaChannelFormatDesc const ch_desc{cudaCreateChannelDesc(
      sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat)};
  CUDA_CHECK(cudaMallocArray(&d__tex_arr, &ch_desc, cv_accretion_disk_img.cols,
                             cv_accretion_disk_img.rows));
  CUDA_CHECK(
      cudaMemcpy2DToArray(d__tex_arr, 0UL, 0UL, cv_accretion_disk_img_f.data,
                          cv_accretion_disk_img.cols * sizeof(float),
                          cv_accretion_disk_img.cols * sizeof(float),
                          cv_accretion_disk_img.rows, cudaMemcpyHostToDevice));
  cudaResourceDesc res_desc{};
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = d__tex_arr;
  cudaTextureDesc tex_desc{};
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = 1;
  cudaCreateTextureObject(&d__disk_tex, &res_desc, &tex_desc, NULL);
}

math::Quaternionf GetRotation(math::Vector3f const &axis, float const &angle) {
  float const half_angle{0.5f * angle};
  math::Vector3f const qv{axis.Normalized() * std::sin(half_angle)};
  return math::Quaternionf{std::cos(half_angle), qv.x(), qv.y(), qv.z()};
}

int last_x{0}, last_y{0};
bool is_controlling{false};
bool is_rotating{false};
bool is_moving_lat{false};
bool is_moving_lon{false};
void MouseControlCallback(int event, int x, int y, int flags, void *userdata) {
  cuda::Camera *const camera{(cuda::Camera *)userdata};
  int const dx{x - last_x};
  int const dy{y - last_y};

  if (cv::EVENT_LBUTTONDOWN == event) {
    last_x = x;
    last_y = y;
    is_rotating = true;
  }
  if (cv::EVENT_LBUTTONUP == event) {
    is_rotating = false;
    return;
  }

  if (cv::EVENT_MBUTTONDOWN == event) {
    last_x = x;
    last_y = y;
    is_moving_lat = true;
  }
  if (cv::EVENT_MBUTTONUP == event) {
    is_moving_lat = false;
    return;
  }

  if (cv::EVENT_MOUSEWHEEL == event) {
    std::cout << "mouse_wheel" << std::endl;
    is_moving_lon = true;
  }

  if (cv::EVENT_MOUSEMOVE == event) {
    if (is_rotating) {
      math::Vector3f const ray1{
          camera->getPixelCameraCoordinate(PixCoord{x, y}).Normalized()};
      math::Vector3f const ray2{
          camera->getPixelCameraCoordinate(PixCoord{last_x, last_y})
              .Normalized()};
      math::Vector3f const axis{ray1.Cross(ray2)};
      float const angle{std::acos(ray1.Dot(ray2))};
      if (angle < 1.0e-6) {
        last_x = x;
        last_y = y;
        return;
      }
      math::Quaternionf const dq{GetRotation(axis, angle)};
      camera->setRotation(camera->getRotation() * dq);
      last_x = x;
      last_y = y;

      return;
    }

    if (is_moving_lat) {
      math::Vector3f const pc1{
          camera->getPixelCameraCoordinate(PixCoord{x, y})};
      math::Vector3f const pc2{
          camera->getPixelCameraCoordinate(PixCoord{last_x, last_y})};
      math::Vector3f const dx{camera->getRotation() * (pc2 - pc1) * 3.0f};
      camera->setPosition(camera->getPosition() + dx);
      last_x = x;
      last_y = y;

      return;
    }

    if (is_moving_lon) {
      if (0 < cv::getMouseWheelDelta(flags)) {
        math::Vector3f const dx{camera->getRotation() *
                                math::Vector3f{0.0f, 0.0f, -0.2f}};
        camera->setPosition(camera->getPosition() + dx);
      } else if (0 > cv::getMouseWheelDelta(flags)) {
        math::Vector3f const dx{camera->getRotation() *
                                math::Vector3f{0.0f, 0.0f, 0.2f}};
        camera->setPosition(camera->getPosition() + dx);
      }
      last_x = x;
      last_y = y;
      is_moving_lon = false;
      std::cout << "moving lon" << std::endl;

      return;
    }
  }

  is_controlling = false;
}

int main(int argc, char **argv) {
  std::string const accretion_disk_texture_path{argv[1]};
  LoadAccretionDiskTexture(accretion_disk_texture_path);

  std::string const background_texture_path{argv[2]};
  auto const background_data{
      LoadBackgroundTexture(background_texture_path, false)};
  cuda::HdriSkyBackground background{background_data};

  const float aspect_ratio = 16.0f / 9.0f;
  const size_t image_width = 1280UL;
  const size_t image_height = static_cast<size_t>(image_width / aspect_ratio);
  const float vfov = 70.0f * M_PIf32 / 180.0f;
  const float focal_length = 1.0f;
  const float fov_height = 2.0f * std::tan(vfov / 2.0f) * focal_length;
  const float fov_width =
      fov_height * (static_cast<float>(image_width) / image_height);
  const float defocus_angle = 0.0f;
  Point3D camera_position{8.0f, 0.0f, 0.5f};
  Point3D lookat{0.0f, 0.0f, 0.0f};
  math::Vector3f vup{0.0f, 0.0f, 1.0f};
  math::Quaternionf camera_rotation =
      getCameraRotation(camera_position, lookat, vup);
  cuda::Camera camera(image_width, image_height, fov_width, fov_height,
                      focal_length, defocus_angle, camera_position,
                      camera_rotation);

  cv::Mat image{image_height, image_width, CV_8UC3};
  uint8_t *d__image{};
  CUDA_CHECK(cudaMalloc((void **)&d__image, image_height * image_width * 3));
  std::cout << "image_size: " << image.cols << "x" << image.rows << std::endl;

  std::string const window_name{"BlackHoleRenderer"};
  cv::namedWindow(window_name);
  cv::setMouseCallback(window_name, MouseControlCallback, &camera);

  while (true) {
    cuda::bhlite::render(camera, background, d__disk_tex, image_width,
                         image_height, d__image);
    CUDA_CHECK(cudaMemcpy(image.data, d__image,
                          image_height * image_width * 3UL,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::imshow(window_name, image);
    int const key{cv::waitKey(2)};
    if ('q' == key || 'Q' == key) {
      break;
    }
  }

  background.Release();

  CUDA_CHECK(cudaFree(d__image));

  return 0;
}