#include "Camera.h"

math::Vector3f getRayColor(const Ray &ray) {
  const math::Vector3f &direction = ray.getDirection();
  const float a = 0.5f * (direction.y() + 1.0f);
  const math::Vector3f color1(1.0f, 1.0f, 1.0f);
  const math::Vector3f color2(0.5f, 0.7, 1.0f);

  return (1.0f - a) * color1 + a * color2;
}

int main(int argc, char **argv) {
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

  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  for (size_t row = 0; row < image_height; ++row) {
    std::clog << "\rScanlines remaining: " << (image_height - row) << " "
              << std::flush;
    for (size_t col = 0; col < image_width; ++col) {
      const Ray ray = camera.getRay(PixCoord(col, row));
      const math::Vector3f color = getRayColor(ray);
      writeColorToOStream(std::cout, color);
      std::cout << std::endl;
    }
  }
  std::clog << "\nDone.                  " << std::endl;

  return 0;
}
