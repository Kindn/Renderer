#include "Camera.h"
#include "HittableList.h"
#include "Sphere.h"

math::Vector3f getRayColor(const Ray &ray, const HittableList &world,
                            RandomNumberGenerator &rng, const int &depth) {
  if (depth <= 0) {
    return math::Vector3f::Zero();
  }

  HitRecord hit_record;
  if (world.hit(ray, Intervalf(0.001, inff), hit_record)) {
    const math::Vector3f direction =
        rng.uniformPoint3DOnUnitHemiSphere3D(hit_record.normal);
    return 0.5f *
           getRayColor(Ray(hit_record.p, direction), world, rng, depth - 1);
  }

  const math::Vector3f &direction = ray.getDirection();
  const float a = 0.5f * (-direction.y() + 1.0f);
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
  const int max_depth = 50;

  const int samples_per_pixel = 10;
  const float max_pert = 0.5f;

  const Camera camera(image_width, image_height, fov_width, fov_height,
                      focal_length, 0.0f, camera_position);

  HittableList world;
  world.add(std::make_shared<Sphere>(Point3D{0.0f, 0.0f, 2.0f}, 1.5f));
  world.add(std::make_shared<Sphere>(Point3D{0.0f, 101.5f, 2.0f}, 100));

  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  RandomNumberGenerator rng;
  for (size_t row = 0; row < image_height; ++row) {
    std::clog << "\rScanlines remaining: " << (image_height - row) << " "
              << std::flush;
    for (size_t col = 0; col < image_width; ++col) {
      math::Vector3f color{0.0f, 0.0f, 0.0f};
      for (int i = 0; i < samples_per_pixel; ++i) {
        const Ray ray = camera.getPerturbedRay(PixCoord(col, row), max_pert);
        color += getRayColor(ray, world, rng, max_depth);
      }
      color /= float(samples_per_pixel);
      writeColorToOStream(std::cout, color);
      std::cout << std::endl;
    }
  }
  std::clog << "\nDone.                  " << std::endl;

  return 0;
}
