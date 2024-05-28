#include "Camera.h"
#include "Sphere.h"
#include "HittableList.h" 

Eigen::Vector3d getRayColor(const Ray &ray, const HittableList &world) {
    HitRecord hit_record; 
    if (world.hit(ray, Intervald::positive(), hit_record)) {
        return 0.5 * Eigen::Vector3d{hit_record.normal.x() + 1.0, 
                                     hit_record.normal.y() + 1.0,
                                     hit_record.normal.z() + 1.0}; 
    }

    const Eigen::Vector3d &direction = ray.getDirection(); 
    const double a = 0.5 * (direction.y() + 1.0); 
    const Eigen::Vector3d color1(1.0, 1.0, 1.0); 
    const Eigen::Vector3d color2(0.5, 0.7, 1.0); 

    return (1.0 - a) * color1 + a * color2; 
}

int main(int argc, char **argv) {
    const double aspect_ratio = 16.0 / 9.0; 
    const size_t image_width = 400; 
    const size_t image_height = static_cast<size_t>(image_width / aspect_ratio); 
    const double fov_height = 2.0; 
    const double fov_width = fov_height * (static_cast<double>(image_width) / image_height); 
    const double focal_length = 1.0; 
    const Point3D camera_position(0, 0, -focal_length); 

    const int samples_per_pixel = 10; 
    const double max_pert = 0.5; 

    const Camera camera(image_width, image_height, 
                        fov_width, fov_height, 
                        focal_length, 
                        0.0, 
                        camera_position); 

    HittableList world; 
    world.add(std::make_shared<Sphere>(Point3D{0.0, 0.0, 1.0}, 0.5)); 
    world.add(std::make_shared<Sphere>(Point3D{0.0, 100.5, 1.0}, 100.0)); 

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n"; 

    RandomNumberGenerator rng; 
    for (size_t row = 0; row < image_height; ++row) {
        std::clog << "\rScanlines remaining: " << (image_height - row) << " " << std::flush; 
        for (size_t col = 0; col < image_width; ++col) {
            Eigen::Vector3d color{0.0, 0.0, 0.0}; 
            for (int i = 0; i < samples_per_pixel; ++i) {
                const Ray ray = camera.getPerturbedRay(PixCoord(col, row), max_pert); 
                color += getRayColor(ray, world); 
            }
            color /= double(samples_per_pixel);
            writeColorToOStream(std::cout, color); 
            std::cout << std::endl; 
        }
    }
    std::clog << "\nDone.                  " << std::endl; 

    return 0; 
}

